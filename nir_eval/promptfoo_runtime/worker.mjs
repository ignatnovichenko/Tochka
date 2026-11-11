/** Isolated NDJSON bridge to the pinned Promptfoo evaluator and HTTP provider. */
import readline from 'node:readline';
import { performance } from 'node:perf_hooks';
import { readFileSync } from 'node:fs';
import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';

// Set these before importing Promptfoo; never persist prompts, credentials or logs.
Object.assign(process.env, {
  PROMPTFOO_DISABLE_TELEMETRY: 'true', PROMPTFOO_DISABLE_UPDATE: 'true',
  PROMPTFOO_DISABLE_ERROR_LOG: 'true', PROMPTFOO_DISABLE_SHARING: 'true',
  PROMPTFOO_DISABLE_TEMPLATE_ENV_VARS: 'true', PROMPTFOO_CACHE_ENABLED: 'false',
  PROMPTFOO_DISABLE_VAR_EXPANSION: 'true',
  PROMPTFOO_LOG_TO_STDERR: 'true', LOG_LEVEL: 'error',
  // Per-request AbortSignals below enforce individual deadlines without racing env.
  REQUEST_TIMEOUT_MS: '3600000',
});
const protocolWrite = process.stdout.write.bind(process.stdout);
process.stdout.write = () => true;
process.stderr.write = () => true;
for (const method of ['log', 'error', 'warn', 'info', 'debug']) console[method] = () => {};

function emit(value) {
  let encoded = JSON.stringify(value);
  if (Buffer.byteLength(encoded) > 2_000_000) {
    if (value.event === 'result') {
      encoded = JSON.stringify({ id: value.id, event: 'result', job_id: value.job_id,
        latency_seconds: value.latency_seconds,
        result: { id: value.id, ok: false, error: 'api_response_too_large',
          attempted: value.result?.attempted, started_at_utc: value.result?.started_at_utc } });
    } else {
      encoded = JSON.stringify({ id: value.id, ok: false, error: 'api_response_too_large' });
    }
  }
  protocolWrite(encoded + '\n');
}

// HTTP provider 0.125.1 ignores config.redirect. Isolate each concurrent call's
// response status, size limit and timeout so neither credentials nor 429s cross jobs.
const requestContext = new AsyncLocalStorage();
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, options = {}) => {
  const state = requestContext.getStore();
  const signal = state ? (options.signal ? AbortSignal.any([options.signal, state.signal]) : state.signal) : options.signal;
  const response = await nativeFetch(url, { ...options, signal, redirect: 'error' });
  if (state) state.observedHttp = { status: response.status, retry_after: response.headers.get('retry-after') };
  let bytes = 0;
  const body = response.body?.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      bytes += chunk.byteLength;
      if (bytes > 1024 * 1024) {
        if (state) state.oversizedResponse = true;
        throw new Error('response_size_limit');
      }
      controller.enqueue(chunk);
    },
  }));
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
};

const VERSION = '0.125.1';
let runtime;
let runtimeError;
try {
  const installed = JSON.parse(readFileSync(new URL('./node_modules/promptfoo/package.json', import.meta.url), 'utf8'));
  if (installed.version !== VERSION) throw new Error('version');
  runtime = await import('promptfoo');
  runtime.cache.disableCache();
} catch {
  runtimeError = 'promptfoo_runtime_unavailable';
}

function apiErrorCode(status, raw) {
  if (status === 400 && typeof raw === 'string') {
    const message = raw.slice(0, 16384).toLowerCase();
    if (message.includes('hosted outside europe and the united states')) return 'api_http_400_region_restricted';
    if (message.includes('trains on request data')) return 'api_http_400_training_restricted';
  }
  return `api_http_${status}`;
}

function validateRequest(request) {
  if (!request || !Array.isArray(request.messages) || typeof request.model !== 'string'
      || typeof request.base_url !== 'string' || !Number.isFinite(request.timeout_seconds)
      || request.timeout_seconds <= 0 || request.timeout_seconds > 3600
      || (request.api_key != null && typeof request.api_key !== 'string')
      || (request.parameters != null && (typeof request.parameters !== 'object' || Array.isArray(request.parameters)))) return false;
  if (request.blocked_until_ms !== undefined && (!Number.isFinite(request.blocked_until_ms) || request.blocked_until_ms < 0)) return false;
  if (Object.keys(request.parameters || {}).some((key) => ['model', 'messages', 'stream', 'tools', 'tool_choice'].includes(key))) return false;
  if (typeof request.session_id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(request.session_id)) return false;
  try {
    const endpoint = new URL(request.base_url);
    return ['http:', 'https:'].includes(endpoint.protocol) && !endpoint.username && !endpoint.password
      && !endpoint.search && !endpoint.hash && !/[{}]/.test(request.base_url);
  } catch { return false; }
}

async function execute(request) {
  const id = typeof request?.id === 'string' || typeof request?.id === 'number' ? request.id : null;
  const fail = (error, extra = {}) => ({ id, ok: false, error, ...extra });
  if (runtimeError) return fail(runtimeError);
  if (request?.op === 'health') return { id, ok: true, engine: 'promptfoo', version: VERSION,
    scheduler: 'promptfoo.evaluate' };
  if (!validateRequest(request)) return fail('api_invalid_request');
  const endpoint = new URL(request.base_url);
  endpoint.pathname = endpoint.pathname.replace(/\/$/, '') + '/chat/completions';
  const body = { ...request.parameters, model: request.model, messages: request.messages, stream: false };
  const headers = { 'Content-Type': 'application/json', 'User-Agent': 'Tochka/0.5.0',
    'x-opencode-session': request.session_id };
  if (request.api_key) headers.Authorization = 'Bearer {{api_key}}';
  const state = { signal: AbortSignal.timeout(Math.ceil(request.timeout_seconds * 1000)),
    observedHttp: null, oversizedResponse: false };
  const start = performance.now();
  return requestContext.run(state, async () => {
    try {
      const provider = await runtime.loadApiProvider(endpoint.href, { options: { config: {
        method: 'POST', headers, body: '{{prompt}}', maxRetries: 0,
        redirect: 'error', validateStatus: () => true,
        transformResponse: (json) => ({ output: json }),
      } } });
      const result = await provider.callApi(JSON.stringify(body), {
        vars: { api_key: request.api_key || '' }, bustCache: true,
      });
      const status = result.metadata?.http?.status;
      if (Number.isInteger(status) && (status < 200 || status >= 300)) {
        const retryAfter = result.metadata?.http?.headers?.['retry-after'];
        return fail(apiErrorCode(status, result.raw), { status, retry_after: typeof retryAfter === 'string' ? retryAfter : null });
      }
      if (result.error || !Number.isInteger(status)) {
        return fail(state.signal.aborted ? 'api_timeout' : (state.oversizedResponse ? 'api_response_too_large' : 'api_connection_unavailable'));
      }
      if (typeof result.raw !== 'string') return fail('api_invalid_json');
      return { id, ok: true, raw_response: result.raw, engine: 'promptfoo', version: VERSION,
        latency_ms: performance.now() - start };
    } catch (error) {
      if (state.observedHttp && (state.observedHttp.status < 200 || state.observedHttp.status >= 300)) {
        return fail(`api_http_${state.observedHttp.status}`, state.observedHttp);
      }
      if (state.oversizedResponse) return fail('api_response_too_large');
      return fail(state.signal.aborted || error?.name === 'TimeoutError' || error?.name === 'AbortError'
        ? 'api_timeout' : 'api_connection_unavailable');
    }
  });
}

function cooldownSeconds(retryAfter) {
  let value = typeof retryAfter === 'string' && retryAfter.trim() ? Number(retryAfter) : NaN;
  if (!Number.isFinite(value) && typeof retryAfter === 'string') value = (Date.parse(retryAfter) - Date.now()) / 1000;
  return Math.max(1, Math.min(Number.isFinite(value) ? value : 60, 7 * 86400));
}

async function evaluateMatrix(request) {
  const id = request.id;
  const fail = (error) => ({ id, ok: false, event: 'complete', error, engine: 'promptfoo',
    version: VERSION, scheduler: 'promptfoo.evaluate' });
  if (runtimeError) return fail(runtimeError);
  if (!Array.isArray(request.jobs) || request.jobs.length > 10000
      || !Number.isInteger(request.max_concurrency) || request.max_concurrency < 1 || request.max_concurrency > 32) return fail('api_invalid_matrix');
  const jobIds = new Set();
  for (const job of request.jobs) {
    if (!validateRequest(job) || typeof job.job_id !== 'string' || !job.job_id || job.job_id.length > 256
        || typeof job.provider_id !== 'string' || !job.provider_id || job.provider_id.length > 256
        || jobIds.has(job.job_id)) return fail('api_invalid_matrix');
    jobIds.add(job.job_id);
  }
  if (!request.jobs.length) return { id, ok: true, event: 'complete', engine: 'promptfoo', version: VERSION,
    scheduler: 'promptfoo.evaluate', completed: 0, test_count: 0, total_calls: 0, max_active_requests: 0 };
  const jobs = new Map(request.jobs.map((job, index) => [`j${index}`, job]));
  const groups = new Map();
  const completed = new Set();
  const dispatched = new Set();
  const cooldowns = new Map();
  let activeRequests = 0;
  let maxActiveRequests = 0;
  let totalRequests = 0;
  for (const job of request.jobs) {
    if (job.blocked_until_ms) {
      const key = createHash('sha256').update(JSON.stringify([job.base_url.replace(/\/$/, ''), job.api_key || ''])).digest('hex');
      cooldowns.set(key, Math.max(cooldowns.get(key) || 0, job.blocked_until_ms));
    }
  }
  for (const job of request.jobs) if (!groups.has(job.provider_id)) groups.set(job.provider_id, `tochka-provider-${groups.size}`);
  const providers = [...groups].map(([providerId, opaqueId]) => ({
    id: () => opaqueId,
    label: opaqueId,
    config: { maxRetries: 0 },
    handlesOwnRetries: true,
    async callApi(_prompt, context) {
      const job = jobs.get(context.vars.job_key);
      if (!job || job.provider_id !== providerId || dispatched.has(job.job_id)) return { error: 'tochka_matrix_identity_error' };
      dispatched.add(job.job_id);
      const start = performance.now();
      // The key never enters provider IDs, test vars, PF metadata, or persisted files.
      const cooldownKey = createHash('sha256').update(JSON.stringify([job.base_url.replace(/\/$/, ''), job.api_key || ''])).digest('hex');
      const blockedUntil = cooldowns.get(cooldownKey) || 0;
      let result;
      if (blockedUntil > Date.now()) {
        result = { id, ok: false, attempted: false, error: 'api_cooldown_active', retry_after_seconds: (blockedUntil - Date.now()) / 1000 };
      } else {
        const startedAt = new Date().toISOString();
        emit({ id, event: 'started', job_id: job.job_id, started_at_utc: startedAt });
        activeRequests++;
        maxActiveRequests = Math.max(maxActiveRequests, activeRequests);
        totalRequests++;
        try {
          result = { ...await execute({ ...job, id }), attempted: true, started_at_utc: startedAt };
        } finally { activeRequests--; }
        if (result.status === 429) {
          result.retry_after_seconds = cooldownSeconds(result.retry_after);
          cooldowns.set(cooldownKey, Date.now() + result.retry_after_seconds * 1000);
        }
      }
      completed.add(job.job_id);
      emit({ id, event: 'result', job_id: job.job_id, result,
        latency_seconds: (performance.now() - start) / 1000 });
      // PF schedules the matrix; Tochka interprets its own result/quality contract.
      // No raw API body, secrets or invented numeric grade enter PF's result store.
      return result.ok ? { output: 'tochka-response-completed' } : { error: result.error || 'tochka-provider-error' };
    },
  }));
  const tests = [...jobs].map(([jobKey, job]) => ({
    vars: { job_key: jobKey }, providers: [groups.get(job.provider_id)],
    options: { disableVarExpansion: true },
  }));
  try {
    await runtime.evaluate({ prompts: ['{{job_key}}'], providers, tests,
      sharing: false, writeLatestResults: false, tracing: { enabled: false } }, {
      maxConcurrency: request.max_concurrency, cache: false, repeat: 1,
      showProgressBar: false, silent: true, timeoutMs: 0,
    });
    if (completed.size !== request.jobs.length) return fail('promptfoo_matrix_incomplete');
    return { id, ok: true, event: 'complete', engine: 'promptfoo', version: VERSION,
      scheduler: 'promptfoo.evaluate', completed: completed.size, test_count: request.jobs.length,
      total_calls: totalRequests, max_active_requests: maxActiveRequests };
  } catch {
    return fail('promptfoo_evaluation_failed');
  }
}

for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  let result;
  try {
    if (Buffer.byteLength(line) > 64_000_000) throw new Error('request_size_limit');
    const request = JSON.parse(line);
    result = request?.op === 'evaluate' ? await evaluateMatrix(request) : await execute(request);
  } catch { result = { id: null, ok: false, error: 'api_invalid_request' }; }
  emit(result);
}
