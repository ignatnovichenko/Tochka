/** Isolated NDJSON bridge to the pinned promptfoo HTTP provider. */
import readline from 'node:readline';
import { performance } from 'node:perf_hooks';
import { readFileSync } from 'node:fs';

// Set these before importing promptfoo; never persist prompts, credentials or logs.
Object.assign(process.env, {
  PROMPTFOO_DISABLE_TELEMETRY: 'true', PROMPTFOO_DISABLE_UPDATE: 'true',
  PROMPTFOO_DISABLE_ERROR_LOG: 'true', PROMPTFOO_DISABLE_SHARING: 'true',
  PROMPTFOO_DISABLE_TEMPLATE_ENV_VARS: 'true', PROMPTFOO_CACHE_ENABLED: 'false',
  PROMPTFOO_LOG_TO_STDERR: 'true', LOG_LEVEL: 'error',
});
// Only our protocol writer can emit stdout. Dependency diagnostics may contain bodies.
const protocolWrite = process.stdout.write.bind(process.stdout);
process.stdout.write = () => true;
process.stderr.write = () => true;
for (const method of ['log', 'error', 'warn', 'info', 'debug']) console[method] = () => {};

// 0.125.1's HTTP provider does not forward config.redirect to fetch.
const nativeFetch = globalThis.fetch;
let observedHttp = null;
let oversizedResponse = false;
globalThis.fetch = async (url, options = {}) => {
  const response = await nativeFetch(url, { ...options, redirect: 'error' });
  observedHttp = { status: response.status, retry_after: response.headers.get('retry-after') };
  let bytes = 0;
  const body = response.body?.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      bytes += chunk.byteLength;
      if (bytes > 1024 * 1024) {
        oversizedResponse = true;
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

async function execute(request) {
  const id = typeof request?.id === 'string' || typeof request?.id === 'number' ? request.id : null;
  const fail = (error, extra = {}) => ({ id, ok: false, error, ...extra });
  if (runtimeError) return fail(runtimeError);
  if (request?.op === 'health') return { id, ok: true, engine: 'promptfoo', version: VERSION };
  if (!request || !Array.isArray(request.messages) || typeof request.model !== 'string'
      || typeof request.base_url !== 'string' || !Number.isFinite(request.timeout_seconds)
      || request.timeout_seconds <= 0 || request.timeout_seconds > 86400
      || (request.api_key != null && typeof request.api_key !== 'string')
      || (request.parameters != null && (typeof request.parameters !== 'object' || Array.isArray(request.parameters)))) {
    return fail('api_invalid_request');
  }
  let endpoint;
  try {
    endpoint = new URL(request.base_url);
    if (!['http:', 'https:'].includes(endpoint.protocol) || endpoint.username || endpoint.password
        || endpoint.search || endpoint.hash || /[{}]/.test(request.base_url)) return fail('api_invalid_request');
    endpoint.pathname = endpoint.pathname.replace(/\/$/, '') + '/chat/completions';
  } catch { return fail('api_invalid_request'); }
  process.env.REQUEST_TIMEOUT_MS = String(Math.ceil(request.timeout_seconds * 1000));
  const body = { ...request.parameters, model: request.model, messages: request.messages, stream: false };
  const headers = { 'Content-Type': 'application/json', 'User-Agent': 'Tochka/0.4.0' };
  if (typeof request.session_id !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(request.session_id)) {
    return fail('api_invalid_request');
  }
  headers['x-opencode-session'] = request.session_id;
  // Supply the key as a variable so literal braces in a credential are not templated.
  if (request.api_key) headers.Authorization = 'Bearer {{api_key}}';
  observedHttp = null;
  oversizedResponse = false;
  const start = performance.now();
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
    if (result.error || !Number.isInteger(status)) return fail('api_connection_unavailable');
    // Ignore PF normalized/estimated usage and cost. Python checks raw provider evidence.
    if (typeof result.raw !== 'string') return fail('api_invalid_json');
    return { id, ok: true, raw_response: result.raw, engine: 'promptfoo', version: VERSION,
      latency_ms: performance.now() - start };
  } catch (error) {
    // PF's fetch layer throws for 429 even with validateStatus:true.
    if (observedHttp && (observedHttp.status < 200 || observedHttp.status >= 300)) {
      return fail(`api_http_${observedHttp.status}`, observedHttp);
    }
    if (oversizedResponse) return fail('api_response_too_large');
    const isTimeout = error?.name === 'TimeoutError' || error?.name === 'AbortError';
    return fail(isTimeout ? 'api_timeout' : 'api_connection_unavailable');
  }
}

for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  let result;
  try { result = await execute(JSON.parse(line)); }
  catch { result = { id: null, ok: false, error: 'api_invalid_request' }; }
  let encoded = JSON.stringify(result);
  if (Buffer.byteLength(encoded) > 2 * 1024 * 1024) {
    encoded = JSON.stringify({ id: result.id, ok: false, error: 'api_response_too_large' });
  }
  protocolWrite(encoded + '\n');
}
