# NIR evaluation report

Only rows marked model-inference represent actual model calls.

| Model | Config | Source | N | Correct | Accuracy | Wilson 95% CI | p50 s | Errors |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | direct | model-inference | 1 | 0 | 0.0% | 0.0%–79.3% | 10.3352 | 0 |

## Interpretation

- All attempted cases, including provider and parse errors, are in the accuracy denominator.
- Displayed 95% Wilson interval uses the first repetition for each distinct task; it does not count repeats as extra tasks.
- Additional 95% percentile bootstrap resamples tasks (2000 draws, seed 42); repeats are not independent tasks.
- Small pilots and all-zero/all-one samples give limited, sometimes degenerate intervals, not proof of certainty.
- Program exact match is token equality, not official symbolic program accuracy.
- No hidden LLM judge is used. Missing cost, quantization, TTFT and usage stay unknown.
- Control and replay runs are not model-quality or model-latency measurements.
