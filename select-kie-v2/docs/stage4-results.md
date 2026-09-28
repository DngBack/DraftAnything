# Stage 4 results

| Condition | Official accuracy | Normalized | High-confidence errors/decisions |
|---|---:|---:|---:|
| qwen2b-baseline | 28.17% | 29.17% | 31/65 |
| qwen2b-length_control | 26.83% | 27.67% | 35/70 |
| qwen2b-clarified | 70.00% | 72.17% | 11/153 |
| qwen9b-baseline | 50.67% | 52.67% | 71/251 |
| qwen9b-length_control | 49.50% | 51.67% | 83/266 |
| qwen9b-clarified | 76.00% | 79.50% | 23/308 |

Paired document bootstrap95% intervals:

- qwen2b-clarified minus qwen2b-length_control: +43.17 percentage points; CI [+40.00, +46.67].
- qwen9b-clarified minus qwen9b-length_control: +26.50 percentage points; CI [+23.33, +29.50].
- qwen2b-length_control minus qwen2b-baseline: -1.33 percentage points; CI [-3.33, +0.67].
- qwen9b-length_control minus qwen9b-baseline: -1.17 percentage points; CI [-2.50, +0.33].
- qwen9b-baseline minus qwen2b-baseline: +22.50 percentage points; CI [+19.00, +26.17].
- qwen9b-length_control minus qwen2b-length_control: +22.67 percentage points; CI [+19.17, +26.50].
- qwen9b-clarified minus qwen2b-clarified: +6.00 percentage points; CI [+3.50, +8.50].

Human audit remains pending. 9B batch4 encountered a multi-page memory failure and resumed batch2 without reducing pages or resolution; failed attempts, loading and rendering are excluded from generate-only timings. The targeted stratum cannot estimate population binding prevalence; use the random100 separately. These are scale and schema diagnostics within one model family, not a new relational architecture or evidence of a guaranteed CVPR score.
