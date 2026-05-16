# CosyVoiceNet Timings

These timings are provided as practical reference points, not universal
guarantees. Exact performance depends on hardware, drivers, TorchSharp/native
library versions, optimization profile, text length, selected endpoint, prompt
audio length, and whether a model is already warm.

## Hardware

- CPU: AMD Ryzen 9 5900X 12-Core Processor, 24 logical processors.
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, driver 595.79.
- Runtime: Windows x64, .NET `net10.0`, TorchSharp `0.106.0`.
- Benchmark date: 2026-05-15 and 2026-05-16.

## Generation-Only Timings

Generation-only timings in seconds. Model load time and first-time saved-voice
clone time are excluded. CPU 1 thread and CPU 8 threads are explicit runtime
thread settings. CUDA uses the CUDA backend from the default Windows build.

Test text:

```text
A comprehensive test to compare the TTS pipeline between original CosyVoice Python pipeline and the new C# Pipeline
```

TTS characters: 115. Prompt WAV:
`CosyVoiceNet\asset\zero_shot_prompt.wav`, 3.48 seconds. Speaker policy:
first available SFT/instruct speaker for non-clonable voices, and
`zero_shot_prompt.wav` for clonable endpoints.

| Model | Endpoint | Speaker | CPU 1 thread | CPU 8 threads | CUDA | Audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Fun-CosyVoice3-0.5B` | `zero_shot` |  | 103.591 | 69.280 | 13.472 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `cross_lingual` |  | 65.653 | 40.051 | 45.774 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `instruct2` |  | 83.447 | 49.589 | 48.439 | 8.360 |
| `Fun-CosyVoice3-0.5B` | `saved_zero_shot` |  | 127.600 | 65.133 | 60.687 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `saved_cross_lingual` |  | 84.272 | 40.208 | 48.688 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `saved_instruct2` |  | 95.728 | 46.970 | 51.667 | 8.360 |
| `CosyVoice2-0.5B` | `zero_shot` |  | 97.715 | 62.149 | 73.678 | 11.680 |
| `CosyVoice2-0.5B` | `cross_lingual` |  | 90.492 | 57.296 | 66.858 | 12.040 |
| `CosyVoice2-0.5B` | `instruct2` |  | 52.317 | 38.133 | 67.619 | 7.320 |
| `CosyVoice2-0.5B` | `saved_zero_shot` |  | 86.728 | 62.527 | 94.495 | 11.680 |
| `CosyVoice2-0.5B` | `saved_cross_lingual` |  | 87.285 | 66.696 | 78.212 | 12.040 |
| `CosyVoice2-0.5B` | `saved_instruct2` |  | 62.121 | 38.379 | 69.090 | 7.320 |
| `CosyVoice-300M` | `zero_shot` |  | 172.879 | 74.114 | 204.735 | 7.755 |
| `CosyVoice-300M` | `cross_lingual` |  | 132.209 | 62.094 | 178.128 | 8.150 |
| `CosyVoice-300M` | `saved_zero_shot` |  | 178.791 | 66.419 | 216.309 | 7.755 |
| `CosyVoice-300M` | `saved_cross_lingual` |  | 133.008 | 56.394 | 188.426 | 8.150 |
| `CosyVoice-300M-SFT` | `sft` | `中文女` | 89.488 | 44.531 | 71.783 | 6.815 |
| `CosyVoice-300M-Instruct` | `instruct` | `中文女` | 67.671 | 32.705 | 43.771 | 5.271 |

## Repeat Generation Trend

This test checks whether repeated generation gets slower when the same model
instance remains loaded. Each row is five generations with no model reload
between repeats.

- Optimization profile: `Balanced`.
- CPU path: 8 Torch threads, 1 interop thread.
- CUDA path: CUDA backend.
- Repeats: 5 generations per loaded model.
- Verdict is based primarily on run 2 to run 5, so first-call warmup does not
  hide steady-state drift.

| Model | Endpoint | Backend | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Average | Last vs run 2 | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Fun-CosyVoice3-0.5B` | `zero_shot` | CPU 8 threads | 41.657 | 38.834 | 38.678 | 38.716 | 38.526 | 39.282 | -0.308 (-0.8%) | No increasing trend |
| `Fun-CosyVoice3-0.5B` | `zero_shot` | CUDA | 7.589 | 6.054 | 6.031 | 6.077 | 6.028 | 6.356 | -0.026 (-0.4%) | No increasing trend |
| `CosyVoice2-0.5B` | `zero_shot` | CPU 8 threads | 35.276 | 34.257 | 34.231 | 34.534 | 34.102 | 34.480 | -0.155 (-0.5%) | No increasing trend |
| `CosyVoice2-0.5B` | `zero_shot` | CUDA | 7.846 | 7.596 | 7.431 | 7.452 | 7.428 | 7.551 | -0.167 (-2.2%) | No increasing trend |
| `CosyVoice-300M` | `zero_shot` | CPU 8 threads | 43.070 | 42.895 | 42.748 | 43.020 | 42.283 | 42.803 | -0.611 (-1.4%) | No increasing trend |
| `CosyVoice-300M` | `zero_shot` | CUDA | 6.654 | 6.028 | 5.931 | 5.978 | 5.993 | 6.117 | -0.035 (-0.6%) | No increasing trend |
| `CosyVoice-300M-SFT` | `sft` | CPU 8 threads | 27.450 | 26.725 | 27.088 | 26.695 | 26.674 | 26.926 | -0.051 (-0.2%) | No increasing trend |
| `CosyVoice-300M-SFT` | `sft` | CUDA | 4.784 | 4.762 | 4.667 | 4.667 | 4.710 | 4.718 | -0.052 (-1.1%) | No increasing trend |
| `CosyVoice-300M-Instruct` | `instruct` | CPU 8 threads | 20.419 | 21.611 | 21.890 | 20.664 | 20.895 | 21.096 | -0.716 (-3.3%) | No increasing trend |
| `CosyVoice-300M-Instruct` | `instruct` | CUDA | 4.196 | 4.092 | 4.096 | 4.086 | 4.145 | 4.123 | 0.053 (1.3%) | No increasing trend |

In this run, generation did not get progressively slower for any tested model.
The first generation is often slower due to warmup, then steady-state timings
stay flat within normal noise.
