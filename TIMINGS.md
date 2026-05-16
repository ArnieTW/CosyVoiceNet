# CosyVoiceNet Optimization Profile Timing Matrix

These timings are practical reference points, not universal guarantees. Exact
performance depends on hardware, drivers, TorchSharp/native library versions,
optimization profile, text length, selected endpoint, prompt audio length, and
whether a model is already warm.

## Hardware

- CPU: AMD Ryzen 9 5900X 12-Core Processor, 24 logical processors.
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, driver 595.79.
- Runtime: Windows x64, .NET `net10.0`, TorchSharp `0.106.0`.
- Benchmark date: 2026-05-16.

Generation-only timings are in seconds. Model load and first-time saved-voice
clone time are excluded.

Text:

```text
A comprehensive test to compare the TTS pipeline between original CosyVoice Python pipeline and the new C# Pipeline
```

Prompt WAV: `CosyVoiceNet\asset\zero_shot_prompt.wav` (3.48 s). Clonable
endpoints use this WAV; SFT/instruct-only endpoints use the first available
built-in speaker.

## Compatibility

| Model | Endpoint | Speaker | CPU 1 thread | CPU 8 threads | CUDA | Audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Fun-CosyVoice3-0.5B` | `zero_shot` |  | 96.792 | 42.341 | 8.745 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `cross_lingual` |  | 64.064 | 27.758 | 5.523 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `instruct2` |  | 72.963 | 31.720 | 5.877 | 8.360 |
| `Fun-CosyVoice3-0.5B` | `saved_zero_shot` |  | 98.772 | 42.114 | 6.870 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `saved_cross_lingual` |  | 63.700 | 26.775 | 4.881 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `saved_instruct2` |  | 72.812 | 31.067 | 5.432 | 8.360 |
| `CosyVoice2-0.5B` | `zero_shot` |  | 72.313 | 37.109 | 8.480 | 11.680 |
| `CosyVoice2-0.5B` | `cross_lingual` |  | 69.426 | 37.727 | 7.789 | 12.040 |
| `CosyVoice2-0.5B` | `instruct2` |  | 42.421 | 23.208 | 5.999 | 7.320 |
| `CosyVoice2-0.5B` | `saved_zero_shot` |  | 71.121 | 37.387 | 8.449 | 11.680 |
| `CosyVoice2-0.5B` | `saved_cross_lingual` |  | 69.635 | 37.987 | 7.310 | 12.040 |
| `CosyVoice2-0.5B` | `saved_instruct2` |  | 42.084 | 23.201 | 5.647 | 7.320 |
| `CosyVoice-300M` | `zero_shot` |  | 141.827 | 54.710 | 6.856 | 7.755 |
| `CosyVoice-300M` | `cross_lingual` |  | 107.417 | 45.151 | 6.777 | 8.150 |
| `CosyVoice-300M` | `saved_zero_shot` |  | 141.908 | 55.348 | 6.689 | 7.755 |
| `CosyVoice-300M` | `saved_cross_lingual` |  | 108.027 | 44.514 | 6.278 | 8.150 |
| `CosyVoice-300M-SFT` | `sft` | 中文女 | 71.799 | 32.534 | 5.148 | 6.815 |
| `CosyVoice-300M-Instruct` | `instruct` | 中文女 | 52.972 | 24.561 | 4.737 | 5.271 |

## Balanced

| Model | Endpoint | Speaker | CPU 1 thread | CPU 8 threads | CUDA | Audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Fun-CosyVoice3-0.5B` | `zero_shot` |  | 101.123 | 37.395 | 6.788 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `cross_lingual` |  | 64.397 | 23.454 | 4.814 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `instruct2` |  | 74.018 | 27.344 | 5.065 | 8.360 |
| `Fun-CosyVoice3-0.5B` | `saved_zero_shot` |  | 97.828 | 36.412 | 5.848 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `saved_cross_lingual` |  | 63.022 | 22.939 | 4.253 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `saved_instruct2` |  | 70.917 | 26.644 | 4.569 | 8.360 |
| `CosyVoice2-0.5B` | `zero_shot` |  | 69.485 | 32.710 | 7.235 | 11.680 |
| `CosyVoice2-0.5B` | `cross_lingual` |  | 66.624 | 31.927 | 6.569 | 12.040 |
| `CosyVoice2-0.5B` | `instruct2` |  | 39.745 | 20.176 | 5.092 | 7.320 |
| `CosyVoice2-0.5B` | `saved_zero_shot` |  | 66.078 | 32.190 | 7.111 | 11.680 |
| `CosyVoice2-0.5B` | `saved_cross_lingual` |  | 63.573 | 31.787 | 6.219 | 12.040 |
| `CosyVoice2-0.5B` | `saved_instruct2` |  | 39.181 | 20.162 | 4.744 | 7.320 |
| `CosyVoice-300M` | `zero_shot` |  | 123.747 | 37.808 | 5.967 | 7.755 |
| `CosyVoice-300M` | `cross_lingual` |  | 96.983 | 32.612 | 5.966 | 8.150 |
| `CosyVoice-300M` | `saved_zero_shot` |  | 125.958 | 37.412 | 5.734 | 7.755 |
| `CosyVoice-300M` | `saved_cross_lingual` |  | 95.645 | 31.911 | 5.519 | 8.150 |
| `CosyVoice-300M-SFT` | `sft` | 中文女 | 62.991 | 23.949 | 4.569 | 6.815 |
| `CosyVoice-300M-Instruct` | `instruct` | 中文女 | 46.412 | 18.083 | 4.157 | 5.271 |

## Throughput

| Model | Endpoint | Speaker | CPU 1 thread | CPU 8 threads | CUDA | Audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Fun-CosyVoice3-0.5B` | `zero_shot` |  | 89.482 | 37.066 | 6.846 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `cross_lingual` |  | 59.157 | 23.134 | 4.776 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `instruct2` |  | 66.716 | 26.896 | 5.099 | 8.360 |
| `Fun-CosyVoice3-0.5B` | `saved_zero_shot` |  | 87.715 | 35.845 | 5.837 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `saved_cross_lingual` |  | 58.006 | 23.154 | 4.217 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `saved_instruct2` |  | 66.227 | 26.741 | 4.569 | 8.360 |
| `CosyVoice2-0.5B` | `zero_shot` |  | 64.514 | 32.910 | 7.247 | 11.680 |
| `CosyVoice2-0.5B` | `cross_lingual` |  | 63.197 | 31.956 | 6.599 | 12.040 |
| `CosyVoice2-0.5B` | `instruct2` |  | 38.966 | 20.106 | 5.041 | 7.320 |
| `CosyVoice2-0.5B` | `saved_zero_shot` |  | 64.465 | 33.183 | 7.109 | 11.680 |
| `CosyVoice2-0.5B` | `saved_cross_lingual` |  | 62.535 | 32.007 | 6.233 | 12.040 |
| `CosyVoice2-0.5B` | `saved_instruct2` |  | 38.777 | 22.043 | 4.790 | 7.320 |
| `CosyVoice-300M` | `zero_shot` |  | 122.910 | 48.325 | 5.929 | 7.755 |
| `CosyVoice-300M` | `cross_lingual` |  | 96.258 | 36.489 | 5.956 | 8.150 |
| `CosyVoice-300M` | `saved_zero_shot` |  | 126.357 | 39.861 | 5.771 | 7.755 |
| `CosyVoice-300M` | `saved_cross_lingual` |  | 94.662 | 32.081 | 5.472 | 8.150 |
| `CosyVoice-300M-SFT` | `sft` | 中文女 | 63.724 | 23.926 | 4.606 | 6.815 |
| `CosyVoice-300M-Instruct` | `instruct` | 中文女 | 46.139 | 18.014 | 4.084 | 5.271 |

## LowMemory

| Model | Endpoint | Speaker | CPU 1 thread | CPU 8 threads | CUDA | Audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Fun-CosyVoice3-0.5B` | `zero_shot` |  | 89.197 | 37.283 | 6.701 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `cross_lingual` |  | 58.576 | 23.528 | 4.812 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `instruct2` |  | 66.693 | 27.337 | 5.093 | 8.360 |
| `Fun-CosyVoice3-0.5B` | `saved_zero_shot` |  | 87.645 | 36.110 | 5.857 | 11.160 |
| `Fun-CosyVoice3-0.5B` | `saved_cross_lingual` |  | 57.662 | 22.997 | 4.219 | 7.160 |
| `Fun-CosyVoice3-0.5B` | `saved_instruct2` |  | 66.011 | 26.661 | 4.572 | 8.360 |
| `CosyVoice2-0.5B` | `zero_shot` |  | 63.932 | 32.959 | 7.214 | 11.680 |
| `CosyVoice2-0.5B` | `cross_lingual` |  | 63.340 | 31.982 | 6.525 | 12.040 |
| `CosyVoice2-0.5B` | `instruct2` |  | 39.130 | 20.041 | 5.105 | 7.320 |
| `CosyVoice2-0.5B` | `saved_zero_shot` |  | 64.675 | 32.050 | 7.196 | 11.680 |
| `CosyVoice2-0.5B` | `saved_cross_lingual` |  | 63.266 | 31.701 | 6.202 | 12.040 |
| `CosyVoice2-0.5B` | `saved_instruct2` |  | 38.936 | 19.758 | 4.712 | 7.320 |
| `CosyVoice-300M` | `zero_shot` |  | 123.353 | 37.614 | 5.992 | 7.755 |
| `CosyVoice-300M` | `cross_lingual` |  | 94.907 | 32.711 | 5.983 | 8.150 |
| `CosyVoice-300M` | `saved_zero_shot` |  | 124.020 | 37.685 | 5.767 | 7.755 |
| `CosyVoice-300M` | `saved_cross_lingual` |  | 94.540 | 32.191 | 5.415 | 8.150 |
| `CosyVoice-300M-SFT` | `sft` | 中文女 | 62.413 | 23.858 | 4.604 | 6.815 |
| `CosyVoice-300M-Instruct` | `instruct` | 中文女 | 46.384 | 18.491 | 4.051 | 5.271 |
