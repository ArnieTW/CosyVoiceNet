# CosyVoiceNet

CosyVoiceNet is a C#/.NET port and integration layer for the
[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) text-to-speech
pipeline. It is based on the original CosyVoice Python implementation, but has
been adjusted for .NET usage, lazy model downloads, high-level application
integration, CPU/CUDA backend selection, saved voice reuse, and runtime
optimization profiles.

The goal is simple: external apps should be able to request CosyVoice capability
through one C# facade, without requiring Python at runtime.

## What It Does

- Runs CosyVoice models from .NET through TorchSharp and the local C# pipeline.
- Supports CPU and CUDA backend selection from a single CUDA-capable build.
- Lets callers set a global backend and override backend per model/request.
- Auto-downloads a model only when that specific model is requested.
- Exposes model capabilities without initializing model weights.
- Supports zero-shot, cross-lingual, instruct, instruct2, SFT, and saved voices
  depending on the selected model.
- Can clone a prompt WAV into a named saved voice, then generate future TTS by
  voice name.
- Returns generated audio as bytes. CosyVoiceNet does not decide where external
  apps store WAV files.
- Uses `CosyVoiceRuntimeOptions` for logging, profiling, thread counts, cache
  strategy, affinity, and optimization behavior.

## Supported Models

| Model alias | Local model | Upstream model | Main modes |
| --- | --- | --- | --- |
| `cosyvoice3` | `Fun-CosyVoice3-0.5B` | [`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | zero-shot, cross-lingual, instruct2, saved voices |
| `cosyvoice2` | `CosyVoice2-0.5B` | [`FunAudioLLM/CosyVoice2-0.5B`](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | zero-shot, cross-lingual, instruct2, saved voices |
| `cosyvoice` | `CosyVoice-300M` | [`FunAudioLLM/CosyVoice-300M`](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | zero-shot, cross-lingual, saved voices |
| `sft` | `CosyVoice-300M-SFT` | [`FunAudioLLM/CosyVoice-300M-SFT`](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) | built-in SFT speakers |
| `instruct` | `CosyVoice-300M-Instruct` | [`FunAudioLLM/CosyVoice-300M-Instruct`](https://huggingface.co/FunAudioLLM/CosyVoice-300M-Instruct) | original 300M instruct route |

The model registry lives in `CosyVoiceNet.cli.CosyVoiceModels`. The downloader is
lazy: asking for `cosyvoice3` does not download every other model.

## Build

CosyVoiceNet targets `net10.0`.

```powershell
dotnet restore CosyVoiceNet\CosyVoiceNet.csproj
dotnet build CosyVoiceNet\CosyVoiceNet.csproj
```

The default Windows build references the CUDA TorchSharp runtime package. A
caller can still force CPU at runtime with `CosyVoiceBackend.Cpu`.

For a CPU-only build:

```powershell
dotnet build CosyVoiceNet\CosyVoiceNet.csproj -p:TorchUseCuda=false
```

## Generation Timings

Benchmark hardware used:

- CPU: AMD Ryzen 9 5900X 12-Core Processor, 24 logical processors.
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, driver 595.79.
- Runtime: Windows x64, .NET `net10.0`, TorchSharp `0.106.0`.
- Benchmark date: 2026-05-15.

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

## High-Level API

External applications should normally use `CosyVoiceReturner` or the
`ICosyVoiceReturner` interface instead of directly constructing the lower-level
runtime classes.

```csharp
using CosyVoiceNet;
using CosyVoiceNet.cli;

ICosyVoiceReturner tts = CosyVoiceReturner.Shared;
```

### List Models

This is static capability information and does not initialize model weights.

```csharp
foreach (var model in tts.GetModels())
{
    Console.WriteLine($"{model.LocalName} downloaded={model.IsDownloaded} features={model.Features}");
}
```

### Download A Model With Progress

```csharp
var progress = new Progress<CosyVoiceDownloadProgress>(item =>
{
    var percent = item.ModelPercent?.ToString("0.0") ?? "?";
    Console.WriteLine($"{item.Stage}: {item.Message} ({percent}%)");
});

var modelDirectory = tts.EnsureModelDownloaded("cosyvoice3", progress);
Console.WriteLine(modelDirectory);
```

### Backend Selection

The global backend defaults to `Auto`, which attempts CUDA and falls back to CPU
when CUDA is unavailable. You can make the choice explicit globally:

```csharp
tts.SetGlobalBackend(CosyVoiceBackend.Cuda);
```

Or per model/load/generation request:

```csharp
var loaded = tts.LoadModel(
    model: "cosyvoice3",
    backend: CosyVoiceBackend.Cpu,
    runtimeOptions: new CosyVoiceRuntimeOptions
    {
        OptimizationProfile = CosyVoiceOptimizationProfile.Balanced,
        CpuThreads = 8,
        CpuInteropThreads = 1
    });

Console.WriteLine($"{loaded.LocalName} active backend: {loaded.ActiveBackend}");
```

### Generate From A Prompt WAV

Zero-shot generation needs both a prompt WAV and the transcript of that WAV.

```csharp
var result = tts.Generate(new CosyVoiceTtsRequest
{
    Model = "cosyvoice3",
    Text = "A quick CosyVoiceNet test from the C# pipeline.",
    PromptText = "This is the exact transcript of the prompt audio.",
    PromptWav = @"D:\voices\prompt.wav",
    Backend = CosyVoiceBackend.Cuda,
    RuntimeOptions = new CosyVoiceRuntimeOptions
    {
        OptimizationProfile = CosyVoiceOptimizationProfile.Throughput
    }
});

File.WriteAllBytes(@"D:\voices\out.wav", result.WavBytes);
Console.WriteLine($"Generated {result.DurationSeconds:0.00}s in {result.InferenceTime.TotalSeconds:0.00}s");
```

`CosyVoiceTtsResult.WavBytes` is the playable WAV payload. `RawFloat32Bytes`
contains the raw mono float32 samples used before WAV packaging.

### Clone And Reuse A Voice

```csharp
var savedVoice = tts.CloneAndSaveVoice(new CosyVoiceCloneRequest(
    Model: "cosyvoice3",
    VoiceName: "my_voice",
    PromptText: "This is the exact transcript of the prompt audio.",
    PromptWav: @"D:\voices\prompt.wav",
    Backend: CosyVoiceBackend.Cuda));

var result = tts.Generate(new CosyVoiceTtsRequest
{
    Model = "cosyvoice3",
    Text = "This uses the saved voice without reprocessing the prompt WAV.",
    Voice = savedVoice,
    Backend = CosyVoiceBackend.Cuda
});
```

Saved voices are model-specific. First-time generation from a provided WAV voice
selector can clone that WAV automatically, then later calls reuse the saved
voice name.

### List Voices

```csharp
var voices = tts.GetVoices(
    model: "cosyvoice3",
    providedWavs: new[] { @"D:\voices\guest.wav" },
    ensureDownloaded: false);

foreach (var voice in voices)
{
    Console.WriteLine($"{voice.Id} kind={voice.Kind} cloneOnFirstUse={voice.RequiresClone}");
}
```

For clone-capable models, provided WAVs and integrated prompt WAVs are exposed as
voice options without forcing model initialization. Built-in SFT voices are
reported for SFT models.

### Instruct And Cross-Lingual

Leave `Mode` as `CosyVoiceTtsMode.Auto` for normal use. The facade chooses the
route from the fields you provide:

- `InstructText` chooses `Instruct2` for CosyVoice2/CosyVoice3 or `Instruct`
  for the 300M instruct model.
- `CrossLingual = true` chooses the cross-lingual route.
- `Voice` chooses SFT for SFT-only models, otherwise a saved/provided voice.
- `PromptWav` without a voice chooses zero-shot.

```csharp
var result = tts.Generate(new CosyVoiceTtsRequest
{
    Model = "cosyvoice3",
    Text = "Say this with a calm, clear streaming voice.",
    Voice = "my_voice",
    InstructText = "Use a relaxed and natural delivery.",
    Backend = CosyVoiceBackend.Cuda
});
```

## Runtime Options

Runtime behavior is controlled through objects passed to the API, not through
environment variables. Useful knobs include:

- `OptimizationProfile`: `Compatibility`, `Balanced`, `Throughput`, or
  `LowMemory`.
- `CpuThreads` and `CpuInteropThreads`: Torch CPU worker settings.
- `CpuProcessorAffinityMask`: optional process affinity mask for CPU pinning.
- `QwenKvCacheBackend` and `LegacyTransformerCacheBackend`: cache strategy.
- `QwenAttentionBackend`, `QwenMlpBackend`, and `SamplingBackend`: lower-level
  generation implementation choices.
- `Logger` and `Profiler`: opt-in diagnostics sinks.
- `TraceTextInput`, `TracePromptTrim`, `TraceLlmInputShapes`, and
  `TraceGeneratedTokens`: debugging traces that should stay off during normal
  generation.

## Notes On The Port

CosyVoiceNet is intentionally close to the original CosyVoice behavior where it
matters for model logic, token preparation, prompt handling, flow generation, and
vocoder output. It is still not a direct packaging of the Python code. The C#
runtime has been adapted and optimized for:

- .NET application embedding.
- No Python dependency at runtime.
- Lazy model acquisition.
- Reusable high-level request/response types.
- CPU/CUDA backend management.
- Saved voice workflows.
- Per-request and per-model runtime options.
- Profiling and logging through explicit API hooks.

## Kudos

Huge thanks to the FunAudioLLM team for CosyVoice and the research/model releases
that made this port possible:

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice 3.0 demo and paper links](https://funaudiollm.github.io/cosyvoice3/)
- [CosyVoice 2.0 demo and paper links](https://funaudiollm.github.io/cosyvoice2/)
- [CosyVoice 1.0 demo](https://fun-audio-llm.github.io)

The upstream CosyVoice project also acknowledges major work from:

- [FunASR](https://github.com/modelscope/FunASR)
- [FunCodec](https://github.com/modelscope/FunCodec)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
- [WeNet](https://github.com/wenet-e2e/wenet)

Please keep upstream license and attribution requirements in mind when
redistributing models or derived code. The bundled upstream CosyVoice source tree
uses the Apache License 2.0.

## Citations

If you use this project in published work, cite the original CosyVoice papers:

```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice2,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Socials And Support

- ArnieTW links and socials: [linktr.ee/ArnieTW](https://linktr.ee/ArnieTW)
- GitHub: [github.com/ArnieTW](https://github.com/ArnieTW)

If CosyVoiceNet saves you time, please consider donating or supporting
development through the ArnieTW Linktree. It helps keep the port maintained,
tested, and improving.
