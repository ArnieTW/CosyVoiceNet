using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CosyVoiceApp;
using CosyVoiceNet.cli;
using CosyVoiceNet.Utils;
using RuntimeCosyVoice = CosyVoiceNet.cli.CosyVoice;
using RuntimeInferenceResult = CosyVoiceNet.cli.CosyVoice.InferenceResult;

namespace CosyVoiceNet
{
    /// <summary>
    /// High-level CosyVoiceNet facade for external applications.
    /// Prefer this interface over direct use of the lower-level <c>CosyVoiceNet.cli</c> runtime classes.
    /// </summary>
    public interface ICosyVoiceReturner
    {
        /// <summary>
        /// Returns the known CosyVoice model candidates and their local download state.
        /// This does not initialize model weights.
        /// </summary>
        IReadOnlyList<CosyVoiceModelCapability> GetModels();

        /// <summary>
        /// Lists voices for a model, reusing a cached runtime model when one is already loaded.
        /// Clone-capable models can also expose provided WAV files as selectable <c>wav:path</c> voices.
        /// </summary>
        IReadOnlyList<CosyVoiceVoiceDescriptor> GetVoices(
            string model = "cosyvoice3",
            IEnumerable<string>? providedWavs = null,
            bool ensureDownloaded = false,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null);

        /// <summary>
        /// Ensures a model is present locally and reports downloader progress.
        /// </summary>
        string EnsureModelDownloaded(string model = "cosyvoice3", IProgress<CosyVoiceDownloadProgress>? downloadProgress = null);

        /// <summary>
        /// Deletes a downloaded model directory and evicts any cached runtime instances for that model.
        /// </summary>
        bool DeleteModel(string model);

        /// <summary>
        /// Sets the default backend used when a request does not explicitly provide one.
        /// </summary>
        void SetGlobalBackend(CosyVoiceBackend backend);

        /// <summary>
        /// Loads or returns a cached model instance and reports the active backend/device layout.
        /// </summary>
        CosyVoiceLoadedModelInfo LoadModel(
            string model = "cosyvoice3",
            CosyVoiceBackend? backend = null,
            bool fp16 = false,
            bool reload = false,
            CosyVoiceRuntimeOptions? runtimeOptions = null,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null);

        /// <summary>
        /// Returns the model runtimes currently cached in memory.
        /// </summary>
        IReadOnlyList<CosyVoiceLoadedModelInfo> GetLoadedModels();

        /// <summary>
        /// Drops cached runtime instances for one model.
        /// </summary>
        bool UnloadModel(string model);

        /// <summary>
        /// Creates a named saved voice from a prompt WAV and transcript.
        /// </summary>
        string CloneAndSaveVoice(CosyVoiceCloneRequest request);

        /// <summary>
        /// Deletes a saved cloned voice for a clone-capable model.
        /// </summary>
        bool DeleteSavedVoice(string model, string voice);

        /// <summary>
        /// Creates a deterministic saved voice for a provided WAV if it has not already been created.
        /// </summary>
        string CloneProvidedWavIfNeeded(CosyVoiceProvidedWavCloneRequest request);

        /// <summary>
        /// Generates TTS and returns both raw float32 samples and playable WAV bytes.
        /// </summary>
        CosyVoiceTtsResult Generate(CosyVoiceTtsRequest request);

        /// <summary>
        /// Drops cached model instances. The next generation request will reload the requested model.
        /// </summary>
        void ClearLoadedModels();
    }

    /// <summary>
    /// Main high-level entry point for CosyVoiceNet.
    /// It hides model selection, lazy loading, backend selection, saved voice reuse, and WAV packaging.
    /// </summary>
    public sealed class CosyVoiceReturner : ICosyVoiceReturner
    {
        public const int PlaybackLeadInMilliseconds = 250;
        public const int MaxInternalSilenceMilliseconds = 450;

        private readonly object _sync = new();
        private readonly Dictionary<ModelCacheKey, RuntimeCosyVoice> _models = new();

        /// <summary>
        /// Shared process-wide facade for simple app integration.
        /// Create a separate instance only when you need an isolated model cache.
        /// </summary>
        public static CosyVoiceReturner Shared { get; } = new();

        public IReadOnlyList<CosyVoiceModelCapability> GetModels()
        {
            return CosyVoiceModels.All
                .Select(candidate => new CosyVoiceModelCapability(
                    candidate.Kind,
                    candidate.LocalName,
                    candidate.RemoteId,
                    candidate.RuntimeType.Name,
                    candidate.Features,
                    candidate.IsDownloaded,
                    Path.GetFullPath(candidate.LocalDirectory),
                    candidate.Aliases))
                .ToArray();
        }

        public IReadOnlyList<CosyVoiceVoiceDescriptor> GetVoices(
            string model = "cosyvoice3",
            IEnumerable<string>? providedWavs = null,
            bool ensureDownloaded = false,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null)
        {
            var candidate = CosyVoiceModels.Find(model);
            if (ensureDownloaded)
                candidate.EnsureDownloaded(downloadProgress);

            RuntimeCosyVoice? loaded = null;
            lock (_sync)
            {
                loaded = _models
                    .Where(entry => string.Equals(entry.Key.Model, candidate.LocalName, StringComparison.OrdinalIgnoreCase))
                    .Select(entry => entry.Value)
                    .FirstOrDefault();
            }

            var voiceWavs = IncludeIntegratedVoiceWavs(candidate, providedWavs);
            var options = loaded is not null
                ? loaded.ListAvailableVoiceOptions(voiceWavs)
                : candidate.ListAvailableVoiceOptions(voiceWavs, ensureDownloaded: false);

            return options
                .Select(option => new CosyVoiceVoiceDescriptor(
                    option.Id,
                    option.DisplayName,
                    option.Kind,
                    option.WavPath,
                    option.RequiresClone))
                .ToArray();
        }

        private static IReadOnlyList<string> IncludeIntegratedVoiceWavs(
            CosyVoiceModelCandidate candidate,
            IEnumerable<string>? providedWavs)
        {
            var wavs = new List<string>();
            if (providedWavs is not null)
                wavs.AddRange(providedWavs.Where(wav => !string.IsNullOrWhiteSpace(wav)));

            if (!candidate.Supports(CosyVoiceModelFeatures.ZeroShot)
                && !candidate.Supports(CosyVoiceModelFeatures.CrossLingual)
                && !candidate.Supports(CosyVoiceModelFeatures.Instruct2)
                && !candidate.Supports(CosyVoiceModelFeatures.SavedVoice))
            {
                return wavs;
            }

            foreach (var key in new[] { "zero_shot_prompt.wav", "cross_lingual_prompt.wav" })
            {
                if (AppHost.Assets.TryGetValue(key, out var path) && File.Exists(path))
                    wavs.Add(path);
            }

            return wavs
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        public string EnsureModelDownloaded(
            string model = "cosyvoice3",
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null)
        {
            var candidate = CosyVoiceModels.Find(model);
            return candidate.EnsureDownloaded(downloadProgress);
        }

        public bool DeleteModel(string model)
        {
            var candidate = CosyVoiceModels.Find(model);
            var localDirectory = Path.GetFullPath(candidate.LocalDirectory);

            EvictModelInstances(candidate.LocalName);

            if (!Directory.Exists(localDirectory))
                return false;

            Directory.Delete(localDirectory, recursive: true);
            return true;
        }

        public void SetGlobalBackend(CosyVoiceBackend backend)
        {
            RuntimeCosyVoice.SetGlobalBackend(backend);
        }

        public CosyVoiceLoadedModelInfo LoadModel(
            string model = "cosyvoice3",
            CosyVoiceBackend? backend = null,
            bool fp16 = false,
            bool reload = false,
            CosyVoiceRuntimeOptions? runtimeOptions = null,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null)
        {
            var instance = GetModel(model, backend, fp16, reload, runtimeOptions, downloadProgress);
            var candidate = CosyVoiceModels.Find(model);
            return DescribeLoadedModel(candidate, instance);
        }

        public IReadOnlyList<CosyVoiceLoadedModelInfo> GetLoadedModels()
        {
            lock (_sync)
            {
                return _models
                    .Select(entry => DescribeLoadedModel(CosyVoiceModels.Find(entry.Key.Model), entry.Value))
                    .OrderBy(model => model.LocalName, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(model => model.RequestedBackend.ToString(), StringComparer.OrdinalIgnoreCase)
                    .ToArray();
            }
        }

        public bool UnloadModel(string model)
        {
            var candidate = CosyVoiceModels.Find(model);
            return EvictModelInstances(candidate.LocalName);
        }

        public string CloneAndSaveVoice(CosyVoiceCloneRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var model = GetModel(request.Model, request.Backend, request.Fp16, reload: false, request.RuntimeOptions, request.DownloadProgress);
            model.ApplyRuntimeOptions();
            return model.CloneAndSaveVoice(
                request.VoiceName,
                request.PromptText,
                request.PromptWav,
                request.Overwrite,
                request.TextFrontend);
        }

        public bool DeleteSavedVoice(string model, string voice)
        {
            if (string.IsNullOrWhiteSpace(voice))
                throw new ArgumentException("Voice is required.", nameof(voice));

            var candidate = CosyVoiceModels.Find(model);
            if (!candidate.Supports(CosyVoiceModelFeatures.SavedVoice))
                throw new NotSupportedException($"{candidate.LocalName} does not support cloned saved voices.");

            var savedVoicesDir = Path.Combine(Path.GetFullPath(candidate.LocalDirectory), "cosyvoice-net-voices");
            if (!Directory.Exists(savedVoicesDir))
                return false;

            var deleted = false;
            foreach (var path in Directory.EnumerateFiles(savedVoicesDir, "*.json", SearchOption.TopDirectoryOnly))
            {
                string? savedName = null;
                try
                {
                    using var stream = File.OpenRead(path);
                    using var doc = System.Text.Json.JsonDocument.Parse(stream);
                    if (doc.RootElement.TryGetProperty("Name", out var nameProperty))
                        savedName = nameProperty.GetString();
                }
                catch
                {
                    continue;
                }

                if (!string.Equals(savedName, voice.Trim(), StringComparison.OrdinalIgnoreCase))
                    continue;

                File.Delete(path);
                deleted = true;
            }

            if (deleted)
                EvictModelInstances(candidate.LocalName);

            return deleted;
        }

        public string CloneProvidedWavIfNeeded(CosyVoiceProvidedWavCloneRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            var model = GetModel(request.Model, request.Backend, request.Fp16, reload: false, request.RuntimeOptions, request.DownloadProgress);
            model.ApplyRuntimeOptions();
            return model.CloneProvidedWavIfNeeded(
                request.PromptWav,
                request.PromptText,
                request.VoiceName,
                request.TextFrontend);
        }

        public CosyVoiceTtsResult Generate(CosyVoiceTtsRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            if (string.IsNullOrWhiteSpace(request.Text))
                throw new ArgumentException("TTS text cannot be empty.", nameof(request));

            var candidate = CosyVoiceModels.Find(request.Model);
            var loadSw = Stopwatch.StartNew();
            var model = GetModel(candidate.LocalName, request.Backend, request.Fp16, request.ReloadModel, request.RuntimeOptions, request.DownloadProgress);
            loadSw.Stop();
            model.ApplyRuntimeOptions();

            Common.SetAllRandomSeed(request.Seed ?? 0);

            var preparedRequest = request.NormalizeTextForTts
                ? request with { Text = TtsTextPreprocessor.PrepareForTts(request.Text), NormalizeTextForTts = false }
                : request;

            var inferSw = Stopwatch.StartNew();
            var mode = ResolveMode(preparedRequest, candidate);
            var previousSamplingTopK = model.SamplingTopK;
            var previousNormalizeTextForTts = model.NormalizeTextForTts;
            RuntimeInferenceResult[] chunks;
            try
            {
                model.SamplingTopK = preparedRequest.SamplingTopK;
                model.NormalizeTextForTts = preparedRequest.NormalizeTextForTts;
                chunks = RunInference(model, candidate, preparedRequest, mode).ToArray();
            }
            finally
            {
                model.SamplingTopK = previousSamplingTopK;
                model.NormalizeTextForTts = previousNormalizeTextForTts;
            }
            inferSw.Stop();

            var rawBytes = CombineChunks(chunks.Select(chunk => chunk.TtsSpeech));
            var cleanedRawBytes = preparedRequest.CompressInternalSilence
                ? WavHelper.CompressInternalSilenceInFloat32Bytes(
                    rawBytes,
                    model.SampleRate,
                    maxSilenceMilliseconds: preparedRequest.MaxInternalSilenceMilliseconds)
                : rawBytes;
            var playableRawBytes = WavHelper.PrependSilenceToFloat32Bytes(cleanedRawBytes, model.SampleRate, silenceMilliseconds: PlaybackLeadInMilliseconds);
            var wavBytes = WavHelper.CreateWavFromFloat32Bytes(playableRawBytes, model.SampleRate);
            var samples = playableRawBytes.Length / sizeof(float);

            return new CosyVoiceTtsResult(
                candidate.LocalName,
                candidate.Kind,
                model.GetType().Name,
                mode,
                preparedRequest.Voice,
                preparedRequest.PromptWav,
                preparedRequest.InstructText,
                model.RequestedBackend,
                model.ActiveBackend,
                model.Device,
                model.LlmDevice,
                model.FlowDevice,
                model.HiftDevice,
                model.SampleRate,
                samples,
                samples / (double)model.SampleRate,
                loadSw.Elapsed,
                inferSw.Elapsed,
                playableRawBytes,
                wavBytes);
        }

        public void ClearLoadedModels()
        {
            lock (_sync)
                _models.Clear();
        }

        private bool EvictModelInstances(string model)
        {
            var removed = false;
            lock (_sync)
            {
                foreach (var key in _models.Keys
                    .Where(key => string.Equals(key.Model, model, StringComparison.OrdinalIgnoreCase))
                    .ToArray())
                {
                    _models.Remove(key);
                    removed = true;
                }
            }

            return removed;
        }

        private RuntimeCosyVoice GetModel(
            string model,
            CosyVoiceBackend? backend,
            bool fp16,
            bool reload,
            CosyVoiceRuntimeOptions? runtimeOptions,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress)
        {
            var candidate = CosyVoiceModels.Find(model);
            var resolvedBackend = backend ?? RuntimeCosyVoice.GlobalBackend;
            var resolvedRuntimeOptions = runtimeOptions?.Clone() ?? new CosyVoiceRuntimeOptions();
            var key = new ModelCacheKey(candidate.LocalName, resolvedBackend, fp16, RuntimeOptionsCacheKey.From(resolvedRuntimeOptions, resolvedBackend));

            lock (_sync)
            {
                if (!reload && _models.TryGetValue(key, out var existing))
                    return existing;
            }

            // Candidate.Create performs the lazy download if the model is not present locally.
            // Keep it outside the cache lock so status/progress readers are not blocked while large files download.
            var created = candidate.Create(fp16: fp16, backend: resolvedBackend, downloadProgress: downloadProgress, runtimeOptions: resolvedRuntimeOptions);

            lock (_sync)
            {
                if (!reload && _models.TryGetValue(key, out var existing))
                    return existing;

                _models[key] = created;
                return created;
            }
        }

        private static CosyVoiceLoadedModelInfo DescribeLoadedModel(CosyVoiceModelCandidate candidate, RuntimeCosyVoice model)
        {
            return new CosyVoiceLoadedModelInfo(
                candidate.LocalName,
                candidate.Kind,
                candidate.RemoteId,
                model.GetType().Name,
                model.RequestedBackend,
                model.ActiveBackend,
                model.Device,
                model.LlmDevice,
                model.FlowDevice,
                model.HiftDevice,
                model.SampleRate);
        }

        private static CosyVoiceTtsMode ResolveMode(CosyVoiceTtsRequest request, CosyVoiceModelCandidate candidate)
        {
            if (request.Mode != CosyVoiceTtsMode.Auto)
                return request.Mode;

            // Instruct text takes priority because it changes the prompt contract for instruct-capable models.
            if (!string.IsNullOrWhiteSpace(request.InstructText))
            {
                if (candidate.Supports(CosyVoiceModelFeatures.Instruct2))
                    return CosyVoiceTtsMode.Instruct2;
                if (candidate.Supports(CosyVoiceModelFeatures.Instruct))
                    return CosyVoiceTtsMode.Instruct;
            }

            if (request.CrossLingual)
                return CosyVoiceTtsMode.CrossLingual;

            // A named voice means SFT for SFT-only models, otherwise a saved voice or provided WAV selector.
            if (!string.IsNullOrWhiteSpace(request.Voice))
                return candidate.Supports(CosyVoiceModelFeatures.Sft)
                    ? CosyVoiceTtsMode.Sft
                    : CosyVoiceTtsMode.SavedVoice;

            if (!string.IsNullOrWhiteSpace(request.PromptWav))
                return CosyVoiceTtsMode.ZeroShot;

            if (candidate.Supports(CosyVoiceModelFeatures.Sft))
                return CosyVoiceTtsMode.Sft;

            throw new InvalidOperationException(
                $"Could not infer a TTS mode for {candidate.LocalName}. Provide Voice, PromptWav, InstructText, or CrossLingual.");
        }

        private static IEnumerable<RuntimeInferenceResult> RunInference(
            RuntimeCosyVoice model,
            CosyVoiceModelCandidate candidate,
            CosyVoiceTtsRequest request,
            CosyVoiceTtsMode mode)
        {
            return mode switch
            {
                CosyVoiceTtsMode.Sft => model.InferenceSft(
                    request.Text,
                    RequireVoice(request, candidate),
                    request.Stream,
                    request.Speed,
                    request.TextFrontend),

                CosyVoiceTtsMode.ZeroShot => model.InferenceZeroShot(
                    request.Text,
                    request.PromptText,
                    RequirePromptWav(request),
                    request.ZeroShotSpeakerId ?? string.Empty,
                    request.Stream,
                    request.Speed,
                    request.TextFrontend),

                CosyVoiceTtsMode.CrossLingual => RunCrossLingual(model, request),
                CosyVoiceTtsMode.Instruct => RunInstruct(model, candidate, request),
                CosyVoiceTtsMode.Instruct2 => RunInstruct2(model, request),
                CosyVoiceTtsMode.SavedVoice => RunVoice(model, request),
                _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported TTS mode.")
            };
        }

        private static IEnumerable<RuntimeInferenceResult> RunVoice(RuntimeCosyVoice model, CosyVoiceTtsRequest request)
        {
            return model.InferenceWithVoice(
                request.Text,
                RequireVoice(request, null),
                request.PromptText,
                request.InstructText,
                request.CrossLingual,
                request.Stream,
                request.Speed,
                request.TextFrontend);
        }

        private static IEnumerable<RuntimeInferenceResult> RunCrossLingual(RuntimeCosyVoice model, CosyVoiceTtsRequest request)
        {
            if (!string.IsNullOrWhiteSpace(request.Voice))
                return model.InferenceWithVoice(
                    request.Text,
                    request.Voice,
                    request.PromptText,
                    request.InstructText,
                    crossLingual: true,
                    request.Stream,
                    request.Speed,
                    request.TextFrontend);

            return model.InferenceCrossLingual(
                request.Text,
                RequirePromptWav(request),
                request.ZeroShotSpeakerId ?? string.Empty,
                request.Stream,
                request.Speed,
                request.TextFrontend);
        }

        private static IEnumerable<RuntimeInferenceResult> RunInstruct(
            RuntimeCosyVoice model,
            CosyVoiceModelCandidate candidate,
            CosyVoiceTtsRequest request)
        {
            if (!candidate.Supports(CosyVoiceModelFeatures.Instruct))
                return RunVoice(model, request);

            return model.InferenceInstruct(
                request.Text,
                RequireVoice(request, candidate),
                request.InstructText,
                request.Stream,
                request.Speed,
                request.TextFrontend);
        }

        private static IEnumerable<RuntimeInferenceResult> RunInstruct2(RuntimeCosyVoice model, CosyVoiceTtsRequest request)
        {
            if (!string.IsNullOrWhiteSpace(request.Voice))
                return model.InferenceWithVoice(
                    request.Text,
                    request.Voice,
                    request.PromptText,
                    request.InstructText,
                    crossLingual: false,
                    request.Stream,
                    request.Speed,
                    request.TextFrontend);

            if (model is CosyVoice2 cosyVoice2)
            {
                return cosyVoice2.InferenceInstruct2(
                    request.Text,
                    request.InstructText,
                    RequirePromptWav(request),
                    request.ZeroShotSpeakerId ?? string.Empty,
                    request.Stream,
                    request.Speed,
                    request.TextFrontend);
            }

            return RunVoice(model, request);
        }

        private static string RequireVoice(CosyVoiceTtsRequest request, CosyVoiceModelCandidate? candidate)
        {
            if (!string.IsNullOrWhiteSpace(request.Voice))
                return request.Voice;

            if (candidate != null)
            {
                var firstVoice = candidate.ListAvailableSpks().FirstOrDefault();
                if (!string.IsNullOrWhiteSpace(firstVoice))
                    return firstVoice;
            }

            throw new ArgumentException("A voice is required for this TTS mode.", nameof(request));
        }

        private static string RequirePromptWav(CosyVoiceTtsRequest request)
        {
            if (!string.IsNullOrWhiteSpace(request.PromptWav))
                return request.PromptWav;

            throw new ArgumentException("A prompt WAV is required for this TTS mode.", nameof(request));
        }

        private static byte[] CombineChunks(IEnumerable<byte[]> chunks)
        {
            var parts = chunks.ToArray();
            var length = parts.Sum(part => part.Length);
            var result = new byte[length];
            var offset = 0;
            foreach (var part in parts)
            {
                Buffer.BlockCopy(part, 0, result, offset, part.Length);
                offset += part.Length;
            }

            return result;
        }

        private readonly record struct RuntimeOptionsCacheKey(
            CosyVoiceOptimizationProfile OptimizationProfile,
            LegacyTransformerCacheBackend LegacyTransformerCacheBackend,
            QwenKvCacheBackend QwenKvCacheBackend,
            QwenAttentionBackend QwenAttentionBackend,
            QwenMlpBackend QwenMlpBackend,
            CosyVoiceSamplingBackend SamplingBackend,
            int? CpuThreads,
            int? CpuInteropThreads,
            long? CpuProcessorAffinityMask)
        {
            public static RuntimeOptionsCacheKey From(CosyVoiceRuntimeOptions options, CosyVoiceBackend backend)
            {
                var resolved = options.Resolve(backend);
                return new RuntimeOptionsCacheKey(
                    resolved.OptimizationProfile,
                    resolved.LegacyTransformerCacheBackend,
                    resolved.QwenKvCacheBackend,
                    resolved.QwenAttentionBackend,
                    resolved.QwenMlpBackend,
                    resolved.SamplingBackend,
                    NormalizePositive(resolved.CpuThreads),
                    NormalizePositive(resolved.CpuInteropThreads),
                    NormalizePositive(resolved.CpuProcessorAffinityMask));
            }

            private static int? NormalizePositive(int? value) => value > 0 ? value : null;

            private static long? NormalizePositive(long? value) => value > 0 ? value : null;
        }

        private readonly record struct ModelCacheKey(string Model, CosyVoiceBackend Backend, bool Fp16, RuntimeOptionsCacheKey RuntimeOptions);
    }

    /// <summary>
    /// Explicit TTS generation route. Use <see cref="Auto"/> for normal external app calls.
    /// </summary>
    public enum CosyVoiceTtsMode
    {
        /// <summary>Infer the best route from request fields and model capabilities.</summary>
        Auto,

        /// <summary>Speaker fine-tuning route for SFT models and built-in speakers.</summary>
        Sft,

        /// <summary>Zero-shot route using a prompt WAV and prompt transcript.</summary>
        ZeroShot,

        /// <summary>Cross-lingual route using a prompt WAV or saved/provided voice.</summary>
        CrossLingual,

        /// <summary>Original 300M instruct route using an SFT speaker and instruction text.</summary>
        Instruct,

        /// <summary>CosyVoice2/FunCosyVoice3 instruct route using a prompt WAV or saved/provided voice.</summary>
        Instruct2,

        /// <summary>Saved voice route, including lazy cloning from <c>wav:path</c> voice selectors.</summary>
        SavedVoice
    }

    /// <summary>
    /// High-level TTS request. In normal use, set <see cref="Text"/>, optionally set <see cref="Voice"/>,
    /// <see cref="PromptWav"/>, <see cref="PromptText"/>, or <see cref="InstructText"/>, and leave
    /// <see cref="Mode"/> as <see cref="CosyVoiceTtsMode.Auto"/>.
    /// </summary>
    public sealed record CosyVoiceTtsRequest
    {
        /// <summary>Model name, alias, local folder name, or remote model id. Defaults to Fun-CosyVoice3.</summary>
        public string Model { get; init; } = "cosyvoice3";

        /// <summary>Text to synthesize.</summary>
        public string Text { get; init; } = string.Empty;

        /// <summary>
        /// Built-in speaker id, saved voice name, or provided WAV selector returned by <see cref="ICosyVoiceReturner.GetVoices"/>.
        /// </summary>
        public string? Voice { get; init; }

        /// <summary>Transcript of <see cref="PromptWav"/> for zero-shot cloning and zero-shot generation.</summary>
        public string PromptText { get; init; } = string.Empty;

        /// <summary>Prompt WAV path used for zero-shot, cross-lingual, instruct2, or first-time provided-WAV cloning.</summary>
        public string? PromptWav { get; init; }

        /// <summary>Optional low-level zero-shot speaker id when using the direct zero-shot route.</summary>
        public string? ZeroShotSpeakerId { get; init; }

        /// <summary>Instruction text for instruct-capable models.</summary>
        public string InstructText { get; init; } = string.Empty;

        /// <summary>When true, prefer cross-lingual behavior in auto mode or saved voice generation.</summary>
        public bool CrossLingual { get; init; }

        /// <summary>Passes streaming preference to the model. The facade still combines returned chunks.</summary>
        public bool Stream { get; init; }

        /// <summary>Generation speed multiplier.</summary>
        public double Speed { get; init; } = 1.0;

        /// <summary>Enables CosyVoice text frontend normalization.</summary>
        public bool TextFrontend { get; init; } = true;

        /// <summary>Expands compact identifiers like FullTimeSlob or ArnieTW into speakable words before tokenization.</summary>
        public bool NormalizeTextForTts { get; init; }

        /// <summary>Optional per-request backend. If omitted, the global backend is used.</summary>
        public CosyVoiceBackend? Backend { get; init; }

        /// <summary>Requests fp16 model loading where supported.</summary>
        public bool Fp16 { get; init; }

        /// <summary>Forces the cached model instance to be recreated before generation.</summary>
        public bool ReloadModel { get; init; }

        /// <summary>Optional sampling seed for reproducible request output.</summary>
        public int? Seed { get; init; }

        /// <summary>
        /// LLM top-k sampling width. The facade defaults to 25 to preserve the original CosyVoice sampling behavior;
        /// set to 1 only when deterministic greedy generation is explicitly preferred.
        /// </summary>
        public int SamplingTopK { get; init; } = 25;

        /// <summary>Shortens excessive low-energy gaps produced inside the generated audio.</summary>
        public bool CompressInternalSilence { get; init; } = true;

        /// <summary>Maximum internal low-energy gap kept when <see cref="CompressInternalSilence"/> is enabled.</summary>
        public int MaxInternalSilenceMilliseconds { get; init; } = CosyVoiceReturner.MaxInternalSilenceMilliseconds;

        /// <summary>Receives progress while the requested model is being auto-downloaded.</summary>
        public IProgress<CosyVoiceDownloadProgress>? DownloadProgress { get; init; }

        /// <summary>Optional runtime settings such as Torch CPU thread counts and Qwen KV-cache behavior.</summary>
        public CosyVoiceRuntimeOptions? RuntimeOptions { get; init; }

        /// <summary>Explicit mode override. Leave as <see cref="CosyVoiceTtsMode.Auto"/> for normal app calls.</summary>
        public CosyVoiceTtsMode Mode { get; init; } = CosyVoiceTtsMode.Auto;
    }

    /// <summary>
    /// Request to create or overwrite a named saved voice from a prompt WAV and transcript.
    /// </summary>
    /// <param name="Model">Model name or alias.</param>
    /// <param name="VoiceName">Name to save and later pass as <see cref="CosyVoiceTtsRequest.Voice"/>.</param>
    /// <param name="PromptText">Transcript of the prompt WAV.</param>
    /// <param name="PromptWav">Prompt WAV path.</param>
    /// <param name="Overwrite">Whether to replace an existing saved voice with the same name.</param>
    /// <param name="TextFrontend">Whether to apply text frontend normalization to the prompt transcript.</param>
    /// <param name="Backend">Optional backend used while cloning.</param>
    /// <param name="Fp16">Requests fp16 model loading where supported.</param>
    /// <param name="DownloadProgress">Receives progress while the requested model is being auto-downloaded.</param>
    public sealed record CosyVoiceCloneRequest(
        string Model,
        string VoiceName,
        string PromptText,
        string PromptWav,
        bool Overwrite = true,
        bool TextFrontend = true,
        CosyVoiceBackend? Backend = null,
        bool Fp16 = false,
        CosyVoiceRuntimeOptions? RuntimeOptions = null,
        IProgress<CosyVoiceDownloadProgress>? DownloadProgress = null);

    /// <summary>
    /// Request to lazily clone a provided WAV into a deterministic saved voice.
    /// Repeated calls with the same WAV and prompt text reuse the same saved voice.
    /// </summary>
    /// <param name="Model">Model name or alias.</param>
    /// <param name="PromptWav">Prompt WAV path.</param>
    /// <param name="PromptText">Transcript of the prompt WAV.</param>
    /// <param name="VoiceName">Optional explicit saved voice name. If omitted, a deterministic name is generated.</param>
    /// <param name="TextFrontend">Whether to apply text frontend normalization to the prompt transcript.</param>
    /// <param name="Backend">Optional backend used while cloning.</param>
    /// <param name="Fp16">Requests fp16 model loading where supported.</param>
    /// <param name="DownloadProgress">Receives progress while the requested model is being auto-downloaded.</param>
    public sealed record CosyVoiceProvidedWavCloneRequest(
        string Model,
        string PromptWav,
        string PromptText,
        string? VoiceName = null,
        bool TextFrontend = true,
        CosyVoiceBackend? Backend = null,
        bool Fp16 = false,
        CosyVoiceRuntimeOptions? RuntimeOptions = null,
        IProgress<CosyVoiceDownloadProgress>? DownloadProgress = null);

    /// <summary>
    /// Result of a high-level TTS generation request.
    /// </summary>
    /// <param name="Model">Resolved local model name.</param>
    /// <param name="ModelKind">Resolved model kind.</param>
    /// <param name="Runtime">Runtime class used under the facade.</param>
    /// <param name="Mode">Generation route used.</param>
    /// <param name="Voice">Requested voice selector, if any.</param>
    /// <param name="PromptWav">Prompt WAV path, if any.</param>
    /// <param name="InstructText">Instruction text, if any.</param>
    /// <param name="RequestedBackend">Backend requested by the caller or global default.</param>
    /// <param name="ActiveBackend">Backend actually active after model loading.</param>
    /// <param name="Device">Primary model device.</param>
    /// <param name="LlmDevice">LLM component device.</param>
    /// <param name="FlowDevice">Flow component device.</param>
    /// <param name="HiftDevice">HiFT vocoder component device.</param>
    /// <param name="SampleRate">Output sample rate.</param>
    /// <param name="Samples">Number of mono float32 samples.</param>
    /// <param name="DurationSeconds">Output duration in seconds.</param>
    /// <param name="LoadTime">Time spent obtaining the cached or newly loaded model.</param>
    /// <param name="InferenceTime">Time spent generating audio chunks.</param>
    /// <param name="RawFloat32Bytes">Raw little-endian float32 mono sample bytes.</param>
    /// <param name="WavBytes">Playable PCM16 WAV bytes. The facade does not write these bytes to disk.</param>
    public sealed record CosyVoiceTtsResult(
        string Model,
        CosyVoiceModelKind ModelKind,
        string Runtime,
        CosyVoiceTtsMode Mode,
        string? Voice,
        string? PromptWav,
        string? InstructText,
        CosyVoiceBackend RequestedBackend,
        CosyVoiceBackend ActiveBackend,
        string Device,
        string LlmDevice,
        string FlowDevice,
        string HiftDevice,
        int SampleRate,
        int Samples,
        double DurationSeconds,
        TimeSpan LoadTime,
        TimeSpan InferenceTime,
        byte[] RawFloat32Bytes,
        byte[] WavBytes);

    /// <summary>
    /// Voice option returned to external apps.
    /// </summary>
    /// <param name="Id">Value to pass as <see cref="CosyVoiceTtsRequest.Voice"/>.</param>
    /// <param name="DisplayName">Human-readable label.</param>
    /// <param name="Kind">Voice kind: built-in, saved, or provided WAV.</param>
    /// <param name="WavPath">Source WAV path for provided-WAV options.</param>
    /// <param name="RequiresClone">True when first use will create a saved cloned voice.</param>
    public sealed record CosyVoiceVoiceDescriptor(
        string Id,
        string DisplayName,
        string Kind,
        string? WavPath,
        bool RequiresClone);

    /// <summary>
    /// Static capability description for a known model candidate.
    /// </summary>
    public sealed record CosyVoiceModelCapability(
        CosyVoiceModelKind Kind,
        string LocalName,
        string RemoteId,
        string Runtime,
        CosyVoiceModelFeatures Features,
        bool IsDownloaded,
        string LocalDirectory,
        IReadOnlyList<string> Aliases);

    /// <summary>
    /// Runtime model state returned after loading or retrieving a cached model.
    /// </summary>
    public sealed record CosyVoiceLoadedModelInfo(
        string LocalName,
        CosyVoiceModelKind Kind,
        string RemoteId,
        string Runtime,
        CosyVoiceBackend RequestedBackend,
        CosyVoiceBackend ActiveBackend,
        string Device,
        string LlmDevice,
        string FlowDevice,
        string HiftDevice,
        int SampleRate);
}
