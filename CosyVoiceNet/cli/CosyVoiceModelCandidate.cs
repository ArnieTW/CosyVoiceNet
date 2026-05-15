using System;
using System.Collections.Generic;
using System.Linq;

namespace CosyVoiceNet.cli
{
    public enum CosyVoiceModelKind
    {
        FunCosyVoice3_0_5B,
        CosyVoice2_0_5B,
        CosyVoice300M,
        CosyVoice300MSft,
        CosyVoice300MInstruct
    }

    [Flags]
    public enum CosyVoiceModelFeatures
    {
        None = 0,
        Sft = 1,
        ZeroShot = 2,
        CrossLingual = 4,
        Instruct = 8,
        Instruct2 = 16,
        SavedVoice = 32
    }

    public sealed class CosyVoiceModelCandidate
    {
        internal CosyVoiceModelCandidate(
            CosyVoiceModelKind kind,
            string localName,
            string remoteId,
            Type runtimeType,
            CosyVoiceModelFeatures features,
            params string[] aliases)
        {
            Kind = kind;
            LocalName = localName;
            RemoteId = remoteId;
            RuntimeType = runtimeType;
            Features = features;

            var allAliases = new[] { localName, remoteId }
                .Concat(aliases ?? Array.Empty<string>())
                .Where(alias => !string.IsNullOrWhiteSpace(alias))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            Aliases = Array.AsReadOnly(allAliases);
        }

        public CosyVoiceModelKind Kind { get; }
        public string LocalName { get; }
        public string RemoteId { get; }
        public Type RuntimeType { get; }
        public CosyVoiceModelFeatures Features { get; }
        public IReadOnlyList<string> Aliases { get; }
        public string LocalDirectory => CosyVoice.GetLocalModelDir(LocalName);
        public bool IsDownloaded => CosyVoice.IsModelAvailable(LocalName);

        public bool Supports(CosyVoiceModelFeatures feature)
        {
            return (Features & feature) == feature;
        }

        public bool Matches(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return false;

            var normalized = value.Trim().Replace('\\', '/').TrimEnd('/');
            var fileName = System.IO.Path.GetFileName(normalized);
            return Aliases.Any(alias => string.Equals(alias, normalized, StringComparison.OrdinalIgnoreCase)) ||
                   Aliases.Any(alias => string.Equals(alias, fileName, StringComparison.OrdinalIgnoreCase));
        }

        public string EnsureDownloaded(IProgress<CosyVoiceDownloadProgress>? progress = null)
        {
            CosyVoice.EnsureModelDir(LocalName, progress);
            return LocalDirectory;
        }

        public IReadOnlyList<string> ListAvailableSpks(IEnumerable<string>? providedWavs = null, bool ensureDownloaded = false)
        {
            return CosyVoice.ListAvailableSpks(LocalName, providedWavs, ensureDownloaded);
        }

        public IReadOnlyList<CosyVoice.CosyVoiceVoiceOption> ListAvailableVoiceOptions(
            IEnumerable<string>? providedWavs = null,
            bool ensureDownloaded = false)
        {
            return CosyVoice.ListAvailableVoiceOptions(LocalName, providedWavs, ensureDownloaded);
        }

        public CosyVoice Create(
            bool loadJit = false,
            bool loadTrt = false,
            bool loadVllm = false,
            bool fp16 = false,
            int trtConcurrent = 1,
            CosyVoiceBackend? backend = null,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null,
            CosyVoiceRuntimeOptions? runtimeOptions = null)
        {
            EnsureDownloaded(downloadProgress);

            return Kind switch
            {
                CosyVoiceModelKind.FunCosyVoice3_0_5B => CreateCosyVoice3(loadJit, loadTrt, loadVllm, fp16, trtConcurrent, backend, runtimeOptions),
                CosyVoiceModelKind.CosyVoice2_0_5B => new CosyVoice2(LocalName, loadJit, loadTrt, loadVllm, fp16, trtConcurrent, backend, runtimeOptions),
                CosyVoiceModelKind.CosyVoice300M => new CosyVoice(LocalName, loadJit, loadTrt, fp16, trtConcurrent, backend, runtimeOptions),
                CosyVoiceModelKind.CosyVoice300MSft => new CosyVoice(LocalName, loadJit, loadTrt, fp16, trtConcurrent, backend, runtimeOptions),
                CosyVoiceModelKind.CosyVoice300MInstruct => new CosyVoice(LocalName, loadJit, loadTrt, fp16, trtConcurrent, backend, runtimeOptions),
                _ => throw new ArgumentOutOfRangeException(nameof(Kind), Kind, "Unsupported CosyVoice model candidate.")
            };
        }

        private CosyVoice3 CreateCosyVoice3(
            bool loadJit,
            bool loadTrt,
            bool loadVllm,
            bool fp16,
            int trtConcurrent,
            CosyVoiceBackend? backend,
            CosyVoiceRuntimeOptions? runtimeOptions)
        {
            if (loadJit)
                throw new NotSupportedException($"{LocalName} uses the CosyVoice3 runtime, which does not expose JIT module loading.");

            return new CosyVoice3(LocalName, loadTrt, loadVllm, fp16, trtConcurrent, backend, runtimeOptions);
        }

        public override string ToString()
        {
            return $"{LocalName} ({RemoteId}) -> {RuntimeType.Name}";
        }
    }

    public static class CosyVoiceModels
    {
        private static readonly CosyVoiceModelCandidate[] Candidates =
        {
            new(
                CosyVoiceModelKind.FunCosyVoice3_0_5B,
                "Fun-CosyVoice3-0.5B",
                "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                typeof(CosyVoice3),
                CosyVoiceModelFeatures.ZeroShot |
                CosyVoiceModelFeatures.CrossLingual |
                CosyVoiceModelFeatures.Instruct2 |
                CosyVoiceModelFeatures.SavedVoice,
                "fun-cosyvoice3",
                "cosyvoice3",
                "cosyvoice3-0.5b",
                "fun-cosyvoice3-0.5b-2512"),

            new(
                CosyVoiceModelKind.CosyVoice2_0_5B,
                "CosyVoice2-0.5B",
                "FunAudioLLM/CosyVoice2-0.5B",
                typeof(CosyVoice2),
                CosyVoiceModelFeatures.ZeroShot |
                CosyVoiceModelFeatures.CrossLingual |
                CosyVoiceModelFeatures.Instruct2 |
                CosyVoiceModelFeatures.SavedVoice,
                "cosyvoice2",
                "cosyvoice2-0.5b"),

            new(
                CosyVoiceModelKind.CosyVoice300M,
                "CosyVoice-300M",
                "FunAudioLLM/CosyVoice-300M",
                typeof(CosyVoice),
                CosyVoiceModelFeatures.ZeroShot |
                CosyVoiceModelFeatures.CrossLingual |
                CosyVoiceModelFeatures.SavedVoice,
                "cosyvoice",
                "cosyvoice-300m"),

            new(
                CosyVoiceModelKind.CosyVoice300MSft,
                "CosyVoice-300M-SFT",
                "FunAudioLLM/CosyVoice-300M-SFT",
                typeof(CosyVoice),
                CosyVoiceModelFeatures.Sft,
                "cosyvoice-sft",
                "cosyvoice-300m-sft",
                "sft"),

            new(
                CosyVoiceModelKind.CosyVoice300MInstruct,
                "CosyVoice-300M-Instruct",
                "FunAudioLLM/CosyVoice-300M-Instruct",
                typeof(CosyVoice),
                CosyVoiceModelFeatures.Instruct,
                "cosyvoice-instruct",
                "cosyvoice-300m-instruct",
                "instruct")
        };

        public static IReadOnlyList<CosyVoiceModelCandidate> All { get; } = Array.AsReadOnly(Candidates);

        public static CosyVoiceModelCandidate FunCosyVoice3 => Get(CosyVoiceModelKind.FunCosyVoice3_0_5B);
        public static CosyVoiceModelCandidate CosyVoice2 => Get(CosyVoiceModelKind.CosyVoice2_0_5B);
        public static CosyVoiceModelCandidate CosyVoice300M => Get(CosyVoiceModelKind.CosyVoice300M);
        public static CosyVoiceModelCandidate CosyVoice300MSft => Get(CosyVoiceModelKind.CosyVoice300MSft);
        public static CosyVoiceModelCandidate CosyVoice300MInstruct => Get(CosyVoiceModelKind.CosyVoice300MInstruct);

        public static CosyVoiceModelCandidate Get(CosyVoiceModelKind kind)
        {
            return Candidates.First(candidate => candidate.Kind == kind);
        }

        public static bool TryFind(string nameOrAlias, out CosyVoiceModelCandidate? candidate)
        {
            candidate = Candidates.FirstOrDefault(item => item.Matches(nameOrAlias));
            return candidate != null;
        }

        public static CosyVoiceModelCandidate Find(string nameOrAlias)
        {
            if (TryFind(nameOrAlias, out var candidate))
                return candidate!;

            var availableNames = string.Join(", ", Candidates.Select(item => item.LocalName));
            throw new ArgumentException($"Unknown CosyVoice model '{nameOrAlias}'. Known candidates: {availableNames}.", nameof(nameOrAlias));
        }

        public static CosyVoice Create(
            CosyVoiceModelKind kind,
            bool loadJit = false,
            bool loadTrt = false,
            bool loadVllm = false,
            bool fp16 = false,
            int trtConcurrent = 1,
            CosyVoiceBackend? backend = null,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null,
            CosyVoiceRuntimeOptions? runtimeOptions = null)
        {
            return Get(kind).Create(loadJit, loadTrt, loadVllm, fp16, trtConcurrent, backend, downloadProgress, runtimeOptions);
        }

        public static CosyVoice Create(
            string nameOrAlias,
            bool loadJit = false,
            bool loadTrt = false,
            bool loadVllm = false,
            bool fp16 = false,
            int trtConcurrent = 1,
            CosyVoiceBackend? backend = null,
            IProgress<CosyVoiceDownloadProgress>? downloadProgress = null,
            CosyVoiceRuntimeOptions? runtimeOptions = null)
        {
            return Find(nameOrAlias).Create(loadJit, loadTrt, loadVllm, fp16, trtConcurrent, backend, downloadProgress, runtimeOptions);
        }
    }
}
