using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using CosyVoiceNet.Utils;

namespace CosyVoiceNet.cli
{
    public enum CosyVoiceOptimizationProfile
    {
        Auto = 0,
        Compatibility = 1,
        Balanced = 2,
        Throughput = 3,
        LowMemory = 4
    }

    public enum QwenKvCacheBackend
    {
        Auto = 0,
        Standard = 1,
        Preallocated = 2,
        Disabled = 3
    }

    public enum LegacyTransformerCacheBackend
    {
        Auto = 0,
        Standard = 1,
        Preallocated = 2
    }

    public enum QwenAttentionBackend
    {
        Auto = 0,
        Manual = 1,
        ScaledDotProductAttention = 2
    }

    public enum QwenMlpBackend
    {
        Auto = 0,
        SeparateProjections = 1,
        FusedGateUpProjection = 2
    }

    public interface ICosyVoiceProfiler
    {
        void Record(string name, double milliseconds, IReadOnlyDictionary<string, string>? tags = null);
    }

    public enum CosyVoiceLogLevel
    {
        Trace = 0,
        Debug = 1,
        Information = 2,
        Warning = 3,
        Error = 4
    }

    public interface ICosyVoiceLogger
    {
        void Log(CosyVoiceLogLevel level, string message, Exception? exception = null, IReadOnlyDictionary<string, string>? tags = null);
    }

    public sealed record CosyVoiceLogEvent(
        DateTimeOffset Timestamp,
        CosyVoiceLogLevel Level,
        string Message,
        string? Exception,
        IReadOnlyDictionary<string, string> Tags);

    public sealed class CosyVoiceLogCollector : ICosyVoiceLogger
    {
        private readonly ConcurrentQueue<CosyVoiceLogEvent> _events = new();

        public void Log(CosyVoiceLogLevel level, string message, Exception? exception = null, IReadOnlyDictionary<string, string>? tags = null)
        {
            if (string.IsNullOrWhiteSpace(message))
                return;

            _events.Enqueue(new CosyVoiceLogEvent(
                DateTimeOffset.UtcNow,
                level,
                message,
                exception?.ToString(),
                tags is null
                    ? new Dictionary<string, string>()
                    : new Dictionary<string, string>(tags, StringComparer.OrdinalIgnoreCase)));
        }

        public IReadOnlyList<CosyVoiceLogEvent> SnapshotEvents()
        {
            return _events.ToArray();
        }
    }

    public sealed class CosyVoiceDelegateLogger : ICosyVoiceLogger
    {
        private readonly Action<CosyVoiceLogEvent> _sink;

        public CosyVoiceDelegateLogger(Action<CosyVoiceLogEvent> sink)
        {
            _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void Log(CosyVoiceLogLevel level, string message, Exception? exception = null, IReadOnlyDictionary<string, string>? tags = null)
        {
            if (string.IsNullOrWhiteSpace(message))
                return;

            _sink(new CosyVoiceLogEvent(
                DateTimeOffset.UtcNow,
                level,
                message,
                exception?.ToString(),
                tags is null
                    ? new Dictionary<string, string>()
                    : new Dictionary<string, string>(tags, StringComparer.OrdinalIgnoreCase)));
        }
    }

    internal static class CosyVoiceLog
    {
        public static void Write(ICosyVoiceLogger? logger, CosyVoiceLogLevel level, string message, Exception? exception = null, IReadOnlyDictionary<string, string>? tags = null)
        {
            logger?.Log(level, message, exception, tags);
        }
    }

    public sealed record CosyVoiceProfileEvent(
        string Name,
        double Milliseconds,
        IReadOnlyDictionary<string, string> Tags);

    public sealed record CosyVoiceProfileSummary(
        string Name,
        int Count,
        double TotalMilliseconds,
        double AverageMilliseconds,
        double MinMilliseconds,
        double MaxMilliseconds);

    public sealed class CosyVoiceProfileCollector : ICosyVoiceProfiler
    {
        private readonly ConcurrentQueue<CosyVoiceProfileEvent> _events = new();

        public void Record(string name, double milliseconds, IReadOnlyDictionary<string, string>? tags = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                return;

            _events.Enqueue(new CosyVoiceProfileEvent(
                name,
                milliseconds,
                tags is null
                    ? new Dictionary<string, string>()
                    : new Dictionary<string, string>(tags, StringComparer.OrdinalIgnoreCase)));
        }

        public IReadOnlyList<CosyVoiceProfileEvent> SnapshotEvents()
        {
            return _events.ToArray();
        }

        public IReadOnlyList<CosyVoiceProfileSummary> SnapshotSummary()
        {
            return _events
                .GroupBy(item => item.Name, StringComparer.OrdinalIgnoreCase)
                .Select(group =>
                {
                    var values = group.Select(item => item.Milliseconds).ToArray();
                    return new CosyVoiceProfileSummary(
                        group.Key,
                        values.Length,
                        values.Sum(),
                        values.Average(),
                        values.Min(),
                        values.Max());
                })
                .OrderByDescending(item => item.TotalMilliseconds)
                .ToArray();
        }
    }

    public sealed class CosyVoiceRuntimeOptions
    {
        public CosyVoiceOptimizationProfile OptimizationProfile { get; set; } = CosyVoiceOptimizationProfile.Balanced;

        public LegacyTransformerCacheBackend LegacyTransformerCacheBackend { get; set; } = LegacyTransformerCacheBackend.Auto;

        public QwenKvCacheBackend QwenKvCacheBackend { get; set; } = QwenKvCacheBackend.Auto;

        [Obsolete("Use QwenKvCacheBackend instead.")]
        public bool PreallocateQwenKvCache
        {
            get => QwenKvCacheBackend == QwenKvCacheBackend.Preallocated;
            set => QwenKvCacheBackend = value ? QwenKvCacheBackend.Preallocated : QwenKvCacheBackend.Standard;
        }

        public QwenAttentionBackend QwenAttentionBackend { get; set; } = QwenAttentionBackend.Auto;

        public QwenMlpBackend QwenMlpBackend { get; set; } = QwenMlpBackend.Auto;

        public CosyVoiceSamplingBackend SamplingBackend { get; set; } = CosyVoiceSamplingBackend.LogitsDevice;

        /// <summary>
        /// Optional ONNX backend override. Leave null to use the active Torch backend.
        /// </summary>
        public CosyVoiceBackend? OnnxBackend { get; set; }

        /// <summary>
        /// Torch CPU intra-op thread count. Leave null to keep Torch's current/default value.
        /// </summary>
        public int? CpuThreads { get; set; }

        /// <summary>
        /// Torch CPU inter-op thread count. Leave null to use CosyVoiceNet's low-latency CPU default.
        /// </summary>
        public int? CpuInteropThreads { get; set; }

        /// <summary>
        /// Optional diagnostics sink. Leave null for silent library behavior.
        /// </summary>
        public ICosyVoiceLogger? Logger { get; set; }

        public ICosyVoiceProfiler? Profiler { get; set; }

        /// <summary>
        /// Logs tokenization details for incoming TTS text. This is intentionally opt-in because it repeats tokenization work.
        /// </summary>
        public bool TraceTextInput { get; set; }

        /// <summary>
        /// Logs prompt boundary trimming decisions.
        /// </summary>
        public bool TracePromptTrim { get; set; }

        /// <summary>
        /// Logs LLM tensor shapes before token generation.
        /// </summary>
        public bool TraceLlmInputShapes { get; set; }

        /// <summary>
        /// Logs the head of generated and appended speech tokens. This stores token IDs during generation, so keep it off for normal use.
        /// </summary>
        public bool TraceGeneratedTokens { get; set; }

        /// <summary>
        /// Applies CosyVoiceNet text preparation by default for direct CosyVoice callers. High-level request APIs can override per request.
        /// </summary>
        public bool NormalizeTextForTts { get; set; }

        public ResolvedCosyVoiceRuntimeOptions Resolve(CosyVoiceBackend backend)
        {
            var profile = OptimizationProfile == CosyVoiceOptimizationProfile.Auto
                ? CosyVoiceOptimizationProfile.Balanced
                : OptimizationProfile;

            return new ResolvedCosyVoiceRuntimeOptions(
                profile,
                ResolveLegacyTransformerCacheBackend(profile),
                ResolveQwenKvCacheBackend(profile),
                ResolveQwenAttentionBackend(profile),
                ResolveQwenMlpBackend(profile),
                ResolveSamplingBackend(backend),
                CpuThreads,
                CpuInteropThreads,
                Logger,
                Profiler,
                TraceTextInput,
                TracePromptTrim,
                TraceLlmInputShapes,
                TraceGeneratedTokens,
                NormalizeTextForTts);
        }

        public CosyVoiceRuntimeOptions Clone()
        {
            return new CosyVoiceRuntimeOptions
            {
                OptimizationProfile = OptimizationProfile,
                LegacyTransformerCacheBackend = LegacyTransformerCacheBackend,
                QwenKvCacheBackend = QwenKvCacheBackend,
                QwenAttentionBackend = QwenAttentionBackend,
                QwenMlpBackend = QwenMlpBackend,
                SamplingBackend = SamplingBackend,
                OnnxBackend = OnnxBackend,
                CpuThreads = CpuThreads,
                CpuInteropThreads = CpuInteropThreads,
                Logger = Logger,
                Profiler = Profiler,
                TraceTextInput = TraceTextInput,
                TracePromptTrim = TracePromptTrim,
                TraceLlmInputShapes = TraceLlmInputShapes,
                TraceGeneratedTokens = TraceGeneratedTokens,
                NormalizeTextForTts = NormalizeTextForTts
            };
        }

        private LegacyTransformerCacheBackend ResolveLegacyTransformerCacheBackend(CosyVoiceOptimizationProfile profile)
        {
            if (LegacyTransformerCacheBackend != LegacyTransformerCacheBackend.Auto)
                return LegacyTransformerCacheBackend;

            return profile switch
            {
                CosyVoiceOptimizationProfile.Compatibility => LegacyTransformerCacheBackend.Standard,
                CosyVoiceOptimizationProfile.LowMemory => LegacyTransformerCacheBackend.Standard,
                CosyVoiceOptimizationProfile.Balanced => LegacyTransformerCacheBackend.Preallocated,
                CosyVoiceOptimizationProfile.Throughput => LegacyTransformerCacheBackend.Preallocated,
                _ => LegacyTransformerCacheBackend.Preallocated
            };
        }

        private QwenKvCacheBackend ResolveQwenKvCacheBackend(CosyVoiceOptimizationProfile profile)
        {
            if (QwenKvCacheBackend != QwenKvCacheBackend.Auto)
                return QwenKvCacheBackend;

            return profile switch
            {
                CosyVoiceOptimizationProfile.Compatibility => QwenKvCacheBackend.Standard,
                CosyVoiceOptimizationProfile.LowMemory => QwenKvCacheBackend.Standard,
                CosyVoiceOptimizationProfile.Balanced => QwenKvCacheBackend.Preallocated,
                CosyVoiceOptimizationProfile.Throughput => QwenKvCacheBackend.Preallocated,
                _ => QwenKvCacheBackend.Preallocated
            };
        }

        private QwenAttentionBackend ResolveQwenAttentionBackend(CosyVoiceOptimizationProfile profile)
        {
            if (QwenAttentionBackend != QwenAttentionBackend.Auto)
                return QwenAttentionBackend;

            return profile switch
            {
                CosyVoiceOptimizationProfile.Compatibility => QwenAttentionBackend.Manual,
                _ => QwenAttentionBackend.Auto
            };
        }

        private QwenMlpBackend ResolveQwenMlpBackend(CosyVoiceOptimizationProfile profile)
        {
            if (QwenMlpBackend != QwenMlpBackend.Auto)
                return QwenMlpBackend;

            return profile switch
            {
                CosyVoiceOptimizationProfile.Compatibility => QwenMlpBackend.SeparateProjections,
                CosyVoiceOptimizationProfile.LowMemory => QwenMlpBackend.SeparateProjections,
                CosyVoiceOptimizationProfile.Balanced => QwenMlpBackend.FusedGateUpProjection,
                CosyVoiceOptimizationProfile.Throughput => QwenMlpBackend.FusedGateUpProjection,
                _ => QwenMlpBackend.FusedGateUpProjection
            };
        }

        private CosyVoiceSamplingBackend ResolveSamplingBackend(CosyVoiceBackend backend)
        {
            return backend == CosyVoiceBackend.Cpu && SamplingBackend == CosyVoiceSamplingBackend.Cuda
                ? CosyVoiceSamplingBackend.Cpu
                : SamplingBackend;
        }

    }

    public sealed record ResolvedCosyVoiceRuntimeOptions(
        CosyVoiceOptimizationProfile OptimizationProfile,
        LegacyTransformerCacheBackend LegacyTransformerCacheBackend,
        QwenKvCacheBackend QwenKvCacheBackend,
        QwenAttentionBackend QwenAttentionBackend,
        QwenMlpBackend QwenMlpBackend,
        CosyVoiceSamplingBackend SamplingBackend,
        int? CpuThreads,
        int? CpuInteropThreads,
        ICosyVoiceLogger? Logger,
        ICosyVoiceProfiler? Profiler,
        bool TraceTextInput,
        bool TracePromptTrim,
        bool TraceLlmInputShapes,
        bool TraceGeneratedTokens,
        bool NormalizeTextForTts);
}
