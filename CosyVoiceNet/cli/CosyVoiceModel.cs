using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Diagnostics;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using Tensor = TorchSharp.torch.Tensor;
using CosyVoiceNet.hifigan;
using CosyVoiceNet.Utils;
using CosyVoiceNet;
using CosyVoiceNet.LLM;
namespace CosyVoiceNet.cli
{
    // Fully aligned with cosyvoice/cli/model.py
    public class CosyVoiceModel : IDisposable
    {
        public string Device { get; private set; }
        public string LlmDevice { get; private set; }
        public string FlowDevice { get; private set; }
        public string HiftDevice { get; private set; }
        public CosyVoiceBackend RequestedBackend { get; private set; }
        public CosyVoiceBackend ActiveBackend { get; private set; }
        public CosyVoiceBackend LlmBackend { get; private set; }
        public CosyVoiceBackend FlowBackend { get; private set; }
        public CosyVoiceBackend HiftBackend { get; private set; }
        public nn.Module Llm;
        public nn.Module Flow;
        public nn.Module Hift;
        public bool Fp16;
        public int SamplingTopK { get; set; } = 25;
        public ICosyVoiceLogger? Logger { get; set; }
        public bool TraceLlmInputShapes { get; set; }
        public bool TraceGeneratedTokens { get; set; }
        private QwenAttentionBackend _qwenAttentionBackend = QwenAttentionBackend.Auto;
        public QwenAttentionBackend QwenAttentionBackend
        {
            get => _qwenAttentionBackend;
            set
            {
                _qwenAttentionBackend = value;
                if (Llm is TransformerLM transformerLm)
                    transformerLm.QwenAttentionBackend = value;
            }
        }

        private QwenMlpBackend _qwenMlpBackend = QwenMlpBackend.Auto;
        public QwenMlpBackend QwenMlpBackend
        {
            get => _qwenMlpBackend;
            set
            {
                _qwenMlpBackend = value;
                if (Llm is TransformerLM transformerLm)
                    transformerLm.QwenMlpBackend = value;
            }
        }

        private QwenKvCacheBackend _qwenKvCacheBackend = QwenKvCacheBackend.Standard;
        public QwenKvCacheBackend QwenKvCacheBackend
        {
            get => Llm is Qwen2LM qwen2 ? qwen2.QwenKvCacheBackend : _qwenKvCacheBackend;
            set
            {
                _qwenKvCacheBackend = value;
                if (Llm is Qwen2LM qwen2)
                    qwen2.QwenKvCacheBackend = value;
            }
        }

        private LegacyTransformerCacheBackend _legacyTransformerCacheBackend = LegacyTransformerCacheBackend.Standard;
        public LegacyTransformerCacheBackend LegacyTransformerCacheBackend
        {
            get => Llm is TransformerLM transformerLm ? transformerLm.LegacyTransformerCacheBackend : _legacyTransformerCacheBackend;
            set
            {
                _legacyTransformerCacheBackend = value;
                if (Llm is TransformerLM transformerLm)
                    transformerLm.LegacyTransformerCacheBackend = value;
            }
        }

        private ICosyVoiceProfiler? _profiler;
        public ICosyVoiceProfiler? Profiler
        {
            get => _profiler;
            set
            {
                _profiler = value;
                if (Llm is TransformerLM transformerLm)
                    transformerLm.Profiler = value;
            }
        }

        public bool PreallocateQwenKvCache
        {
            get => QwenKvCacheBackend == QwenKvCacheBackend.Preallocated;
            set => QwenKvCacheBackend = value ? QwenKvCacheBackend.Preallocated : QwenKvCacheBackend.Standard;
        }

        public int TokenMinHopLen;
        public int TokenMaxHopLen;
        public int TokenOverlapLen = 20;

        public int MelOverlapLen;
        public Tensor MelWindow;

        public int MelCacheLen = 20;
        public int SourceCacheLen;
        public Tensor SpeechWindow;

        public int StreamScaleFactor = 1;
        protected readonly object _lock = new object();

        public readonly Dictionary<string, List<int>> TtsSpeechTokenDict = new Dictionary<string, List<int>>();
        public readonly Dictionary<string, bool> LlmEndDict = new Dictionary<string, bool>();
        public readonly Dictionary<string, Tensor> MelOverlapDict = new Dictionary<string, Tensor>();
        public readonly Dictionary<string, Tensor> FlowCacheDict = new Dictionary<string, Tensor>();
        public readonly Dictionary<string, Dictionary<string, Tensor>> HiftCacheDict = new Dictionary<string, Dictionary<string, Tensor>>();
        public readonly List<int> SilentTokens = new List<int>();

        protected ILLMInference iLLM { get { return Llm as ILLMInference; } }
        protected IFlowInference iFlow { get { return Flow as IFlowInference; } }
        protected IHiftInference iHiFT { get { return Hift as IHiftInference; } }

        protected void RecordProfile(string name, double milliseconds, IReadOnlyDictionary<string, string>? tags = null)
        {
            Profiler?.Record(name, milliseconds, tags);
        }

        protected Stopwatch StartProfileTimer(string? device = null)
        {
            SynchronizeIfCuda(device);
            return Stopwatch.StartNew();
        }

        protected void StopProfileTimer(string name, Stopwatch stopwatch, string? device = null, IReadOnlyDictionary<string, string>? tags = null)
        {
            SynchronizeIfCuda(device);
            stopwatch.Stop();
            RecordProfile(name, stopwatch.Elapsed.TotalMilliseconds, tags);
        }

        public CosyVoiceModel(nn.Module llm, nn.Module flow, nn.Module hift, bool fp16 = false, CosyVoiceBackend backend = CosyVoiceBackend.Auto)
        {
            ApplyBackendSelection(CosyVoiceBackendResolver.Resolve(backend), moveModules: false);
            Llm = llm ?? throw new ArgumentNullException(nameof(llm));
            Flow = flow ?? throw new ArgumentNullException(nameof(flow));
            Hift = hift ?? throw new ArgumentNullException(nameof(hift));
            Fp16 = fp16;
            if (Llm is TransformerLM transformerLm)
            {
                transformerLm.Profiler = Profiler;
                transformerLm.LegacyTransformerCacheBackend = LegacyTransformerCacheBackend;
                transformerLm.QwenKvCacheBackend = QwenKvCacheBackend;
                transformerLm.QwenAttentionBackend = QwenAttentionBackend;
                transformerLm.QwenMlpBackend = QwenMlpBackend;
            }

            try
            {
                var inputFrameRate = iFlow.InputFrameRate;
                TokenMinHopLen = 2 * Math.Max(1, (int)inputFrameRate);
                TokenMaxHopLen = 4 * TokenMinHopLen;
                MelOverlapLen = (int)(TokenOverlapLen / Math.Max(1, (double)inputFrameRate) * 22050.0 / 256.0);
                if (MelOverlapLen < 1) MelOverlapLen = 1;

                // Create Hamming window as tensor
                MelWindow = CreateHammingWindowTensor(Math.Max(1, 2 * MelOverlapLen));
                SourceCacheLen = MelCacheLen * 256;
                SpeechWindow = CreateHammingWindowTensor(Math.Max(1, 2 * SourceCacheLen));
            }
            catch
            {
                TokenMinHopLen = 20;
                TokenMaxHopLen = 80;
                MelOverlapLen = 1;
                MelWindow = torch.ones(2);
                SourceCacheLen = MelCacheLen * 256;
                SpeechWindow = torch.ones(2 * SourceCacheLen);
            }
        }

        protected static Tensor CreateHammingWindowTensor(int length)
        {
            if (length == 1) return torch.tensor(0.5f);

            var window = new float[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = (float)(0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (length - 1)));
            }
            return torch.tensor(window);
        }

        public void Load(string llmModel, string flowModel, string hiftModel)
        {
            if (!string.IsNullOrWhiteSpace(llmModel))
            {
                try
                {
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, $"Loading LLM from {llmModel}.");
                    LoadModuleWeights(Llm, llmModel, "LLM");
                    Llm.to(torch.device(LlmDevice));
                    Llm.eval();
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, "Successfully loaded LLM weights.");
                }
                catch (Exception ex) 
                { 
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Error, "Failed to load LLM model.", ex);
                    throw;
                }
            }

            if (!string.IsNullOrWhiteSpace(flowModel))
            {
                try
                {
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, $"Loading Flow from {flowModel}.");
                    LoadModuleWeights(Flow, flowModel, "Flow");
                    MoveFlowToConfiguredDevice();
                    Flow.eval();
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, "Successfully loaded Flow weights.");
                }
                catch (Exception ex) 
                { 
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Error, "Failed to load Flow model.", ex);
                    throw;
                }
            }

            if (!string.IsNullOrWhiteSpace(hiftModel))
            {
                try
                {
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, $"Loading HiFT from {hiftModel}.");
                    LoadHiftWeights(Hift, hiftModel);
                    Hift.to(torch.device(HiftDevice));
                    Hift.eval();
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, "Successfully loaded HiFT weights.");
                }
                catch (Exception ex) 
                { 
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Error, "Failed to load Hift model.", ex);
                    throw;
                }
            }
        }

        public void SetBackend(CosyVoiceBackend backend)
        {
            ApplyBackendSelection(CosyVoiceBackendResolver.Resolve(backend), moveModules: true);
        }

        public void SetComponentBackends(CosyVoiceBackend llmBackend, CosyVoiceBackend flowBackend, CosyVoiceBackend hiftBackend)
        {
            var llmSelection = CosyVoiceBackendResolver.Resolve(llmBackend);
            var flowSelection = CosyVoiceBackendResolver.Resolve(flowBackend);
            var hiftSelection = CosyVoiceBackendResolver.Resolve(hiftBackend);

            RequestedBackend = llmBackend == flowBackend && flowBackend == hiftBackend ? llmSelection.RequestedBackend : CosyVoiceBackend.Auto;
            ActiveBackend = llmSelection.ActiveBackend == flowSelection.ActiveBackend && flowSelection.ActiveBackend == hiftSelection.ActiveBackend
                ? llmSelection.ActiveBackend
                : CosyVoiceBackend.Auto;
            Device = llmSelection.Device == flowSelection.Device && flowSelection.Device == hiftSelection.Device
                ? llmSelection.Device
                : "mixed";

            LlmBackend = llmSelection.ActiveBackend;
            FlowBackend = flowSelection.ActiveBackend;
            HiftBackend = hiftSelection.ActiveBackend;
            LlmDevice = llmSelection.Device;
            FlowDevice = flowSelection.Device;
            HiftDevice = hiftSelection.Device;

            Llm?.to(torch.device(LlmDevice));
            MoveFlowToConfiguredDevice();
            Hift?.to(torch.device(HiftDevice));
            ClearRuntimeCaches();
        }

        private void ApplyBackendSelection(CosyVoiceBackendSelection selection, bool moveModules)
        {
            RequestedBackend = selection.RequestedBackend;
            ActiveBackend = selection.ActiveBackend;
            Device = selection.Device;
            LlmBackend = selection.ActiveBackend;
            FlowBackend = selection.ActiveBackend;
            HiftBackend = selection.ActiveBackend;
            LlmDevice = selection.Device;
            FlowDevice = selection.Device;
            HiftDevice = selection.Device;

            if (!moveModules)
                return;

            var device = torch.device(Device);
            Llm?.to(device);
            MoveFlowToConfiguredDevice();
            Hift?.to(device);
            MelWindow = MelWindow?.to(device);
            SpeechWindow = SpeechWindow?.to(device);
            ClearRuntimeCaches();
        }

        private void ClearRuntimeCaches()
        {
            lock (_lock)
            {
                foreach (var tensor in MelOverlapDict.Values)
                    tensor?.Dispose();
                foreach (var tensor in FlowCacheDict.Values)
                    tensor?.Dispose();
                foreach (var cache in HiftCacheDict.Values)
                    DisposeTensorCache(cache);

                TtsSpeechTokenDict.Clear();
                LlmEndDict.Clear();
                MelOverlapDict.Clear();
                FlowCacheDict.Clear();
                HiftCacheDict.Clear();
            }
        }

        protected void CleanupRuntimeState(string uuid)
        {
            lock (_lock)
            {
                TtsSpeechTokenDict.Remove(uuid);
                LlmEndDict.Remove(uuid);

                if (MelOverlapDict.Remove(uuid, out var melOverlap))
                    melOverlap?.Dispose();
                if (FlowCacheDict.Remove(uuid, out var flowCache))
                    flowCache?.Dispose();
                if (HiftCacheDict.Remove(uuid, out var hiftCache))
                    DisposeTensorCache(hiftCache);
            }
        }

        private static void DisposeTensorCache(Dictionary<string, Tensor> cache)
        {
            if (cache is null)
                return;

            foreach (var tensor in cache.Values)
                tensor?.Dispose();
            cache.Clear();
        }

        protected static void DisposeTensorIfDifferent(Tensor previous, Tensor current)
        {
            if (previous is null || ReferenceEquals(previous, current))
                return;

            previous.Dispose();
        }

        protected static void DisposeTensorCacheIfDifferent(Dictionary<string, Tensor> previous, Dictionary<string, Tensor> current)
        {
            if (previous is null || ReferenceEquals(previous, current))
                return;

            foreach (var pair in previous)
            {
                if (current is not null &&
                    current.TryGetValue(pair.Key, out var replacement) &&
                    ReferenceEquals(pair.Value, replacement))
                {
                    continue;
                }

                pair.Value?.Dispose();
            }

            previous.Clear();
        }

        protected static Dictionary<string, Tensor> BuildSpeechOutput(Tensor ttsSpeech)
        {
            if (ttsSpeech.device.type == DeviceType.CPU)
                return new Dictionary<string, Tensor> { ["tts_speech"] = ttsSpeech };

            var cpuSpeech = ttsSpeech.cpu();
            ttsSpeech.Dispose();
            return new Dictionary<string, Tensor> { ["tts_speech"] = cpuSpeech };
        }

        protected static Tensor ExtractTensorOrDefault(Dictionary<string, object> dict, string key, List<Tensor> ownedDefaults, Func<Tensor> createDefault)
        {
            var tensor = ExtractTensor(dict, key);
            if (tensor is not null)
                return tensor;

            tensor = createDefault();
            ownedDefaults.Add(tensor);
            return tensor;
        }

        protected static void DisposeOwnedDefaults(List<Tensor> ownedDefaults)
        {
            foreach (var tensor in ownedDefaults)
                tensor?.Dispose();
            ownedDefaults.Clear();
        }

        public virtual void Dispose()
        {
            ClearRuntimeCaches();
            MelWindow?.Dispose();
            SpeechWindow?.Dispose();
            Llm?.Dispose();
            Flow?.Dispose();
            Hift?.Dispose();
        }

        private void MoveFlowToConfiguredDevice()
        {
            if (Flow is null)
                return;

            Flow.to(torch.device(FlowDevice));
            Flow.to(UseFlowFp16 ? ScalarType.Float16 : ScalarType.Float32);
        }

        protected void SynchronizeIfCuda()
        {
            SynchronizeIfCuda(Device);
        }

        protected void SynchronizeIfCuda(string device)
        {
            if (device == "cuda")
                torch.cuda.synchronize(torch.device(device));
        }

        protected bool UseFlowFp16 => Fp16 && FlowDevice == "cuda";

        protected Tensor ToFlowFloat(Tensor tensor)
        {
            tensor = tensor.to(FlowDevice);
            return UseFlowFp16 ? tensor.to(ScalarType.Float16) : tensor;
        }

        protected Tensor ToHiftMel(Tensor tensor)
        {
            return tensor.to(ScalarType.Float32).to(HiftDevice);
        }

        private static void LoadModuleWeights(nn.Module module, string modelPath, string label, ISet<string> allowedMissingKeys = null)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"{label} checkpoint not found.", modelPath);

            var stateDict = PtDirectLoader.LoadTensorsFromPt(modelPath);
            if (stateDict.Count == 0)
                throw new InvalidOperationException($"{label} checkpoint did not contain any tensors that CosyVoiceNet can load.");

            var moduleStateDict = module.state_dict();
            NormalizeLegacyWeightNormKeys(stateDict, moduleStateDict);
            var shapeMismatches = stateDict
                .Where(kv => moduleStateDict.TryGetValue(kv.Key, out var target) && !target.shape.SequenceEqual(kv.Value.shape))
                .Take(20)
                .Select(kv => $"{kv.Key}: module={string.Join("x", moduleStateDict[kv.Key].shape)}, checkpoint={string.Join("x", kv.Value.shape)}")
                .ToArray();
            if (shapeMismatches.Length != 0)
                throw new InvalidOperationException($"{label} checkpoint shape mismatch. {string.Join("; ", shapeMismatches)}");

            IList<string> missingKeys;
            IList<string> unexpectedKeys;
            try
            {
                (missingKeys, unexpectedKeys) = module.load_state_dict(stateDict, strict: true);
            }
            catch (InvalidOperationException)
            {
                (missingKeys, unexpectedKeys) = module.load_state_dict(stateDict, strict: false);
            }

            var missingNotAllowed = missingKeys
                .Where(key => !key.EndsWith(".pos_enc.pe", StringComparison.Ordinal) &&
                              (allowedMissingKeys == null || !allowedMissingKeys.Contains(key)))
                .ToArray();

            if (missingNotAllowed.Length != 0 || unexpectedKeys.Count != 0)
            {
                throw new InvalidOperationException(
                    $"{label} checkpoint key mismatch. Missing: {string.Join(", ", missingNotAllowed.Take(20))}" +
                    $"{(missingNotAllowed.Length > 20 ? $" (+{missingNotAllowed.Length - 20} more)" : string.Empty)}; " +
                    $"Unexpected: {string.Join(", ", unexpectedKeys.Take(20))}" +
                    $"{(unexpectedKeys.Count > 20 ? $" (+{unexpectedKeys.Count - 20} more)" : string.Empty)}");
            }

        }

        private static void NormalizeLegacyWeightNormKeys(IDictionary<string, Tensor> stateDict, IDictionary<string, Tensor> moduleStateDict)
        {
            var additions = new List<(string key, Tensor value)>();
            var removals = new List<string>();

            foreach (var key in stateDict.Keys.ToArray())
            {
                string suffix;
                if (key.EndsWith(".weight_g", StringComparison.Ordinal))
                    suffix = ".parametrizations.weight.original0";
                else if (key.EndsWith(".weight_v", StringComparison.Ordinal))
                    suffix = ".parametrizations.weight.original1";
                else
                    continue;

                var prefix = key[..^".weight_g".Length];
                if (key.EndsWith(".weight_v", StringComparison.Ordinal))
                    prefix = key[..^".weight_v".Length];
                var mappedKey = prefix + suffix;
                if (moduleStateDict.ContainsKey(mappedKey))
                {
                    additions.Add((mappedKey, stateDict[key]));
                    removals.Add(key);
                }
            }

            foreach (var key in stateDict.Keys.ToArray())
            {
                const string pythonBlockNorm = ".block.2.";
                if (!key.Contains(pythonBlockNorm, StringComparison.Ordinal))
                    continue;

                var mappedKey = key.Replace(pythonBlockNorm, ".block.1.", StringComparison.Ordinal);
                if (moduleStateDict.ContainsKey(mappedKey) && !stateDict.ContainsKey(mappedKey))
                {
                    additions.Add((mappedKey, stateDict[key]));
                    removals.Add(key);
                }
            }

            foreach (var key in removals)
                stateDict.Remove(key);
            foreach (var (key, value) in additions)
                stateDict[key] = value;
        }

        private static void LoadHiftWeights(nn.Module hift, string modelPath)
        {
            var allowedMissing = new HashSet<string>(StringComparer.Ordinal)
            {
                "_stftWindow",
                "stft_window",
                "m_source._uv",
                "m_source.l_sin_gen.rand_ini",
                "m_source.l_sin_gen.sine_waves",
            };
            try
            {
                LoadModuleWeights(hift, modelPath, "HiFT", allowedMissing);
            }
            catch (Exception directLoadError)
            {
                // Python strips a leading "generator." prefix from hift.pt before loading.
                // Loading through a wrapper with a generator child gives the native loader the same key shape.
                try
                {
                    using var wrapper = new GeneratorCheckpointWrapper(hift);
                    LoadModuleWeights(wrapper, modelPath, "HiFT", new HashSet<string>(allowedMissing.Select(key => "generator." + key), StringComparer.Ordinal));
                }
                catch (Exception wrapperLoadError)
                {
                    throw new InvalidOperationException(
                        $"Failed to load HiFT checkpoint directly or as a generator-prefixed checkpoint. Direct load: {directLoadError.Message}; wrapper load: {wrapperLoadError.Message}",
                        wrapperLoadError);
                }
            }
        }

        private sealed class GeneratorCheckpointWrapper : nn.Module
        {
            public readonly nn.Module generator;

            public GeneratorCheckpointWrapper(nn.Module generator) : base("GeneratorCheckpointWrapper")
            {
                this.generator = generator ?? throw new ArgumentNullException(nameof(generator));
                RegisterComponents();
            }
        }

        // LLM job - runs in background thread
        protected virtual void LlmJob(Tensor text, Tensor promptText, Tensor llmPromptSpeechToken, Tensor llmEmbedding, string uuid)
        {
            var perf = false;
            if (TraceLlmInputShapes)
            {
                Logger?.Log(
                    CosyVoiceLogLevel.Trace,
                    "LLM input shapes.",
                    tags: new Dictionary<string, string>
                    {
                        ["text"] = ShapeOf(text),
                        ["prompt_text"] = ShapeOf(promptText),
                        ["prompt_speech"] = ShapeOf(llmPromptSpeechToken),
                        ["embedding"] = ShapeOf(llmEmbedding)
                    });
            }

            SynchronizeIfCuda(LlmDevice);
            var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            var profileSw = StartProfileTimer(LlmDevice);
            int curSilentTokenNum = 0;
            const int maxSilentTokenNum = 5;
            var traceTokens = TraceGeneratedTokens;
            var generatedTokens = new List<int>();
            var appendedTokens = new List<int>();
            var generatedTokenCount = 0;
            var appendedTokenCount = 0;
            using (torch.inference_mode())
            using (torch.NewDisposeScope())
            {
                var samplingTopK = SamplingTopK > 0 ? SamplingTopK : 25;

                var tokenGenerator = iLLM.inference(
                    text: text.to(LlmDevice),
                    textLen: torch.tensor(new long[] { text.shape[1] }, dtype: ScalarType.Int64).to(LlmDevice),
                    promptText: promptText.to(LlmDevice),
                    promptTextLen: torch.tensor(new long[] { promptText.shape[1] }, dtype: ScalarType.Int64).to(LlmDevice),
                    promptSpeechToken: llmPromptSpeechToken.to(LlmDevice),
                    promptSpeechTokenLen: torch.tensor(new long[] { llmPromptSpeechToken.shape[1] }, dtype: ScalarType.Int64).to(LlmDevice),
                    embedding: llmEmbedding.to(LlmDevice),
                    sampling: samplingTopK,
                    uuid: uuid
                );

                if (tokenGenerator != null)
                {
                    foreach (var token in tokenGenerator)
                    {
                        generatedTokenCount++;
                        if (traceTokens)
                            generatedTokens.Add(token);
                        if (SilentTokens.Contains(token))
                        {
                            curSilentTokenNum++;
                            if (curSilentTokenNum > maxSilentTokenNum)
                                continue;
                        }
                        else
                        {
                            curSilentTokenNum = 0;
                        }

                        lock (_lock)
                        {
                            TtsSpeechTokenDict[uuid].Add(token);
                        }
                        appendedTokenCount++;
                        if (traceTokens)
                            appendedTokens.Add(token);
                    }
                }
            }

            if (traceTokens)
            {
                Logger?.Log(
                    CosyVoiceLogLevel.Trace,
                    "LLM generated speech tokens.",
                    tags: new Dictionary<string, string>
                    {
                        ["uuid"] = uuid,
                        ["generated_count"] = generatedTokens.Count.ToString(),
                        ["appended_count"] = appendedTokens.Count.ToString(),
                        ["generated_head"] = string.Join(", ", generatedTokens.Take(64)),
                        ["appended_head"] = string.Join(", ", appendedTokens.Take(64))
                    });
            }

            if (perf)
            {
                SynchronizeIfCuda(LlmDevice);
                Console.WriteLine($"[CosyVoicePerf] llm_job_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
            }
            StopProfileTimer("llm.job_total", profileSw, LlmDevice, new Dictionary<string, string>
            {
                ["generated_tokens"] = generatedTokenCount.ToString(),
                ["appended_tokens"] = appendedTokenCount.ToString()
            });

            lock (_lock)
            {
                LlmEndDict[uuid] = true;
            }
        }

        private static string ShapeOf(Tensor tensor)
        {
            if (tensor is null)
                return "<null>";
            return "[" + string.Join(",", tensor.shape) + "]";
        }

        // Voice conversion job - runs in background thread
        protected virtual void VcJob(Tensor sourceSpeechToken, string uuid)
        {
            lock (_lock)
            {
                TtsSpeechTokenDict[uuid] = sourceSpeechToken.data<int>().ToArray().ToList();
                LlmEndDict[uuid] = true;
            }
        }

        // Token to waveform conversion
        public virtual Tensor Token2Wav(Tensor token, Tensor promptToken, Tensor promptFeat, Tensor embedding, string uuid, bool finalize = false, double speed = 1.0)
        {
            using (torch.no_grad())
            {
                // Flow inference
                var flowResult = iFlow.Inference(
                        token: token.to(ScalarType.Int32).to(FlowDevice),
                        token_len: torch.tensor(new long[] { token.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        prompt_token: promptToken.to(FlowDevice),
                        prompt_token_len: torch.tensor(new long[] { promptToken.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        prompt_feat: ToFlowFloat(promptFeat),
                        prompt_feat_len: torch.tensor(new long[] { promptFeat.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        embedding: ToFlowFloat(embedding),
                        cache: FlowCacheDict.ContainsKey(uuid) ? FlowCacheDict[uuid] : null
                    );

                var ttsMel = ((ValueTuple<Tensor, Tensor>)flowResult).Item1;
                var previousFlowCache = FlowCacheDict.TryGetValue(uuid, out var oldFlowCache) ? oldFlowCache : null;
                var nextFlowCache = ((ValueTuple<Tensor, Tensor>)flowResult).Item2;
                FlowCacheDict[uuid] = nextFlowCache;
                DisposeTensorIfDifferent(previousFlowCache, nextFlowCache);

                // Mel overlap fade in out
                if (MelOverlapDict[uuid].shape[2] > 0)
                {
                    ttsMel = Common.FadeInOut(ttsMel, MelOverlapDict[uuid].to(ttsMel.device), MelWindow.to(ttsMel.device));
                }
                ttsMel = ToHiftMel(ttsMel);

                // Append hift cache
                Tensor hiftCacheSource;
                if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                {
                    var hiftCacheMel = HiftCacheDict[uuid]["mel"];
                    hiftCacheSource = HiftCacheDict[uuid]["source"];
                    ttsMel = torch.cat(new[] { hiftCacheMel, ttsMel }, dim: 2);
                }
                else
                {
                    hiftCacheSource = torch.zeros(1, 1, 0, device: torch.device(HiftDevice));
                }

                Tensor ttsSpeech, ttsSource;

                // Keep overlap mel and hift cache
                if (!finalize)
                {
                    var previousMelOverlap = MelOverlapDict.TryGetValue(uuid, out var oldMelOverlap) ? oldMelOverlap : null;
                    var nextMelOverlap = ttsMel[.., .., ^MelOverlapLen..];
                    MelOverlapDict[uuid] = nextMelOverlap;
                    DisposeTensorIfDifferent(previousMelOverlap, nextMelOverlap);
                    ttsMel = ttsMel[.., .., ..^MelOverlapLen];

                    // HiFT inference
                    var hiftResult = iHiFT.Inference(ttsMel, hiftCacheSource);
                    ttsSpeech = ((ValueTuple<Tensor, Tensor>)hiftResult).Item1;
                    ttsSource = ((ValueTuple<Tensor, Tensor>)hiftResult).Item2;

                    if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                    {
                        ttsSpeech = Common.FadeInOut(ttsSpeech, HiftCacheDict[uuid]["speech"], SpeechWindow.to(ttsSpeech.device));
                    }

                    var previousHiftCache = HiftCacheDict.TryGetValue(uuid, out var oldHiftCache) ? oldHiftCache : null;
                    var nextHiftCache = new Dictionary<string, Tensor>
                    {
                        ["mel"] = ttsMel[.., .., ^MelCacheLen..],
                        ["source"] = ttsSource[.., .., ^SourceCacheLen..],
                        ["speech"] = ttsSpeech[.., ^SourceCacheLen..]
                    };
                    HiftCacheDict[uuid] = nextHiftCache;
                    DisposeTensorCacheIfDifferent(previousHiftCache, nextHiftCache);

                    ttsSpeech = ttsSpeech[.., ..^SourceCacheLen];
                }
                else
                {
                    // Speed adjustment for non-streaming mode
                    if (speed != 1.0)
                    {
                        if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                            throw new InvalidOperationException("Speed change only supports non-stream inference mode");

                        ttsMel = interpolate(ttsMel, size: new long[] { (long)(ttsMel.shape[2] / speed) }, mode: InterpolationMode.Linear);
                    }

                    // HiFT inference
                    var hiftResult = iHiFT.Inference(ttsMel, hiftCacheSource);
                    ttsSpeech = ((ValueTuple<Tensor, Tensor>)hiftResult).Item1;

                    if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                    {
                        ttsSpeech = Common.FadeInOut(ttsSpeech, HiftCacheDict[uuid]["speech"], SpeechWindow.to(ttsSpeech.device));
                    }
                }

                return ttsSpeech;
            }
        }

        // Main TTS method - accepts model_input dictionary like Python
        public virtual IEnumerable<Dictionary<string, Tensor>> Tts(
            Dictionary<string, object> modelInput,
            bool stream = false,
            double speed = 1.0)
        {
            // Extract tensors from modelInput dictionary
            var ownedDefaults = new List<Tensor>();
            var text = ExtractTensorOrDefault(modelInput, "text", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var flowEmbedding = ExtractTensorOrDefault(modelInput, "flow_embedding", ownedDefaults, () => torch.zeros(0, 192));
            var llmEmbedding = ExtractTensorOrDefault(modelInput, "llm_embedding", ownedDefaults, () => torch.zeros(0, 192));
            var promptText = ExtractTensorOrDefault(modelInput, "prompt_text", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var llmPromptSpeechToken = ExtractTensorOrDefault(modelInput, "llm_prompt_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var flowPromptSpeechToken = ExtractTensorOrDefault(modelInput, "flow_prompt_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var promptSpeechFeat = ExtractTensorOrDefault(modelInput, "prompt_speech_feat", ownedDefaults, () => torch.zeros(1, 0, 80));
            var sourceSpeechToken = ExtractTensorOrDefault(modelInput, "source_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));

            // Generate UUID for this inference session
            var thisUuid = Guid.NewGuid().ToString();

            // Initialize session state
            lock (_lock)
            {
                TtsSpeechTokenDict[thisUuid] = new List<int>();
                LlmEndDict[thisUuid] = false;
                HiftCacheDict[thisUuid] = null;
                MelOverlapDict[thisUuid] = torch.zeros(1, 80, 0);
                FlowCacheDict[thisUuid] = torch.zeros(1, 80, 0, 2);
            }

            Thread llmThread = null;
            try
            {
            // Start LLM thread
            if (sourceSpeechToken.shape[1] == 0)
            {
                llmThread = new Thread(() => LlmJob(text, promptText, llmPromptSpeechToken, llmEmbedding, thisUuid));
            }
            else
            {
                llmThread = new Thread(() => VcJob(sourceSpeechToken, thisUuid));
            }
            llmThread.Start();

            if (stream)
            {
                // Streaming mode
                var tokenHopLen = TokenMinHopLen;

                while (true)
                {
                    Thread.Sleep(100);

                    int currentTokenCount;
                    lock (_lock)
                    {
                        currentTokenCount = TtsSpeechTokenDict[thisUuid].Count;
                    }

                    if (currentTokenCount >= tokenHopLen + TokenOverlapLen)
                    {
                        List<int> tokens;
                        lock (_lock)
                        {
                            tokens = TtsSpeechTokenDict[thisUuid].Take(tokenHopLen + TokenOverlapLen).ToList();
                        }

                        var thisTtsSpeechToken = torch.tensor(tokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                        var thisTtsSpeech = Token2Wav(
                            token: thisTtsSpeechToken,
                            promptToken: flowPromptSpeechToken,
                            promptFeat: promptSpeechFeat,
                            embedding: flowEmbedding,
                            uuid: thisUuid,
                            finalize: false
                        );

                        thisTtsSpeechToken.Dispose();
                        yield return BuildSpeechOutput(thisTtsSpeech);

                        lock (_lock)
                        {
                            TtsSpeechTokenDict[thisUuid] = TtsSpeechTokenDict[thisUuid].Skip(tokenHopLen).ToList();
                        }

                        // Increase token hop length progressively
                        tokenHopLen = Math.Min(TokenMaxHopLen, (int)(tokenHopLen * StreamScaleFactor));
                    }

                    bool isLlmEnd;
                    lock (_lock)
                    {
                        isLlmEnd = LlmEndDict[thisUuid];
                        currentTokenCount = TtsSpeechTokenDict[thisUuid].Count;
                    }

                    if (isLlmEnd && currentTokenCount < tokenHopLen + TokenOverlapLen)
                        break;
                }

                llmThread.Join();

                // Process remaining tokens
                List<int> remainingTokens;
                lock (_lock)
                {
                    remainingTokens = TtsSpeechTokenDict[thisUuid].ToList();
                }

                if (remainingTokens.Count > 0)
                {
                    var thisTtsSpeechToken = torch.tensor(remainingTokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                    var thisTtsSpeech = Token2Wav(
                        token: thisTtsSpeechToken,
                        promptToken: flowPromptSpeechToken,
                        promptFeat: promptSpeechFeat,
                        embedding: flowEmbedding,
                        uuid: thisUuid,
                        finalize: true
                    );

                    thisTtsSpeechToken.Dispose();
                    yield return BuildSpeechOutput(thisTtsSpeech);
                }
            }
            else
            {
                // Non-streaming mode
                List<int> allTokens;
                if (sourceSpeechToken.shape[1] == 0)
                {
                    llmThread.Join();
                    lock (_lock)
                    {
                        allTokens = TtsSpeechTokenDict[thisUuid].ToList();
                    }
                }
                else
                {
                    llmThread.Join();
                    lock (_lock)
                    {
                        allTokens = TtsSpeechTokenDict[thisUuid].ToList();
                    }
                }


                var thisTtsSpeechToken = torch.tensor(allTokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                var thisTtsSpeech = Token2Wav(
                    token: thisTtsSpeechToken,
                    promptToken: flowPromptSpeechToken,
                    promptFeat: promptSpeechFeat,
                    embedding: flowEmbedding,
                    uuid: thisUuid,
                    finalize: true,
                    speed: speed
                );

                thisTtsSpeechToken.Dispose();
                yield return BuildSpeechOutput(thisTtsSpeech);
            }
            }
            finally
            {
                if (llmThread?.IsAlive == true)
                    llmThread.Join();
                CleanupRuntimeState(thisUuid);
                DisposeOwnedDefaults(ownedDefaults);
            }

            // TorchSharp manages CUDA memory automatically
        }

        // Helper method to extract tensor from dictionary
        protected static Tensor ExtractTensor(Dictionary<string, object> dict, string key)
        {
            if (!dict.TryGetValue(key, out var value)) return null;
            return value as Tensor;
        }

        public void LoadJit(string textEncoderPath, string llmPath, string flowEncoderPath)
        {
            Console.WriteLine("[CosyVoiceModel] Loading JIT models...");
            // TODO: Implement JIT loading
        }

        public void LoadTrt(string estimatorPlanPath, string estimatorOnnxPath, int trtConcurrent, bool fp16)
        {
            Console.WriteLine("[CosyVoiceModel] Loading TensorRT models...");
            // TODO: Implement TensorRT loading
        }

        public void LoadVllm(string vllmPath)
        {
            Console.WriteLine($"Loading VLLM from {vllmPath}");
            // TODO: Implement vLLM loading
        }
    }

    // CosyVoice2 Model - Different token2wav and tts logic
    public class CosyVoice2Model : CosyVoiceModel
    {
        public int TokenHopLen;

        public CosyVoice2Model(nn.Module llm, nn.Module flow, nn.Module hift, bool fp16 = false, CosyVoiceBackend backend = CosyVoiceBackend.Auto) 
            : base(llm, flow, hift, fp16, backend)
        {
            // Override initialization for CosyVoice2
            TokenHopLen = 25;
            TokenMinHopLen = 25; // Not used in V2, but keep for compatibility
            TokenMaxHopLen = 4 * TokenHopLen;
            StreamScaleFactor = 2;
            MelCacheLen = 8;
            SourceCacheLen = MelCacheLen * 480;
            SpeechWindow = CreateHammingWindowTensor(2 * SourceCacheLen);
        }

        // CosyVoice2 has different token2wav implementation
        public override Tensor Token2Wav(Tensor token, Tensor promptToken, Tensor promptFeat, Tensor embedding, string uuid, bool finalize = false, double speed = 1.0)
        {
            // For V2, we need token_offset parameter - get it from context or default to 0
            return Token2WavWithOffset(token, promptToken, promptFeat, embedding, 0, uuid, false, finalize, speed);
        }

        public virtual Tensor Token2WavWithOffset(Tensor token, Tensor promptToken, Tensor promptFeat, Tensor embedding, int tokenOffset, string uuid, bool streaming = false, bool finalize = false, double speed = 1.0)
        {
            using (torch.no_grad())
            {
                var totalSw = Profiler is not null ? StartProfileTimer() : null;
                // Flow inference with streaming parameter
                var flowSw = Profiler is not null ? StartProfileTimer(FlowDevice) : null;
                var flowResult = iFlow.Inference(
                    token: token.to(ScalarType.Int32).to(FlowDevice),
                    token_len: torch.tensor(new long[] { token.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                    prompt_token: promptToken.to(FlowDevice),
                    prompt_token_len: torch.tensor(new long[] { promptToken.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                    prompt_feat: ToFlowFloat(promptFeat),
                    prompt_feat_len: torch.tensor(new long[] { promptFeat.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                    embedding: ToFlowFloat(embedding),
                    streaming: streaming,
                    finalize: finalize
                );
                if (flowSw is not null)
                {
                    StopProfileTimer("cv2.flow.inference", flowSw, FlowDevice, new Dictionary<string, string>
                    {
                        ["streaming"] = streaming.ToString(),
                        ["finalize"] = finalize.ToString(),
                        ["tokens"] = token.shape[1].ToString()
                    });
                }
                var ttsMel = ((ValueTuple<Tensor, Tensor>)flowResult).Item1;

                // Get token_mel_ratio from Flow
                var tokenMelRatio = iFlow.token_mel_ratio;
                // Slice from token_offset
                ttsMel = ttsMel[.., .., (tokenOffset * tokenMelRatio)..];
                ttsMel = ToHiftMel(ttsMel);

                // Append hift cache
                Tensor hiftCacheSource;
                if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                {
                    var hiftCacheMel = HiftCacheDict[uuid]["mel"];
                    hiftCacheSource = HiftCacheDict[uuid]["source"];
                    ttsMel = torch.cat(new[] { hiftCacheMel, ttsMel }, dim: 2);
                }
                else
                {
                    hiftCacheSource = torch.zeros(1, 1, 0, device: torch.device(HiftDevice));
                }

                Tensor ttsSpeech, ttsSource;

                // Keep overlap mel and hift cache
                if (!finalize)
                {
                    // HiFT inference
                    var hiftSw = Profiler is not null ? StartProfileTimer(HiftDevice) : null;
                    var hiftResult = iHiFT.Inference(ttsMel, hiftCacheSource);
                    if (hiftSw is not null)
                    {
                        StopProfileTimer("cv2.hift.inference", hiftSw, HiftDevice, new Dictionary<string, string>
                        {
                            ["streaming"] = streaming.ToString(),
                            ["finalize"] = finalize.ToString(),
                            ["mel_frames"] = ttsMel.shape[2].ToString()
                        });
                    }
                    ttsSpeech = ((ValueTuple<Tensor, Tensor>)hiftResult).Item1;
                    ttsSource = ((ValueTuple<Tensor, Tensor>)hiftResult).Item2;

                    if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                    {
                        ttsSpeech = Common.FadeInOut(ttsSpeech, HiftCacheDict[uuid]["speech"], SpeechWindow.to(ttsSpeech.device));
                    }

                    var previousHiftCache = HiftCacheDict.TryGetValue(uuid, out var oldHiftCache) ? oldHiftCache : null;
                    var nextHiftCache = new Dictionary<string, Tensor>
                    {
                        ["mel"] = ttsMel[.., .., ^MelCacheLen..],
                        ["source"] = ttsSource[.., .., ^SourceCacheLen..],
                        ["speech"] = ttsSpeech[.., ^SourceCacheLen..]
                    };
                    HiftCacheDict[uuid] = nextHiftCache;
                    DisposeTensorCacheIfDifferent(previousHiftCache, nextHiftCache);

                    ttsSpeech = ttsSpeech[.., ..^SourceCacheLen];
                }
                else
                {
                    // Speed adjustment
                    if (speed != 1.0)
                    {
                        if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                            throw new InvalidOperationException("Speed change only supports non-stream inference mode");

                        ttsMel = interpolate(ttsMel, size: new long[] { (long)(ttsMel.shape[2] / speed) }, mode: InterpolationMode.Linear);
                    }

                    // HiFT inference
                    var hiftSw = Profiler is not null ? StartProfileTimer(HiftDevice) : null;
                    var hiftResult = iHiFT.Inference(ttsMel, hiftCacheSource);
                    if (hiftSw is not null)
                    {
                        StopProfileTimer("cv2.hift.inference", hiftSw, HiftDevice, new Dictionary<string, string>
                        {
                            ["streaming"] = streaming.ToString(),
                            ["finalize"] = finalize.ToString(),
                            ["mel_frames"] = ttsMel.shape[2].ToString()
                        });
                    }
                    ttsSpeech = ((ValueTuple<Tensor, Tensor>)hiftResult).Item1;

                    if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                    {
                        ttsSpeech = Common.FadeInOut(ttsSpeech, HiftCacheDict[uuid]["speech"], SpeechWindow.to(ttsSpeech.device));
                    }
                }

                if (totalSw is not null)
                {
                    StopProfileTimer("cv2.token2wav.total", totalSw, HiftDevice, new Dictionary<string, string>
                    {
                        ["streaming"] = streaming.ToString(),
                        ["finalize"] = finalize.ToString(),
                        ["tokens"] = token.shape[1].ToString()
                    });
                }
                return ttsSpeech;
            }
        }

        // CosyVoice2 has different TTS implementation with token_offset
        public override IEnumerable<Dictionary<string, Tensor>> Tts(
            Dictionary<string, object> modelInput,
            bool stream = false,
            double speed = 1.0)
        {
            var perf = false;
            var ttsSw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            // Extract tensors from modelInput dictionary
            var ownedDefaults = new List<Tensor>();
            var text = ExtractTensorOrDefault(modelInput, "text", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var flowEmbedding = ExtractTensorOrDefault(modelInput, "flow_embedding", ownedDefaults, () => torch.zeros(0, 192));
            var llmEmbedding = ExtractTensorOrDefault(modelInput, "llm_embedding", ownedDefaults, () => torch.zeros(0, 192));
            var promptText = ExtractTensorOrDefault(modelInput, "prompt_text", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var llmPromptSpeechToken = ExtractTensorOrDefault(modelInput, "llm_prompt_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var flowPromptSpeechToken = ExtractTensorOrDefault(modelInput, "flow_prompt_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));
            var promptSpeechFeat = ExtractTensorOrDefault(modelInput, "prompt_speech_feat", ownedDefaults, () => torch.zeros(1, 0, 80));
            var sourceSpeechToken = ExtractTensorOrDefault(modelInput, "source_speech_token", ownedDefaults, () => torch.zeros(1, 0, dtype: ScalarType.Int32));

            var thisUuid = Guid.NewGuid().ToString();

            lock (_lock)
            {
                TtsSpeechTokenDict[thisUuid] = new List<int>();
                LlmEndDict[thisUuid] = false;
                HiftCacheDict[thisUuid] = null;
            }

            Thread llmThread = null;
            try
            {
            // Start LLM thread
            if (sourceSpeechToken.shape[1] == 0)
            {
                llmThread = new Thread(() => LlmJob(text, promptText, llmPromptSpeechToken, llmEmbedding, thisUuid));
            }
            else
            {
                llmThread = new Thread(() => VcJob(sourceSpeechToken, thisUuid));
            }
            llmThread.Start();

            if (stream)
            {
                // Streaming mode with token_offset
                int tokenOffset = 0;
                var preLookaheadLen = iFlow.PreLookaheadLen;
                var promptTokenPad = (int)Math.Ceiling((double)flowPromptSpeechToken.shape[1] / TokenHopLen) * TokenHopLen - (int)flowPromptSpeechToken.shape[1];

                while (true)
                {
                    Thread.Sleep(100);

                    var thisTokenHopLen = (tokenOffset == 0) ? TokenHopLen + promptTokenPad : TokenHopLen;

                    int currentTokenCount;
                    lock (_lock)
                    {
                        currentTokenCount = TtsSpeechTokenDict[thisUuid].Count;
                    }

                    if (currentTokenCount - tokenOffset >= thisTokenHopLen + preLookaheadLen)
                    {
                        List<int> tokens;
                        lock (_lock)
                        {
                            tokens = TtsSpeechTokenDict[thisUuid].Take(tokenOffset + thisTokenHopLen + preLookaheadLen).ToList();
                        }

                        var thisTtsSpeechToken = torch.tensor(tokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                        var thisTtsSpeech = Token2WavWithOffset(
                            token: thisTtsSpeechToken,
                            promptToken: flowPromptSpeechToken,
                            promptFeat: promptSpeechFeat,
                            embedding: flowEmbedding,
                            tokenOffset: tokenOffset,
                            uuid: thisUuid,
                            streaming: stream,
                            finalize: false
                        );

                        tokenOffset += thisTokenHopLen;
                        TokenHopLen = Math.Min(TokenMaxHopLen, TokenHopLen * StreamScaleFactor);

                        thisTtsSpeechToken.Dispose();
                        yield return BuildSpeechOutput(thisTtsSpeech);
                    }

                    bool isLlmEnd;
                    lock (_lock)
                    {
                        isLlmEnd = LlmEndDict[thisUuid];
                        currentTokenCount = TtsSpeechTokenDict[thisUuid].Count;
                    }

                    if (isLlmEnd && currentTokenCount - tokenOffset < thisTokenHopLen + preLookaheadLen)
                        break;
                }

                llmThread.Join();

                // Process remaining tokens
                List<int> remainingTokens;
                lock (_lock)
                {
                    remainingTokens = TtsSpeechTokenDict[thisUuid].ToList();
                }

                if (remainingTokens.Count > 0)
                {
                    var thisTtsSpeechToken = torch.tensor(remainingTokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                    var thisTtsSpeech = Token2WavWithOffset(
                        token: thisTtsSpeechToken,
                        promptToken: flowPromptSpeechToken,
                        promptFeat: promptSpeechFeat,
                        embedding: flowEmbedding,
                        tokenOffset: tokenOffset,
                        uuid: thisUuid,
                        finalize: true
                    );

                    thisTtsSpeechToken.Dispose();
                    yield return BuildSpeechOutput(thisTtsSpeech);
                }
            }
            else
            {
                // Non-streaming mode
                List<int> allTokens;
                if (sourceSpeechToken.shape[1] == 0)
                {
                    llmThread.Join();
                    lock (_lock)
                    {
                        allTokens = TtsSpeechTokenDict[thisUuid].ToList();
                    }
                }
                else
                {
                    llmThread.Join();
                    lock (_lock)
                    {
                        allTokens = TtsSpeechTokenDict[thisUuid].ToList();
                    }
                }
                if (perf)
                    Console.WriteLine($"[CosyVoicePerf] llm_join_ms={ttsSw!.Elapsed.TotalMilliseconds:0.000}");

                var thisTtsSpeechToken = torch.tensor(allTokens.ToArray(), dtype: ScalarType.Int32).unsqueeze(0);
                var thisTtsSpeech = Token2WavWithOffset(
                    token: thisTtsSpeechToken,
                    promptToken: flowPromptSpeechToken,
                    promptFeat: promptSpeechFeat,
                    embedding: flowEmbedding,
                    tokenOffset: 0,
                    uuid: thisUuid,
                    finalize: true,
                    speed: speed
                );
                if (perf)
                    Console.WriteLine($"[CosyVoicePerf] tts_total_before_cpu_ms={ttsSw!.Elapsed.TotalMilliseconds:0.000}");

                thisTtsSpeechToken.Dispose();
                yield return BuildSpeechOutput(thisTtsSpeech);
            }
            }
            finally
            {
                if (llmThread?.IsAlive == true)
                    llmThread.Join();
                CleanupRuntimeState(thisUuid);
                DisposeOwnedDefaults(ownedDefaults);
            }

            // TorchSharp manages CUDA memory automatically
        }
    }

    // CosyVoice3 Model - Different token2wav with mel cache instead of HiFT cache
    public class CosyVoice3Model : CosyVoice2Model
    {
        public CosyVoice3Model(nn.Module llm, nn.Module flow, nn.Module hift, bool fp16 = false, CosyVoiceBackend backend = CosyVoiceBackend.Auto) 
            : base(llm, flow, hift, fp16, backend)
        {
            // FSQ silent and breath tokens
            SilentTokens.AddRange(new[] { 1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323 });
        }

        // CosyVoice3 has different token2wav - uses mel cache differently
        public override Tensor Token2WavWithOffset(Tensor token, Tensor promptToken, Tensor promptFeat, Tensor embedding, int tokenOffset, string uuid, bool streaming = false, bool finalize = false, double speed = 1.0)
        {
            if (Flow is IFlowInference iFlow && Hift is IHiftInference iHift )
            {
                using (torch.inference_mode())
                using (torch.NewDisposeScope())
                {
                    var perf = false;
                    var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
                    var flowProfileSw = StartProfileTimer(FlowDevice);
                    // Flow inference
                    var flowResult = iFlow.Inference(
                        token: token.to(ScalarType.Int32).to(FlowDevice),
                        token_len: torch.tensor(new long[] { token.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        prompt_token: promptToken.to(FlowDevice),
                        prompt_token_len: torch.tensor(new long[] { promptToken.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        prompt_feat: ToFlowFloat(promptFeat),
                        prompt_feat_len: torch.tensor(new long[] { promptFeat.shape[1] }, dtype: ScalarType.Int64).to(FlowDevice),
                        embedding: ToFlowFloat(embedding),
                        streaming: streaming,
                        finalize: finalize
                    );
                    if (perf)
                    {
                        SynchronizeIfCuda(FlowDevice);
                        Console.WriteLine($"[CosyVoicePerf] flow_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                    }
                    StopProfileTimer("flow.inference", flowProfileSw, FlowDevice, new Dictionary<string, string>
                    {
                        ["tokens"] = token.shape[1].ToString(),
                        ["streaming"] = streaming.ToString(),
                        ["finalize"] = finalize.ToString()
                    });
                    sw?.Restart();
                    var ttsMel = ((ValueTuple<Tensor, Tensor>)flowResult).Item1;

                    // Get token_mel_ratio
                    var tokenMelRatio = iFlow.token_mel_ratio;
                    ttsMel = ttsMel[.., .., (tokenOffset * tokenMelRatio)..];
                    ttsMel = ToHiftMel(ttsMel);

                    // Append mel cache (different from V2!)
                    if (HiftCacheDict.ContainsKey(uuid) && HiftCacheDict[uuid] != null)
                    {
                        var hiftCache = HiftCacheDict[uuid];
                        var hiftCacheMel = hiftCache["mel"];
                        ttsMel = torch.cat(new[] { hiftCacheMel, ttsMel }, dim: 2);
                        hiftCache["mel"] = ttsMel;
                        DisposeTensorIfDifferent(hiftCacheMel, ttsMel);
                    }
                    else
                    {
                        HiftCacheDict[uuid] = new Dictionary<string, Tensor>
                        {
                            ["mel"] = ttsMel,
                            ["speech_offset"] = torch.tensor(0L, dtype: ScalarType.Int64)
                        };
                    }

                    // Speed adjustment
                    if (speed != 1.0)
                    {
                        if (tokenOffset != 0 || !finalize)
                            throw new InvalidOperationException("Speed change only supports non-stream inference mode");

                        ttsMel = interpolate(ttsMel, size: new long[] { (long)(ttsMel.shape[2] / speed) }, mode: InterpolationMode.Linear);
                    }

                    // HiFT inference
                    var hiftProfileSw = StartProfileTimer(HiftDevice);
                    var hiftResult = iHift.Inference(ttsMel, finalize);
                    if (perf)
                    {
                        SynchronizeIfCuda(HiftDevice);
                        Console.WriteLine($"[CosyVoicePerf] hift_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                    }
                    StopProfileTimer("hift.inference", hiftProfileSw, HiftDevice, new Dictionary<string, string>
                    {
                        ["mel_frames"] = ttsMel.shape[2].ToString(),
                        ["finalize"] = finalize.ToString()
                    });
                    var ttsSpeech = ((ValueTuple<Tensor, Tensor>)hiftResult).Item1;

                    // Slice from speech_offset
                    var previousSpeechOffset = HiftCacheDict[uuid]["speech_offset"];
                    var speechOffset = (int)previousSpeechOffset.item<long>();
                    ttsSpeech = ttsSpeech[.., speechOffset..];
                    var nextSpeechOffset = torch.tensor((long)(speechOffset + (int)ttsSpeech.shape[1]), dtype: ScalarType.Int64);
                    HiftCacheDict[uuid]["speech_offset"] = nextSpeechOffset;
                    DisposeTensorIfDifferent(previousSpeechOffset, nextSpeechOffset);

                    return ttsSpeech.MoveToOuterDisposeScope();
                }
            }
            return null;
        }
    }
}
