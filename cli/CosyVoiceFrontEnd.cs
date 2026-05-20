// Exported from CosyVoice/cosyvoice/cli/frontend.py

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using System.Numerics.Tensors;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using System.Text.RegularExpressions;
using Humanizer;
using CosyVoice.Tokenizer;
using System.Collections;
using CosyVoiceNet.Utilities;

namespace CosyVoiceNet.cli
{
    public class CosyVoiceFrontEnd : IDisposable
    {
        private readonly object _tokenizer; // CosyVoice2Tokenizer or CosyVoice3Tokenizer
        private readonly Func<torch.Tensor, torch.Tensor> _featExtractor; // Updated to return 3D array
        private readonly InferenceSession _campplusSession;
        private readonly InferenceSession _speechTokenizerSession;
        private readonly object _zeroShotPromptCacheLock = new object();
        private readonly Dictionary<string, Dictionary<string, object>> _zeroShotPromptCache = new Dictionary<string, Dictionary<string, object>>(StringComparer.Ordinal);
        private const int ZeroShotPromptAudioProcessingVersion = 5;
        public readonly Dictionary<string, object> Spk2Info = new Dictionary<string, object>(StringComparer.Ordinal);
        public readonly string AllowedSpecial;
        public ICosyVoiceLogger? Logger { get; set; }
        public bool TracePromptTrim { get; set; }
        public CosyVoiceBackend CampPlusOnnxBackend { get; }
        public CosyVoiceBackend SpeechTokenizerOnnxBackend { get; }

        public CosyVoiceFrontEnd(Func<object> getTokenizer, Func<torch.Tensor, torch.Tensor> featExtractor,
                                 string campplusModel, string speechTokenizerModel,
                                 string spk2info = null, string allowedSpecial = "all",
                                 CosyVoiceBackend onnxBackend = CosyVoiceBackend.Cpu,
                                 ICosyVoiceLogger? logger = null)
        {
            Logger = logger;
            _tokenizer = getTokenizer?.Invoke() ?? throw new ArgumentNullException(nameof(getTokenizer));
            _featExtractor = featExtractor ?? throw new ArgumentNullException(nameof(featExtractor));
            AllowedSpecial = allowedSpecial;

            // CampPlus optimizations in the .NET CPU runtime change speaker embeddings
            // noticeably, while the unoptimized graph matches Python ONNX Runtime.
            _campplusSession = CreateOnnxSession(
                campplusModel,
                CreateCampPlusOptions,
                onnxBackend,
                "campplus",
                out var campplusBackend);
            CampPlusOnnxBackend = campplusBackend;

            _speechTokenizerSession = CreateOnnxSession(
                speechTokenizerModel,
                CreateSpeechTokenizerOptions,
                onnxBackend,
                "speech_tokenizer",
                out var speechTokenizerBackend);
            SpeechTokenizerOnnxBackend = speechTokenizerBackend;

            // Load spk2info map
            if (!string.IsNullOrEmpty(spk2info) && File.Exists(spk2info))
            {
                try
                {
                    Spk2Info = LoadSpk2Info(spk2info);
                }
                catch (Exception ex)
                {
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Warning, "Failed to load spk2info.", ex);
                }
            }
        }

        private static SessionOptions CreateCampPlusOptions()
        {
            EnsureOnnxRuntimeEnvironment();
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
                IntraOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };
        }

        private static SessionOptions CreateSpeechTokenizerOptions()
        {
            EnsureOnnxRuntimeEnvironment();
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };
        }

        private static void EnsureOnnxRuntimeEnvironment()
        {
            _ = OrtEnv.Instance();
        }

        private InferenceSession CreateOnnxSession(
            string modelPath,
            Func<SessionOptions> createOptions,
            CosyVoiceBackend requestedBackend,
            string component,
            out CosyVoiceBackend activeBackend)
        {
            if (requestedBackend == CosyVoiceBackend.Cuda)
            {
                try
                {
                    var cudaOptions = createOptions();
                    cudaOptions.AppendExecutionProvider_CUDA(0);
                    var session = new InferenceSession(modelPath, cudaOptions);
                    activeBackend = CosyVoiceBackend.Cuda;
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Information, $"ONNX {component} session is using CUDA.", null, new Dictionary<string, string>
                    {
                        ["component"] = component,
                        ["model"] = modelPath,
                        ["backend"] = "cuda"
                    });
                    return session;
                }
                catch (Exception ex)
                {
                    CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Warning, $"ONNX {component} CUDA initialization failed. Falling back to CPU.", ex, new Dictionary<string, string>
                    {
                        ["component"] = component,
                        ["model"] = modelPath,
                        ["requested_backend"] = "cuda",
                        ["fallback_backend"] = "cpu"
                    });
                    Console.WriteLine($"[CosyVoice] ONNX {component} CUDA initialization failed. Falling back to CPU. {ex.Message}");
                }
            }

            var cpuOptions = createOptions();
            var cpuSession = new InferenceSession(modelPath, cpuOptions);
            activeBackend = CosyVoiceBackend.Cpu;
            CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Debug, $"ONNX {component} session is using CPU.", null, new Dictionary<string, string>
            {
                ["component"] = component,
                ["model"] = modelPath,
                ["backend"] = "cpu"
            });
            return cpuSession;
        }

        private Dictionary<string, object> LoadSpk2Info(string spk2infoPath)
        {
            try
            {
                if (!File.Exists(spk2infoPath))
                    return new Dictionary<string, object>();

                var loaded = PickleUnpickler.Unpickle(spk2infoPath);
                var converted = ConvertPickleValue(loaded);
                if (converted is Dictionary<string, object> dict)
                    return dict;

                throw new InvalidDataException("spk2info.pt is not a valid speaker dictionary.");
            }
            catch (Exception ex)
            {
                CosyVoiceLog.Write(Logger, CosyVoiceLogLevel.Warning, "Failed to load spk2info.", ex);
                return new Dictionary<string, object>();
            }
        }

        private static object ConvertPickleValue(object value)
        {
            switch (value)
            {
                case PickleUnpickler.TorchTensorPlaceholder tensor:
                    return PlaceholderToTensor(tensor);
                case PickleUnpickler.TorchParameterPlaceholder parameter when parameter.Tensor != null:
                    return PlaceholderToTensor(parameter.Tensor);
                case IDictionary<string, object> typed:
                    return typed.ToDictionary(kv => kv.Key, kv => ConvertPickleValue(kv.Value), StringComparer.Ordinal);
                case IDictionary dictionary:
                    var result = new Dictionary<string, object>(StringComparer.Ordinal);
                    foreach (DictionaryEntry entry in dictionary)
                    {
                        if (entry.Key == null || entry.Value == null)
                            continue;

                        result[entry.Key.ToString() ?? string.Empty] = ConvertPickleValue(entry.Value);
                    }
                    return result;
                case object[] array:
                    return array.Select(ConvertPickleValue).ToArray();
                default:
                    return value;
            }
        }

        private static torch.Tensor PlaceholderToTensor(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var shape = tensor.Shape?.Select(dim => (long)dim).ToArray() ?? Array.Empty<long>();
            var dtype = tensor.StorageDType ?? string.Empty;

            if (dtype.Contains("LongStorage", StringComparison.OrdinalIgnoreCase))
                return torch.tensor(ReadInt64Storage(tensor), dtype: torch.int64).view(shape);

            if (dtype.Contains("IntStorage", StringComparison.OrdinalIgnoreCase))
                return torch.tensor(ReadInt32Storage(tensor), dtype: torch.int32).view(shape);

            if (dtype.Contains("ShortStorage", StringComparison.OrdinalIgnoreCase))
                return torch.tensor(ReadInt16Storage(tensor), dtype: torch.int16).view(shape);

            if (dtype.Contains("ByteStorage", StringComparison.OrdinalIgnoreCase) ||
                dtype.Contains("CharStorage", StringComparison.OrdinalIgnoreCase) ||
                dtype.Contains("BoolStorage", StringComparison.OrdinalIgnoreCase))
            {
                return torch.tensor(ReadByteStorage(tensor), dtype: torch.uint8).view(shape);
            }

            return torch.tensor(tensor.ToFloat32Array(), dtype: torch.float32).view(shape);
        }

        private static long[] ReadInt64Storage(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var bytes = tensor.StorageBytes ?? throw new InvalidDataException("Tensor storage bytes are missing.");
            var count = ElementCount(tensor);
            var offset = (tensor.StorageOffset ?? 0) * sizeof(long);
            var result = new long[count];
            for (var i = 0; i < count; i++)
                result[i] = BitConverter.ToInt64(bytes, checked((int)(offset + i * sizeof(long))));
            return result;
        }

        private static int[] ReadInt32Storage(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var bytes = tensor.StorageBytes ?? throw new InvalidDataException("Tensor storage bytes are missing.");
            var count = ElementCount(tensor);
            var offset = (tensor.StorageOffset ?? 0) * sizeof(int);
            var result = new int[count];
            for (var i = 0; i < count; i++)
                result[i] = BitConverter.ToInt32(bytes, checked((int)(offset + i * sizeof(int))));
            return result;
        }

        private static short[] ReadInt16Storage(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var bytes = tensor.StorageBytes ?? throw new InvalidDataException("Tensor storage bytes are missing.");
            var count = ElementCount(tensor);
            var offset = (tensor.StorageOffset ?? 0) * sizeof(short);
            var result = new short[count];
            for (var i = 0; i < count; i++)
                result[i] = BitConverter.ToInt16(bytes, checked((int)(offset + i * sizeof(short))));
            return result;
        }

        private static byte[] ReadByteStorage(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var bytes = tensor.StorageBytes ?? throw new InvalidDataException("Tensor storage bytes are missing.");
            var count = ElementCount(tensor);
            var offset = checked((int)(tensor.StorageOffset ?? 0));
            var result = new byte[count];
            Buffer.BlockCopy(bytes, offset, result, 0, count);
            return result;
        }

        private static int ElementCount(PickleUnpickler.TorchTensorPlaceholder tensor)
        {
            var shape = tensor.Shape ?? Array.Empty<int>();
            if (shape.Length == 0)
                return 1;

            checked
            {
                var count = 1;
                foreach (var dim in shape)
                    count *= dim;
                return count;
            }
        }

        public (torch.Tensor speechToken, torch.Tensor speechTokenLen) ExtractSpeechToken(string wavPath)
        {
            try
            {
                using var mono = CosyVoiceNet.Utils.FileUtils.load_wav(wavPath, 16000);
                return ExtractSpeechToken(mono);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex}");
                // Return empty Tensors to match signature
                return (torch.zeros(new long[] { 1, 0 }, dtype: torch.int32), torch.zeros(new long[] { 1 }, dtype: torch.int32));
            }
        }


        public (torch.Tensor speechFeat, torch.Tensor featLen) ExtractSpeechFeat(string wavPath)
        {
            try
            {
                // 1. Load waveform at 24kHz [1, T]
                using var speech = CosyVoiceNet.Utils.FileUtils.load_wav(wavPath, 24000);
                return ExtractSpeechFeat(speech);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Frontend] ExtractSpeechFeat failed: {ex.Message}");
                // Return empty tensors to avoid null issues
                return (torch.zeros(new long[] { 1, 0, 80 }), torch.zeros(new long[] { 1 }));
            }
        }


        private static (torch.Tensor token, torch.Tensor len) TextTokenTensors((int[] token, int len) textToken)
        {
            var token = torch.tensor(textToken.token, dtype: torch.int64).unsqueeze(0);
            var len = torch.tensor(new[] { textToken.len }, dtype: torch.int32);
            return (token, len);
        }

        public Dictionary<string, object> FrontendSft(string text, string spkId)
        {
            var (textToken, textTokenLen) = TextTokenTensors(ExtractTextToken(text));

            if (!Spk2Info.TryGetValue(spkId, out var spkInfoObj))
                throw new KeyNotFoundException($"Speaker '{spkId}' was not found in spk2info.");

            object embedding;
            if (spkInfoObj is IDictionary<string, object> spkDict && spkDict.TryGetValue("embedding", out var embObj))
            {
                embedding = embObj;
            }
            else if (spkInfoObj is IDictionary<string, torch.Tensor> spkTensorDict && spkTensorDict.TryGetValue("embedding", out var embTensor))
            {
                embedding = embTensor;
            }
            else
            {
                throw new KeyNotFoundException($"Speaker '{spkId}' does not contain an 'embedding' entry.");
            }

            return new Dictionary<string, object>
            {
                ["text"] = textToken,
                ["text_len"] = textTokenLen,
                ["llm_embedding"] = embedding,
                ["flow_embedding"] = embedding
            };
        }

        public Dictionary<string, object> FrontendZeroShot(string text, string promptText, string promptWav, int sampleRate, string zeroShotSpkId = "")
        {
            var (textToken, textTokenLen) = TextTokenTensors(ExtractTextToken(text));
            if (string.IsNullOrEmpty(zeroShotSpkId))
            {
                var modelInput = new Dictionary<string, object>(GetZeroShotPromptInput(promptText, promptWav, sampleRate));
                modelInput["text"] = textToken;
                modelInput["text_len"] = textTokenLen;
                return modelInput;
            }
            else if (Spk2Info.TryGetValue(zeroShotSpkId, out var spkObj) && spkObj is IDictionary<string, object> spkDict)
            {
                var modelInput = new Dictionary<string, object>(spkDict);
                modelInput["text"] = textToken;
                modelInput["text_len"] = textTokenLen;
                return modelInput;
            }
            else
            {
                throw new KeyNotFoundException($"Speaker '{zeroShotSpkId}' was not found in spk2info.");
            }
        }

        private (torch.Tensor speechToken, torch.Tensor speechTokenLen) ExtractSpeechToken(torch.Tensor mono16k)
        {
            if (mono16k.shape[1] / 16000.0 > 30.0)
                throw new InvalidOperationException("audio longer than 30s");

            using var feat = CosyVoiceNet.TorchSharpUtils.WhisperLikeLogMelSpectogram.WhisperLogMelSpectogram(mono16k, 128);

            long T = feat.shape[2];
            var contiguous = feat.contiguous().to(torch.float32).cpu();
            var data = contiguous.data<float>().ToArray();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_speechTokenizerSession.InputMetadata.Keys.ElementAt(0),
                    new DenseTensor<float>(data, new int[] { 1, 128, (int)T })),
                NamedOnnxValue.CreateFromTensor(_speechTokenizerSession.InputMetadata.Keys.ElementAt(1),
                    new DenseTensor<int>(new int[] { (int)T }, new int[] { 1 }))
            };

            using var results = _speechTokenizerSession.Run(inputs);
            var output = ReadOnnxIntOutput(results.First());

            var speechToken = torch.tensor(output, dtype: torch.int32).unsqueeze(0);
            var speechTokenLen = torch.tensor(new int[] { (int)speechToken.shape[1] }, dtype: torch.int32);

            return (speechToken, speechTokenLen);
        }

        private (torch.Tensor speechFeat, torch.Tensor featLen) ExtractSpeechFeat(torch.Tensor speech24k)
        {
            var speechFeat = _featExtractor(speech24k).squeeze(0).transpose(0, 1);
            speechFeat = speechFeat.unsqueeze(0);
            var speechFeatLen = torch.tensor(new int[] { (int)speechFeat.shape[1] }, dtype: torch.int32);
            return (speechFeat, speechFeatLen);
        }

        private torch.Tensor TrimPromptBoundarySilence(torch.Tensor speech, int sampleRate)
        {
            if (speech.shape.Length < 2 || speech.shape[1] <= 0)
                return speech.clone();

            var totalSamples = checked((int)speech.shape[1]);
            using var mono = speech.squeeze(0).contiguous().to(torch.float32).cpu();
            var samples = mono.data<float>().ToArray();
            if (samples.Length == 0)
                return speech.clone();

            var peak = samples.Select(MathF.Abs).DefaultIfEmpty(0f).Max();
            if (peak <= 0f)
                return speech.clone();

            var threshold = Math.Max(0.0025f, peak * 0.015f);
            var window = Math.Max(1, sampleRate / 50);
            var firstSpeech = -1;
            var lastSpeechExclusive = -1;

            for (var start = 0; start < samples.Length; start += window)
            {
                var exclusiveEnd = Math.Min(samples.Length, start + window);
                double sumSquares = 0;
                for (var i = start; i < exclusiveEnd; i++)
                    sumSquares += samples[i] * samples[i];

                var rms = Math.Sqrt(sumSquares / Math.Max(1, exclusiveEnd - start));
                if (rms >= threshold)
                {
                    if (firstSpeech < 0)
                        firstSpeech = start;
                    lastSpeechExclusive = exclusiveEnd;
                }
            }

            if (firstSpeech < 0 || lastSpeechExclusive <= firstSpeech)
                return speech.clone();

            var keepPadding = Math.Max(1, sampleRate * 80 / 1000);
            var trimStart = Math.Max(0, firstSpeech - keepPadding);
            var trimEnd = Math.Min(totalSamples, lastSpeechExclusive + keepPadding);
            var trimLength = trimEnd - trimStart;
            if (trimStart == 0 && trimLength == totalSamples)
                return speech.clone();

            if (TracePromptTrim)
            {
                var removedStartMs = trimStart * 1000.0 / sampleRate;
                var removedEndMs = (totalSamples - trimEnd) * 1000.0 / sampleRate;
                Logger?.Log(
                    CosyVoiceLogLevel.Trace,
                    "Prompt boundary trim applied.",
                    tags: new Dictionary<string, string>
                    {
                        ["sample_rate"] = sampleRate.ToString(),
                        ["removed_start_ms"] = removedStartMs.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture),
                        ["removed_end_ms"] = removedEndMs.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)
                    });
            }

            return speech.narrow(1, trimStart, trimLength).contiguous();
        }

        private static int[] ReadOnnxIntOutput(DisposableNamedOnnxValue value)
        {
            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<int> intTensor)
                return intTensor.ToArray();
            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<long> longTensor)
                return longTensor.Select(v => checked((int)v)).ToArray();
            if (value.Value is IEnumerable<int> intValues)
                return intValues.ToArray();
            if (value.Value is IEnumerable<long> longValues)
                return longValues.Select(v => checked((int)v)).ToArray();

            throw new InvalidOperationException($"Unsupported speech tokenizer output type: {value.Value?.GetType().FullName ?? "null"}");
        }

        private Dictionary<string, object> GetZeroShotPromptInput(string promptText, string promptWav, int sampleRate)
        {
            var cacheKey = BuildZeroShotPromptCacheKey(promptText, promptWav, sampleRate);
            lock (_zeroShotPromptCacheLock)
            {
                if (_zeroShotPromptCache.TryGetValue(cacheKey, out var cached))
                    return cached;
            }

            var created = CreateZeroShotPromptInput(promptText, promptWav, sampleRate);
            lock (_zeroShotPromptCacheLock)
            {
                if (_zeroShotPromptCache.TryGetValue(cacheKey, out var cached))
                {
                    DisposeTensorDictionary(created);
                    return cached;
                }

                if (_zeroShotPromptCache.Count >= 8)
                    ClearZeroShotPromptCacheNoLock();

                _zeroShotPromptCache[cacheKey] = created;
                return created;
            }
        }

        private void ClearZeroShotPromptCacheNoLock()
            => ClearZeroShotPromptCacheNoLock(new List<torch.Tensor>());

        private void ClearZeroShotPromptCacheNoLock(List<torch.Tensor> disposed)
        {
            foreach (var entry in _zeroShotPromptCache.Values)
                DisposeTensorDictionary(entry, disposed);
            _zeroShotPromptCache.Clear();
        }

        internal static void DisposeTensorDictionary(IDictionary<string, object> dict)
            => DisposeTensorDictionary(dict, new List<torch.Tensor>());

        private static void DisposeTensorDictionary(IDictionary<string, object> dict, List<torch.Tensor> disposed)
        {
            if (dict is null)
                return;

            foreach (var value in dict.Values)
            {
                if (value is not torch.Tensor tensor)
                    continue;

                if (disposed.Any(existing => ReferenceEquals(existing, tensor)))
                    continue;

                tensor.Dispose();
                disposed.Add(tensor);
            }
        }

        public void Dispose()
        {
            var disposed = new List<torch.Tensor>();
            lock (_zeroShotPromptCacheLock)
            {
                ClearZeroShotPromptCacheNoLock(disposed);
            }

            foreach (var value in Spk2Info.Values)
            {
                if (value is IDictionary<string, object> dict)
                    DisposeTensorDictionary(dict, disposed);
            }

            Spk2Info.Clear();
            _campplusSession?.Dispose();
            _speechTokenizerSession?.Dispose();
        }

        private static string BuildZeroShotPromptCacheKey(string promptText, string promptWav, int sampleRate)
        {
            var fullPath = Path.GetFullPath(promptWav);
            var info = new FileInfo(fullPath);
            var lastWriteTicks = info.Exists ? info.LastWriteTimeUtc.Ticks : 0;
            var length = info.Exists ? info.Length : 0;
            return string.Join("|", ZeroShotPromptAudioProcessingVersion.ToString(System.Globalization.CultureInfo.InvariantCulture), sampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture), fullPath, lastWriteTicks, length, promptText);
        }

        private Dictionary<string, object> CreateZeroShotPromptInput(string promptText, string promptWav, int sampleRate)
        {
            var perf = false;
            var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            var (promptTextToken, promptTextLen) = TextTokenTensors(ExtractTextToken(promptText));
            using var speech24k = CosyVoiceNet.Utils.FileUtils.load_wav(promptWav, 24000);
            var (speechFeat, speechFeatLenTensor) = ExtractSpeechFeat(speech24k);
            using var speech16k = CosyVoiceNet.Utils.FileUtils.load_wav(promptWav, 16000);
            var (speechToken, speechTokenLenTensor) = ExtractSpeechToken(speech16k);

            int speechFeatLen = (int)speechFeat.shape[1];
            int speechTokenLen = (int)speechToken.shape[1];
            if (sampleRate == 24000)
            {
                int tokenLen = Math.Min(speechFeatLen / 2, speechTokenLen);

                speechFeat = speechFeat.narrow(1, 0, 2 * tokenLen);
                speechFeatLen = 2 * tokenLen;

                speechToken = speechToken.narrow(1, 0, tokenLen);
                speechTokenLen = tokenLen;
            }

            var embedding = ExtractSpkEmbedding(speech16k);
            if (perf)
                Console.WriteLine($"[CosyVoicePerf] zero_shot_prompt_clone_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
            return new Dictionary<string, object>
            {
                ["prompt_text"] = promptTextToken,
                ["prompt_text_len"] = promptTextLen,
                ["llm_prompt_speech_token"] = speechToken,
                ["llm_prompt_speech_token_len"] = torch.tensor(new[] { speechTokenLen }, dtype: torch.int32),
                ["flow_prompt_speech_token"] = speechToken,
                ["flow_prompt_speech_token_len"] = torch.tensor(new[] { speechTokenLen }, dtype: torch.int32),
                ["prompt_speech_feat"] = speechFeat,
                ["prompt_speech_feat_len"] = torch.tensor(new[] { speechFeatLen }, dtype: torch.int32),
                ["llm_embedding"] = embedding,
                ["flow_embedding"] = embedding
            };
        }

        // Embedding extraction (Python: _extract_spk_embedding)
        private torch.Tensor ExtractSpkEmbedding(string wavPath)
        {
            // 1. Load waveform [1, T]
            using var speech = CosyVoiceNet.Utils.FileUtils.load_wav(wavPath, 16000);
            return ExtractSpkEmbedding(speech);
        }

        private torch.Tensor ExtractSpkEmbedding(torch.Tensor speech16k)
        {
            // 2. Compute fbank features [Frames, 80]
            using var feat = TorchSharpUtils.KaldiLikeFBank.FBank(speech16k, 16000, 80);
            // 3. Normalize features: feat - feat.mean(dim=0, keepdim=True)
            using var normFeat = feat - feat.mean(new long[] { 0 }, keepdim: true);

            // 4. Prepare for ONNX: feat.unsqueeze(dim=0) -> [1, Frames, 80]
            using var inputFeat = normFeat.unsqueeze(0);
            var dense = inputFeat.contiguous().to(torch.float32).cpu();
            float[] flatData = dense.data<float>().ToArray();

            var shape = new int[] { 1, (int)dense.shape[1], (int)dense.shape[2] };

            // 5. Run Campplus ONNX session
            var inputName = _campplusSession.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(flatData, shape))
            };

            using var results = _campplusSession.Run(inputs);
            var embedding = results.First().AsEnumerable<float>().ToArray();

            // 6. Return as [1, embedding_dim] - matches torch.tensor([embedding])
            return torch.tensor(embedding).unsqueeze(0);
        }


        // Advanced text normalization (Python: text_normalize)
        public IEnumerable<string> TextNormalize(string text, bool split, bool textFrontend)
        {
            if (text == string.Empty)
                return split ? new[] { text } : new[] { text };

            if (text.Contains("<|") && text.Contains("|>"))
                textFrontend = false;

            if (!textFrontend)
                return new[] { text };

            text = text.Trim();
            if (text.Length == 0)
                return split ? Array.Empty<string>() : new[] { text };

            bool isChinese = Utils.FrontendUtils.ContainsChinese(text);// text.Any(c => c >= 0x4E00 && c <= 0x9FFF);
            List<string> texts;

            if (isChinese)
            {
                text = text.Replace("\n", "");
                text = ReplaceBlank(text);
                text = ReplaceCornerMark(text);
                text = text.Replace(".", "。");
                text = text.Replace(" - ", "，");
                text = RemoveBracket(text);
                text = Regex.Replace(text, "[，,、]+$", "。", RegexOptions.Compiled);

                Func<string, int> tokenizer = s => ExtractTextToken(s).len;
                texts = Utils.FrontendUtils.SplitParagraph(text, tokenizer, "zh", tokenMaxN: 80, tokenMinN: 60, mergeLen: 20, commaSplit: false).ToList();
            }
            else
            {
                text = SpellOutNumber(text);
                Func<string, int> tokenizer = s => ExtractTextToken(s).len;
                texts = Utils.FrontendUtils.SplitParagraph(text, tokenizer, "en", tokenMaxN: 80, tokenMinN: 60, mergeLen: 20, commaSplit: false).ToList();
            }

            texts = texts.Where(s => !IsOnlyPunctuation(s)).ToList();
            return split ? texts : new[] { text };
        }

        // --- Helper methods for normalization ---
        private string ReplaceBlank(string text) => text.Replace(" ", "");
        private string ReplaceCornerMark(string text) => text.Replace("“", "\"").Replace("”", "\"").Replace("‘", "'").Replace("’", "'");
        private string RemoveBracket(string text)
        {
            var brackets = new[] { '(', ')', '[', ']', '{', '}', '（', '）', '【', '】', '『', '』', '「', '」' };
            return new string(text.Where(c => !brackets.Contains(c)).ToArray());
        }

        private bool IsOnlyPunctuation(string s) => s.All(char.IsPunctuation);

        // --- Update FrontendVc to use embedding extraction ---
        public Dictionary<string, object> FrontendVc(string sourceWav, string promptWav, int sampleRate)
        {
            var (promptSpeechToken, promptSpeechTokenLen) = ExtractSpeechToken(promptWav);
            var (promptSpeechFeat, promptSpeechFeatLen) = ExtractSpeechFeat(promptWav);

            var embedding = ExtractSpkEmbedding(promptWav); // Use real embedding
            var (sourceSpeechToken, sourceSpeechTokenLen) = ExtractSpeechToken(sourceWav);
            var modelInput = new Dictionary<string, object>
            {
                ["source_speech_token"] = sourceSpeechToken,
                ["source_speech_token_len"] = sourceSpeechTokenLen,
                ["flow_prompt_speech_token"] = promptSpeechToken,
                ["flow_prompt_speech_token_len"] = promptSpeechTokenLen,
                ["prompt_speech_feat"] = promptSpeechFeat,
                ["prompt_speech_feat_len"] = promptSpeechFeatLen,
                ["flow_embedding"] = embedding
            };
            return modelInput;
        }
        // Cross-lingual frontend (Python: frontend_cross_lingual)
        public Dictionary<string, object> FrontendCrossLingual(string text, string promptWav, int sampleRate, string zeroShotSpkId = "")
        {
            var modelInput = FrontendZeroShot(text, "", promptWav, sampleRate, zeroShotSpkId);

            modelInput.Remove("prompt_text");
            modelInput.Remove("prompt_text_len");
            modelInput.Remove("llm_prompt_speech_token");
            modelInput.Remove("llm_prompt_speech_token_len");

            return modelInput;
        }

        // Instruct frontend (Python: frontend_instruct)
        public Dictionary<string, object> FrontendInstruct(string text, string spkId, string instructText)
        {
            var modelInput = FrontendSft(text, spkId);

            // In instruct mode Python removes llm_embedding to avoid leakage.
            modelInput.Remove("llm_embedding");

            var (instructTok, instructLen) = TextTokenTensors(ExtractTextToken(instructText));
            modelInput["prompt_text"] = instructTok;
            modelInput["prompt_text_len"] = instructLen;

            return modelInput;
        }

        public (int[] token, int len) ExtractTextToken(string text)
        {
            if (string.IsNullOrEmpty(text))
                return (Array.Empty<int>(), 0);

            try
            {
                if (_tokenizer is CosyVoiceTokenizerFactory.CosyVoice2Tokenizer cv2tok)
                {
                    var tokens = EncodeTextWithCosyVoiceSpecials(cv2tok, text);
                    return (tokens, tokens.Length);
                }

                if (_tokenizer is CosyVoiceTokenizerFactory.CosyVoiceWhisperTokenizer whisperTok)
                {
                    var tokens = whisperTok.Encode(text);
                    return (tokens, tokens.Length);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Frontend] ExtractTextToken failed: {ex.Message}");
            }

            return (Array.Empty<int>(), 0);
        }

        private static int[] EncodeTextWithCosyVoiceSpecials(CosyVoiceTokenizerFactory.CosyVoice2Tokenizer tokenizer, string text)
        {
            const string EndOfPrompt = "<|endofprompt|>";
            const int EndOfPromptToken = 151646;

            if (!text.Contains(EndOfPrompt, StringComparison.Ordinal))
                return tokenizer.Encode(text).ToArray();

            var result = new List<int>();
            var parts = text.Split(EndOfPrompt, StringSplitOptions.None);
            for (var i = 0; i < parts.Length; i++)
            {
                if (parts[i].Length > 0)
                    result.AddRange(tokenizer.Encode(parts[i]));

                if (i < parts.Length - 1)
                    result.Add(EndOfPromptToken);
            }

            return result.ToArray();
        }

        public Dictionary<string, object> FrontendInstruct2(string text, string instructText, string promptWav, int sampleRate, string zeroShotSpkId)
        {
            // Python: model_input = self.frontend_zero_shot(...); remove llm_prompt_speech_token/len
            var modelInput = FrontendZeroShot(text, instructText, promptWav, sampleRate, zeroShotSpkId);
            if (!string.IsNullOrEmpty(zeroShotSpkId))
            {
                // Saved voices reuse cloned voice tensors, but instruct2 still needs the instruction as prompt_text.
                var (instructTok, instructLen) = TextTokenTensors(ExtractTextToken(instructText));
                modelInput["prompt_text"] = instructTok;
                modelInput["prompt_text_len"] = instructLen;
            }

            modelInput.Remove("llm_prompt_speech_token");
            modelInput.Remove("llm_prompt_speech_token_len");
            return modelInput;
        }
        // Number to words (Python: spell_out_number)
        private string SpellOutNumber(string text)
        {
            // Use Humanizer for English number-to-words conversion
            if (string.IsNullOrEmpty(text)) return text;
            // Replace all integer numbers in the text with their word equivalents
            return System.Text.RegularExpressions.Regex.Replace(
                text,
                @"\b\d+\b",
                match => int.Parse(match.Value).ToWords(),
                System.Text.RegularExpressions.RegexOptions.Compiled
            );
        }
    }
}
