// Exported from #file:cosyvoice/llm/llm.py
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;
using CosyVoiceNet.cli;
using CosyVoiceNet.Transformers;
using CosyVoiceNet.Utils;
using CosyVoiceNet.TorchSharpUtils;

namespace CosyVoiceNet.LLM
{
    public class TransformerLM : torch.nn.Module<Dictionary<string, Tensor>, Device, Dictionary<string, Tensor>>, ILLMInference
    {
        public const int IGNORE_ID = -1;
        public int llm_input_size;
        public int speech_token_size;

        public ManagedEmbedding text_embedding;
        public ManagedEmbedding speech_embedding;
        public ManagedEmbedding llm_embedding;
        public ManagedLinear text_encoder_affine_layer;
        public ManagedLinear llm_decoder;
        public ManagedLinear spk_embed_affine_layer;

        public torch.nn.Module? text_encoder;
        public torch.nn.Module? llm;

        protected static Tensor TokenIndex(int value, Device device)
        {
            return torch.tensor(new[] { value }, dtype: ScalarType.Int64).to(device);
        }
        public Func<Tensor, List<int>, int, int> sampling;
        public LegacyTransformerCacheBackend LegacyTransformerCacheBackend { get; set; } = LegacyTransformerCacheBackend.Standard;
        public QwenKvCacheBackend QwenKvCacheBackend { get; set; } = QwenKvCacheBackend.Standard;
        public ICosyVoiceProfiler? Profiler { get; set; }
        public QwenAttentionBackend QwenAttentionBackend { get; set; } = QwenAttentionBackend.Auto;
        public QwenMlpBackend QwenMlpBackend { get; set; } = QwenMlpBackend.Auto;

        [Obsolete("Use QwenKvCacheBackend instead.")]
        public bool UsePreallocatedQwenKvCache
        {
            get => QwenKvCacheBackend == QwenKvCacheBackend.Preallocated;
            set => QwenKvCacheBackend = value ? QwenKvCacheBackend.Preallocated : QwenKvCacheBackend.Standard;
        }

        public int sos;
        public int task_id;
        public int eos_token;
        public int fill_token;
        public List<int> stop_token_ids;
        public List<int> mix_ratio;
        public Dictionary<string, Queue<int>> vllm_output_queue;
        public dynamic speech_token_extractor;
        public bool online_feature;
        public string onnx_path;
        public dynamic CriterionCE;

        public TransformerLM(int textEncoderInputSize, int llmInputSize, int llmOutputSize, int textTokenSize, int speechTokenSize,
            torch.nn.Module textEncoder, torch.nn.Module llm, Func<Tensor, List<int>, int, int> sampling,
            bool lengthNormalizedLoss = true, float lsmWeight = 0.0f, int spkEmbedDim = 192)
            : base("TransformerLM")
        {
            this.llm_input_size = llmInputSize;
            this.speech_token_size = speechTokenSize;
            this.text_encoder = textEncoder;
            this.llm = llm;
            this.sampling = sampling;

            this.text_embedding = new ManagedEmbedding(textTokenSize, textEncoderInputSize);
            var textEncoderOutputSize = ((dynamic)textEncoder).output_size();
            this.text_encoder_affine_layer = new ManagedLinear(textEncoderOutputSize, llmInputSize);
            this.llm_embedding = new ManagedEmbedding(2, llmInputSize);
            this.llm_decoder = new ManagedLinear(llmOutputSize, speechTokenSize + 1);
            this.speech_embedding = new ManagedEmbedding(speechTokenSize, llmInputSize);
            this.spk_embed_affine_layer = new ManagedLinear(spkEmbedDim, llmInputSize);

            this.sos = 0;
            this.task_id = 1;
            this.eos_token = speechTokenSize;
            this.fill_token = speechTokenSize + 2;
            this.stop_token_ids = new List<int> { speechTokenSize };
            this.mix_ratio = new List<int> { 5, 15 };
            this.vllm_output_queue = new Dictionary<string, Queue<int>>();
            this.online_feature = false;
            this.onnx_path = string.Empty;

            this.CriterionCE = new LabelSmoothingLoss(
                size: speechTokenSize + 1,
                paddingIdx: IGNORE_ID,
                smoothing: lsmWeight,
                normalizeLength: lengthNormalizedLoss
            );
            RegisterComponents();
        }

        // Protected no-op constructor: allows subclasses to bypass full module initialisation
        // (mirrors Python's torch.nn.Module.__init__(self) bypass pattern used in CosyVoice3LM).
        protected TransformerLM(string name) : base(name) { }

        public (Tensor, Tensor) encode(Tensor text, Tensor text_lengths, int decoding_chunk_size = 1, int num_decoding_left_chunks = -1)
        {
            // Python: encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
            var encoder_result = ((dynamic)text_encoder).Forward(text, text_lengths, decoding_chunk_size, num_decoding_left_chunks);
            Tensor encoder_out = encoder_result.Item1;
            Tensor encoder_mask = encoder_result.Item2;
            var encoder_out_lens = encoder_mask.squeeze(1).sum(1);
            encoder_out = text_encoder_affine_layer.forward(encoder_out);
            return (encoder_out, encoder_out_lens);
        }

        public List<Tensor> unpad_sequence(Tensor padded, Tensor lengths)
        {
            var unpadded = new List<Tensor>();
            for (int i = 0; i < padded.shape[0]; i++)
            {
                unpadded.Add(padded[i].narrow(0, 0, ScalarToInt(lengths[i])));
            }
            return unpadded;
        }

        public (Tensor, Tensor) pad_unpad_sequence(Tensor sos_emb, Tensor embedding, Tensor text_token, Tensor text_token_len, Tensor task_id_emb, Tensor speech_token, Tensor speech_token_len)
        {
            var text_token_unpadded = unpad_sequence(text_token, text_token_len.cpu());
            var speech_token_unpadded = unpad_sequence(speech_token, speech_token_len.cpu());
            var lm_input = new List<Tensor>();
            for (int i = 0; i < text_token_unpadded.Count; i++)
            {
                var concat = torch.cat(new Tensor[] { sos_emb.squeeze(0), embedding[i], text_token_unpadded[i], task_id_emb.squeeze(0), speech_token_unpadded[i] }, 0);
                lm_input.Add(concat);
            }
            var lm_input_len = torch.tensor(lm_input.ConvertAll(t => t.shape[0]), torch.int32);
            var lm_input_padded = torch.nn.utils.rnn.pad_sequence(lm_input, true, IGNORE_ID);
            return (lm_input_padded, lm_input_len);
        }


        // Forward method for TransformerLM (batch dict version)
        public override Dictionary<string, Tensor> forward(Dictionary<string, Tensor> batch, Device device)
        {
            // Implements the batch forward logic as in Python
            var text_token = batch["text_token"].to(device);
            var text_token_len = batch["text_token_len"].to(device);
            var speech_token = batch["speech_token"].to(device);
            var speech_token_len = batch["speech_token_len"].to(device);
            var embedding = batch["embedding"].to(device);

            // Prepare llm_target
            var batchSize = text_token.shape[0];
            var lm_targetList = new List<Tensor>();
            for (int i = 0; i < batchSize; i++)
            {
                var ignoreList = new List<int>();
                for (int j = 0; j < 2 + ScalarToInt(text_token_len[i]); j++) ignoreList.Add(IGNORE_ID);
                var speechList = new List<int>();
                for (int j = 0; j < ScalarToInt(speech_token_len[i]); j++) speechList.Add(ScalarToInt(speech_token[i, j]));
                ignoreList.AddRange(speechList);
                ignoreList.Add(speech_token_size);
                lm_targetList.Add(torch.tensor(ignoreList.ToArray(), dtype: ScalarType.Int32));
            }
            var lm_target = torch.nn.utils.rnn.pad_sequence(lm_targetList, true, IGNORE_ID).to(device);

            // Encode text_token
            var text_token_emb = text_embedding.forward(text_token);
            var (encoded, encoded_lens) = encode(text_token_emb, text_token_len);

            // Embedding projection
            embedding = torch.nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding).unsqueeze(1);

            // sos and task_id
            var sos_emb = llm_embedding.GetWeight()[sos].reshape(1, 1, -1);
            var task_id_emb = llm_embedding.GetWeight()[task_id].reshape(1, 1, -1);

            // Encode speech_token
            var speech_token_emb = speech_embedding.forward(speech_token);

            // Unpad and pad
            var (lm_input, lm_input_len) = pad_unpad_sequence(sos_emb, embedding, encoded, encoded_lens, task_id_emb, speech_token_emb, speech_token_len);

            // Run lm forward
            var lm_result = ((dynamic)llm).forward(lm_input, lm_input_len.to(device));
            Tensor lm_output = lm_result.Item1;
            //Tensor lm_output_mask = lm_result.Item2; // Unused
            var logits = llm_decoder.forward(lm_output);
            var loss = CriterionCE.forward(logits, lm_target);
            var result = new Dictionary<string, Tensor> { { "loss", loss } };
            return result;
        }

        // Inference method for TransformerLM
        public virtual IEnumerable<int> inference(
            Tensor text, Tensor text_len, Tensor prompt_text, Tensor prompt_text_len,
            Tensor prompt_speech_token, Tensor prompt_speech_token_len, Tensor embedding,
            int sampling = 25, float max_token_text_ratio = 20, float min_token_text_ratio = 2, string uuid = "")
        {
            var device = text.device;
            text = torch.cat(new Tensor[] { prompt_text, text }, 1);
            text_len += prompt_text_len;
            text = text_embedding.forward(text);

            // Encode text
            var (encoded, encoded_lens) = encode(text, text_len);

            // Encode embedding
            if (embedding.shape[0] != 0)
            {
                embedding = torch.nn.functional.normalize(embedding, dim: 1);
                embedding = spk_embed_affine_layer.forward(embedding).unsqueeze(1);
            }
            else
            {
                embedding = torch.zeros(new long[] { 1, 0, llm_input_size }, dtype: text.dtype).to(device);
            }

            // Prepare LLM input
            var sos_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(sos, device)).reshape(1, 1, -1);
            var task_id_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(task_id, device)).reshape(1, 1, -1);
            Tensor prompt_speech_token_emb;
            if (prompt_speech_token_len.numel() != 0 && ScalarToInt(prompt_speech_token_len.flatten()[0]) != 0)
                prompt_speech_token_emb = speech_embedding.forward(prompt_speech_token);
            else
                prompt_speech_token_emb = torch.zeros(new long[] { 1, 0, llm_input_size }, dtype: text.dtype).to(device);
            var lm_input = torch.cat(new Tensor[] { sos_emb, embedding, encoded, task_id_emb, prompt_speech_token_emb }, 1);

            // Calculate min/max lengths
            var min_len = (int)((encoded_lens - prompt_text_len) * min_token_text_ratio);
            var max_len = (int)((encoded_lens - prompt_text_len) * max_token_text_ratio);

            if (LegacyTransformerCacheBackend == LegacyTransformerCacheBackend.Preallocated && llm is TransformerEncoder transformerEncoder)
            {
                foreach (var token in InferenceWithPreallocatedTransformerCache(transformerEncoder, lm_input, sampling, min_len, max_len))
                {
                    yield return token;
                }

                yield break;
            }

            // Decode step by step
            var out_tokens = new List<int>();
            var offset = 0;
            var att_cache = torch.zeros(new long[] { 0, 0, 0, 0 }, dtype: lm_input.dtype, device: lm_input.device);
            var cnn_cache = torch.zeros(new long[] { 0, 0, 0, 0 }, dtype: lm_input.dtype, device: lm_input.device);
            var oneTokenAttMask = torch.ones(new long[] { 1, 1, 1 }, dtype: ScalarType.Bool, device: lm_input.device);
            try
            {
                for (int i = 0; i < max_len; i++)
                {
                    int? emittedToken = null;
                    var previousInput = lm_input;
                    var previousAttCache = att_cache;
                    var previousCnnCache = cnn_cache;
                    var previousInputLength = (int)lm_input.shape[1];

                    using (torch.NewDisposeScope())
                    {
                        var attMask = previousInputLength == 1
                            ? oneTokenAttMask
                            : torch.tril(torch.ones(new long[] { 1, previousInputLength, previousInputLength }, dtype: ScalarType.Bool, device: lm_input.device));
                        var chunkResult = CallLlmForwardChunk(lm_input, offset, -1, att_cache, cnn_cache, attMask);
                        Tensor y_pred = chunkResult.y;

                        var logits = llm_decoder.forward(y_pred.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Single(-1), TensorIndex.Ellipsis }));
                        var logp = logits.log_softmax(-1);
                        var top_id = sampling_ids(logp.squeeze(0), out_tokens, sampling, ignore_eos: i < min_len);
                        if (top_id == speech_token_size)
                        {
                            att_cache = null!;
                            cnn_cache = null!;
                            lm_input = null!;
                            break;
                        }

                        emittedToken = top_id;
                        out_tokens.Add(top_id);
                        att_cache = chunkResult.attCache.MoveToOuterDisposeScope();
                        cnn_cache = chunkResult.cnnCache.MoveToOuterDisposeScope();
                        lm_input = speech_embedding.GetWeight()[top_id].reshape(1, 1, -1).MoveToOuterDisposeScope();
                    }

                    previousAttCache?.Dispose();
                    previousCnnCache?.Dispose();
                    previousInput?.Dispose();

                    if (!emittedToken.HasValue)
                        break;

                    offset += previousInputLength;
                    yield return emittedToken.Value;
                }
            }
            finally
            {
                oneTokenAttMask?.Dispose();
                att_cache?.Dispose();
                cnn_cache?.Dispose();
                lm_input?.Dispose();
            }
        }

        private IEnumerable<int> InferenceWithPreallocatedTransformerCache(
            TransformerEncoder encoder,
            Tensor lmInput,
            int sampling,
            int minLen,
            int maxLen)
        {
            var outTokens = new List<int>();
            var offset = 0;
            var cache = new TransformerChunkPreallocatedCache(lmInput.shape[1] + maxLen);
            var oneTokenAttMask = torch.ones(new long[] { 1, 1, 1 }, dtype: ScalarType.Bool, device: lmInput.device);

            try
            {
                for (int i = 0; i < maxLen; i++)
                {
                    int? emittedToken = null;
                    var previousInput = lmInput;
                    var previousInputLength = (int)lmInput.shape[1];

                    using (torch.NewDisposeScope())
                    {
                        var attMask = previousInputLength == 1
                            ? oneTokenAttMask
                            : torch.tril(torch.ones(new long[] { 1, previousInputLength, previousInputLength }, dtype: ScalarType.Bool, device: lmInput.device));
                        var chunkResult = encoder.ForwardChunkPreallocated(lmInput, offset, -1, cache, attMask);
                        Tensor yPred = chunkResult.y;

                        var logits = llm_decoder.forward(yPred.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Single(-1), TensorIndex.Ellipsis }));
                        var logp = logits.log_softmax(-1);
                        var topId = sampling_ids(logp.squeeze(0), outTokens, sampling, ignore_eos: i < minLen);
                        if (topId == speech_token_size)
                        {
                            lmInput = null!;
                            break;
                        }

                        emittedToken = topId;
                        outTokens.Add(topId);
                        lmInput = speech_embedding.GetWeight()[topId].reshape(1, 1, -1).MoveToOuterDisposeScope();
                    }

                    previousInput?.Dispose();

                    if (!emittedToken.HasValue)
                        break;

                    offset += previousInputLength;
                    yield return emittedToken.Value;
                }
            }
            finally
            {
                oneTokenAttMask?.Dispose();
                cache.Dispose();
                lmInput?.Dispose();
            }
        }

        private (Tensor y, Tensor attCache, Tensor cnnCache) CallLlmForwardChunk(
            Tensor lmInput,
            int offset,
            int requiredCacheSize,
            Tensor attCache,
            Tensor cnnCache,
            Tensor attMask)
        {
            return llm switch
            {
                TransformerEncoder encoder => encoder.ForwardChunk(lmInput, offset, requiredCacheSize, attCache, cnnCache, attMask),
                _ => ((dynamic)llm).forward_chunk(lmInput, offset, requiredCacheSize, attCache, cnnCache, attMask)
            };
        }

        public int sampling_ids(Tensor weightedScores, List<int> decodedTokens, int samp, bool ignore_eos = true)
        {
            if (ignore_eos)
            {
                weightedScores[(long)speech_token_size] = float.NegativeInfinity;
            }
            if (sampling is not null)
                return sampling(weightedScores, decodedTokens, samp);

            if (samp > 0)
            {
                return Common.RandomSampling(weightedScores, decodedTokens, true);
            }
            return ScalarToInt(weightedScores.argmax(-1));
        }

        protected static int ScalarToInt(Tensor value)
        {
            if (value is null || value.numel() == 0)
                return 0;

            var scalar = value.flatten()[0];
            return scalar.dtype switch
            {
                ScalarType.Int64 => checked((int)scalar.item<long>()),
                ScalarType.Int32 => scalar.item<int>(),
                ScalarType.Int16 => scalar.item<short>(),
                ScalarType.Byte => scalar.item<byte>(),
                ScalarType.Float32 => checked((int)scalar.item<float>()),
                ScalarType.Float64 => checked((int)scalar.item<double>()),
                _ => checked((int)scalar.to_type(ScalarType.Int64).item<long>())
            };
        }
    }


    // Exported from cosyvoice/llm/llm.py Qwen2Encoder
    public class Qwen2Encoder : torch.nn.Module<Tensor, Tensor>
    {
        public readonly string PretrainPath;
        public readonly Qwen2ForCausalLM model;
        public ICosyVoiceProfiler? Profiler
        {
            get => model.Profiler;
            set => model.Profiler = value;
        }
        public QwenAttentionBackend QwenAttentionBackend
        {
            get => model.AttentionBackend;
            set => model.AttentionBackend = value;
        }
        public QwenMlpBackend QwenMlpBackend
        {
            get => model.MlpBackend;
            set => model.MlpBackend = value;
        }

        public Qwen2Encoder(string pretrain_path = "") : base("Qwen2Encoder")
        {
            PretrainPath = pretrain_path ?? string.Empty;
            var config = LoadConfig(PretrainPath);
            model = new TorchSharpUtils.Qwen2ForCausalLM(
                vocabSize: config.vocabSize,
                hiddenSize: config.hiddenSize,
                numAttentionHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                numHiddenLayers: config.numHiddenLayers,
                intermediateSize: config.intermediateSize,
                rmsNormEps: config.rmsNormEps,
                ropeTheta: config.ropeTheta
            );
            register_module("model", model);
        }

        // Module<Tensor,Tensor> contract — returns embed_tokens output (used by Qwen2LM embed_tokens access)
        public override Tensor forward(Tensor input_ids)
        {
            return model.model.embed_tokens.forward(input_ids);
        }

        // Full-sequence forward: (inputs_embeds, xs_lens) -> (hidden_states, mask)
        // Mirrors Python: outs.hidden_states[-1] from HuggingFace Qwen2.
        public (Tensor, Tensor) forward(Tensor xs, Tensor xs_lens)
        {
            var T = xs.shape[1];
            var validMask = MakeValidMask(xs_lens, T);                              // (B, T) bool, True=valid
            var additiveMask = MakeCausalAttentionMask(validMask, T, T);
            var hidden = RunWithEmbeds(xs, additiveMask);
            return (hidden, validMask.unsqueeze(1));                                 // mask: (B,1,T)
        }

        // Autoregressive one-step forward. The managed Qwen path has no native KV cache,
        // The cache stores per-layer K/V tensors so generation can process only new tokens.
        public (Tensor, object) forward_one_step(Tensor xs, Tensor masks, object cache = null)
        {
            var kvCache = cache as Qwen2KvCache;
            var preallocatedCache = cache as Qwen2PreallocatedKvCache;
            Tensor additiveMask = null;

            if (kvCache is null && (preallocatedCache is null || preallocatedCache.Length == 0))
            {
                var seqLen = xs.shape[1];
                var validMask = torch.ones(new long[] { xs.shape[0], seqLen }, dtype: ScalarType.Bool, device: xs.device);
                additiveMask = MakeCausalAttentionMask(validMask, seqLen, seqLen);
            }

            if (preallocatedCache is not null)
            {
                var preallocatedResult = model.ForwardEmbedsWithPreallocatedCache(xs, additiveMask, preallocatedCache);
                return (preallocatedResult.hiddenStates, preallocatedResult.cache);
            }

            var result = model.ForwardEmbedsWithCache(xs, additiveMask, kvCache);
            return (result.hiddenStates, result.cache);
        }

        // Runs through all Qwen2 decoder layers and final norm.
        // Mirrors Python hidden_states[-1].
        private Tensor RunWithEmbeds(Tensor inputs_embeds, Tensor attentionMask)
        {
            return model.ForwardEmbeds(inputs_embeds, attentionMask);
        }

        private static Tensor MakeValidMask(Tensor lengths, long maxLen)
        {
            var batch = lengths.shape[0];
            var range = torch.arange(maxLen, device: lengths.device)
                             .unsqueeze(0).expand(new long[] { batch, maxLen });
            return range < lengths.unsqueeze(1);  // True = valid (not padding)
        }

        private static Tensor MakeCausalAttentionMask(Tensor validMask, long queryLen, long keyLen)
        {
            var causal = torch.tril(torch.ones(new long[] { queryLen, keyLen }, dtype: ScalarType.Bool, device: validMask.device));
            var mask = validMask.unsqueeze(1).unsqueeze(1).logical_and(causal.unsqueeze(0).unsqueeze(0));
            return mask.logical_not().to(ScalarType.Float32) * -1e9f;
        }

        private static (int vocabSize, int hiddenSize, int numAttentionHeads, int numKeyValueHeads, int numHiddenLayers, int intermediateSize, float rmsNormEps, float ropeTheta) LoadConfig(string pretrainPath)
        {
            var configPath = string.IsNullOrWhiteSpace(pretrainPath) ? string.Empty : Path.Combine(pretrainPath, "config.json");
            if (!File.Exists(configPath))
                return (151936, 896, 14, 2, 24, 4864, 1e-6f, 10000.0f);

            var json = JObject.Parse(File.ReadAllText(configPath));
            return (
                json.Value<int?>("vocab_size") ?? 151936,
                json.Value<int?>("hidden_size") ?? 896,
                json.Value<int?>("num_attention_heads") ?? 14,
                json.Value<int?>("num_key_value_heads") ?? json.Value<int?>("num_attention_heads") ?? 2,
                json.Value<int?>("num_hidden_layers") ?? 24,
                json.Value<int?>("intermediate_size") ?? 4864,
                json.Value<float?>("rms_norm_eps") ?? 1e-6f,
                json.Value<float?>("rope_theta") ?? 10000.0f
            );
        }
    }

    public class Qwen2LM : TransformerLM
    {
        // Protected no-op constructor for subclasses that manage their own modules
        protected Qwen2LM(string name) : base(name) { }

        public Qwen2LM(int llmInputSize, int llmOutputSize, int speechTokenSize, torch.nn.Module llm, Func<Tensor, List<int>, int, int> sampling, bool lengthNormalizedLoss = true, float lsmWeight = 0.0f, List<int> mixRatio = null)
            : base("Qwen2LM")
        {
            this.llm_input_size = llmInputSize;
            this.speech_token_size = speechTokenSize;
            this.sos = 0;
            this.task_id = 1;
            this.eos_token = speechTokenSize;
            this.fill_token = speechTokenSize + 2;
            this.llm = llm;
            this.sampling = sampling;
            this.llm_embedding = new ManagedEmbedding(2, llmInputSize);
            this.llm_decoder = new ManagedLinear(llmOutputSize, speechTokenSize + 3);
            this.speech_embedding = new ManagedEmbedding(speechTokenSize + 3, llmInputSize);
            this.CriterionCE = new LabelSmoothingLoss(
                size: speechTokenSize + 3,
                paddingIdx: IGNORE_ID,
                smoothing: lsmWeight,
                normalizeLength: lengthNormalizedLoss
            );
            this.mix_ratio = mixRatio ?? new List<int> { 5, 15 };
            this.stop_token_ids = new List<int> { speechTokenSize, speechTokenSize + 1, speechTokenSize + 2 };
            this.vllm_output_queue = new Dictionary<string, Queue<int>>();
            this.online_feature = false;
            this.onnx_path = string.Empty;
            RegisterComponents();
        }

        // PrepareLmInputTarget as in Python
        public (Tensor, Tensor, Tensor) prepare_lm_input_target(
            Tensor sos_emb,
            Tensor text_token,
            Tensor text_token_emb,
            Tensor text_token_len,
            Tensor task_id_emb,
            Tensor speech_token,
            Tensor speech_token_emb,
            Tensor speech_token_len)
        {
            var lm_target = new List<Tensor>();
            var lm_input = new List<Tensor>();
            var text_token_unpadded = unpad_sequence(text_token, text_token_len.cpu());
            var speech_token_unpadded = unpad_sequence(speech_token, speech_token_len.cpu());
            var text_token_emb_unpadded = unpad_sequence(text_token_emb, text_token_len.cpu());
            var speech_token_emb_unpadded = unpad_sequence(speech_token_emb, speech_token_len.cpu());
            for (int i = 0; i < text_token_unpadded.Count; i++)
            {
                var concat = torch.cat(new Tensor[] { sos_emb.squeeze(0), text_token_emb_unpadded[i], task_id_emb.squeeze(0), speech_token_emb_unpadded[i] }, 0);
                lm_input.Add(concat);
                var target = torch.cat(new Tensor[] { text_token_unpadded[i], speech_token_unpadded[i] }, 0);
                lm_target.Add(target);
            }
            var lm_input_len = torch.tensor(lm_input.ConvertAll(t => t.shape[0]), torch.int32);
            var lm_input_padded = torch.nn.utils.rnn.pad_sequence(lm_input, true, IGNORE_ID);
            var lm_target_padded = torch.nn.utils.rnn.pad_sequence(lm_target, true, IGNORE_ID);
            return (lm_target_padded, lm_input_padded, lm_input_len);
        }

        // Forward method for Qwen2LM
        public override Dictionary<string, Tensor> forward(Dictionary<string, Tensor> batch, Device device)
        {
            // 1. encode text_token
            var text_token = batch["text_token"];
            var text_token_len = batch["text_token_len"];
            Tensor text_token_emb = ((dynamic)llm).forward(text_token); // Simulate embed_tokens

            // 2. encode speech_token
            var speech_token = batch["speech_token"];
            var speech_token_len = batch["speech_token_len"];
            var speech_token_emb = speech_embedding.forward(speech_token);

            // 3. sos and task_id
            var sos_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(sos, device)).reshape(1, 1, -1);
            var task_id_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(task_id, device)).reshape(1, 1, -1);

            // 4. prepare llm_input/target
            var (lm_target, lm_input, lm_input_len) = prepare_lm_input_target(sos_emb, text_token, text_token_emb, text_token_len, task_id_emb, speech_token, speech_token_emb, speech_token_len);
            lm_target = lm_target.to(device);

            // 5. run lm forward
            var lm_result = ((dynamic)llm).forward(lm_input, lm_input_len.to(device));
            Tensor lm_output = lm_result.Item1;
            //Tensor lm_output_mask = lm_result.Item2; // Unused
            var logits = llm_decoder.forward(lm_output);
            var loss = CriterionCE.forward(logits, lm_target.to(device));
            // acc = th_accuracy(logits.view(-1, self.llm_decoder.out_features), lm_target, ignore_label=IGNORE_ID)
            var result = new Dictionary<string, Tensor> { { "loss", loss } };
            return result;
        }

        // Inference method for Qwen2LM
        public override IEnumerable<int> inference(
            Tensor text, Tensor text_len, Tensor prompt_text, Tensor prompt_text_len,
            Tensor prompt_speech_token, Tensor prompt_speech_token_len, Tensor embedding,
            int sampling = 25, float max_token_text_ratio = 20, float min_token_text_ratio = 2, string uuid = "")
        {
            var device = text.device;
            text = torch.cat(new Tensor[] { prompt_text, text }, 1);
            text_len += prompt_text_len;
            Tensor text_emb = ((dynamic)llm).forward(text); // Simulate embed_tokens

            // 3. concat llm_input
            var sos_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(sos, device)).reshape(1, 1, -1);
            var task_id_emb = llm_embedding.GetWeight().index_select(0, TokenIndex(task_id, device)).reshape(1, 1, -1);
            Tensor prompt_speech_token_emb;
            if (TensorFirstInt(prompt_speech_token_len) != 0)
                prompt_speech_token_emb = speech_embedding.forward(prompt_speech_token);
            else
                prompt_speech_token_emb = torch.zeros(new long[] { 1, 0, llm_input_size }, dtype: text_emb.dtype).to(device);
            var lm_input = torch.cat(new Tensor[] { sos_emb, text_emb, task_id_emb, prompt_speech_token_emb }, 1);

            // 4. cal min/max_length
            var targetTextLen = TensorFirstInt(text_len) - TensorFirstInt(prompt_text_len);
            var min_len = (int)(targetTextLen * min_token_text_ratio);
            var max_len = (int)(targetTextLen * max_token_text_ratio);

            foreach (var token in InferenceWrapper(lm_input, sampling, min_len, max_len))
                yield return token;
        }

        protected IEnumerable<int> InferenceWrapper(Tensor lmInput, int sampling, int minLen, int maxLen)
        {
            if (QwenKvCacheBackend == QwenKvCacheBackend.Disabled)
            {
                foreach (var token in InferenceWrapperWithoutKvCache(lmInput, sampling, minLen, maxLen))
                    yield return token;
                yield break;
            }

            var outTokens = new List<int>();
            object cache = QwenKvCacheBackend == QwenKvCacheBackend.Preallocated
                ? new Qwen2PreallocatedKvCache(lmInput.shape[1] + maxLen)
                : null;
            var preallocatedCache = cache is Qwen2PreallocatedKvCache;
            var tokenCount = 0;
            var profileEnabled = Profiler is not null;
            if (llm is Qwen2Encoder qwenEncoder)
            {
                qwenEncoder.Profiler = Profiler;
                qwenEncoder.QwenAttentionBackend = QwenAttentionBackend;
                qwenEncoder.QwenMlpBackend = QwenMlpBackend;
            }
            var decodeTotal = profileEnabled ? Stopwatch.StartNew() : null;
            double forwardMs = 0;
            double logitsSamplingMs = 0;
            try
            {
                for (int i = 0; i < maxLen; i++)
                {
                    int? emittedToken = null;
                    Tensor previousInput = lmInput;
                    object previousCache = preallocatedCache ? null : cache;

                    using (torch.NewDisposeScope())
                    {
                        Tensor masks = null!;
                        var stepSw = profileEnabled ? Stopwatch.StartNew() : null;
                        var stepResult = ((dynamic)llm).forward_one_step(lmInput, masks, cache);
                        if (stepSw is not null)
                        {
                            stepSw.Stop();
                            forwardMs += stepSw.Elapsed.TotalMilliseconds;
                        }
                        Tensor yPred = stepResult.Item1;
                        var nextCache = stepResult.Item2;
                        stepSw?.Restart();
                        var logp = llm_decoder.forward(yPred[TensorIndex.Colon, -1]).log_softmax(-1);
                        var topId = sampling_ids(logp.squeeze(0), outTokens, sampling, ignore_eos: i < minLen);
                        if (stepSw is not null)
                        {
                            stepSw.Stop();
                            logitsSamplingMs += stepSw.Elapsed.TotalMilliseconds;
                        }

                        if (stop_token_ids.Contains(topId))
                        {
                            if (preallocatedCache)
                                DisposeCache(cache);
                            else
                                DisposeCache(previousCache);
                            previousInput?.Dispose();
                            cache = null;
                            lmInput = null;
                            break;
                        }

                        outTokens.Add(topId);
                        tokenCount++;
                        emittedToken = topId;
                        cache = MoveCacheToOuter(nextCache);
                        lmInput = speech_embedding.GetWeight()[topId].reshape(1, 1, -1).MoveToOuterDisposeScope();
                    }

                    DisposeCache(previousCache);
                    previousInput?.Dispose();

                    if (!emittedToken.HasValue)
                        break;
                    yield return emittedToken.Value;
                }
            }
            finally
            {
                decodeTotal?.Stop();
                if (Profiler is not null && decodeTotal is not null)
                {
                    Profiler.Record("llm.qwen.decode_total", decodeTotal.Elapsed.TotalMilliseconds, new Dictionary<string, string>
                {
                    ["tokens"] = tokenCount.ToString(),
                    ["preallocated_kv"] = preallocatedCache.ToString()
                });
                    Profiler.Record("llm.qwen.forward_one_step", forwardMs, new Dictionary<string, string>
                {
                    ["tokens"] = tokenCount.ToString(),
                    ["preallocated_kv"] = preallocatedCache.ToString()
                });
                    Profiler.Record("llm.qwen.logits_sampling", logitsSamplingMs, new Dictionary<string, string>
                {
                    ["tokens"] = tokenCount.ToString()
                });
                }
                DisposeCache(cache);
                lmInput?.Dispose();
            }
        }

        private IEnumerable<int> InferenceWrapperWithoutKvCache(Tensor lmInput, int sampling, int minLen, int maxLen)
        {
            var outTokens = new List<int>();
            try
            {
                for (int i = 0; i < maxLen; i++)
                {
                    int? emittedToken = null;
                    Tensor previousInput = lmInput;

                    using (torch.NewDisposeScope())
                    {
                        Tensor masks = null!;
                        var stepResult = ((dynamic)llm).forward_one_step(lmInput, masks, null);
                        Tensor yPred = stepResult.Item1;
                        DisposeCache(stepResult.Item2);

                        var logp = llm_decoder.forward(yPred[TensorIndex.Colon, -1]).log_softmax(-1);
                        var topId = sampling_ids(logp.squeeze(0), outTokens, sampling, ignore_eos: i < minLen);

                        if (stop_token_ids.Contains(topId))
                        {
                            previousInput?.Dispose();
                            lmInput = null;
                            break;
                        }

                        outTokens.Add(topId);
                        emittedToken = topId;
                        var nextTokenEmb = speech_embedding.GetWeight()[topId].reshape(1, 1, -1);
                        lmInput = torch.cat(new[] { lmInput, nextTokenEmb }, 1).MoveToOuterDisposeScope();
                    }

                    previousInput?.Dispose();

                    if (!emittedToken.HasValue)
                        break;
                    yield return emittedToken.Value;
                }
            }
            finally
            {
                lmInput?.Dispose();
            }
        }

        private static long CacheLength(object cache)
        {
            return cache switch
            {
                Qwen2KvCache qwenCache when qwenCache.Layers.Count > 0 => qwenCache.Layers[0].Key.shape[2],
                Qwen2PreallocatedKvCache preallocatedCache => preallocatedCache.Length,
                Tensor tensorCache => tensorCache.shape[1],
                _ => 0
            };
        }

        private static object MoveCacheToOuter(object cache)
        {
            if (cache is Qwen2PreallocatedKvCache preallocatedCache)
            {
                foreach (var layer in preallocatedCache.Layers)
                {
                    if (layer.Key is not null)
                        layer.Key = layer.Key.MoveToOuterDisposeScope();
                    if (layer.Value is not null)
                        layer.Value = layer.Value.MoveToOuterDisposeScope();
                }
                return preallocatedCache;
            }

            if (cache is not Qwen2KvCache qwenCache)
                return cache;

            var moved = new Qwen2KvCache();
            foreach (var layer in qwenCache.Layers)
            {
                moved.Layers.Add(new Qwen2LayerCache
                {
                    Key = layer.Key.MoveToOuterDisposeScope(),
                    Value = layer.Value.MoveToOuterDisposeScope()
                });
            }
            return moved;
        }

        private static void DisposeCache(object cache)
        {
            if (cache is Qwen2KvCache qwenCache)
            {
                foreach (var layer in qwenCache.Layers)
                {
                    layer.Key?.Dispose();
                    layer.Value?.Dispose();
                }
            }
            else if (cache is Qwen2PreallocatedKvCache preallocatedCache)
            {
                foreach (var layer in preallocatedCache.Layers)
                {
                    layer.Key?.Dispose();
                    layer.Value?.Dispose();
                }
            }
            else if (cache is Tensor tensor)
            {
                tensor.Dispose();
            }
        }

        protected static int TensorFirstInt(Tensor value)
        {
            if (value is null || value.numel() == 0)
                return 0;

            var scalar = value.flatten()[0];
            return scalar.dtype switch
            {
                ScalarType.Int64 => checked((int)scalar.item<long>()),
                ScalarType.Int32 => scalar.item<int>(),
                ScalarType.Int16 => scalar.item<short>(),
                ScalarType.Byte => scalar.item<byte>(),
                ScalarType.Float32 => checked((int)scalar.item<float>()),
                ScalarType.Float64 => checked((int)scalar.item<double>()),
                _ => checked((int)scalar.to_type(ScalarType.Int64).item<long>())
            };
        }

        protected static bool TensorContainsToken(Tensor value, long tokenId)
        {
            if (value is null || value.numel() == 0)
                return false;
            var flat = value.flatten();
            for (long i = 0; i < flat.numel(); i++)
            {
                if (TensorFirstLong(flat[i]) == tokenId)
                    return true;
            }
            return false;
        }

        protected static long TensorFirstLong(Tensor value)
        {
            if (value is null || value.numel() == 0)
                return 0;

            var scalar = value.flatten()[0];
            return scalar.dtype switch
            {
                ScalarType.Int64 => scalar.item<long>(),
                ScalarType.Int32 => scalar.item<int>(),
                ScalarType.Int16 => scalar.item<short>(),
                ScalarType.Byte => scalar.item<byte>(),
                ScalarType.Float32 => checked((long)scalar.item<float>()),
                ScalarType.Float64 => checked((long)scalar.item<double>()),
                _ => scalar.to_type(ScalarType.Int64).item<long>()
            };
        }
    }

    public class CosyVoice3LM : Qwen2LM
    {
        // Python: calls torch.nn.Module.__init__(self) directly, registering ONLY llm, llm_decoder, speech_embedding.
        public CosyVoice3LM(
            int llmInputSize,
            int llmOutputSize,
            int speechTokenSize,
            torch.nn.Module llm,
            Func<Tensor, List<int>, int, int> sampling,
            bool lengthNormalizedLoss = true,
            float lsmWeight = 0.0f,
            List<int> mixRatio = null)
            : base("CosyVoice3LM")   // bypass full Qwen2LM/TransformerLM module creation
        {
            this.llm_input_size   = llmInputSize;
            this.speech_token_size = speechTokenSize;
            this.sampling         = sampling;

            // Python: self.sos = speech_token_size + 0, eos = +1, task_id = +2, fill = +3
            this.sos        = speechTokenSize + 0;
            this.eos_token  = speechTokenSize + 1;
            this.task_id    = speechTokenSize + 2;
            this.fill_token = speechTokenSize + 3;

            // Python: llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
            this.llm_decoder      = new ManagedLinear(llmOutputSize, speechTokenSize + 200, hasBias: false);
            // Python: speech_embedding = nn.Embedding(speech_token_size + 200, llm_input_size)
            this.speech_embedding = new ManagedEmbedding(speechTokenSize + 200, llmInputSize);
            this.llm              = (torch.nn.Module<Tensor, Tensor>)llm;

            this.CriterionCE = new LabelSmoothingLoss(
                size: speechTokenSize + 200,
                paddingIdx: IGNORE_ID,
                smoothing: lsmWeight,
                normalizeLength: lengthNormalizedLoss
            );
            this.mix_ratio     = mixRatio ?? new List<int> { 5, 15 };
            this.stop_token_ids = new List<int>();
            for (int i = 0; i < 200; i++)
                this.stop_token_ids.Add(speechTokenSize + i);
            this.vllm_output_queue = new Dictionary<string, Queue<int>>();

            RegisterComponents();  // registers: llm, llm_decoder, speech_embedding
        }

        // Python: uses speech_embedding.weight[sos/task_id] (not llm_embedding)
        public override Dictionary<string, Tensor> forward(Dictionary<string, Tensor> batch, Device device)
        {
            var text_token     = batch["text_token"].to(device);
            var text_token_len = batch["text_token_len"].to(device);
            var text_token_emb = ((Qwen2Encoder)llm).model.model.embed_tokens.forward(text_token);

            var speech_token     = batch["speech_token"].to(device);
            var speech_token_len = batch["speech_token_len"].to(device);
            var speech_token_emb = speech_embedding.forward(speech_token);

            var sos_emb     = speech_embedding.weight.index_select(0, TokenIndex(sos, device)).reshape(1, 1, -1);
            var task_id_emb = speech_embedding.weight.index_select(0, TokenIndex(task_id, device)).reshape(1, 1, -1);

            var (lm_target, lm_input, lm_input_len) = prepare_lm_input_target(
                sos_emb, text_token, text_token_emb, text_token_len,
                task_id_emb, speech_token, speech_token_emb, speech_token_len);
            lm_target = lm_target.to(device);

            var lm_result = ((Qwen2Encoder)llm).forward(lm_input, lm_input_len.to(device));
            Tensor lm_output = lm_result.Item1;
            var logits = llm_decoder.forward(lm_output);
            var loss   = CriterionCE.forward(logits, lm_target.to(device));
            return new Dictionary<string, Tensor> { { "loss", loss } };
        }

        // Python: uses speech_embedding.weight[sos/task_id] during inference
        public override IEnumerable<int> inference(
            Tensor text, Tensor text_len, Tensor prompt_text, Tensor prompt_text_len,
            Tensor prompt_speech_token, Tensor prompt_speech_token_len, Tensor embedding,
            int sampling = 25, float max_token_text_ratio = 20, float min_token_text_ratio = 2, string uuid = "")
        {
            var device   = text.device;
            text         = torch.cat(new Tensor[] { prompt_text, text }, 1);
            text_len    += prompt_text_len;
            if (!TensorContainsToken(text, 151646))
                throw new InvalidOperationException("<|endofprompt|> not detected in CosyVoice3 prompt_text, check your input.");
            var text_emb = ((Qwen2Encoder)llm).model.model.embed_tokens.forward(text);

            var sos_emb     = speech_embedding.weight.index_select(0, TokenIndex(sos, device)).reshape(1, 1, -1);
            var task_id_emb = speech_embedding.weight.index_select(0, TokenIndex(task_id, device)).reshape(1, 1, -1);
            Tensor prompt_speech_token_emb;
            if (TensorFirstInt(prompt_speech_token_len) != 0)
                prompt_speech_token_emb = speech_embedding.forward(prompt_speech_token);
            else
                prompt_speech_token_emb = torch.zeros(new long[] { 1, 0, llm_input_size }, dtype: text_emb.dtype).to(device);

            var lm_input = torch.cat(new Tensor[] { sos_emb, text_emb, task_id_emb, prompt_speech_token_emb }, 1);

            var targetTextLen = TensorFirstInt(text_len) - TensorFirstInt(prompt_text_len);
            var min_len = (int)(targetTextLen * min_token_text_ratio);
            var max_len = (int)(targetTextLen * max_token_text_ratio);

            foreach (var token in InferenceWrapper(lm_input, sampling, min_len, max_len))
                yield return token;
        }
    }
}
