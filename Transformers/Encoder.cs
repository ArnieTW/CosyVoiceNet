// Equivalent Python file: cosyvoice/transformer/encoder.py
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using CosyVoiceNet.Utils;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public sealed class TransformerChunkPreallocatedCache : IDisposable
    {
        private readonly List<PreallocatedAttentionCache> attentionLayers = new();
        private readonly List<Tensor?> cnnLayers = new();

        public TransformerChunkPreallocatedCache(long maxLength)
        {
            if (maxLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxLength), "Cache length must be greater than zero.");

            MaxLength = maxLength;
        }

        public long MaxLength { get; }
        public long Length { get; set; }

        public PreallocatedAttentionCache GetAttentionLayer(int index)
        {
            while (attentionLayers.Count <= index)
            {
                attentionLayers.Add(new PreallocatedAttentionCache());
            }

            return attentionLayers[index];
        }

        public Tensor? GetCnnLayer(int index)
        {
            return index < cnnLayers.Count ? cnnLayers[index] : null;
        }

        public void SetCnnLayer(int index, Tensor? tensor)
        {
            while (cnnLayers.Count <= index)
            {
                cnnLayers.Add(null);
            }

            cnnLayers[index]?.Dispose();
            cnnLayers[index] = tensor;
        }

        public void Dispose()
        {
            foreach (var attentionLayer in attentionLayers)
            {
                attentionLayer.Dispose();
            }

            foreach (var cnnLayer in cnnLayers)
            {
                cnnLayer?.Dispose();
            }

            attentionLayers.Clear();
            cnnLayers.Clear();
            Length = 0;
        }
    }

    public class TransformerEncoder : torch.nn.Module<(Tensor, Tensor, int, int), (Tensor, Tensor, Tensor)>
    {
        public readonly ModuleList<nn.Module> encoders = new ModuleList<nn.Module>();
        public readonly nn.Module<Tensor, Tensor> after_norm;
        private readonly bool normalizeBefore;
        private readonly bool useDynamicChunk;
        private readonly bool useDynamicLeftChunk;
        private readonly GlobalCMVN? globalCmvn;
        public readonly EncoderEmbedding embed;
        private readonly int outputSize;
        private readonly int staticChunkSize;
        private readonly bool gradientCheckpointing;
        private Tensor? oneTokenChunkMask;
        private string? oneTokenChunkMaskKey;
        private Tensor? emptyMaskPad;
        private string? emptyMaskPadKey;
        private Tensor? emptyCnnCache;
        private string? emptyCnnCacheKey;

        public TransformerEncoder(int inputSize, int outputSize = 256, int attentionHeads = 4, int linearUnits = 2048, int numBlocks = 6,
            double dropoutRate = 0.1, double positionalDropoutRate = 0.1, double attentionDropoutRate = 0.0, string inputLayer = "conv2d",
            string posEncLayerType = "abs_pos", bool normalizeBefore = true, int staticChunkSize = 0, bool useDynamicChunk = false,
            GlobalCMVN? globalCmvn = null, bool useDynamicLeftChunk = false, bool gradientCheckpointing = false, bool keyBias = true,
            string selfAttentionLayerType = "selfattn", string activationType = "relu", bool conformerLayer = false,
            bool macaronStyle = true, bool useCnnModule = false, int cnnModuleKernel = 15) : base("TransformerEncoder")
        {
            this.outputSize = outputSize;
            this.normalizeBefore = normalizeBefore;
            this.useDynamicChunk = useDynamicChunk;
            this.useDynamicLeftChunk = useDynamicLeftChunk;
            this.globalCmvn = globalCmvn;
            this.staticChunkSize = staticChunkSize;
            this.gradientCheckpointing = gradientCheckpointing;
            this.after_norm = nn.LayerNorm(outputSize, eps: 1e-5);

            this.embed = new EncoderEmbedding(inputSize, outputSize, dropoutRate, inputLayer, posEncLayerType, positionalDropoutRate);

            Func<Tensor, Tensor> activation = ResolveActivation(activationType);

            for (int i = 0; i < numBlocks; i++)
            {
                nn.Module selfAttn = string.Equals(selfAttentionLayerType, "rel_selfattn", StringComparison.OrdinalIgnoreCase)
                    ? new RelPositionMultiHeadedAttention(attentionHeads, outputSize, attentionDropoutRate, keyBias)
                    : new MultiHeadedAttention(attentionHeads, outputSize, attentionDropoutRate, keyBias);
                var feedForward = new PositionwiseFeedForward(outputSize, linearUnits, dropoutRate, activation);
                if (conformerLayer)
                {
                    nn.Module? feedForwardMacaron = macaronStyle ? new PositionwiseFeedForward(outputSize, linearUnits, dropoutRate, activation) : null;
                    nn.Module? convModule = useCnnModule ? new ConvolutionModule(outputSize, cnnModuleKernel, activation) : null;
                    encoders.append(new ConformerEncoderLayer(outputSize, selfAttn, feedForward, feedForwardMacaron, convModule, dropoutRate, normalizeBefore));
                }
                else
                {
                    encoders.append(new TransformerEncoderLayer(outputSize, selfAttn, feedForward, dropoutRate, normalizeBefore));
                }
            }

            RegisterComponents();
        }

        public int output_size() => outputSize;

        public (Tensor, Tensor, Tensor) Forward(Tensor xs, Tensor xsLens, int decodingChunkSize = 0, int numDecodingLeftChunks = -1)
            => forward((xs, xsLens, decodingChunkSize, numDecodingLeftChunks));

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, int, int) inputs)
        {
            var (xs, xsLens, decodingChunkSize, numDecodingLeftChunks) = inputs;

            int T = (int)xs.shape[1];
            var masks = ~MakePadMask(xsLens, T).unsqueeze(1);

            if (globalCmvn != null)
            {
                xs = globalCmvn.Forward(xs);
            }

            var (embeddedXs, posEmb, updatedMasks) = embed.Forward(xs, masks);
            masks = updatedMasks;

            var chunkMasks = AddOptionalChunkMask(embeddedXs, masks, decodingChunkSize, numDecodingLeftChunks);

            if (gradientCheckpointing)
            {
                xs = ForwardLayersCheckpointed(embeddedXs, chunkMasks, posEmb, masks);
            }
            else
            {
                xs = ForwardLayers(embeddedXs, chunkMasks, posEmb, masks);
            }

            if (normalizeBefore)
            {
                xs = after_norm.forward(xs);
            }

            return (xs, masks, masks);
        }

        public (Tensor, Tensor) ForwardChunkByChunk(Tensor xs, int decodingChunkSize, int numDecodingLeftChunks = -1)
        {
            if (decodingChunkSize <= 0) throw new ArgumentException("Decoding chunk size must be greater than 0.");
            if (staticChunkSize <= 0 && !useDynamicChunk) throw new InvalidOperationException("Model must be trained with static or dynamic chunking.");

            int subsampling = embed.SubsamplingRate;
            int context = embed.RightContext + 1;
            int stride = subsampling * decodingChunkSize;
            int decodingWindow = (decodingChunkSize - 1) * subsampling + context;
            int numFrames = (int)xs.shape[1];

            var attCache = torch.zeros(new long[] { 0, 0, 0, 0 }, device: xs.device);
            var cnnCache = torch.zeros(new long[] { 0, 0, 0, 0 }, device: xs.device);
            var outputs = new List<Tensor>();
            int offset = 0;
            int requiredCacheSize = decodingChunkSize * numDecodingLeftChunks;

            for (int cur = 0; cur < numFrames - context + 1; cur += stride)
            {
                int end = Math.Min(cur + decodingWindow, numFrames);
                var chunkXs = xs.index(new TensorIndex[] { TensorIndex.Slice(0, 1), TensorIndex.Slice(cur, end), TensorIndex.Ellipsis });
                var attMask = torch.ones(new long[] { 1, 1, chunkXs.shape[1] }, device: xs.device, dtype: ScalarType.Bool); // Added attMask.
                var (y, newAttCache, newCnnCache) = ForwardChunk(chunkXs, offset, requiredCacheSize, attCache, cnnCache, attMask);
                outputs.Add(y);
                offset += (int)y.shape[1];
                attCache = newAttCache;
                cnnCache = newCnnCache;
            }

            var ys = torch.cat(outputs.ToArray(), 1);
            var masks = torch.ones(new long[] { 1, 1, ys.shape[1] }, device: ys.device, dtype: ScalarType.Bool);
            return (ys, masks);
        }

        public Tensor ForwardOneStep(Tensor xs, Tensor attCache, Tensor cnnCache, Tensor attMask)
        {
            var tmpMasks = torch.ones(new long[] { 1, xs.shape[1] }, device: xs.device, dtype: ScalarType.Bool).unsqueeze(1);

            if (globalCmvn != null)
            {
                xs = globalCmvn.Forward(xs);
            }

            var (embeddedXs, posEmb, _) = embed.Forward(xs, tmpMasks);

            // Implement one-step forward logic
            // Placeholder for caching logic
            return embeddedXs;
        }

        public Tensor EmbedTokens(Tensor tokens)
        {
            // Implement embedding logic for tokens
            return embed.Forward(tokens, null).Item1;
        }

        private Tensor MakePadMask(Tensor lengths, int maxLength)
        {
            var range = torch.arange(maxLength, device: lengths.device).unsqueeze(0);
            return range >= lengths.unsqueeze(1);
        }

        private Tensor AddOptionalChunkMask(Tensor xs, Tensor masks, int decodingChunkSize, int numDecodingLeftChunks)
        {
            return Mask.AddOptionalChunkMask(xs, masks, useDynamicChunk, useDynamicLeftChunk,
                decodingChunkSize, staticChunkSize, numDecodingLeftChunks);
        }

        public virtual Tensor ForwardLayers(Tensor xs, Tensor masks, Tensor posEmb, Tensor maskPad)
        {
            foreach (var layer in encoders)
            {
                var res = ((dynamic)layer).forward((xs, masks, posEmb, maskPad, (Tensor?)null, (Tensor?)null));
                xs = res.Item1;
            }

            return xs;
        }

        private static Func<Tensor, Tensor> ResolveActivation(string activationType)
        {
            return activationType.ToLowerInvariant() switch
            {
                "swish" => Activation.Swish,
                "silu" => x => torch.nn.functional.silu(x),
                "gelu" => x => torch.nn.functional.gelu(x),
                "tanh" => x => torch.tanh(x),
                "hardtanh" => x => torch.nn.functional.hardtanh(x),
                "selu" => x => torch.nn.functional.selu(x),
                _ => x => torch.nn.functional.relu(x)
            };
        }

        public Tensor ForwardLayersCheckpointed(Tensor xs, Tensor masks, Tensor posEmb, Tensor maskPad)
        {
            foreach (var layer in encoders)
            {
                var res = ((dynamic)layer).forward((xs, masks, posEmb, maskPad, (Tensor?)null, (Tensor?)null)); // Replace with actual checkpointing logic if available
                xs = res.Item1;
            }

            return xs;
        }

        public (Tensor, Tensor, Tensor) ForwardChunk(Tensor xs, int offset, int requiredCacheSize, Tensor attCache, Tensor cnnCache, Tensor attMask)
        {
            var tmpMasks = xs.shape[1] == 1
                ? GetOneTokenChunkMask(xs.device)
                : torch.ones(new long[] { 1, xs.shape[1] }, device: xs.device, dtype: ScalarType.Bool).unsqueeze(1);

            if (globalCmvn != null)
            {
                xs = globalCmvn.Forward(xs);
            }

            var (embeddedXs, _, _) = embed.Forward(xs, tmpMasks, offset);

            var elayers = attCache.shape[0];
            var cacheT1 = attCache.shape.Length > 2 ? attCache.shape[2] : 0;
            var chunkSize = embeddedXs.shape[1];
            var attentionKeySize = cacheT1 + chunkSize;
            var posEmb = embed.PositionEncoding(offset - cacheT1, (int)attentionKeySize, applyDropout: false, embeddedXs.device);

            long nextCacheStart;
            if (requiredCacheSize < 0)
                nextCacheStart = 0;
            else if (requiredCacheSize == 0)
                nextCacheStart = attentionKeySize;
            else
                nextCacheStart = Math.Max(attentionKeySize - requiredCacheSize, 0);

            var newAttCaches = new List<Tensor>();
            var newCnnCaches = new List<Tensor>();
            var fakeMaskPad = GetEmptyMaskPad(xs.device);
            var layerIndex = 0;
            foreach (var layer in encoders)
            {
                var layerAttCache = elayers > 0 ? attCache.narrow(0, layerIndex, 1) : attCache;
                var layerCnnCache = cnnCache.shape[0] > 0 ? cnnCache[layerIndex] : cnnCache;
                var res = ((dynamic)layer).forward((embeddedXs, attMask, posEmb, fakeMaskPad, layerAttCache, layerCnnCache));
                embeddedXs = res.Item1;

                Tensor newAttCache = res.Item3;
                Tensor newCnnCache = res.Item4;
                newAttCaches.Add(newAttCache.narrow(2, nextCacheStart, attentionKeySize - nextCacheStart).contiguous().clone());
                newCnnCaches.Add(newCnnCache.unsqueeze(0));
                layerIndex++;
            }

            if (normalizeBefore)
            {
                embeddedXs = after_norm.forward(embeddedXs);
            }

            Tensor rAttCache;
            if (newAttCaches.Count == 0)
            {
                rAttCache = torch.zeros(new long[] { 0, 0, 0, 0 }, dtype: embeddedXs.dtype, device: embeddedXs.device);
            }
            else
            {
                var first = newAttCaches[0];
                rAttCache = torch.zeros(new long[] { newAttCaches.Count, first.shape[1], first.shape[2], first.shape[3] }, dtype: embeddedXs.dtype, device: embeddedXs.device);
                for (var i = 0; i < newAttCaches.Count; i++)
                {
                    rAttCache[i].copy_(newAttCaches[i].squeeze(0));
                }
            }
            var rCnnCache = newCnnCaches.Count == 0 || newCnnCaches[0].numel() == 0
                ? torch.zeros(new long[] { layerIndex, 0, 0, 0 }, dtype: embeddedXs.dtype, device: embeddedXs.device)
                : torch.cat(newCnnCaches.ToArray(), dim: 0);
            return (embeddedXs, rAttCache, rCnnCache);
        }

        public (Tensor y, TransformerChunkPreallocatedCache cache) ForwardChunkPreallocated(
            Tensor xs,
            int offset,
            int requiredCacheSize,
            TransformerChunkPreallocatedCache cache,
            Tensor attMask)
        {
            if (requiredCacheSize >= 0)
                throw new NotSupportedException("Preallocated legacy transformer cache currently supports unbounded cache mode only.");

            var tmpMasks = xs.shape[1] == 1
                ? GetOneTokenChunkMask(xs.device)
                : torch.ones(new long[] { 1, xs.shape[1] }, device: xs.device, dtype: ScalarType.Bool).unsqueeze(1);

            if (globalCmvn != null)
            {
                xs = globalCmvn.Forward(xs);
            }

            var (embeddedXs, _, _) = embed.Forward(xs, tmpMasks, offset);
            var cacheT1 = cache.Length;
            var chunkSize = embeddedXs.shape[1];
            var attentionKeySize = cacheT1 + chunkSize;
            if (attentionKeySize > cache.MaxLength)
                throw new InvalidOperationException($"Legacy transformer cache exhausted: required {attentionKeySize}, allocated {cache.MaxLength}.");

            var posEmb = embed.PositionEncoding(offset - cacheT1, (int)attentionKeySize, applyDropout: false, embeddedXs.device);
            var fakeMaskPad = GetEmptyMaskPad(xs.device);
            var emptyCnn = GetEmptyCnnCache(embeddedXs.dtype, embeddedXs.device);
            var layerIndex = 0;

            foreach (var layer in encoders)
            {
                var layerAttCache = cache.GetAttentionLayer(layerIndex);
                var layerCnnCache = cache.GetCnnLayer(layerIndex) ?? emptyCnn;
                (Tensor x, Tensor mask, Tensor? cnnCache) res = layer switch
                {
                    ConformerEncoderLayer conformer => conformer.ForwardWithPreallocatedAttention(embeddedXs, attMask, posEmb, fakeMaskPad, layerAttCache, layerCnnCache, cache.MaxLength),
                    TransformerEncoderLayer transformer => transformer.ForwardWithPreallocatedAttention(embeddedXs, attMask, posEmb, fakeMaskPad, layerAttCache, layerCnnCache, cache.MaxLength),
                    _ => throw new NotSupportedException($"Preallocated legacy transformer cache is not supported for encoder layer {layer.GetType().Name}.")
                };

                embeddedXs = res.x;
                if (res.cnnCache is not null && res.cnnCache.numel() > 0)
                {
                    cache.SetCnnLayer(layerIndex, res.cnnCache.MoveToOuterDisposeScope());
                }

                layerIndex++;
            }

            if (normalizeBefore)
            {
                embeddedXs = after_norm.forward(embeddedXs);
            }

            cache.Length = attentionKeySize;
            return (embeddedXs, cache);
        }

        private Tensor GetOneTokenChunkMask(Device device)
        {
            var key = device.ToString();
            if (oneTokenChunkMask is not null && string.Equals(oneTokenChunkMaskKey, key, StringComparison.Ordinal))
                return oneTokenChunkMask;

            oneTokenChunkMask?.Dispose();
            oneTokenChunkMask = torch.ones(new long[] { 1, 1, 1 }, dtype: ScalarType.Bool, device: device).DetachFromDisposeScope();
            oneTokenChunkMaskKey = key;
            return oneTokenChunkMask;
        }

        private Tensor GetEmptyMaskPad(Device device)
        {
            var key = device.ToString();
            if (emptyMaskPad is not null && string.Equals(emptyMaskPadKey, key, StringComparison.Ordinal))
                return emptyMaskPad;

            emptyMaskPad?.Dispose();
            emptyMaskPad = torch.ones(new long[] { 0, 0, 0 }, dtype: ScalarType.Bool, device: device).DetachFromDisposeScope();
            emptyMaskPadKey = key;
            return emptyMaskPad;
        }

        private Tensor GetEmptyCnnCache(ScalarType dtype, Device device)
        {
            var key = $"{device}:{dtype}";
            if (emptyCnnCache is not null && string.Equals(emptyCnnCacheKey, key, StringComparison.Ordinal))
                return emptyCnnCache;

            emptyCnnCache?.Dispose();
            emptyCnnCache = torch.zeros(new long[] { 0, 0, 0 }, dtype: dtype, device: device).DetachFromDisposeScope();
            emptyCnnCacheKey = key;
            return emptyCnnCache;
        }
    }

    public class LayerNormalization
    {
        private readonly int dModel;

        public LayerNormalization(int dModel)
        {
            this.dModel = dModel;
        }

        public Tensor Forward(Tensor input)
        {
            var mean = input.mean();
            var std = input.std();
            return (input - mean) / (std + 1e-5f);
        }
    }

    public class EncoderEmbedding : nn.Module<(Tensor xs, Tensor masks), (Tensor xs, Tensor posEmb, Tensor masks)>
    {
        public readonly nn.Module<Tensor, Tensor> @out;
        public readonly PositionalEncoding pos_enc;
        private readonly bool legacy;

        public int SubsamplingRate { get; private set; } = 1;
        public int RightContext { get; private set; } = 0;

        public EncoderEmbedding(int inputSize, int outputSize, double dropoutRate, string inputLayer, string posEncLayerType, double positionalDropoutRate)
            : base("EncoderEmbedding")
        {
            legacy = string.Equals(inputLayer, "linear_legacy", StringComparison.OrdinalIgnoreCase);
            @out = legacy
                ? nn.Sequential(
                    nn.Linear(inputSize, outputSize),
                    nn.LayerNorm(outputSize, eps: 1e-5),
                    nn.Dropout(dropoutRate),
                    nn.ReLU())
                : nn.Sequential(
                    nn.Linear(inputSize, outputSize),
                    nn.LayerNorm(outputSize, eps: 1e-5),
                    nn.Dropout(dropoutRate));
            pos_enc = posEncLayerType.ToLowerInvariant() switch
            {
                "rel_pos" => new RelPositionalEncoding(outputSize, positionalDropoutRate),
                "rel_pos_espnet" => new EspnetRelPositionalEncoding(outputSize, positionalDropoutRate),
                "no_pos" => new NoPositionalEncoding(outputSize, positionalDropoutRate),
                _ => new PositionalEncoding(outputSize, positionalDropoutRate)
            };
            RegisterComponents();
        }

        public (Tensor, Tensor, Tensor) Forward(Tensor xs, Tensor masks, long offset = 0)
        {
            xs = @out.forward(xs);
            var (encoded, posEmb) = pos_enc.ForwardWithPosition(xs, offset);
            return (encoded, posEmb, masks);
        }

        public Tensor PositionEncoding(long offset, int size, bool applyDropout = true, Device? device = null)
            => pos_enc.position_encoding(offset, size, applyDropout, device);

        public override (Tensor xs, Tensor posEmb, Tensor masks) forward((Tensor xs, Tensor masks) inputs)
            => Forward(inputs.xs, inputs.masks);
    }

    public class Embedding : EncoderEmbedding
    {
        public Embedding(int inputSize, int outputSize, double dropoutRate, string posEncLayerType, double positionalDropoutRate)
            : base(inputSize, outputSize, dropoutRate, "linear", posEncLayerType, positionalDropoutRate)
        {
        }
    }

    public class GlobalCMVN
    {
        public Tensor Forward(Tensor xs)
        {
            var mean = xs.mean(new long[] { 0 }, true);
            var std = xs.std(new long[] { 0 }, true);
            return (xs - mean) / (std + 1e-5f);
        }
    }

    public class ConformerEncoder : TransformerEncoder
    {
        public ConformerEncoder(int inputSize, int outputSize = 256, int attentionHeads = 4, int linearUnits = 2048, int numBlocks = 6,
            double dropoutRate = 0.1, double positionalDropoutRate = 0.1, double attentionDropoutRate = 0.0, string inputLayer = "conv2d",
            string posEncLayerType = "rel_pos", bool normalizeBefore = true, int staticChunkSize = 0, bool useDynamicChunk = false,
            GlobalCMVN? globalCmvn = null, bool useDynamicLeftChunk = false, int positionwiseConvKernelSize = 1, bool macaronStyle = true,
            string selfAttentionLayerType = "rel_selfattn", string activationType = "swish", bool useCnnModule = true, int cnnModuleKernel = 15,
            bool causal = false, string cnnModuleNorm = "batch_norm")
            : base(inputSize, outputSize, attentionHeads, linearUnits, numBlocks, dropoutRate, positionalDropoutRate, attentionDropoutRate,
                  inputLayer, posEncLayerType, normalizeBefore, staticChunkSize, useDynamicChunk, globalCmvn, useDynamicLeftChunk,
                  keyBias: true, selfAttentionLayerType: selfAttentionLayerType, activationType: activationType,
                  conformerLayer: true, macaronStyle: macaronStyle, useCnnModule: useCnnModule, cnnModuleKernel: cnnModuleKernel)
        {
        }
    }
}
