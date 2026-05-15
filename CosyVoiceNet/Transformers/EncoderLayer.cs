// Equivalent Python file: cosyvoice/transformer/encoder_layer.py
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public class TransformerEncoderLayer : torch.nn.Module<(Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?), (Tensor, Tensor, Tensor, Tensor)>
    {
        public nn.Module self_attn;
        public nn.Module feed_forward;
        public readonly nn.Module<Tensor, Tensor> norm1;
        public readonly nn.Module<Tensor, Tensor> norm2;
        public readonly nn.Module<Tensor, Tensor> dropout;
        public readonly bool NormalizeBefore;

        public TransformerEncoderLayer(int size, dynamic selfAttn, dynamic feedForward, double dropoutRate, bool normalizeBefore = true) : base("TransformerEncoderLayer")
        {
            self_attn = (nn.Module)selfAttn;
            feed_forward = (nn.Module)feedForward;
            norm1 = torch.nn.LayerNorm(size, eps: 1e-12);
            norm2 = torch.nn.LayerNorm(size, eps: 1e-12);
            dropout = torch.nn.Dropout(dropoutRate);
            NormalizeBefore = normalizeBefore;

            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor, Tensor) forward((Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?) inputs)
        {
            var (x, mask, posEmb, maskPad, attCache, cnnCache) = inputs;

            var residual = x;
            if (NormalizeBefore) x = norm1.forward(x);
            var res1 = CallSelfAttention(x, mask, posEmb, attCache);
            Tensor xAtt = res1.Item1;
            var newAttCache = res1.Item2;
            x = residual + dropout.forward(xAtt);
            if (!NormalizeBefore) x = norm1.forward(x);

            residual = x;
            if (NormalizeBefore) x = norm2.forward(x);
            x = residual + dropout.forward(((dynamic)feed_forward).forward(x));
            if (!NormalizeBefore) x = norm2.forward(x);

            var fakeCnnCache = torch.zeros(new long[] { 0, 0, 0 }, dtype: x.dtype, device: x.device);
            return (x, mask, newAttCache, fakeCnnCache);
        }

        private (Tensor, Tensor) CallSelfAttention(Tensor x, Tensor mask, Tensor posEmb, Tensor? attCache)
        {
            return self_attn switch
            {
                RelPositionMultiHeadedAttention rel => rel.Forward(x, x, x, mask, posEmb, attCache),
                MultiHeadedAttention att => att.Forward(x, x, x, mask, posEmb, attCache),
                _ => ((dynamic)self_attn).forward((x, x, x, mask, posEmb, attCache))
            };
        }

        public (Tensor x, Tensor mask, Tensor? cnnCache) ForwardWithPreallocatedAttention(
            Tensor x,
            Tensor mask,
            Tensor posEmb,
            Tensor maskPad,
            PreallocatedAttentionCache attCache,
            Tensor? cnnCache,
            long maxCacheLength)
        {
            var residual = x;
            if (NormalizeBefore) x = norm1.forward(x);
            var res1 = CallSelfAttentionPreallocated(x, mask, posEmb, attCache, maxCacheLength);
            x = residual + dropout.forward(res1.output);
            if (!NormalizeBefore) x = norm1.forward(x);

            residual = x;
            if (NormalizeBefore) x = norm2.forward(x);
            x = residual + dropout.forward(((dynamic)feed_forward).forward(x));
            if (!NormalizeBefore) x = norm2.forward(x);

            return (x, mask, null);
        }

        private (Tensor output, long nextLength) CallSelfAttentionPreallocated(
            Tensor x,
            Tensor mask,
            Tensor posEmb,
            PreallocatedAttentionCache attCache,
            long maxCacheLength)
        {
            return self_attn switch
            {
                RelPositionMultiHeadedAttention rel => rel.ForwardWithPreallocatedCache(x, x, x, mask, posEmb, attCache, maxCacheLength),
                MultiHeadedAttention att => att.ForwardWithPreallocatedCache(x, x, x, mask, posEmb, attCache, maxCacheLength),
                _ => throw new NotSupportedException($"Preallocated legacy transformer cache is not supported for attention module {self_attn.GetType().Name}.")
            };
        }
    }

    public class ConformerEncoderLayer : torch.nn.Module<(Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?), (Tensor, Tensor, Tensor, Tensor)>
    {
        public nn.Module self_attn;
        public nn.Module feed_forward;
        public nn.Module? feed_forward_macaron;
        public nn.Module? conv_module;
        public readonly nn.Module<Tensor, Tensor> norm_ff;
        public readonly nn.Module<Tensor, Tensor> norm_mha;
        public readonly nn.Module<Tensor, Tensor> norm_ff_macaron;
        public readonly nn.Module<Tensor, Tensor> norm_conv;
        public readonly nn.Module<Tensor, Tensor> norm_final;
        public readonly nn.Module<Tensor, Tensor> dropout;
        private readonly double FFScale;
        public readonly bool NormalizeBefore;

        public ConformerEncoderLayer(int size, dynamic selfAttn, dynamic feedForward, dynamic feedForwardMacaron, dynamic convModule, double dropoutRate, bool normalizeBefore = true) : base("ConformerEncoderLayer")
        {
            self_attn = (nn.Module)selfAttn;
            feed_forward = (nn.Module)feedForward;
            feed_forward_macaron = feedForwardMacaron == null ? null : (nn.Module)feedForwardMacaron;
            conv_module = convModule == null ? null : (nn.Module)convModule;
            norm_ff = torch.nn.LayerNorm(size, eps: 1e-12);
            norm_mha = torch.nn.LayerNorm(size, eps: 1e-12);
            if (feedForwardMacaron != null)
            {
                norm_ff_macaron = torch.nn.LayerNorm(size, eps: 1e-12);
                FFScale = 0.5;
            }
            else
            {
                norm_ff_macaron = torch.nn.Identity(); // Default to identity if not used
                FFScale = 1.0;
            }
            if (convModule != null)
            {
                norm_conv = torch.nn.LayerNorm(size, eps: 1e-12);
                norm_final = torch.nn.LayerNorm(size, eps: 1e-12);
            }
            else
            {
                norm_conv = torch.nn.Identity(); // Default to identity if not used
                norm_final = torch.nn.Identity();
            }
            dropout = torch.nn.Dropout(dropoutRate);
            NormalizeBefore = normalizeBefore;

            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor, Tensor) forward((Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?) inputs)
        {
            var (x, mask, posEmb, maskPad, attCache, cnnCache) = inputs;

            if (feed_forward_macaron != null)
            {
                var residual = x;
                if (NormalizeBefore) x = norm_ff_macaron.forward(x);
                x = residual + FFScale * dropout.forward(((dynamic)feed_forward_macaron).forward(x));
                if (!NormalizeBefore) x = norm_ff_macaron.forward(x);
            }

            var mhaResidual = x;
            if (NormalizeBefore) x = norm_mha.forward(x);
            var mhaRes = CallSelfAttention(x, mask, posEmb, attCache);
            Tensor xAtt = mhaRes.Item1;
            var newAttCache = mhaRes.Item2;
            x = mhaResidual + dropout.forward(xAtt);
            if (!NormalizeBefore) x = norm_mha.forward(x);

            Tensor newCnnCache = torch.zeros(new long[] { 0, 0, 0 }, dtype: x.dtype, device: x.device);
            if (conv_module != null)
            {
                var convResidual = x;
                if (NormalizeBefore) x = norm_conv.forward(x);
                var convRes = ((dynamic)conv_module).forward((x, maskPad, cnnCache));
                x = convRes.Item1;
                newCnnCache = convRes.Item2;
                x = convResidual + dropout.forward(x);
                if (!NormalizeBefore) x = norm_conv.forward(x);
            }

            var ffResidual = x;
            if (NormalizeBefore) x = norm_ff.forward(x);
            x = ffResidual + FFScale * dropout.forward(((dynamic)feed_forward).forward(x));
            if (!NormalizeBefore) x = norm_ff.forward(x);

            if (conv_module != null)
            {
                x = norm_final.forward(x);
            }

            return (x, mask, newAttCache, newCnnCache);
        }

        private (Tensor, Tensor) CallSelfAttention(Tensor x, Tensor mask, Tensor posEmb, Tensor? attCache)
        {
            return self_attn switch
            {
                RelPositionMultiHeadedAttention rel => rel.Forward(x, x, x, mask, posEmb, attCache),
                MultiHeadedAttention att => att.Forward(x, x, x, mask, posEmb, attCache),
                _ => ((dynamic)self_attn).forward((x, x, x, mask, posEmb, attCache))
            };
        }

        public (Tensor x, Tensor mask, Tensor? cnnCache) ForwardWithPreallocatedAttention(
            Tensor x,
            Tensor mask,
            Tensor posEmb,
            Tensor maskPad,
            PreallocatedAttentionCache attCache,
            Tensor? cnnCache,
            long maxCacheLength)
        {
            if (feed_forward_macaron != null)
            {
                var residual = x;
                if (NormalizeBefore) x = norm_ff_macaron.forward(x);
                x = residual + FFScale * dropout.forward(((dynamic)feed_forward_macaron).forward(x));
                if (!NormalizeBefore) x = norm_ff_macaron.forward(x);
            }

            var mhaResidual = x;
            if (NormalizeBefore) x = norm_mha.forward(x);
            var mhaRes = CallSelfAttentionPreallocated(x, mask, posEmb, attCache, maxCacheLength);
            x = mhaResidual + dropout.forward(mhaRes.output);
            if (!NormalizeBefore) x = norm_mha.forward(x);

            Tensor? newCnnCache = null;
            if (conv_module != null)
            {
                var convResidual = x;
                if (NormalizeBefore) x = norm_conv.forward(x);
                var convRes = ((dynamic)conv_module).forward((x, maskPad, cnnCache));
                x = convRes.Item1;
                newCnnCache = convRes.Item2;
                x = convResidual + dropout.forward(x);
                if (!NormalizeBefore) x = norm_conv.forward(x);
            }

            var ffResidual = x;
            if (NormalizeBefore) x = norm_ff.forward(x);
            x = ffResidual + FFScale * dropout.forward(((dynamic)feed_forward).forward(x));
            if (!NormalizeBefore) x = norm_ff.forward(x);

            if (conv_module != null)
            {
                x = norm_final.forward(x);
            }

            return (x, mask, newCnnCache);
        }

        private (Tensor output, long nextLength) CallSelfAttentionPreallocated(
            Tensor x,
            Tensor mask,
            Tensor posEmb,
            PreallocatedAttentionCache attCache,
            long maxCacheLength)
        {
            return self_attn switch
            {
                RelPositionMultiHeadedAttention rel => rel.ForwardWithPreallocatedCache(x, x, x, mask, posEmb, attCache, maxCacheLength),
                MultiHeadedAttention att => att.ForwardWithPreallocatedCache(x, x, x, mask, posEmb, attCache, maxCacheLength),
                _ => throw new NotSupportedException($"Preallocated legacy transformer cache is not supported for attention module {self_attn.GetType().Name}.")
            };
        }
    }
}
