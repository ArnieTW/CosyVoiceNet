using System;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public class DecoderLayer : torch.nn.Module<(Tensor, Tensor, Tensor, Tensor, Tensor?), (Tensor, Tensor, Tensor, Tensor)>
    {
        public int Size { get; }
        public dynamic SelfAttn;
        public dynamic SrcAttn;
        public dynamic FeedForward;
        private readonly dynamic Norm1;
        private readonly dynamic Norm2;
        private readonly dynamic Norm3;
        private readonly dynamic Dropout;
        public readonly bool NormalizeBefore;

        public DecoderLayer(int size, dynamic selfAttn, dynamic srcAttn, dynamic feedForward, double dropoutRate, bool normalizeBefore = true) : base("DecoderLayer")
        {
            Size = size;
            SelfAttn = selfAttn;
            SrcAttn = srcAttn;
            FeedForward = feedForward;
            Norm1 = torch.nn.LayerNorm(size, eps: 1e-5);
            Norm2 = torch.nn.LayerNorm(size, eps: 1e-5);
            Norm3 = torch.nn.LayerNorm(size, eps: 1e-5);
            Dropout = torch.nn.Dropout(dropoutRate);
            NormalizeBefore = normalizeBefore;

            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor, Tensor) forward((Tensor, Tensor, Tensor, Tensor, Tensor?) inputs)
        {
            var (tgt, tgtMask, memory, memoryMask, cache) = inputs;

            var residual = tgt;
            if (NormalizeBefore)
            {
                tgt = Norm1.forward(tgt);
            }

            Tensor tgtQ;
            Tensor tgtQMask;
            if (cache is null)
            {
                tgtQ = tgt;
                tgtQMask = tgtMask;
            }
            else
            {
                // Compute only the last frame query
                var last = (int)tgt.shape[1] - 1;
                tgtQ = tgt.narrow(1, last, 1);
                residual = residual.narrow(1, last, 1);
                tgtQMask = tgtMask.narrow(1, last, 1);
            }

            var selfOut = SelfAttn.forward(tgtQ, tgt, tgt, tgtQMask);
            var x = residual + Dropout.forward(selfOut.Item1);
            if (!NormalizeBefore)
            {
                x = Norm1.forward(x);
            }

            if (SrcAttn != null)
            {
                residual = x;
                if (NormalizeBefore)
                {
                    x = Norm2.forward(x);
                }
                var srcOut = SrcAttn.forward(x, memory, memory, memoryMask);
                x = residual + Dropout.forward(srcOut.Item1);
                if (!NormalizeBefore)
                {
                    x = Norm2.forward(x);
                }
            }

            residual = x;
            if (NormalizeBefore)
            {
                x = Norm3.forward(x);
            }
            x = residual + Dropout.forward(FeedForward.forward(x));
            if (!NormalizeBefore)
            {
                x = Norm3.forward(x);
            }

            if (cache is not null)
            {
                x = torch.cat(new Tensor[] { cache, x }, 1);
            }

            return (x, tgtMask, memory, memoryMask);
        }
    }
}

// Equivalent Python file: cosyvoice/transformer/decoder_layer.py
