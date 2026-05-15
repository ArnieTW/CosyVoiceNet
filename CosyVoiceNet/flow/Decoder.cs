// Exported from CosyVoice\cosyvoice\flow\decoder.py
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using CosyVoiceNet.Matcha;
using CosyVoiceNet.Utils;
using static TorchSharp.torch;

namespace CosyVoiceNet.flow
{
    public class Transpose : nn.Module<Tensor, Tensor>
    {
        private readonly int dim0;
        private readonly int dim1;

        public Transpose(int dim0, int dim1) : base("Transpose")
        {
            this.dim0 = dim0;
            this.dim1 = dim1;
        }

        public override Tensor forward(Tensor x) => x.transpose(dim0, dim1);
    }

    public class CausalConv1d : nn.Module<Tensor, Tensor>
    {
        public readonly Parameter weight;
        public readonly Parameter bias;
        private readonly int causalPadding;
        private readonly long dilation;
        private readonly long groups;

        public CausalConv1d(int inChannels, int outChannels, int kernelSize, int stride = 1, int dilation = 1, int groups = 1, bool hasBias = true) : base("CausalConv1d")
        {
            if (stride != 1)
                throw new ArgumentException("Stride must be 1 for causal convolution.");

            causalPadding = (kernelSize - 1) * dilation;
            this.dilation = dilation;
            this.groups = groups;
            weight = nn.Parameter(torch.empty(new long[] { outChannels, inChannels / groups, kernelSize }));
            bias = hasBias ? nn.Parameter(torch.empty(outChannels)) : null;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = nn.functional.pad(x, new long[] { causalPadding, 0 });
            return nn.functional.conv1d(x, weight, bias, 1L, 0L, dilation, groups);
        }
    }

    public class CausalBlock1D : Block1D
    {
        public CausalBlock1D(int dim, int dimOut, int groups = 8) : base("CausalBlock1D")
        {
            block = nn.Sequential(
                new CausalConv1d(dim, dimOut, 3),
                new Transpose(1, 2),
                nn.LayerNorm(dimOut),
                new Transpose(1, 2),
                nn.Mish());
            RegisterComponents();
        }

    }

    public class CausalResnetBlock1D : ResnetBlock1D
    {
        public CausalResnetBlock1D(int dim, int dimOut, int timeEmbDim, int groups = 8)
            : base("CausalResnetBlock1D")
        {
            this.timeEmbDim = timeEmbDim;
            mlp = nn.Sequential(nn.Mish(), nn.Linear(timeEmbDim, dimOut));
            block1 = new CausalBlock1D(dim, dimOut, groups);
            block2 = new CausalBlock1D(dimOut, dimOut, groups);
            res_conv = nn.Conv1d(dim, dimOut, 1);
            RegisterComponents();
        }
    }

    public class ConditionalDecoder : nn.Module<Tensor, Tensor>
    {
        public readonly int in_channels;
        public readonly int out_channels;
        public readonly SinusoidalPosEmb time_embeddings;
        public readonly TimestepEmbedding time_mlp;
        public readonly ModuleList<nn.Module> down_blocks;
        public readonly ModuleList<nn.Module> mid_blocks;
        public readonly ModuleList<nn.Module> up_blocks;
        public readonly Block1D final_block;
        public readonly nn.Module<Tensor, Tensor> final_proj;

        protected readonly int[] channels;
        protected readonly int attentionHeadDim;
        protected readonly int nBlocks;
        protected readonly int numHeads;
        protected readonly float dropout;
        protected readonly string actFn;

        private readonly record struct AttentionMaskCacheKey(
            string Device,
            ScalarType DType,
            long InputLength,
            IntPtr MaskHandle,
            bool Streaming);

        public ConditionalDecoder(
            int inChannels,
            int outChannels,
            int[] channels = null,
            float dropout = 0.05f,
            int attentionHeadDim = 64,
            int nBlocks = 1,
            int numMidBlocks = 2,
            int numHeads = 4,
            string actFn = "snake") : base("ConditionalDecoder")
        {
            this.in_channels = inChannels;
            this.out_channels = outChannels;
            this.channels = channels ?? new[] { 256, 256 };
            this.attentionHeadDim = attentionHeadDim;
            this.nBlocks = nBlocks;
            this.numHeads = numHeads;
            this.dropout = dropout;
            this.actFn = actFn;

            time_embeddings = new SinusoidalPosEmb(inChannels);
            var timeEmbedDim = this.channels[0] * 4;
            time_mlp = new TimestepEmbedding(inChannels, timeEmbedDim, "silu");
            down_blocks = new ModuleList<nn.Module>();
            mid_blocks = new ModuleList<nn.Module>();
            up_blocks = new ModuleList<nn.Module>();

            var outputChannel = inChannels;
            for (var i = 0; i < this.channels.Length; i++)
            {
                var inputChannel = outputChannel;
                outputChannel = this.channels[i];
                var isLast = i == this.channels.Length - 1;
                var resnet = CreateResnet(inputChannel, outputChannel, timeEmbedDim);
                var transformerBlocks = CreateTransformerBlocks(outputChannel);
                nn.Module downsample = isLast
                    ? CreateLastConv(outputChannel)
                    : new Downsample1D(outputChannel);
                down_blocks.append(new ModuleList<nn.Module>(new nn.Module[] { resnet, transformerBlocks, downsample }));
            }

            for (var i = 0; i < numMidBlocks; i++)
            {
                var inputChannel = this.channels[^1];
                outputChannel = this.channels[^1];
                var resnet = CreateResnet(inputChannel, outputChannel, timeEmbedDim);
                var transformerBlocks = CreateTransformerBlocks(outputChannel);
                mid_blocks.append(new ModuleList<nn.Module>(new nn.Module[] { resnet, transformerBlocks }));
            }

            var reversed = new int[this.channels.Length + 1];
            for (var i = 0; i < this.channels.Length; i++)
                reversed[i] = this.channels[this.channels.Length - 1 - i];
            reversed[^1] = this.channels[0];

            for (var i = 0; i < reversed.Length - 1; i++)
            {
                var inputChannel = reversed[i] * 2;
                outputChannel = reversed[i + 1];
                var isLast = i == reversed.Length - 2;
                var resnet = CreateResnet(inputChannel, outputChannel, timeEmbedDim);
                var transformerBlocks = CreateTransformerBlocks(outputChannel);
                nn.Module upsample = isLast
                    ? CreateLastConv(outputChannel)
                    : new Upsample1D(outputChannel, useConvTranspose: true);
                up_blocks.append(new ModuleList<nn.Module>(new nn.Module[] { resnet, transformerBlocks, upsample }));
            }

            final_block = CreateFinalBlock(reversed[^1], reversed[^1]);
            final_proj = nn.Conv1d(reversed[^1], outChannels, 1);
            RegisterComponents();
        }

        protected virtual ResnetBlock1D CreateResnet(int inputChannel, int outputChannel, int timeEmbedDim)
            => new ResnetBlock1D(inputChannel, outputChannel, timeEmbedDim);

        protected virtual Block1D CreateFinalBlock(int inputChannel, int outputChannel)
            => new Block1D(inputChannel, outputChannel);

        protected virtual nn.Module CreateLastConv(int channels)
            => nn.Conv1d(channels, channels, 3, padding: 1);

        private ModuleList<nn.Module> CreateTransformerBlocks(int outputChannel)
        {
            var transformerBlocks = new ModuleList<nn.Module>();
            for (var i = 0; i < nBlocks; i++)
                transformerBlocks.append(new BasicTransformerBlock(outputChannel, numHeads, attentionHeadDim, dropout, actFn));
            return transformerBlocks;
        }

        public override Tensor forward(Tensor x)
            => throw new InvalidOperationException("Use Forward(x, mask, mu, t, spks, cond, streaming).");

        public virtual Tensor Forward(Tensor x, Tensor mask, Tensor mu, Tensor t, Tensor spks = null, Tensor cond = null, bool streaming = false)
        {
            t = time_embeddings.Forward(t).to(t.dtype);
            t = time_mlp.forward(t);

            x = PackChannel(x, mu);
            if (spks is not null)
            {
                spks = spks.unsqueeze(-1).expand(spks.shape[0], spks.shape[1], x.shape[^1]);
                x = PackChannel(x, spks);
            }
            if (cond is not null)
                x = PackChannel(x, cond);

            var hiddens = new List<Tensor>();
            var masks = new List<Tensor> { mask };
            var attentionMaskCache = new Dictionary<AttentionMaskCacheKey, Tensor>();

            var downIndex = 0;
            foreach (ModuleList<nn.Module> downBlock in down_blocks)
            {
                var maskDown = masks[^1];
                x = ((ResnetBlock1D)downBlock[0]).Forward(x, maskDown, t);
                x = x.transpose(1, 2).contiguous();
                var attnMask = GetAttentionMask(attentionMaskCache, x, maskDown, streaming);
                foreach (BasicTransformerBlock transformerBlock in (ModuleList<nn.Module>)downBlock[1])
                    x = transformerBlock.Forward(x, attnMask, t);
                x = x.transpose(1, 2).contiguous();
                hiddens.Add(x);
                x = ((dynamic)downBlock[2]).forward(x * maskDown);
                masks.Add(maskDown[.., .., TensorIndex.Slice(null, null, 2)]);
                downIndex++;
            }

            masks.RemoveAt(masks.Count - 1);
            var maskMid = masks[^1];
            var midIndex = 0;
            foreach (ModuleList<nn.Module> midBlock in mid_blocks)
            {
                x = ((ResnetBlock1D)midBlock[0]).Forward(x, maskMid, t);
                x = x.transpose(1, 2).contiguous();
                var attnMask = GetAttentionMask(attentionMaskCache, x, maskMid, streaming);
                foreach (BasicTransformerBlock transformerBlock in (ModuleList<nn.Module>)midBlock[1])
                    x = transformerBlock.Forward(x, attnMask, t);
                x = x.transpose(1, 2).contiguous();
                midIndex++;
            }

            Tensor maskUp = null;
            var upIndex = 0;
            foreach (ModuleList<nn.Module> upBlock in up_blocks)
            {
                maskUp = masks[^1];
                masks.RemoveAt(masks.Count - 1);
                var skip = hiddens[^1];
                hiddens.RemoveAt(hiddens.Count - 1);
                x = PackChannel(AlignTime(x, skip.shape[^1]), skip);
                x = ((ResnetBlock1D)upBlock[0]).Forward(x, maskUp, t);
                x = x.transpose(1, 2).contiguous();
                var attnMask = GetAttentionMask(attentionMaskCache, x, maskUp, streaming);
                foreach (BasicTransformerBlock transformerBlock in (ModuleList<nn.Module>)upBlock[1])
                    x = transformerBlock.Forward(x, attnMask, t);
                x = x.transpose(1, 2).contiguous();
                x = ((dynamic)upBlock[2]).forward(x * maskUp);
                upIndex++;
            }

            x = final_block.Forward(x, maskUp);
            var output = final_proj.forward(x * maskUp);
            return output * mask;
        }

        private Tensor GetAttentionMask(Dictionary<AttentionMaskCacheKey, Tensor> cache, Tensor x, Tensor mask, bool streaming)
        {
            var key = new AttentionMaskCacheKey(x.device.ToString(), x.dtype, x.shape[1], mask.Handle, streaming);
            if (!cache.TryGetValue(key, out var attnMask))
            {
                attnMask = BuildAttentionMask(x, mask, streaming);
                cache.Add(key, attnMask);
            }

            return attnMask;
        }

        protected virtual Tensor BuildAttentionMask(Tensor x, Tensor mask, bool streaming)
        {
            var attnMask = Mask.AddOptionalChunkMask(x, mask.to_type(ScalarType.Bool), false, false, 0, 0, -1)
                .repeat(new long[] { 1, x.shape[1], 1 });
            return Common.MaskToBias(attnMask, x.dtype);
        }

        private static Tensor PackChannel(Tensor left, Tensor right)
            => torch.cat(new[] { left, right }, dim: 1);

        private static Tensor AlignTime(Tensor value, long targetLength)
        {
            var currentLength = value.shape[^1];
            if (currentLength == targetLength)
                return value;
            if (currentLength > targetLength)
                return value[.., .., TensorIndex.Slice(0, targetLength)];

            var pad = torch.zeros(new long[] { value.shape[0], value.shape[1], targetLength - currentLength }, device: value.device, dtype: value.dtype);
            return torch.cat(new[] { value, pad }, dim: 2);
        }

    }

    public class CausalConditionalDecoder : ConditionalDecoder
    {
        private readonly int staticChunkSize;
        private readonly int numDecodingLeftChunks;

        public CausalConditionalDecoder(
            int inChannels,
            int outChannels,
            int[] channels = null,
            float dropout = 0.05f,
            int attentionHeadDim = 64,
            int nBlocks = 1,
            int numMidBlocks = 2,
            int numHeads = 4,
            string actFn = "snake",
            int staticChunkSize = 50,
            int numDecodingLeftChunks = 2)
            : base(inChannels, outChannels, channels, dropout, attentionHeadDim, nBlocks, numMidBlocks, numHeads, actFn)
        {
            this.staticChunkSize = staticChunkSize;
            this.numDecodingLeftChunks = numDecodingLeftChunks;
        }

        protected override ResnetBlock1D CreateResnet(int inputChannel, int outputChannel, int timeEmbedDim)
            => new CausalResnetBlock1D(inputChannel, outputChannel, timeEmbedDim);

        protected override Block1D CreateFinalBlock(int inputChannel, int outputChannel)
            => new CausalBlock1D(inputChannel, outputChannel);

        protected override nn.Module CreateLastConv(int channels)
            => new CausalConv1d(channels, channels, 3);

        protected override Tensor BuildAttentionMask(Tensor x, Tensor mask, bool streaming)
        {
            var attnMask = streaming
                ? Mask.AddOptionalChunkMask(x, mask.to_type(ScalarType.Bool), false, false, 0, staticChunkSize, -1)
                : Mask.AddOptionalChunkMask(x, mask.to_type(ScalarType.Bool), false, false, 0, 0, -1)
                    .repeat(new long[] { 1, x.shape[1], 1 });
            return Common.MaskToBias(attnMask, x.dtype);
        }
    }
}
