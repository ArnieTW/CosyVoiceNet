// Equivalent Python file: cosyvoice/transformer/upsample_encoder.py
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using CosyVoiceNet.Utils;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public class Upsample1D : nn.Module<(Tensor, Tensor), (Tensor, Tensor)>
    {
        private readonly int stride;
        public readonly nn.Module<Tensor, Tensor> conv;

        public Upsample1D(int channels, int outChannels, int stride = 2) : base("Upsample1D")
        {
            this.stride = stride;
            conv = nn.Conv1d(channels, outChannels, stride * 2 + 1, stride: 1L, padding: (long)0);
            RegisterComponents();
        }

        public override (Tensor, Tensor) forward((Tensor, Tensor) inputs)
        {
            var (input, inputLengths) = inputs;
            var outputs = nn.functional.interpolate(input, scale_factor: new double[] { stride }, mode: InterpolationMode.Nearest);
            outputs = nn.functional.pad(outputs, (stride * 2, 0));
            outputs = conv.forward(outputs);
            return (outputs, inputLengths * stride);
        }
    }

    // Equivalent Python: cosyvoice/transformer/upsample_encoder.py PreLookaheadLayer
    public class PreLookaheadLayer : nn.Module
    {
        public readonly nn.Module<Tensor, Tensor> conv1;
        public readonly nn.Module<Tensor, Tensor> conv2;
        public readonly int pre_lookahead_len;
        private readonly int conv2_kernel_size;

        public PreLookaheadLayer(int inChannels, int channels, int preLookaheadLen = 1) : base("PreLookaheadLayer")
        {
            pre_lookahead_len = preLookaheadLen;
            conv2_kernel_size = 3;
            conv1 = nn.Conv1d(inChannels, channels, preLookaheadLen + 1, stride: 1L, padding: (long)0);
            conv2 = nn.Conv1d(channels, inChannels, conv2_kernel_size, stride: 1L, padding: (long)0);
            RegisterComponents();
        }

        // inputs/context: (batch, seq, channels)
        public Tensor Forward(Tensor inputs, Tensor context = null)
        {
            var outputs = inputs.transpose(1, 2).contiguous();      // (B, C, T)
            if (context is null || context.shape[1] == 0)
            {
                outputs = nn.functional.pad(outputs, (0, pre_lookahead_len));
            }
            else
            {
                var ctx = context.transpose(1, 2).contiguous();      // (B, C, pre_lookahead_len)
                outputs = torch.cat(new[] { outputs, ctx }, dim: 2);
                int remaining = pre_lookahead_len - (int)ctx.shape[2];
                if (remaining > 0) outputs = nn.functional.pad(outputs, (0, remaining));
            }
            int requiredLen = pre_lookahead_len + 1;
            if (outputs.shape[2] < requiredLen)
            {
                int padAmount = requiredLen - (int)outputs.shape[2];
                outputs = nn.functional.pad(outputs, (0, padAmount));
            }
            outputs = nn.functional.leaky_relu(conv1.forward(outputs));
            outputs = nn.functional.pad(outputs, (conv2_kernel_size - 1, 0));
            outputs = conv2.forward(outputs);
            outputs = outputs.transpose(1, 2).contiguous();          // (B, T, C)
            return outputs + inputs;                                  // residual
        }
    }

    public class UpsampleConformerEncoder : nn.Module
    {
        private readonly int outputSize;
        private readonly bool normalizeBefore;
        private readonly int staticChunkSize;
        public readonly EncoderEmbedding embed;
        public readonly nn.Module<Tensor, Tensor> after_norm;
        public readonly PreLookaheadLayer pre_lookahead_layer;
        public readonly ModuleList<nn.Module> encoders;
        public readonly Upsample1D up_layer;
        public readonly EncoderEmbedding up_embed;
        public readonly ModuleList<nn.Module> up_encoders;

        public UpsampleConformerEncoder(
            int inputSize,
            int outputSize = 256,
            int attentionHeads = 4,
            int linearUnits = 2048,
            int numBlocks = 6,
            double dropoutRate = 0.1,
            double positionalDropoutRate = 0.1,
            double attentionDropoutRate = 0.0,
            string inputLayer = "conv2d",
            string posEncLayerType = "rel_pos",
            bool normalizeBefore = true,
            int staticChunkSize = 0,
            bool useDynamicChunk = false,
            GlobalCMVN globalCmvn = null,
            bool useDynamicLeftChunk = false,
            int positionwiseConvKernelSize = 1,
            bool macaronStyle = true,
            string selfAttentionLayerType = "rel_selfattn",
            string activationType = "swish",
            bool useCnnModule = true,
            int cnnModuleKernel = 15,
            bool causal = false,
            string cnnModuleNorm = "batch_norm",
            bool keyBias = true,
            bool gradientCheckpointing = false)
            : base("UpsampleConformerEncoder")
        {
            this.outputSize = outputSize;
            this.normalizeBefore = normalizeBefore;
            this.staticChunkSize = staticChunkSize;
            this.embed = new EncoderEmbedding(inputSize, outputSize, dropoutRate, inputLayer, posEncLayerType, positionalDropoutRate);
            this.after_norm = torch.nn.LayerNorm(outputSize, eps: 1e-5);
            this.pre_lookahead_layer = new PreLookaheadLayer(outputSize, outputSize, 3);
            this.encoders = new ModuleList<nn.Module>();
            for (int i = 0; i < numBlocks; i++)
            {
                encoders.append(CreateEncoderLayer(outputSize, attentionHeads, linearUnits, dropoutRate, attentionDropoutRate,
                    normalizeBefore, macaronStyle, selfAttentionLayerType, activationType, useCnnModule,
                    cnnModuleKernel, causal, cnnModuleNorm, keyBias));
            }

            this.up_layer = new Upsample1D(outputSize, outputSize, 2);
            this.up_embed = new EncoderEmbedding(inputSize, outputSize, dropoutRate, inputLayer, posEncLayerType, positionalDropoutRate);
            this.up_encoders = new ModuleList<nn.Module>();
            for (int i = 0; i < 4; i++)
            {
                up_encoders.append(CreateEncoderLayer(outputSize, attentionHeads, linearUnits, dropoutRate, attentionDropoutRate,
                    normalizeBefore, macaronStyle, selfAttentionLayerType, activationType, useCnnModule,
                    cnnModuleKernel, causal, cnnModuleNorm, keyBias));
            }
            RegisterComponents();
        }

        public int output_size() => outputSize;

        public (Tensor, Tensor) Forward(Tensor xs, Tensor xsLens, Tensor context = null, int decodingChunkSize = 0, int numDecodingLeftChunks = -1, bool streaming = false)
        {
            context ??= torch.zeros(new long[] { 0, 0, 0 }, device: xs.device, dtype: xs.dtype);
            var t = (int)xs.shape[1];
            var masks = ~Mask.MakePadMask(xsLens, t).unsqueeze(1);
            var embedRes = embed.Forward(xs, masks);
            xs = embedRes.Item1;
            var posEmb = embedRes.Item2;
            masks = embedRes.Item3;

            if (context.shape[1] != 0)
            {
                var contextMasks = torch.ones(new long[] { 1, 1, context.shape[1] }, device: xs.device, dtype: ScalarType.Bool);
                context = embed.Forward(context, contextMasks, xs.shape[1]).Item1;
            }

            var maskPad = masks;
            var chunkMasks = Mask.AddOptionalChunkMask(xs, masks, false, false, 0, streaming ? staticChunkSize : 0, -1);
            xs = pre_lookahead_layer.Forward(xs, context);
            foreach (var encoder in encoders)
            {
                xs = ((dynamic)encoder).forward((xs, chunkMasks, posEmb, maskPad, (Tensor?)null, (Tensor?)null)).Item1;
            }

            xs = xs.transpose(1, 2).contiguous();
            var upResult = up_layer.forward((xs, xsLens));
            xs = upResult.Item1.transpose(1, 2).contiguous();
            xsLens = upResult.Item2;

            t = (int)xs.shape[1];
            masks = ~Mask.MakePadMask(xsLens, t).unsqueeze(1);
            embedRes = up_embed.Forward(xs, masks);
            xs = embedRes.Item1;
            posEmb = embedRes.Item2;
            masks = embedRes.Item3;

            maskPad = masks;
            chunkMasks = Mask.AddOptionalChunkMask(xs, masks, false, false, 0, streaming ? staticChunkSize * 2 : 0, -1);
            foreach (var upEncoder in up_encoders)
            {
                xs = ((dynamic)upEncoder).forward((xs, chunkMasks, posEmb, maskPad, (Tensor?)null, (Tensor?)null)).Item1;
            }

            if (normalizeBefore)
                xs = after_norm.forward(xs);
            return (xs, masks);
        }

        private static ConformerEncoderLayer CreateEncoderLayer(
            int outputSize,
            int attentionHeads,
            int linearUnits,
            double dropoutRate,
            double attentionDropoutRate,
            bool normalizeBefore,
            bool macaronStyle,
            string selfAttentionLayerType,
            string activationType,
            bool useCnnModule,
            int cnnModuleKernel,
            bool causal,
            string cnnModuleNorm,
            bool keyBias)
        {
            nn.Module selfAttn = string.Equals(selfAttentionLayerType, "rel_selfattn", StringComparison.OrdinalIgnoreCase)
                ? new RelPositionMultiHeadedAttention(attentionHeads, outputSize, attentionDropoutRate, keyBias)
                : new MultiHeadedAttention(attentionHeads, outputSize, attentionDropoutRate, keyBias);
            Func<Tensor, Tensor> activation = string.Equals(activationType, "swish", StringComparison.OrdinalIgnoreCase)
                ? Activation.Swish
                : x => torch.nn.functional.relu(x);
            var feedForward = new PositionwiseFeedForward(outputSize, linearUnits, dropoutRate, activation);
            nn.Module feedForwardMacaron = macaronStyle ? new PositionwiseFeedForward(outputSize, linearUnits, dropoutRate, activation) : null;
            nn.Module convModule = useCnnModule ? new ConvolutionModule(outputSize, cnnModuleKernel, activation, cnnModuleNorm, causal) : null;
            return new ConformerEncoderLayer(outputSize, selfAttn, feedForward, feedForwardMacaron, convModule, dropoutRate, normalizeBefore);
        }
    }
}
