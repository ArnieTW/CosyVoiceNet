using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    // Equivalent Python file: cosyvoice/transformers/convolution.py
    public class ConvolutionModule : torch.nn.Module<(Tensor, Tensor?, Tensor?), (Tensor, Tensor)>
    {
        private readonly int channels;
        private readonly int kernelSize;
        private readonly bool useLayerNorm;
        private readonly int lorder;
        public dynamic pointwiseConv1;
        public dynamic depthwiseConv;
        public dynamic norm;
        public dynamic pointwiseConv2;
        public Func<Tensor, Tensor> activation;

        public ConvolutionModule(int channels, int kernelSize = 15, Func<Tensor, Tensor>? activationFn = null, string normType = "batch_norm", bool causal = false, bool bias = true) : base("ConvolutionModule")
        {
            this.channels = channels;
            this.kernelSize = kernelSize;
            this.activation = activationFn ?? (x => torch.nn.functional.relu(x));
            this.lorder = causal ? kernelSize - 1 : 0;
            this.useLayerNorm = normType != "batch_norm";

            // Initialize convolution layers
            pointwiseConv1 = torch.nn.Conv1d(
                in_channels: channels,
                out_channels: 2 * channels,
                kernel_size: 1,
                stride: 1,
                padding: (long)0, // Explicitly cast padding to long
                dilation: 1,
                groups: 1,
                bias: bias
            );      

            int samePadding = (kernelSize - 1) / 2; // Calculate padding for 'same' padding

            depthwiseConv = torch.nn.Conv1d(
                in_channels: channels,
                out_channels: channels,
                kernel_size: kernelSize,
                stride: 1,
                padding: (long)(causal ? 0 : samePadding), // Explicitly cast padding to long
                dilation: 1,
                groups: channels,
                bias: bias
            );

            pointwiseConv2 = torch.nn.Conv1d(
                in_channels: channels,
                out_channels: channels,
                kernel_size: 1,
                stride: 1,
                padding: (long)0, // Explicitly cast padding to long
                dilation: 1,
                groups: 1,
                bias: bias
            );

            // Initialize normalization layer
            if (normType == "batch_norm")
            {
                norm = torch.nn.BatchNorm1d(channels);
            }
            else if (normType == "layer_norm")
            {
                norm = torch.nn.LayerNorm(channels);
            }
            else
            {
                throw new ArgumentException("Unsupported normalization type: " + normType);
            }

            RegisterComponents();
        }

        public override (Tensor, Tensor) forward((Tensor, Tensor?, Tensor?) inputs)
        {
            var (x, maskPad, cache) = inputs;

            x = x.transpose(1, 2);

            if (maskPad is not null && maskPad.shape[2] > 0)
            {
                x = x.masked_fill(maskPad.eq(false), 0.0);
            }

            Tensor newCache;
            if (lorder > 0)
            {
                if (cache is null || cache.shape[2] == 0)
                {
                    x = torch.nn.functional.pad(x, new long[] { lorder, 0 }, 0.0);
                }
                else
                {
                    x = torch.cat(new Tensor[] { cache, x }, 2);
                }
                newCache = x.narrow(2, x.shape[2] - lorder, lorder);
            }
            else
            {
                newCache = torch.zeros(new long[] { 0, 0, 0 }, dtype: x.dtype);
            }

            // Apply pointwise convolution 1 and GLU
            x = pointwiseConv1.forward(x);
            x = torch.nn.functional.glu(x, 1);

            // Apply depthwise convolution
            x = depthwiseConv.forward(x);

            // Apply normalization and activation
            if (useLayerNorm)
            {
                x = x.transpose(1, 2);
                x = activation(norm.forward(x));
                x = x.transpose(1, 2);
            }
            else
            {
                x = activation(norm.forward(x));
            }

            // Apply pointwise convolution 2
            x = pointwiseConv2.forward(x);

            if (maskPad is not null && maskPad.shape[2] > 0)
            {
                x = x.masked_fill(maskPad.eq(false), 0.0);
            }

            return (x.transpose(1, 2), newCache);
        }
    }

    // NOTE: CausalConv1d, CausalConv1dDownSample used by the HiFiGAN vocoder path.
    // Equivalent Python: cosyvoice/transformer/convolution.py CausalConv1d / CausalConv1dDownSample

    // Non-weight-normed CausalConv1d (e.g. source_downs that have plain .weight/.bias keys).
    public class CausalConv1d : nn.Module<Tensor, Tensor>
    {
        public readonly TorchSharp.Modules.Parameter weight;
        public readonly TorchSharp.Modules.Parameter bias;
        public readonly int CausalPadding;

        private readonly long _dilation, _groups;
        private readonly string _causalType;

        public CausalConv1d(long inCh, long outCh, long kernel, long stride = 1, long dilation = 1, long groups = 1, string causalType = "left")
            : base("CausalConv1d")
        {
            _dilation = dilation; _groups = groups; _causalType = causalType;
            CausalPadding = (int)((kernel * dilation - dilation) / 2) * 2 + (int)((kernel + 1) % 2);
            weight = nn.Parameter(zeros(outCh, inCh / groups, kernel));
            bias = nn.Parameter(zeros(outCh));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Tensor padded;
            if (CausalPadding == 0)
                padded = x;
            else if (_causalType == "left")
            {
                var pad = zeros(x.shape[0], x.shape[1], CausalPadding, dtype: x.dtype, device: x.device);
                padded = cat(new[] { pad, x }, dim: 2);
            }
            else
            {
                var pad = zeros(x.shape[0], x.shape[1], CausalPadding, dtype: x.dtype, device: x.device);
                padded = cat(new[] { x, pad }, dim: 2);
            }
            return nn.functional.conv1d(padded, weight, bias, 1L, 0L, _dilation, _groups);
        }

        public Tensor forward(Tensor x, Tensor cache)
        {
            Tensor padded;
            if (_causalType == "left")
                padded = cat(new[] { cache, x }, dim: 2);
            else
                padded = cat(new[] { x, cache }, dim: 2);
            return nn.functional.conv1d(padded, weight, bias, 1L, 0L, _dilation, _groups);
        }
    }

    // Non-weight-normed CausalConv1dDownSample (stride > 1 downsampler).
    public class CausalConv1dDownSample : nn.Module<Tensor, Tensor>
    {
        public readonly TorchSharp.Modules.Parameter weight;
        public readonly TorchSharp.Modules.Parameter bias;

        private readonly long _stride;
        private readonly int _causalPadding;

        public CausalConv1dDownSample(long inCh, long outCh, long kernel, long stride)
            : base("CausalConv1dDownSample")
        {
            _stride = stride;
            _causalPadding = (int)(stride - 1);
            weight = nn.Parameter(zeros(outCh, inCh, kernel));
            bias = nn.Parameter(zeros(outCh));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Tensor padded;
            if (_causalPadding == 0)
                padded = x;
            else
            {
                var pad = zeros(x.shape[0], x.shape[1], _causalPadding, dtype: x.dtype, device: x.device);
                padded = cat(new[] { pad, x }, dim: 2);
            }
            return nn.functional.conv1d(padded, weight, bias, _stride, 0L);
        }
    }
}
