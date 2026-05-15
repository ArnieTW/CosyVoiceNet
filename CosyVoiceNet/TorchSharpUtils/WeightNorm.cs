// Equivalent Python: torch.nn.utils.parametrizations.weight_norm
using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace CosyVoiceNet.TorchSharpUtils
{
    // Innermost parametrization: produces keys "original0" and "original1".
    // original0 = g vector [out_channels, 1, 1]
    // original1 = v tensor [out_channels, in_channels/groups, kernel]
    public class WeightParametrization : nn.Module
    {
        public readonly Parameter original0;
        public readonly Parameter original1;

        public WeightParametrization(long outCh, long inCh, long kernel) : base("weight")
        {
            original0 = nn.Parameter(ones(outCh, 1, 1));
            original1 = nn.Parameter(zeros(outCh, inCh, kernel));
            RegisterComponents();
        }

        public Tensor Compute()
        {
            var vNorm = original1.mul(original1).sum(new long[] { 1, 2 }, keepdim: true).sqrt().clamp(min: 1e-12);
            return original0 * (original1 / vNorm);
        }
    }

    // Parametrizations container: produces submodule key "parametrizations".
    public class WeightNormParametrizations : nn.Module
    {
        public readonly WeightParametrization weight;

        public WeightNormParametrizations(long outCh, long inCh, long kernel) : base("parametrizations")
        {
            weight = new WeightParametrization(outCh, inCh, kernel);
            RegisterComponents();
        }
    }

    // Weight-normed regular Conv1d.
    // State-dict keys: bias | parametrizations.weight.original0/1
    public class WeightNormedConv1d : nn.Module<Tensor, Tensor>
    {
        public readonly Parameter bias;
        public readonly WeightNormParametrizations parametrizations;
        private readonly long _stride, _padding, _dilation, _groups;

        public WeightNormedConv1d(long inCh, long outCh, long kernel, long stride = 1, long padding = 0, long dilation = 1, long groups = 1)
            : base("WeightNormedConv1d")
        {
            _stride = stride; _padding = padding; _dilation = dilation; _groups = groups;
            bias = nn.Parameter(zeros(outCh));
            parametrizations = new WeightNormParametrizations(outCh, inCh / groups, kernel);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
            => nn.functional.conv1d(x, parametrizations.weight.Compute(), bias, _stride, _padding, _dilation, _groups);
    }

    public class WeightNormedConvTranspose1d : nn.Module<Tensor, Tensor>
    {
        public readonly Parameter bias;
        public readonly WeightNormParametrizations parametrizations;
        private readonly long _stride, _padding, _outputPadding, _groups, _dilation;

        public WeightNormedConvTranspose1d(long inCh, long outCh, long kernel, long stride = 1, long padding = 0, long outputPadding = 0, long groups = 1, long dilation = 1)
            : base("WeightNormedConvTranspose1d")
        {
            _stride = stride; _padding = padding; _outputPadding = outputPadding; _groups = groups; _dilation = dilation;
            bias = nn.Parameter(zeros(outCh));
            parametrizations = new WeightNormParametrizations(inCh, outCh / groups, kernel);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
            => nn.functional.conv_transpose1d(x, parametrizations.weight.Compute(), bias, _stride, _padding, _outputPadding, _groups, _dilation);
    }

    // Weight-normed CausalConv1d.
    // State-dict keys: bias | parametrizations.weight.original0/1
    public class WeightNormedCausalConv1d : nn.Module<Tensor, Tensor>
    {
        public readonly Parameter bias;
        public readonly WeightNormParametrizations parametrizations;
        public readonly int CausalPadding;
        private readonly long _dilation, _groups;
        private readonly string _causalType;

        public WeightNormedCausalConv1d(long inCh, long outCh, long kernel, long dilation = 1, long groups = 1, string causalType = "left")
            : base("WeightNormedCausalConv1d")
        {
            _dilation = dilation; _groups = groups; _causalType = causalType;
            CausalPadding = (int)((kernel * dilation - dilation) / 2) * 2 + (int)((kernel + 1) % 2);
            bias = nn.Parameter(zeros(outCh));
            parametrizations = new WeightNormParametrizations(outCh, inCh / groups, kernel);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var w = parametrizations.weight.Compute();
            Tensor padded;
            if (CausalPadding == 0)
            {
                padded = x;
            }
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
            return nn.functional.conv1d(padded, w, bias, 1L, 0L, _dilation, _groups);
        }

        public Tensor forward(Tensor x, Tensor cache)
        {
            var w = parametrizations.weight.Compute();
            Tensor padded;
            if (_causalType == "left")
                padded = cat(new[] { cache, x }, dim: 2);
            else
                padded = cat(new[] { x, cache }, dim: 2);
            return nn.functional.conv1d(padded, w, bias, 1L, 0L, _dilation, _groups);
        }
    }

    // Weight-normed CausalConv1dUpsample (nearest upsample + causal conv).
    // State-dict keys: bias | parametrizations.weight.original0/1
    public class WeightNormedCausalConv1dUpsample : nn.Module<Tensor, Tensor>
    {
        public readonly Parameter bias;
        public readonly WeightNormParametrizations parametrizations;
        private readonly long _upsampleScale;
        private readonly int _causalPadding;

        public WeightNormedCausalConv1dUpsample(long inCh, long outCh, long kernel, long stride)
            : base("WeightNormedCausalConv1dUpsample")
        {
            _upsampleScale = stride;
            _causalPadding = (int)(kernel - 1);
            bias = nn.Parameter(zeros(outCh));
            parametrizations = new WeightNormParametrizations(outCh, inCh, kernel);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = nn.functional.interpolate(x, scale_factor: new double[] { _upsampleScale }, mode: InterpolationMode.Nearest);
            var w = parametrizations.weight.Compute();
            Tensor padded;
            if (_causalPadding == 0)
                padded = x;
            else
            {
                var pad = zeros(x.shape[0], x.shape[1], _causalPadding, dtype: x.dtype, device: x.device);
                padded = cat(new[] { pad, x }, dim: 2);
            }
            return nn.functional.conv1d(padded, w, bias, 1L, 0L);
        }
    }

    // Legacy static helper — kept for any remaining callers.
    public static class WeightNorm
    {
        public static WeightNormedConv1d Apply(long inCh, long outCh, long kernel, long padding = 0, long dilation = 1)
            => new WeightNormedConv1d(inCh, outCh, kernel, padding: padding, dilation: dilation);
    }
}
