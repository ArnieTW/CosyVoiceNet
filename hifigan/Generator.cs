// Equivalent Python file: cosyvoice/hifigan/generator.py
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using CosyVoiceNet.Transformers;
using CosyVoiceNet.TorchSharpUtils;

namespace CosyVoiceNet.hifigan
{
    internal static class HiftDeterministicRandom
    {
        public static Tensor Uniform(long[] shape, ScalarType dtype, Device device, ulong stream)
        {
            var values = new float[CheckedElementCount(shape)];
            for (var i = 0; i < values.Length; i++)
                values[i] = ToUnitFloat(SplitMix64(stream + (ulong)i));

            var cpu = torch.tensor(values, dtype: ScalarType.Float32).view(shape);
            return cpu.to(device).to_type(dtype);
        }

        public static Tensor NormalLike(Tensor reference, ulong stream)
        {
            var shape = reference.shape;
            var values = new float[CheckedElementCount(shape)];
            for (var i = 0; i < values.Length; i += 2)
            {
                var u1 = Math.Max(ToUnitDouble(SplitMix64(stream + (ulong)i)), 1e-12);
                var u2 = ToUnitDouble(SplitMix64(stream + (ulong)i + 1));
                var radius = Math.Sqrt(-2.0 * Math.Log(u1));
                var angle = 2.0 * Math.PI * u2;
                values[i] = (float)(radius * Math.Cos(angle));
                if (i + 1 < values.Length)
                    values[i + 1] = (float)(radius * Math.Sin(angle));
            }

            var cpu = torch.tensor(values, dtype: ScalarType.Float32).view(shape);
            return cpu.to(reference.device).to_type(reference.dtype);
        }

        private static int CheckedElementCount(IReadOnlyList<long> shape)
        {
            long count = 1;
            foreach (var dimension in shape)
                count = checked(count * dimension);
            if (count > int.MaxValue)
                throw new InvalidOperationException($"Deterministic HiFT random tensor is too large: {count} elements.");
            return (int)count;
        }

        private static float ToUnitFloat(ulong value)
            => (float)ToUnitDouble(value);

        private static double ToUnitDouble(ulong value)
            => ((value >> 11) * (1.0 / (1UL << 53)));

        private static ulong SplitMix64(ulong value)
        {
            value += 0x9E3779B97F4A7C15UL;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
            return value ^ (value >> 31);
        }
    }

    public class ResBlock : nn.Module<Tensor, Tensor>
    {
        private readonly ModuleList<nn.Module> convs1;
        private readonly ModuleList<nn.Module> convs2;
        private readonly ModuleList<nn.Module> activations1;
        private readonly ModuleList<nn.Module> activations2;

        public ResBlock(int channels = 512, int kernel_size = 3, int[] dilations = null, bool causal = false) : base("ResBlock")
        {
            dilations ??= new[] { 1, 3, 5 };

            convs1 = nn.ModuleList<nn.Module>();
            convs2 = nn.ModuleList<nn.Module>();
            activations1 = nn.ModuleList<nn.Module>();
            activations2 = nn.ModuleList<nn.Module>();

            foreach (var dilation in dilations)
            {
                if (causal)
                {
                    convs1.append(new WeightNormedCausalConv1d(channels, channels, kernel_size, dilation: dilation, causalType: "left"));
                    convs2.append(new WeightNormedCausalConv1d(channels, channels, kernel_size, dilation: 1, causalType: "left"));
                }
                else
                {
                    convs1.append(new WeightNormedConv1d(channels, channels, kernel_size, padding: GetPadding(kernel_size, dilation), dilation: dilation));
                    convs2.append(new WeightNormedConv1d(channels, channels, kernel_size, padding: GetPadding(kernel_size, 1)));
                }
                activations1.append(new Activation.Snake(channels, alphaLogScale: false));
                activations2.append(new Activation.Snake(channels, alphaLogScale: false));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            for (int idx = 0; idx < convs1.Count; idx++)
            {
                var xt = ((nn.Module<Tensor, Tensor>)activations1[idx]).forward(x);
                xt = ((nn.Module<Tensor, Tensor>)convs1[idx]).forward(xt);
                xt = ((nn.Module<Tensor, Tensor>)activations2[idx]).forward(xt);
                xt = ((nn.Module<Tensor, Tensor>)convs2[idx]).forward(xt);
                x = xt + x;
            }
            return x;
        }

        private static int GetPadding(int kernelSize, int dilation)
            => (kernelSize - 1) / 2 * dilation;
    }

    public class SineGen : nn.Module<Tensor, (Tensor, Tensor, Tensor)>
    {
        public float sine_amp { get; private set; }

        private readonly int sampling_rate;
        private readonly float noise_std;
        private readonly int harmonic_num;
        private readonly int voiced_threshold;

        public SineGen(int samp_rate, int harmonic_num = 0, float sine_amp = 0.1f, float noise_std = 0.003f, int voiced_threshold = 0) : base("SineGen")
        {
            this.sampling_rate = samp_rate;
            this.harmonic_num = harmonic_num;
            this.sine_amp = sine_amp;
            this.noise_std = noise_std;
            this.voiced_threshold = voiced_threshold;
        }

        private Tensor _f02uv(Tensor f0)
            => f0.gt(voiced_threshold).to_type(ScalarType.Float32);

        public override (Tensor, Tensor, Tensor) forward(Tensor f0)
        {
            f0 = f0.transpose(1, 2);
            var F_mat = torch.zeros(new long[] { f0.shape[0], harmonic_num + 1, f0.shape[2] }, dtype: f0.dtype, device: f0.device);
            for (int i = 0; i <= harmonic_num; i++)
            {
                var slice = f0 * (i + 1) / sampling_rate;
                F_mat.select(1, i).copy_(slice.squeeze(1));
            }
            var theta_mat = 2 * Math.PI * F_mat.cumsum(-1).remainder(1);
            var phase_vec = HiftDeterministicRandom.Uniform(new long[] { f0.shape[0], harmonic_num + 1, 1 }, f0.dtype, f0.device, 0x51E63EA3UL) * 2 * Math.PI;
            phase_vec.select(1, 0).zero_();

            var sine_waves = sine_amp * torch.sin(theta_mat + phase_vec);
            var uv = _f02uv(f0);
            var noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3;
            var noise = noise_amp * HiftDeterministicRandom.NormalLike(sine_waves, 0x51E63EA4UL);
            sine_waves = sine_waves * uv + noise;
            return (sine_waves.transpose(1, 2), uv.transpose(1, 2), noise);
        }
    }

    public class SineGen2 : nn.Module<Tensor, (Tensor, Tensor, Tensor)>
    {
        public float sine_amp { get; private set; }

        private readonly int sampling_rate;
        private readonly float noise_std;
        private readonly int harmonic_num;
        private readonly int voiced_threshold;
        private readonly int upsample_scale;
        private readonly bool causal;
        private readonly Tensor rand_ini;
        private readonly Tensor sine_waves;

        public SineGen2(int samp_rate, int upsample_scale, int harmonic_num = 0, float sine_amp = 0.1f, float noise_std = 0.003f, int voiced_threshold = 0, bool causal = false) : base("SineGen2")
        {
            this.sampling_rate = samp_rate;
            this.harmonic_num = harmonic_num;
            this.sine_amp = sine_amp;
            this.noise_std = noise_std;
            this.voiced_threshold = voiced_threshold;
            this.upsample_scale = upsample_scale;
            this.causal = causal;
            if (causal)
            {
                rand_ini = torch.rand(new long[] { 1, harmonic_num + 1 });
                rand_ini.select(1, 0).zero_();
                sine_waves = torch.rand(new long[] { 1, 300 * 24000, harmonic_num + 1 });
            }
        }

        private Tensor _f02sine(Tensor f0_values)
        {
            var rad_values = (f0_values / sampling_rate).remainder(1);
            if (!training && causal)
            {
                rad_values.narrow(1, 0, 1).add_(rand_ini.to(rad_values.device).to(rad_values.dtype).unsqueeze(1));
            }
            else
            {
                var phaseIni = HiftDeterministicRandom.Uniform(new long[] { f0_values.shape[0], f0_values.shape[2] }, f0_values.dtype, f0_values.device, 0x51E63EA5UL);
                phaseIni.select(1, 0).zero_();
                rad_values.narrow(1, 0, 1).add_(phaseIni.unsqueeze(1));
            }

            rad_values = nn.functional.interpolate(rad_values.transpose(1, 2),
                scale_factor: new double[] { 1.0 / upsample_scale }, mode: InterpolationMode.Linear).transpose(1, 2);
            var phase = rad_values.cumsum(1) * 2 * Math.PI;
            phase = nn.functional.interpolate(phase.transpose(1, 2) * upsample_scale,
                scale_factor: new double[] { upsample_scale },
                mode: causal ? InterpolationMode.Nearest : InterpolationMode.Linear).transpose(1, 2);
            return torch.sin(phase);
        }

        public override (Tensor, Tensor, Tensor) forward(Tensor f0)
        {
            var fn = f0 * torch.arange(1, harmonic_num + 2, dtype: f0.dtype, device: f0.device).view(1, 1, -1);
            var sine_waves = _f02sine(fn) * sine_amp;
            var uv = f0.gt(voiced_threshold).to_type(ScalarType.Float32);
            var noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3;
            Tensor noise;
            if (!training && causal)
            {
                noise = noise_amp * this.sine_waves.narrow(1, 0, sine_waves.shape[1]).to(sine_waves.device).to(sine_waves.dtype);
            }
            else
            {
                noise = noise_amp * HiftDeterministicRandom.NormalLike(sine_waves, 0x51E63EA6UL);
            }
            sine_waves = sine_waves * uv + noise;
            return (sine_waves, uv, noise);
        }
    }

    public class SourceModuleHnNSF : nn.Module<Tensor, (Tensor, Tensor, Tensor)>
    {
        private readonly nn.Module<Tensor, (Tensor, Tensor, Tensor)> l_sin_gen;
        private readonly Linear l_linear;
        private readonly Tanh l_tanh;
        private readonly float _sineAmp;
        private readonly bool _causal;
        private readonly Tensor _uv;

        public SourceModuleHnNSF(int sampling_rate, int upsample_scale, int harmonic_num = 0, float sine_amp = 0.1f, float add_noise_std = 0.003f, int voiced_threshold = 0, string sinegen_type = "1", bool causal = false) : base("SourceModuleHnNSF")
        {
            _sineAmp = sine_amp;
            _causal = causal;
            l_sin_gen = sinegen_type == "1"
                ? (nn.Module<Tensor, (Tensor, Tensor, Tensor)>)new SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
                : new SineGen2(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold, causal);
            l_linear = nn.Linear(harmonic_num + 1, 1);
            l_tanh = nn.Tanh();
            if (causal)
                _uv = torch.rand(new long[] { 1, 300 * 24000, 1 });
            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor) forward(Tensor x)
        {
            var (sine_waves, uv, _) = l_sin_gen.forward(x);
            var sine_merge = l_tanh.forward(l_linear.forward(sine_waves));
            var noise = !training && _causal
                ? _uv.narrow(1, 0, uv.shape[1]).to(uv.device).to(uv.dtype) * _sineAmp / 3
                : HiftDeterministicRandom.NormalLike(uv, 0x51E63EA7UL) * _sineAmp / 3;
            return (sine_merge, noise, uv);
        }
    }

    public class HiFTGenerator : nn.Module<Tensor, Tensor>, IHiftInference
    {
        protected readonly nn.Module<Tensor, Tensor> conv_pre;
        protected readonly ModuleList<nn.Module> ups;
        protected readonly ModuleList<nn.Module> source_downs;
        protected readonly ModuleList<nn.Module> source_resblocks;
        protected readonly ModuleList<nn.Module> resblocks;
        protected readonly nn.Module<Tensor, Tensor> conv_post;
        public readonly SourceModuleHnNSF m_source;
        private readonly Tensor stft_window;
        private readonly int _nFft;
        private readonly int _hopLen;
        private readonly float _lreluSlope;
        private readonly float _audioLimit;
        private readonly int _numKernels;
        private readonly int _numUpsamples;
        private readonly int _f0UpsampleScale;
        public readonly nn.Module<Tensor, Tensor> f0_predictor;

        public HiFTGenerator(int in_channels = 80, int base_channels = 512, int nb_harmonics = 8, int sampling_rate = 22050,
            float nsf_alpha = 0.1f, float nsf_sigma = 0.003f, float nsf_voiced_threshold = 10f,
            int[] upsample_rates = null, int[] upsample_kernel_sizes = null,
            Dictionary<string, object> istft_params = null,
            int[] resblock_kernel_sizes = null, int[][] resblock_dilation_sizes = null,
            int[] source_resblock_kernel_sizes = null, int[][] source_resblock_dilation_sizes = null,
            float lrelu_slope = 0.1f, float audio_limit = 0.99f,
            nn.Module<Tensor, Tensor> f0_predictor = null) : base("HiFTGenerator")
        {
            upsample_rates ??= new[] { 8, 8 };
            upsample_kernel_sizes ??= new[] { 16, 16 };
            resblock_kernel_sizes ??= new[] { 3, 7, 11 };
            resblock_dilation_sizes ??= new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
            source_resblock_kernel_sizes ??= new[] { 7, 11 };
            source_resblock_dilation_sizes ??= new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

            int nFft = istft_params != null ? Convert.ToInt32(istft_params["n_fft"]) : 16;
            int hopLen = istft_params != null ? Convert.ToInt32(istft_params["hop_len"]) : 4;

            _nFft = nFft;
            _hopLen = hopLen;
            _lreluSlope = lrelu_slope;
            _audioLimit = audio_limit;
            _numKernels = resblock_kernel_sizes.Length;
            _numUpsamples = upsample_rates.Length;
            this.f0_predictor = f0_predictor;

            int upsampleScale = upsample_rates.Aggregate(1, (a, b) => a * b) * hopLen;
            _f0UpsampleScale = upsampleScale;
            m_source = new SourceModuleHnNSF(sampling_rate, upsampleScale, nb_harmonics, nsf_alpha, nsf_sigma,
                (int)nsf_voiced_threshold, sinegen_type: sampling_rate == 22050 ? "1" : "2", causal: false);

            conv_pre = new WeightNormedConv1d(in_channels, base_channels, 7, padding: 3);

            ups = nn.ModuleList<nn.Module>();
            for (int i = 0; i < upsample_rates.Length; i++)
            {
                int u = upsample_rates[i], k = upsample_kernel_sizes[i];
                int inCh = base_channels / (int)Math.Pow(2, i);
                int outCh = base_channels / (int)Math.Pow(2, i + 1);
                ups.append(new WeightNormedConvTranspose1d(inCh, outCh, k, stride: u, padding: (k - u) / 2));
            }

            int nSrc = nFft + 2;
            var downsample_rates = new int[upsample_rates.Length];
            downsample_rates[0] = 1;
            var rev = upsample_rates.Reverse().ToArray();
            for (int i = 0; i < upsample_rates.Length - 1; i++) downsample_rates[i + 1] = rev[i];
            var cumprod = new int[downsample_rates.Length];
            cumprod[0] = downsample_rates[0];
            for (int i = 1; i < downsample_rates.Length; i++) cumprod[i] = cumprod[i - 1] * downsample_rates[i];
            var cumRatesRev = cumprod.Reverse().ToArray();

            source_downs = nn.ModuleList<nn.Module>();
            source_resblocks = nn.ModuleList<nn.Module>();
            for (int i = 0; i < source_resblock_kernel_sizes.Length; i++)
            {
                int u = cumRatesRev[i];
                int outCh = base_channels / (int)Math.Pow(2, i + 1);
                source_downs.append(u == 1
                    ? nn.Conv1d(nSrc, outCh, 1, stride: 1)
                    : nn.Conv1d(nSrc, outCh, u * 2, stride: u, padding: u / 2));
                source_resblocks.append(new ResBlock(outCh, source_resblock_kernel_sizes[i], source_resblock_dilation_sizes[i], causal: false));
            }

            resblocks = nn.ModuleList<nn.Module>();
            int lastCh = base_channels / (int)Math.Pow(2, upsample_rates.Length);
            for (int i = 0; i < upsample_rates.Length; i++)
            {
                int ch = base_channels / (int)Math.Pow(2, i + 1);
                for (int j = 0; j < resblock_kernel_sizes.Length; j++)
                    resblocks.append(new ResBlock(ch, resblock_kernel_sizes[j], resblock_dilation_sizes[j], causal: false));
                lastCh = ch;
            }

            conv_post = new WeightNormedConv1d(lastCh, nFft + 2, 7, padding: 3);
            stft_window = torch.hann_window(nFft, periodic: true);

            RegisterComponents();
        }

        protected (Tensor, Tensor) _stft(Tensor x)
        {
            var window = stft_window.to(x.device);
            var spec = torch.stft(x, _nFft, _hopLen, _nFft, window: window, return_complex: true);
            return (spec.real, spec.imag);
        }

        protected Tensor _istft(Tensor magnitude, Tensor phase)
        {
            magnitude = torch.clamp(magnitude, max: (Scalar)1e2);
            var real = magnitude * torch.cos(phase);
            var img = magnitude * torch.sin(phase);
            var window = stft_window.to(magnitude.device);
            return torch.istft(torch.complex(real, img), _nFft, _hopLen, _nFft, window: window);
        }

        public Tensor Decode(Tensor x, Tensor s)
        {
            var (s_stft_real, s_stft_imag) = _stft(s.squeeze(1));
            var s_stft = torch.cat(new[] { s_stft_real, s_stft_imag }, dim: 1);

            x = conv_pre.forward(x);
            for (int i = 0; i < _numUpsamples; i++)
            {
                x = nn.functional.leaky_relu(x, _lreluSlope);
                x = ((nn.Module<Tensor, Tensor>)ups[i]).forward(x);

                if (i == _numUpsamples - 1)
                {
                    var leftPad = x.narrow(2, 1, 1);
                    x = torch.cat(new[] { leftPad, x }, dim: 2);
                }

                var si = ((nn.Module<Tensor, Tensor>)source_downs[i]).forward(s_stft);
                si = ((ResBlock)source_resblocks[i]).forward(si);
                x = x + si;

                Tensor xs = null;
                for (int j = 0; j < _numKernels; j++)
                {
                    var rb = ((ResBlock)resblocks[i * _numKernels + j]).forward(x);
                    xs = xs is null ? rb : xs + rb;
                }
                x = xs / _numKernels;
            }
            x = nn.functional.leaky_relu(x);
            x = conv_post.forward(x);
            int half = _nFft / 2 + 1;
            var magnitude = torch.exp(x.narrow(1, 0, half));
            var phase = torch.sin(x.narrow(1, half, _nFft + 2 - half));
            x = _istft(magnitude, phase);
            return torch.clamp(x, -_audioLimit, _audioLimit);
        }

        public override Tensor forward(Tensor x)
        {
            var f0 = f0_predictor.forward(x);
            var s = nn.functional.interpolate(f0.unsqueeze(1), scale_factor: new double[] { _f0UpsampleScale }, mode: InterpolationMode.Nearest);
            s = s.transpose(1, 2);
            var (sine_merge, _, _) = m_source.forward(s);
            return Decode(x, sine_merge.transpose(1, 2));
        }

        public (Tensor, Tensor) Inference(Tensor speechFeat, bool finalize = true)
        {
            return Inference(speechFeat, torch.zeros(1, 1, 0, dtype: speechFeat.dtype, device: speechFeat.device));
        }

        public (Tensor, Tensor) Inference(Tensor speechFeat, Tensor cacheSource)
        {
            var f0 = f0_predictor.forward(speechFeat);
            var s = nn.functional.interpolate(f0.unsqueeze(1), scale_factor: new double[] { _f0UpsampleScale }, mode: InterpolationMode.Nearest);
            s = s.transpose(1, 2);
            var (sineMerge, _, _) = m_source.forward(s);
            s = sineMerge.transpose(1, 2);
            if (cacheSource is not null && cacheSource.shape.Length >= 3 && cacheSource.shape[2] != 0)
            {
                var cacheLen = Math.Min(cacheSource.shape[2], s.shape[2]);
                s[.., .., ..(int)cacheLen] = cacheSource[.., .., ..(int)cacheLen];
            }

            return (Decode(speechFeat, s), s);
        }
    }
}
