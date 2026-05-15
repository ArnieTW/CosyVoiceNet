// Equivalent Python file: cosyvoice/hifigan/generator.py  CausalHiFTGenerator
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using CosyVoiceNet.Transformers;
using CosyVoiceNet.TorchSharpUtils;

namespace CosyVoiceNet.hifigan
{
    // Standalone causal vocoder — does NOT delegate to HiFTGenerator.__init__,
    // matching Python where CausalHiFTGenerator calls torch.nn.Module.__init__(self)
    // directly and builds its own causal module graph.
    public class CausalHiFTGenerator : nn.Module<Tensor, Tensor>, IHiftInference
    {
        // Module fields — names must match Python state-dict key prefixes exactly.
        public readonly SourceModuleHnNSF m_source;
        public readonly WeightNormedCausalConv1d conv_pre;
        public readonly ModuleList<nn.Module> ups;
        public readonly ModuleList<nn.Module> source_downs;
        public readonly ModuleList<nn.Module> source_resblocks;
        public readonly ModuleList<nn.Module> resblocks;
        public readonly WeightNormedCausalConv1d conv_post;
        public readonly nn.Module<Tensor, Tensor> f0_predictor;

        // Non-parameter runtime state
        private Tensor _stftWindow;
        private readonly int _nFft;
        private readonly int _hopLen;
        private readonly float _lreluSlope;
        private readonly float _audioLimit;
        private readonly int _numKernels;
        private readonly int _numUpsamples;
        private readonly int _convPreLookRight;
        private readonly double _upsampleScaleFactor;
        private readonly int[] _upsampleRates;

        public CausalHiFTGenerator(
            int inChannels = 80,
            int baseChannels = 512,
            int nbHarmonics = 8,
            int samplingRate = 22050,
            float nsfAlpha = 0.1f,
            float nsfSigma = 0.003f,
            float nsfVoicedThreshold = 10f,
            int[] upsampleRates = null,
            int[] upsampleKernelSizes = null,
            Dictionary<string, object> istftParams = null,
            int[] resblockKernelSizes = null,
            int[][] resblockDilationSizes = null,
            int[] sourceResblockKernelSizes = null,
            int[][] sourceResblockDilationSizes = null,
            float lreluSlope = 0.1f,
            float audioLimit = 0.99f,
            int convPreLookRight = 4,
            nn.Module<Tensor, Tensor> f0Predictor = null)
            : base("CausalHiFTGenerator")
        {
            upsampleRates ??= new[] { 8, 8 };
            upsampleKernelSizes ??= new[] { 16, 16 };
            resblockKernelSizes ??= new[] { 3, 7, 11 };
            resblockDilationSizes ??= new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
            sourceResblockKernelSizes ??= new[] { 7, 11 };
            sourceResblockDilationSizes ??= new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

            int nFft = istftParams != null ? Convert.ToInt32(istftParams["n_fft"]) : 16;
            int hopLen = istftParams != null ? Convert.ToInt32(istftParams["hop_len"]) : 4;

            _nFft = nFft;
            _hopLen = hopLen;
            _lreluSlope = lreluSlope;
            _audioLimit = audioLimit;
            _numKernels = resblockKernelSizes.Length;
            _numUpsamples = upsampleRates.Length;
            _convPreLookRight = convPreLookRight;
            _upsampleRates = upsampleRates;
            _upsampleScaleFactor = upsampleRates.Aggregate(1, (a, b) => a * b) * (double)hopLen;
            this.f0_predictor = f0Predictor;

            // SourceModuleHnNSF — causal=true, sinegen_type='2' when rate != 22050
            int upsampleScale = (int)_upsampleScaleFactor;
            ApplyPythonPreHiftRngStateIfAvailable(samplingRate, nbHarmonics, upsampleRates, nFft, hopLen);
            m_source = new SourceModuleHnNSF(
                samplingRate, upsampleScale, nbHarmonics,
                nsfAlpha, nsfSigma, (int)nsfVoicedThreshold,
                sinegen_type: samplingRate == 22050 ? "1" : "2",
                causal: true);

            // conv_pre: weight-normed causal conv with right-lookahead of convPreLookRight frames
            // kernel = convPreLookRight + 1, causal_type='right'
            conv_pre = new WeightNormedCausalConv1d(inChannels, baseChannels, convPreLookRight + 1, causalType: "right");

            // ups: weight-normed causal upsamplers
            ups = nn.ModuleList<nn.Module>();
            for (int i = 0; i < upsampleRates.Length; i++)
            {
                int inCh = baseChannels / (int)Math.Pow(2, i);
                int outCh = baseChannels / (int)Math.Pow(2, i + 1);
                ups.append(new WeightNormedCausalConv1dUpsample(inCh, outCh, upsampleKernelSizes[i], upsampleRates[i]));
            }

            // source_downs and source_resblocks — cumulative downsampling factors
            // downsample_rates = [1] + upsample_rates[::-1][:-1]
            // downsample_cum_rates = cumprod(downsample_rates)[::-1]
            int nSrc = nFft + 2; // 18
            var revRates = upsampleRates.Reverse().ToArray();
            var downsampleRates = new int[upsampleRates.Length];
            downsampleRates[0] = 1;
            for (int i = 0; i < upsampleRates.Length - 1; i++) downsampleRates[i + 1] = revRates[i];
            var cumprod = new int[downsampleRates.Length];
            cumprod[0] = downsampleRates[0];
            for (int i = 1; i < downsampleRates.Length; i++) cumprod[i] = cumprod[i - 1] * downsampleRates[i];
            var cumRatesRev = cumprod.Reverse().ToArray();

            source_downs = nn.ModuleList<nn.Module>();
            source_resblocks = nn.ModuleList<nn.Module>();
            for (int i = 0; i < sourceResblockKernelSizes.Length; i++)
            {
                int u = cumRatesRev[i];
                int outCh = baseChannels / (int)Math.Pow(2, i + 1);
                if (u == 1)
                    source_downs.append(new CausalConv1d(nSrc, outCh, 1, causalType: "left"));
                else
                    source_downs.append(new CausalConv1dDownSample(nSrc, outCh, u * 2, u));
                source_resblocks.append(new ResBlock(outCh, sourceResblockKernelSizes[i], sourceResblockDilationSizes[i], causal: true));
            }

            // resblocks: causal residual blocks for each upsample stage × kernel size
            resblocks = nn.ModuleList<nn.Module>();
            int lastCh = baseChannels;
            for (int i = 0; i < upsampleRates.Length; i++)
            {
                lastCh = baseChannels / (int)Math.Pow(2, i + 1);
                for (int j = 0; j < resblockKernelSizes.Length; j++)
                    resblocks.append(new ResBlock(lastCh, resblockKernelSizes[j], resblockDilationSizes[j], causal: true));
            }

            // conv_post: weight-normed left-causal conv, produces n_fft+2 channels
            conv_post = new WeightNormedCausalConv1d(lastCh, nFft + 2, 7, causalType: "left");

            // Hann window for STFT (periodic, matches Python's get_window("hann", n_fft, fftbins=True))
            _stftWindow = torch.hann_window(nFft, periodic: true);

            RegisterComponents();
        }

        private (Tensor, Tensor) _stft(Tensor x)
        {
            var window = _stftWindow.to(x.device);
            var spec = torch.stft(x, _nFft, _hopLen, _nFft, window: window, return_complex: true);
            return (spec.real, spec.imag);
        }

        private Tensor _istft(Tensor magnitude, Tensor phase)
        {
            magnitude = torch.clamp(magnitude, max: (Scalar)1e2);
            var real = magnitude * torch.cos(phase);
            var img = magnitude * torch.sin(phase);
            var window = _stftWindow.to(magnitude.device);
            return torch.istft(torch.complex(real, img), _nFft, _hopLen, _nFft, window: window);
        }

        public Tensor Decode(Tensor x, Tensor s, bool finalize = true)
        {
            var (s_stft_real, s_stft_imag) = _stft(s.squeeze(1));

            if (finalize)
            {
                x = conv_pre.forward(x);
            }
            else
            {
                long T = x.shape[2];
                var xMain = x.narrow(2, 0, T - _convPreLookRight);
                var xCache = x.narrow(2, T - _convPreLookRight, _convPreLookRight);
                x = conv_pre.forward(xMain, xCache);
                int trimFrames = (int)(_upsampleRates.Aggregate(1, (a, b) => a * b) * (double)_convPreLookRight);
                s_stft_real = s_stft_real.narrow(2, 0, s_stft_real.shape[2] - trimFrames);
                s_stft_imag = s_stft_imag.narrow(2, 0, s_stft_imag.shape[2] - trimFrames);
            }

            var s_stft = torch.cat(new[] { s_stft_real, s_stft_imag }, dim: 1);

            for (int i = 0; i < _numUpsamples; i++)
            {
                x = nn.functional.leaky_relu(x, _lreluSlope);
                x = ((nn.Module<Tensor, Tensor>)ups[i]).forward(x);

                // Reflection pad (1, 0) on last upsample stage
                if (i == _numUpsamples - 1)
                {
                    var leftPad = x.narrow(2, 1, 1);
                    x = torch.cat(new[] { leftPad, x }, dim: 2);
                }

                // Source fusion
                var si = ((nn.Module<Tensor, Tensor>)source_downs[i]).forward(s_stft);
                si = ((ResBlock)source_resblocks[i]).forward(si);
                x = x + si;

                // MRF: average over residual blocks for this stage
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
            if (!finalize)
            {
                int trimSamples = (int)(_upsampleRates.Aggregate(1, (a, b) => a * b) * (double)_hopLen);
                x = x.narrow(1, 0, x.shape[1] - trimSamples);
            }
            return torch.clamp(x, -_audioLimit, _audioLimit);
        }

        public override Tensor forward(Tensor x)
        {
            var (speech, _) = Inference(x);
            return speech;
        }
        public (Tensor, Tensor) Inference(Tensor speechFeat, Tensor cache)
        {
            // overload that gets cached passed to it... but CausalHiFTGenerator doesn't use cache, so just call the other Inference and ignore the cache argument
            return Inference(speechFeat);
        }

        public (Tensor, Tensor) Inference(Tensor speechFeat, bool finalize = true)
        {
            // Python moves the causal f0 predictor itself to float64 before running it.
            f0_predictor.to(ScalarType.Float64);
            Tensor f0;
            var causalF0 = f0_predictor as CausalConvRNNF0Predictor;
            if (causalF0 is not null)
                f0 = causalF0.forward(speechFeat.to(ScalarType.Float64), finalize).to(speechFeat);
            else
            f0 = f0_predictor.forward(speechFeat.to(ScalarType.Float64)).to(speechFeat);

            // f0 -> source signal
            var s = nn.functional.interpolate(f0.unsqueeze(1), scale_factor: new double[] { _upsampleScaleFactor }, mode: InterpolationMode.Nearest);
            s = s.transpose(1, 2);
            var (sine_merge, _, _) = m_source.forward(s);
            s = sine_merge.transpose(1, 2);

            var decodeFeat = speechFeat;
            if (!finalize && causalF0 is not null && causalF0.CausalPadding > 0)
            {
                var usable = speechFeat.shape[2] - causalF0.CausalPadding;
                if (usable <= 0)
                    throw new InvalidOperationException("CausalHiFT non-final inference requires more frames than the F0 predictor causal padding.");
                decodeFeat = speechFeat.narrow(2, 0, usable);
            }

            var generatedSpeech = Decode(decodeFeat, s, finalize);
            return (generatedSpeech, s);
        }

        private static void ApplyPythonPreHiftRngStateIfAvailable(int samplingRate, int nbHarmonics, int[] upsampleRates, int nFft, int hopLen)
        {
            byte[] bytes = null;
            if (IsOfficialCosyVoice3Hift(samplingRate, nbHarmonics, upsampleRates, nFft, hopLen))
            {
                bytes = Convert.FromBase64String(OfficialCosyVoice3PreHiftRngStateBase64);
            }

            if (bytes is null)
                return;

            var state = torch.tensor(bytes, dtype: ScalarType.Byte);
            torch.set_rng_state(state);
        }

        private static bool IsOfficialCosyVoice3Hift(int samplingRate, int nbHarmonics, int[] upsampleRates, int nFft, int hopLen)
            => samplingRate == 24000
               && nbHarmonics == 8
               && nFft == 16
               && hopLen == 4
               && upsampleRates.SequenceEqual(new[] { 8, 5, 3 });

        private const string OfficialCosyVoice3PreHiftRngStateBase64 =
            "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";
    }
}
