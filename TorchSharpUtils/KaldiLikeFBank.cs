using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace CosyVoiceNet.TorchSharpUtils
{
    public static class KaldiLikeFBank
    {
        // ------------------------------------------------------------
        // Public API
        // ------------------------------------------------------------
        public static Tensor FBank(
            Tensor waveform,
            int sampleRate = 16000,
            int numMelBins = 80,
            double lowFreq = 20.0,
            double highFreq = 0.0,
            double dither = 0.0,
            double preemph = 0.97,
            double frameLengthMs = 25.0,
            double frameShiftMs = 10.0,
            bool snipEdges = true,
            bool usePower = true,
            bool useLog = true,
            bool useEnergy = false,
            bool subtractMean = false,
            bool htkCompat = false,
            string windowType = "povey",
            double blackmanCoeff = 0.42
        )
        {
            var device = waveform.device;
            var dtype = waveform.dtype;

            (waveform, int windowShift, int windowSize, int paddedWindowSize) = GetWaveformAndWindowProperties(
                waveform,
                channel: 0,
                sampleFrequency: sampleRate,
                frameShiftMs: frameShiftMs,
                frameLengthMs: frameLengthMs,
                roundToPowerOfTwo: true,
                preemphasisCoefficient: preemph
            );

            // 1. Windowing + energy
            var (strided, logEnergy) = GetWindow(
                waveform,
                paddedWindowSize,
                windowSize,
                windowShift,
                windowType,
                blackmanCoeff,
                snipEdges,
                rawEnergy: true,
                energyFloor: 1.0,
                dither: dither,
                removeDcOffset: true,
                preemphasisCoeff: preemph
            );

            // 2. FFT → magnitude or power
            var spectrum = torch.fft.rfft(strided);
            var mag = spectrum.abs();   // magnitude
            if (usePower)
                mag = mag.pow(2);       // power

            // 3. Mel filterbank
            var (melFb, _) = GetMelBanks(
                numMelBins,
                paddedWindowSize,
                sampleRate,
                lowFreq,
                highFreq,
                vtlnLow: 100.0,
                vtlnHigh: -500.0,
                vtlnWarpFactor: 1.0,
                device
            );

            // Pad mel filters to match rfft size
            melFb = nn.functional.pad(melFb, new long[] { 0, 1 }, mode: PaddingModes.Constant, value: 0);

            // 4. Apply mel filters
            var mel = torch.matmul(mag, melFb.transpose(0, 1));

            // 5. Log compression
            if (useLog)
                mel = torch.maximum(mel, GetEpsilon(device, dtype)).log();

            // 6. Energy column
            if (useEnergy)
            {
                var e = logEnergy.unsqueeze(1);  // (m × 1)

                if (htkCompat)
                    mel = torch.cat(new Tensor[] { mel, e }, dim: 1);  // energy last
                else
                    mel = torch.cat(new Tensor[] { e, mel }, dim: 1);  // energy first
            }

            // 7. Mean subtraction
            mel = SubtractColumnMean(mel, subtractMean);
            return mel;
        }

        // ------------------------------------------------------------
        // Mel scale (Kaldi)
        // ------------------------------------------------------------
        private static double MelScaleScalar(double freq)
            => 1127.0 * Math.Log(1.0 + freq / 700.0);

        private static double InverseMelScaleScalar(double mel)
            => 700.0 * (Math.Exp(mel / 1127.0) - 1.0);

        private static Tensor MelScale(Tensor freq)
            => 1127.0 * (1.0 + freq / 700.0).log();

        private static Tensor InverseMelScale(Tensor mel)
            => 700.0 * ((mel / 1127.0).exp() - 1.0);
        private static Tensor VtlnWarpFreq(
            double vtlnLowCutoff,
            double vtlnHighCutoff,
            double lowFreq,
            double highFreq,
            double vtlnWarpFactor,
            Tensor freq)
        {
            if (!(vtlnLowCutoff > lowFreq))
                throw new ArgumentException("be sure to set the vtln_low option higher than low_freq");
            if (!(vtlnHighCutoff < highFreq))
                throw new ArgumentException("be sure to set the vtln_high option lower than high_freq [or negative]");

            double l = vtlnLowCutoff * Math.Max(1.0, vtlnWarpFactor);
            double h = vtlnHighCutoff * Math.Min(1.0, vtlnWarpFactor);
            double scale = 1.0 / vtlnWarpFactor;

            double Fl = scale * l;
            double Fh = scale * h;

            if (!(l > lowFreq && h < highFreq))
                throw new ArgumentException("VTLN inflection points must lie within [low_freq, high_freq]");

            double scaleLeft = (Fl - lowFreq) / (l - lowFreq);
            double scaleRight = (highFreq - Fh) / (highFreq - h);

            var res = torch.empty_like(freq);

            var outsideLowHigh = torch.lt(freq, lowFreq) | torch.gt(freq, highFreq);
            var beforeL = torch.lt(freq, l);
            var beforeH = torch.lt(freq, h);
            var afterH = torch.ge(freq, h);

            // order matters (same as Python)
            res[afterH] = highFreq + scaleRight * (freq[afterH] - highFreq);
            res[beforeH] = scale * freq[beforeH];
            res[beforeL] = lowFreq + scaleLeft * (freq[beforeL] - lowFreq);
            res[outsideLowHigh] = freq[outsideLowHigh];

            return res;
        }

        private static Tensor VtlnWarpMelFreq(
            double vtlnLowCutoff,
            double vtlnHighCutoff,
            double lowFreq,
            double highFreq,
            double vtlnWarpFactor,
            Tensor melFreq)
        {
            var hz = InverseMelScale(melFreq);
            var warpedHz = VtlnWarpFreq(
                vtlnLowCutoff,
                vtlnHighCutoff,
                lowFreq,
                highFreq,
                vtlnWarpFactor,
                hz
            );
            return MelScale(warpedHz);
        }

        // ------------------------------------------------------------
        // Mel filterbank (Kaldi)
        // ------------------------------------------------------------
        public static (Tensor bins, Tensor centerFreqs) GetMelBanks(
            int numBins,
            int windowLengthPadded,
            double sampleFreq,
            double lowFreq,
            double highFreq,
            double vtlnLow,
            double vtlnHigh,
            double vtlnWarpFactor,
            Device device)
        {
            if (numBins <= 3)
                throw new ArgumentException("Must have at least 3 mel bins");

            if (windowLengthPadded % 2 != 0)
                throw new ArgumentException("window_length_padded must be divisible by 2");

            double numFftBins = windowLengthPadded / 2;
            double nyquist = 0.5 * sampleFreq;

            if (highFreq <= 0.0)
                highFreq += nyquist;

            if (!(0.0 <= lowFreq && lowFreq < nyquist &&
                  0.0 < highFreq && highFreq <= nyquist &&
                  lowFreq < highFreq))
                throw new ArgumentException($"Bad low/high freq: {lowFreq}, {highFreq}, nyquist={nyquist}");

            double fftBinWidth = sampleFreq / windowLengthPadded;

            double melLow = MelScaleScalar(lowFreq);
            double melHigh = MelScaleScalar(highFreq);

            double melDelta = (melHigh - melLow) / (numBins + 1);

            if (vtlnHigh < 0.0)
                vtlnHigh += nyquist;

            if (vtlnWarpFactor != 1.0)
            {
                if (!(lowFreq < vtlnLow && vtlnLow < highFreq &&
                      0.0 < vtlnHigh && vtlnHigh < highFreq &&
                      vtlnLow < vtlnHigh))
                    throw new ArgumentException("Bad VTLN parameters");
            }

            // bin index: shape (numBins, 1)
            var bin = torch.arange(numBins, device: device, dtype: float32).unsqueeze(1);

            var leftMel = melLow + bin * melDelta;
            var centerMel = melLow + (bin + 1.0) * melDelta;
            var rightMel = melLow + (bin + 2.0) * melDelta;

            if (vtlnWarpFactor != 1.0)
            {
                leftMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, leftMel);
                centerMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, centerMel);
                rightMel = VtlnWarpMelFreq(vtlnLow, vtlnHigh, lowFreq, highFreq, vtlnWarpFactor, rightMel);
            }

            // center frequencies (numBins)
            var centerFreqs = InverseMelScale(centerMel).squeeze();

            // mel values for FFT bins: shape (1, numFftBins)
            var fftBins = torch.arange(numFftBins, device: device, dtype: float32);
            var mel = MelScale(fftBinWidth * fftBins).unsqueeze(0);

            // slopes: shape (numBins, numFftBins)
            var upSlope = (mel - leftMel) / (centerMel - leftMel);
            var downSlope = (rightMel - mel) / (rightMel - centerMel);

            Tensor binsTensor;

            if (vtlnWarpFactor == 1.0)
            {
                // clamp negative values, take min of slopes
                var zeros = torch.zeros_like(upSlope);
                binsTensor = torch.max(zeros, torch.min(upSlope, downSlope));
            }
            else
            {
                binsTensor = torch.zeros_like(upSlope);

                var upIdx = torch.gt(mel, leftMel) & torch.le(mel, centerMel);
                var downIdx = torch.gt(mel, centerMel) & torch.lt(mel, rightMel);

                binsTensor[upIdx] = upSlope[upIdx];
                binsTensor[downIdx] = downSlope[downIdx];
            }

            return (binsTensor.to(float32), centerFreqs.to(float32));
        }

        private static (Tensor waveform, int windowShift, int windowSize, int paddedWindowSize) GetWaveformAndWindowProperties(
            Tensor waveform,
            int channel,
            double sampleFrequency,
            double frameShiftMs,
            double frameLengthMs,
            bool roundToPowerOfTwo,
            double preemphasisCoefficient)
        {
            // Ensure channel index is valid
            channel = Math.Max(channel, 0);
            if (channel >= waveform.shape[0])
                throw new ArgumentException($"Invalid channel {channel} for size {waveform.shape[0]}");

            // Select channel → shape (num_samples)
            var selected = waveform[channel];

            // Convert ms → samples
            int windowShift = (int)(sampleFrequency * frameShiftMs * 0.001);
            int windowSize = (int)(sampleFrequency * frameLengthMs * 0.001);

            int paddedWindowSize = roundToPowerOfTwo
                ? NextPowerOfTwo(windowSize)
                : windowSize;

            // Assertions (same as Python)
            if (windowSize < 2 || windowSize > selected.shape[0])
                throw new ArgumentException(
                    $"choose a window size {windowSize} that is [2, {selected.shape[0]}]");

            if (windowShift <= 0)
                throw new ArgumentException("`window_shift` must be greater than 0");

            if (paddedWindowSize % 2 != 0)
                throw new ArgumentException(
                    "the padded `window_size` must be divisible by two. " +
                    "use `round_to_power_of_two` or change `frame_length`");

            if (preemphasisCoefficient < 0.0 || preemphasisCoefficient > 1.0)
                throw new ArgumentException("`preemphasis_coefficient` must be between [0,1]");

            if (sampleFrequency <= 0)
                throw new ArgumentException("`sample_frequency` must be greater than zero");

            return (selected, windowShift, windowSize, paddedWindowSize);
        }
        private static Tensor SubtractColumnMean(Tensor tensor, bool subtractMean)
        {
            if (subtractMean)
            {
                var colMeans = tensor.mean(new long[] { 0 }, keepdim: true);
                tensor = tensor - colMeans;
            }
            return tensor;
        }

        // ------------------------------------------------------------
        // Windowing
        // ------------------------------------------------------------
        private static (Tensor, Tensor) GetWindow(
            Tensor waveform,
            int paddedWindowSize,
            int windowSize,
            int windowShift,
            string windowType,
            double blackmanCoeff,
            bool snipEdges,
            bool rawEnergy,
            double energyFloor,
            double dither,
            bool removeDcOffset,
            double preemphasisCoeff)
        {
            var device = waveform.device;
            var dtype = waveform.dtype;

            var epsilon = GetEpsilon(device, dtype);

            var strided = GetStrided(waveform, windowSize, windowShift, snipEdges);

            if (dither != 0.0)
            {
                var randGauss = torch.randn(strided.shape, device: device, dtype: dtype);
                strided = strided + randGauss * dither;
            }

            if (removeDcOffset)
            {
                var rowMeans = strided.mean(dimensions: new long[] { 1 }, keepdim: true);
                strided = strided - rowMeans;
            }

            Tensor logEnergy = null;
            if (rawEnergy)
                logEnergy = GetLogEnergy(strided, epsilon, energyFloor);

            if (preemphasisCoeff != 0.0)
            {
                // 1. Pad on the left with replicate
                var offset = nn.functional.pad(
                    strided.unsqueeze(0),
                    new long[] { 1, 0 },   // pad (left=1, right=0)
                    mode: PaddingModes.Replicate
                ).squeeze(0);              // shape: (m, window_size + 1)

                // 2. Apply preemphasis: x[j] -= coeff * x[j-1]
                var prev = offset.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(0, -1) });
                strided = strided - preemphasisCoeff * prev;
            }

            var window = FeatureWindow(windowType, windowSize, blackmanCoeff, device, dtype)
                         .unsqueeze(0);

            strided = strided * window;

            if (paddedWindowSize != windowSize)
            {
                int padRight = paddedWindowSize - windowSize;
                strided = nn.functional.pad(strided.unsqueeze(0),
                    new long[] { 0, padRight }, PaddingModes.Constant, 0).squeeze(0);
            }

            if (!rawEnergy)
                logEnergy = GetLogEnergy(strided, epsilon, energyFloor);

            return (strided, logEnergy);
        }

        // ------------------------------------------------------------
        // Framing
        // ------------------------------------------------------------
        private static Tensor GetStrided(Tensor waveform, int windowSize, int windowShift, bool snipEdges)
        {
            var device = waveform.device;
            var dtype = waveform.dtype;

            long numSamples = waveform.shape[0];
            (long frameStride, long sampleStride) = (windowShift * waveform.stride(0), waveform.stride(0));

            // ------------------------------------------------------------
            // snip_edges = True  →  only frames that fully fit
            // ------------------------------------------------------------
            if (snipEdges)
            {
                if (numSamples < windowSize)
                    return torch.empty(new long[] { 0, 0 }, dtype: dtype, device: device);

                long m = 1 + (numSamples - windowSize) / windowShift;

                return waveform.as_strided(
                    size: new long[] { m, windowSize },
                    strides: new long[] { frameStride, sampleStride }
                );
            }

            // ------------------------------------------------------------
            // snip_edges = False  →  reflection padding + as_strided
            // ------------------------------------------------------------
            var reversed = torch.flip(waveform, new long[] { 0 });

            long m2 = (numSamples + (windowShift / 2)) / windowShift;
            long pad = windowSize / 2 - windowShift / 2;

            if (pad > 0)
            {
                // pad_left = reversed[-pad:]
                long start = reversed.shape[0] - pad;
                long end = reversed.shape[0];

                var padLeft = reversed.slice(
                    dim: 0,
                    start: start,
                    finish: end,
                    step: 1
                );

                var padRight = reversed;

                waveform = torch.cat(new Tensor[] { padLeft, waveform, padRight }, dim: 0);
            }
            else
            {
                // pad < 0 → trim front: waveform[-pad:]
                long start = -pad;
                long end = waveform.shape[0];

                var trimmed = waveform.slice(
                    dim: 0,
                    start: start,
                    finish: end,
                    step: 1
                );

                waveform = torch.cat(new Tensor[] { trimmed, reversed }, dim: 0);
            }

            return waveform.as_strided(
                size: new long[] { m2, windowSize },
                strides: new long[] { frameStride, sampleStride }
            );
        }


        // ------------------------------------------------------------
        // Window functions
        // ------------------------------------------------------------
        private static Tensor FeatureWindow(
            string windowType,
            int windowSize,
            double blackmanCoeff,
            Device device,
            ScalarType dtype)
        {
            switch (windowType)
            {
                case "hanning":
                case "HANNING":
                    // torch.hann_window(window_size, periodic=False)
                    return torch.hann_window(
                        windowSize,
                        periodic: false,
                        device: device,
                        dtype: dtype
                    );

                case "hamming":
                case "HAMMING":
                    // torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
                    return torch.hamming_window(
                        windowSize,
                        periodic: false,
                        alpha: 0.54f,
                        beta: 0.46f,
                        device: device,
                        dtype: dtype
                    );

                case "povey":
                case "POVEY":
                    // hann_window(periodic=False).pow(0.85)
                    return torch.hann_window(
                        windowSize,
                        periodic: false,
                        device: device,
                        dtype: dtype
                    ).pow(0.85);

                case "rectangular":
                case "RECTANGULAR":
                    return torch.ones(windowSize, device: device, dtype: dtype);

                case "blackman":
                case "BLACKMAN":
                    {
                        // Kaldi-style Blackman (NOT PyTorch's)
                        // a = 2*pi/(window_size - 1)
                        double a = 2.0 * Math.PI / (windowSize - 1);

                        var n = torch.arange(windowSize, device: device, dtype: dtype);

                        return (
                            blackmanCoeff
                            - 0.5 * torch.cos(a * n)
                            + (0.5 - blackmanCoeff) * torch.cos(2 * a * n)
                        ).to(device: device, type: dtype);
                    }

                default:
                    throw new ArgumentException($"Invalid window type: {windowType}");
            }
        }


        // ------------------------------------------------------------
        // Energy
        // ------------------------------------------------------------
        private static Tensor GetEpsilon(Device device, ScalarType dtype)
        {
            double eps = torch.finfo(dtype).eps;
            return torch.tensor(eps, device: device, dtype: dtype);
        }

        private static Tensor GetLogEnergy(Tensor stridedInput, Tensor epsilon, double energyFloor)
        {
            var device = stridedInput.device;
            var dtype = stridedInput.dtype;

            // energy = sum(x^2) over each frame
            var energy = stridedInput.pow(2).sum(1);

            // max(energy, epsilon)
            var logEnergy = torch.maximum(energy, epsilon).log();

            if (energyFloor == 0.0)
                return logEnergy;

            // floor value as tensor
            var floorTensor = torch.tensor(Math.Log(energyFloor), device: device, dtype: dtype);

            // max(logEnergy, log(energy_floor))
            return torch.maximum(logEnergy, floorTensor);
        }

        // ------------------------------------------------------------
        // Utility
        // ------------------------------------------------------------
        private static int NextPowerOfTwo(int x)
        {
            int p = 1;
            while (p < x) p <<= 1;
            return p;
        }
    }

}
