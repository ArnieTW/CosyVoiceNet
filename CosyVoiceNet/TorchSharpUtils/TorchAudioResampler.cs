using System;
using TorchSharp;
using static TorchSharp.torch;

namespace CosyVoiceNet.TorchSharpUtils
{
    public static class TorchAudioResampler
    {
        public static Tensor Resample(
            Tensor waveform,
            int origFreq,
            int newFreq,
            int lowpassFilterWidth = 6,
            double rolloff = 0.99,
            string resamplingMethod = "sinc_interp_hann",
            double? beta = null)
        {
            if (origFreq <= 0 || newFreq <= 0)
                throw new ArgumentException("Original and new frequencies must be positive.");

            if (origFreq == newFreq)
                return waveform;

            int gcd = Gcd(origFreq, newFreq);

            var (kernel, width) = GetSincResampleKernel(
                origFreq,
                newFreq,
                gcd,
                lowpassFilterWidth,
                rolloff,
                resamplingMethod,
                beta,
                waveform.device,
                null
            );

            return ApplySincResampleKernel(waveform, origFreq, newFreq, gcd, kernel, width);
        }

        private static (Tensor kernel, int width) GetSincResampleKernel(
            int origFreq,
            int newFreq,
            int gcd,
            int lowpassFilterWidth,
            double rolloff,
            string resamplingMethod,
            double? beta,
            Device device,
            ScalarType? dtypeOpt)
        {
            if (resamplingMethod == "sinc_interpolation")
                resamplingMethod = "sinc_interp_hann";
            else if (resamplingMethod == "kaiser_window")
                resamplingMethod = "sinc_interp_kaiser";
            else if (resamplingMethod != "sinc_interp_hann" && resamplingMethod != "sinc_interp_kaiser")
                throw new ArgumentException($"Invalid resampling method: {resamplingMethod}");

            origFreq = origFreq / gcd;
            newFreq = newFreq / gcd;

            if (lowpassFilterWidth <= 0)
                throw new ArgumentException("Low pass filter width should be positive.");

            double baseFreq = Math.Min(origFreq, newFreq) * rolloff;

            int width = (int)Math.Ceiling(lowpassFilterWidth * (double)origFreq / baseFreq);

            var idxDtype = dtypeOpt ?? ScalarType.Float64;

            var idx = arange(-width, width + origFreq, dtype: idxDtype, device: device)
                      .unsqueeze(0).unsqueeze(0) / origFreq;

            var t = arange(0, -newFreq, -1, dtype: dtypeOpt ?? ScalarType.Float32, device: device)
                    .unsqueeze(1).unsqueeze(2) / newFreq + idx;

            t *= baseFreq;
            t = t.clamp(-lowpassFilterWidth, lowpassFilterWidth);

            Tensor window;
            if (resamplingMethod == "sinc_interp_hann")
            {
                window = (cos(t * Math.PI / lowpassFilterWidth / 2).pow(2));
            }
            else
            {
                double b = beta ?? 14.769656459379492;
                var betaTensor = tensor((float)b, device: device);
                window = special.i0(betaTensor * sqrt(1 - (t / lowpassFilterWidth).pow(2)))
                         / special.i0(betaTensor);
            }

            t *= Math.PI;

            double scale = baseFreq / origFreq;

            var ones = ones_like(t);
            var kernels = where(t == 0, ones, sin(t) / t);
            kernels *= window * scale;

            if (dtypeOpt is null)
                kernels = kernels.to(ScalarType.Float32);

            return (kernels, width);
        }

        private static Tensor ApplySincResampleKernel(
            Tensor waveform,
            int origFreq,
            int newFreq,
            int gcd,
            Tensor kernel,
            int width)
        {
            if (!waveform.is_floating_point())
                throw new ArgumentException($"Expected floating point waveform, got {waveform.dtype}.");

            origFreq = origFreq / gcd;
            newFreq = newFreq / gcd;

            var shape = waveform.shape;
            var timeDim = shape[^1];

            var flat = waveform.view(-1, timeDim); // [N, T]
            var numWavs = flat.shape[0];
            var length = flat.shape[1];

            // pad (width, width + orig_freq)
            flat = nn.functional.pad(flat, new long[] { width, width + origFreq });

            var x = flat.unsqueeze(1); // [N,1,T+pad]
            var resampled = nn.functional.conv1d(x, kernel, stride: origFreq); // [N,new_freq,T_out]

            resampled = resampled.transpose(1, 2).reshape(numWavs, -1); // [N, T_new]

            var targetLength = (long)Math.Ceiling(newFreq * (double)length / origFreq);
            resampled = resampled.index(new TensorIndex[]
            {
                TensorIndex.Ellipsis,
                TensorIndex.Slice(0, targetLength)
            });

            var outShape = new long[shape.Length];
            for (int i = 0; i < shape.Length - 1; i++)
                outShape[i] = shape[i];
            outShape[^1] = resampled.shape[1];

            return resampled.view(outShape);
        }

        private static int Gcd(int a, int b)
        {
            while (b != 0)
            {
                int t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
    }
}
