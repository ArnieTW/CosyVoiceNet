using System;
using System.Collections.Generic;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
namespace CosyVoiceNet.Matcha
{
    public class MatchaAudio
    {
        public static Tensor? MelBasis = null;
        public static Tensor? HannWindow = null;
        private static string? SpectrogramCacheKey = null;
        public static Tensor dynamic_range_compression_torch( Tensor x, int c = 1, float clip_val = 1e-5f)
        {
            return torch.log( torch.clamp(x, min: clip_val) * c ).to(x.device);
        }
        public static Tensor spectral_normalize_torch(Tensor magnitudes)
        {
            return dynamic_range_compression_torch(magnitudes);
        }
        public static Tensor mel_spectogram( Tensor Y, int n_fft, int num_mels, int sampling_rate, int hop_size, int win_size, int fmin, int fmax, bool center = false )
        {
            if (Y.min().item<float>() < -1.0f)
                Console.WriteLine($"min value is {Y.min().item<float>()}");
            if (Y.max().item<float>() > 1.0f)
                Console.WriteLine($"max value is {Y.max().item<float>()}");
            var device = Y.device;
            var melKey = $"{sampling_rate}_{n_fft}_{num_mels}_{hop_size}_{win_size}_{fmin}_{fmax}_{center}_{device}";
            if ( ( MelBasis is null ) || ( HannWindow is null ) || !string.Equals(SpectrogramCacheKey, melKey, StringComparison.Ordinal) )
            {
                MelBasis = LibrosaLikeFilterbank.Mel(sampling_rate, n_fft, num_mels, fmin, fmax).to(device);
                HannWindow = torch.hann_window(win_size).to(device);
                SpectrogramCacheKey = melKey;
            }
            Y = nn.functional.pad(Y.unsqueeze(1), new long[] { (n_fft - hop_size) / 2, (n_fft - hop_size) / 2 }, mode: PaddingModes.Reflect).squeeze(1);
            var stft = torch.stft(Y, n_fft, hop_length: hop_size, win_length: win_size, window: HannWindow, center: center, pad_mode: PaddingModes.Reflect, normalized: false, onesided: true, return_complex: true);
            var spec = torch.view_as_real(stft);
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9f);
            spec = torch.matmul(MelBasis, spec);
            spec = spectral_normalize_torch(spec);
            return spec;
        }
    }

    public static class LibrosaLikeFilterbank
    {
        public static Tensor Mel(
            int sr,
            int nFft,
            int nMels = 128,
            double fmin = 0.0,
            double? fmaxOpt = null,
            bool htk = false,
            string norm = "slaney",
            ScalarType dtype = ScalarType.Float32,
            Device device = null)
        {
            device ??= CPU;
            double fmax = fmaxOpt ?? (sr / 2.0);

            int nFreqs = 1 + nFft / 2;

            // 1. FFT bin frequencies
            var fftFreqs = FftFrequencies(sr, nFft, device);

            // 2. Mel frequencies (n_mels + 2)
            var melF = MelFrequencies(nMels + 2, fmin, fmax, htk, device);

            // 3. Differences between mel frequencies
            var fdiff = melF[TensorIndex.Slice(1, null)] - melF[TensorIndex.Slice(null, -1)];

            // 4. Ramps: mel_f[i] - fft_freqs
            var ramps = melF.unsqueeze(1) - fftFreqs.unsqueeze(0);

            // 5. Initialize weights
            var weights = zeros(new long[] { nMels, nFreqs }, dtype: float32, device: device);

            for (int i = 0; i < nMels; i++)
            {
                var lower = -ramps[i] / fdiff[i];
                var upper = ramps[i + 2] / fdiff[i + 1];

                var filter = minimum(lower, upper).clamp(0, 1);
                weights[i] = filter;
            }

            // 6. Slaney normalization
            if (norm == "slaney")
            {
                var enorm = 2.0 / (melF[TensorIndex.Slice(2, null)] - melF[TensorIndex.Slice(null, -2)]);
                for (int i = 0; i < nMels; i++)
                    weights[i] *= enorm[i];
            }
            else if (norm != null)
            {
                throw new ArgumentException($"Unsupported norm={norm}");
            }

            return weights.to(dtype);
        }

        // librosa.filters.fft_frequencies
        private static Tensor FftFrequencies(int sr, int nFft, Device device)
        {
            int nFreqs = 1 + nFft / 2;
            return linspace(0.0, sr / 2.0, nFreqs, device: device, dtype: float32);
        }

        // librosa.filters.mel_frequencies
        private static Tensor MelFrequencies(int nMels, double fmin, double fmax, bool htk, Device device)
        {
            var melMin = HzToMel(fmin, htk);
            var melMax = HzToMel(fmax, htk);
            var nmels = linspace(melMin, melMax, nMels, device: device, dtype: float32);
            return MelToHz(nmels, htk);
        }

        public static double HzToMel(double freq, bool htk = false)
        {
            if (htk)
            {
                // HTK mel scale
                return 2595.0 * Math.Log10(1.0 + freq / 700.0);
            }

            // Slaney mel scale (librosa default)
            double f_min = 0.0;
            double f_sp = 200.0 / 3.0;

            // Linear region
            double mel = (freq - f_min) / f_sp;

            // Log region
            double min_log_hz = 1000.0;
            double min_log_mel = (min_log_hz - f_min) / f_sp;  // = 15
            double logstep = Math.Log(6.4) / 27.0;

            if (freq >= min_log_hz)
            {
                mel = min_log_mel + Math.Log(freq / min_log_hz) / logstep;
            }

            return mel;
        }

        public static Tensor MelToHz(Tensor mels, bool htk = false)
        {
            if (htk)
            {
                // HTK mel scale
                return 700.0 * (pow(10.0, mels / 2595.0) - 1.0);
            }

            // Slaney mel scale (librosa default)
            double f_min = 0.0;
            double f_sp = 200.0 / 3.0;

            // Linear region
            var freqs = f_min + f_sp * mels;

            // Log region
            double min_log_hz = 1000.0;
            double min_log_mel = (min_log_hz - f_min) / f_sp;   // = 15
            double logstep = Math.Log(6.4) / 27.0;

            // mels >= min_log_mel → log region
            var log_t = mels >= min_log_mel;

            // freqs[log_t] = min_log_hz * exp(logstep * (mels - min_log_mel))
            var logFreqs = min_log_hz * exp(logstep * (mels - min_log_mel));

            freqs = where(log_t, logFreqs, freqs);

            return freqs;
        }

        public static Tensor MelToHz(double mel, bool htk = false)
        {
            return MelToHz(tensor(mel, dtype: float64), htk);
        }
    }
}
