using CosyVoiceNet.TorchSharpUtils;
using System;
using System.IO;
using System.Linq;
using System.Numerics;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchaudio;

namespace CosyVoiceNet.Tools
{
    internal static class AudioUtils
    {
        // Read PCM WAV files (supports PCM16, PCM32, float32) and return interleaved float samples ([-1,1]).
        public static (Tensor waveform, int sampleRate) LoadWavSoundfileLike(
            string path)
        {
            using var sf = new SoundFileCs(path);
            var tensor = sf.Read(channelsFirst: true);
            int sr = sf.SampleRate;
            return (tensor, sr);
        }

        public static (Tensor speech, int sampleRate) ReadWav(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            if (new string(br.ReadChars(4)) != "RIFF") throw new InvalidDataException("Not a RIFF file");
            br.ReadInt32();
            if (new string(br.ReadChars(4)) != "WAVE") throw new InvalidDataException("Not a WAVE file");

            int sampleRate = 0, channels = 0, bitsPerSample = 0, audioFormat = 0, dataLen = 0;
            long dataPos = -1;

            while (br.BaseStream.Position < br.BaseStream.Length)
            {
                if (br.BaseStream.Length - br.BaseStream.Position < 8) break;
                var chunkId = new string(br.ReadChars(4));
                var chunkSize = br.ReadInt32();
                var chunkStart = br.BaseStream.Position;

                if (chunkId == "fmt ")
                {
                    audioFormat = br.ReadInt16();
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byte rate
                    br.ReadInt16(); // block align
                    bitsPerSample = br.ReadInt16();
                }
                else if (chunkId == "data")
                {
                    dataPos = chunkStart;
                    dataLen = chunkSize;
                }
                br.BaseStream.Position = chunkStart + chunkSize;
                if ((chunkSize & 1) != 0 && br.BaseStream.Position < br.BaseStream.Length) br.BaseStream.Position++;
            }

            if (dataPos < 0) throw new InvalidDataException("No data chunk");

            br.BaseStream.Seek(dataPos, SeekOrigin.Begin);
            byte[] bytes = br.ReadBytes(dataLen);
            int totalSamples = dataLen / (bitsPerSample / 8);
            float[] samples = new float[totalSamples];

            for (int i = 0; i < totalSamples; i++)
            {
                if (bitsPerSample == 16 && audioFormat == 1)
                    samples[i] = BitConverter.ToInt16(bytes, i * 2) / 32768f;
                else if (bitsPerSample == 24 && audioFormat == 1)
                {
                    int idx = i * 3;
                    int val = bytes[idx] | (bytes[idx + 1] << 8) | (bytes[idx + 2] << 16);
                    if ((val & 0x800000) != 0) val |= unchecked((int)0xFF000000);
                    samples[i] = val / 8388608f;
                }
                else if (bitsPerSample == 32 && audioFormat == 3)
                    samples[i] = BitConverter.ToSingle(bytes, i * 4);
                else if (bitsPerSample == 32 && audioFormat == 1)
                    samples[i] = BitConverter.ToInt32(bytes, i * 4) / 2147483648f;
                else if (bitsPerSample == 8 && audioFormat == 1)
                    samples[i] = (bytes[i] - 128) / 128f;
                else
                    throw new NotSupportedException($"Format {audioFormat} @ {bitsPerSample}bit not supported.");
            }

            // torchaudio returns [channels, time]
            // 1. Load as [Frames, Channels] because that's how interleaved WAV data is laid out
            // 2. Transpose to [Channels, Frames] to match torchaudio format
            var tensor = torch.tensor(samples, dtype: float32).view(-1, channels).transpose(0, 1);
            return (tensor.contiguous(), sampleRate);
        }

        // Simple linear resampler for interleaved samples.
        public static float[] ResampleLinear(float[] samples, int srcRate, int dstRate, int channels)
        {
            if (srcRate == dstRate) return samples;
            int srcFrames = samples.Length / channels;
            double ratio = (double)dstRate / srcRate;
            int dstFrames = (int)Math.Round(srcFrames * ratio);
            float[] outSamples = new float[dstFrames * channels];
            for (int f = 0; f < dstFrames; f++)
            {
                double srcPos = f / ratio;
                int i0 = (int)Math.Floor(srcPos);
                int i1 = Math.Min(i0 + 1, srcFrames - 1);
                double t = srcPos - i0;
                for (int c = 0; c < channels; c++)
                {
                    float s0 = samples[i0 * channels + c];
                    float s1 = samples[i1 * channels + c];
                    outSamples[f * channels + c] = (float)((1 - t) * s0 + t * s1);
                }
            }
            return outSamples;
        }

        public static torch.Tensor ResampleSincTorch(torch.Tensor input, int inRate, int outRate, int kernelWidth = 32)
        {
            if (inRate == outRate)
                return input.clone();

            var device = input.device;
            var dtype = input.dtype;

            // Compute L/M ratio
            int g = Gcd(inRate, outRate);
            int L = outRate / g;   // upsample factor
            int M = inRate / g;    // downsample factor

            int kernelHalf = kernelWidth;

            // Build kernel positions
            var t = torch.arange(-kernelHalf, kernelHalf + 1, dtype: torch.float64, device: device);

            // Hann window
            var window = 0.5 * (1 + torch.cos((Math.PI * t) / kernelHalf));

            // sinc
            var sinc = torch.where(
                t == 0,
                torch.tensor(1.0, device: device),
                torch.sin(Math.PI * t) / (Math.PI * t)
            );

            // Final kernel
            var kernel = (sinc * window).to(dtype).unsqueeze(0).unsqueeze(0); // [1,1,K]

            // 1. Upsample by L (insert L-1 zeros between samples)
            var up = torch.zeros(new long[] { 1, input.shape[1] * L }, dtype: dtype, device: device);
            up[0, torch.arange(0, up.shape[1], L)] = input[0];

            // 2. Pad for convolution
            int pad = kernelHalf;
            var padded = torch.nn.functional.pad(up.unsqueeze(1), new long[] { pad, pad });

            // 3. Convolve
            var conv = torch.nn.functional.conv1d(
                padded,
                kernel
            ).squeeze(1); // [1, T_up]

            // 4. Downsample by M
            var down = conv[0, torch.arange(0, conv.shape[1], M)];

            return down.unsqueeze(0); // [1, T_out]
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

        // Convert interleaved samples to mono by averaging channels
        public static float[] ToMono(float[] samples, int channels)
        {
            if (channels == 1) return samples;
            int frames = samples.Length / channels;
            float[] outSamples = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++) sum += samples[i * channels + c];
                outSamples[i] = sum / channels;
            }
            return outSamples;
        }
        public static Tensor fbank(Tensor speech, int sampleRate, int numMels = 80, int nFft = 512, int winLen = 400, int hop = 160)
        {
            // Use the factory to create the module
            using var melTransform = torchaudio.transforms.MelSpectrogram(
                sample_rate: (long)sampleRate,
                n_fft: (long)nFft,
                win_length: (long)winLen,
                hop_length: (long)hop,
                f_min: 0.0,
                f_max: (double)(sampleRate / 2),
                n_mels: (long)numMels,
                window_fn: (len) => torch.hamming_window(len),
                norm: torchaudio.MelNorm.slaney,
                mel_scale: torchaudio.MelScale.htk
            );

            // 1. Get Mel Spectrogram [1, 80, Frames]
            using var melSpec = melTransform.forward(speech);

            // 2. Log compression - matches Kaldi behavior
            using var logMel = torch.log(melSpec.clamp(min: 1e-10));

            // 3. Reshape to [Frames, 80] and detach from graph
            // We use .alias() or .clone() so the returned tensor isn't disposed by the 'using' blocks
            return logMel.squeeze(0).transpose(0, 1).detach().clone();
        }

        private static void ProcessFrame(int frameIdx, float[] samples, double[] window, int winLen, int hop,
                                         int nFft, int fftBins, (int start, int end, double[] weights)[] melFb,
                                         int numMels, float[,] feats)
        {
            // Allocate thread-local buffers
            var fftInput = new double[nFft];
            var fftOutput = new Complex[nFft];
            var power = new double[fftBins];

            ProcessFrameWithBuffers(frameIdx, samples, window, winLen, hop, nFft, fftBins,
                                   melFb, numMels, feats, fftInput, fftOutput, power);
        }

        private static void ProcessFrameWithBuffers(int frameIdx, float[] samples, double[] window, int winLen, int hop,
                                                    int nFft, int fftBins, (int start, int end, double[] weights)[] melFb,
                                                    int numMels, float[,] feats, double[] fftInput, Complex[] fftOutput, double[] power)
        {
            int start = frameIdx * hop;

            // Apply window and prepare FFT input
            Array.Clear(fftInput, 0, nFft);
            for (int i = 0; i < winLen && (start + i) < samples.Length; i++)
            {
                fftInput[i] = samples[start + i] * window[i];
            }

            // Perform FFT using optimized implementation
            FFTOptimized(fftInput, fftOutput, nFft);

            // Compute power spectrum
            for (int k = 0; k < fftBins; k++)
            {
                double mag = fftOutput[k].Magnitude;
                power[k] = (mag * mag) / nFft;
            }

            // Apply mel filters (optimized sparse operation)
            for (int m = 0; m < numMels; m++)
            {
                var filter = melFb[m];
                double sum = 0.0;

                // Only iterate over non-zero filter weights
                for (int k = filter.start; k < filter.end && k < fftBins; k++)
                {
                    sum += filter.weights[k - filter.start] * power[k];
                }

                feats[frameIdx, m] = (float)sum;
            }
        }

        private static void NormalizeFeatures(float[,] feats, int frames, int numMels)
        {
            for (int m = 0; m < numMels; m++)
            {
                double mean = 0.0;
                for (int f = 0; f < frames; f++)
                {
                    feats[f, m] = (float)Math.Log(Math.Max(1e-10, feats[f, m]));
                    mean += feats[f, m];
                }
                mean /= frames;

                for (int f = 0; f < frames; f++)
                {
                    feats[f, m] -= (float)mean;
                }
            }
        }

        // Optimize mel filterbank by storing only non-zero ranges
        private static (int start, int end, double[] weights)[] OptimizeMelFilterbank(double[][] melFb, int fftBins)
        {
            var optimized = new (int start, int end, double[] weights)[melFb.Length];

            for (int m = 0; m < melFb.Length; m++)
            {
                // Find first and last non-zero indices
                int start = 0;
                int end = fftBins;

                for (int k = 0; k < fftBins; k++)
                {
                    if (melFb[m][k] > 0)
                    {
                        start = k;
                        break;
                    }
                }

                for (int k = fftBins - 1; k >= start; k--)
                {
                    if (melFb[m][k] > 0)
                    {
                        end = k + 1;
                        break;
                    }
                }

                // Copy only non-zero weights
                int length = end - start;
                var weights = new double[length];
                Array.Copy(melFb[m], start, weights, 0, length);

                optimized[m] = (start, end, weights);
            }

            return optimized;
        }

        // Compute log-mel spectrogram (features shaped [n_mels, time]) used by whisper-like processing
        public static Tensor WhisperLogMelSpectrogram(Tensor speech, int sampleRate = 16000, int nMels = 128)
        {
            using var scope = torch.NewDisposeScope();

            // 1. WHISPER PADDING: OpenAI pads 200 samples on both sides with reflect mode
            // This is critical for matching the center-alignment of the Python STFT
            using var paddedSpeech = torch.nn.functional.pad(speech, new long[] { 200, 200 }, PaddingModes.Reflect);

            using var melTransform = torchaudio.transforms.MelSpectrogram(
                sample_rate: 16000L,
                n_fft: 400L,
                win_length: 400L,
                hop_length: 160L,
                f_min: 0.0,
                f_max: 8000.0,
                n_mels: (long)nMels,
                window_fn: (len) => torch.hann_window(len),
                power: 2.0,
                center: false, // We already padded manually to match Whisper exactly
                norm: torchaudio.MelNorm.slaney,
                mel_scale: torchaudio.MelScale.slaney
            );

            // 2. Get Power Spectrogram [1, 128, T]
            using var melSpec = melTransform.forward(paddedSpeech);

            // 3. WHISPER FRAME TRUNCATION: Python code does stft[..., :-1]
            // Without this, your feature length is off by 1, which shifts all tokens
            long timeSteps = melSpec.shape[2];
            using var narrowed = melSpec.narrow(2, 0, timeSteps - 1);

            // 4. LOG SCALING: log10(clamp(mel_spec))
            using var logSpec = torch.log10(narrowed.clamp(min: 1e-10));

            // 5. GLOBAL NORMALIZATION: This centers the values for the codebook
            // Formula: (maximum(log_spec, log_spec.max() - 8) + 4) / 4
            using var maxVal = logSpec.max();
            using var shifted = logSpec.maximum(maxVal - 8.0);
            var finalTensor = (shifted + 4.0) / 4.0;

            // Move to CPU and detach to prevent handle invalidation errors
            return scope.MoveToOuter(finalTensor.detach().cpu());
        }

        // Mel filterbank generation: returns array[numMels][fftBins]
        private static double[][] MelFilterBank(int numMels, int nFft, int sampleRate, int lowFreq, int highFreq)
        {
            int fftBins = nFft / 2 + 1;
            double lowMel = HzToMel(lowFreq);
            double highMel = HzToMel(highFreq);
            double[] mels = new double[numMels + 2];
            for (int i = 0; i < mels.Length; i++) mels[i] = lowMel + (highMel - lowMel) * i / (numMels + 1);
            double[] hz = mels.Select(m => MelToHz(m)).ToArray();
            int[] bin = hz.Select(h => (int)Math.Floor((nFft + 1) * h / sampleRate)).ToArray();

            var fb = new double[numMels][];
            for (int m = 0; m < numMels; m++)
            {
                fb[m] = new double[fftBins];
                int f_m_minus = bin[m];
                int f_m = bin[m + 1];
                int f_m_plus = bin[m + 2];
                for (int k = f_m_minus; k < f_m; k++) if (k >= 0 && k < fftBins) fb[m][k] = (k - f_m_minus) / (double)(f_m - f_m_minus);
                for (int k = f_m; k < f_m_plus; k++) if (k >= 0 && k < fftBins) fb[m][k] = (f_m_plus - k) / (double)(f_m_plus - f_m);
            }
            return fb;
        }

        private static double HzToMel(double hz) => 2595.0 * Math.Log10(1 + hz / 700.0);
        private static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        private static void FFTOptimized(double[] input, Complex[] output, int n)
        {
            var buffer = new Complex[n];
            for (var i = 0; i < n && i < input.Length; i++)
            {
                buffer[i] = new Complex(input[i], 0.0);
            }

            FFTInplace(buffer);
            var outputSize = Math.Min(output.Length, n / 2 + 1);
            for (var i = 0; i < outputSize; i++)
            {
                output[i] = buffer[i];
            }
        }

        private static void FFTInplace(Complex[] buffer)
        {
            int n = buffer.Length;
            int bits = (int)Math.Log2(n);
            // bit reversal
            for (int j = 1, i = 0; j < n; j++)
            {
                int bit = n >> 1;
                for (; i >= bit; bit >>= 1) i -= bit;
                i += bit;
                if (j < i)
                {
                    var tmp = buffer[j]; buffer[j] = buffer[i]; buffer[i] = tmp;
                }
            }
            for (int len = 2; len <= n; len <<= 1)
            {
                double angle = -2.0 * Math.PI / len;
                Complex wlen = new Complex(Math.Cos(angle), Math.Sin(angle));
                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    for (int j = 0; j < len / 2; j++)
                    {
                        var u = buffer[i + j];
                        var v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
        }

        private static readonly object _whisperMelLock = new object();
        private static float[]? _whisperMel128x201;

        private static float[] GetWhisperMel128x201()
        {
            if (_whisperMel128x201 != null)
                return _whisperMel128x201;

            lock (_whisperMelLock)
            {
                if (_whisperMel128x201 != null)
                    return _whisperMel128x201;

                var candidates = new[]
                {
                    Path.Combine(AppContext.BaseDirectory, "asset", "whisper_mel_128_f32.bin"),
                    Path.Combine(Directory.GetCurrentDirectory(), "CosyVoiceNet", "asset", "whisper_mel_128_f32.bin"),
                    Path.Combine(Directory.GetCurrentDirectory(), "asset", "whisper_mel_128_f32.bin")
                };

                var melPath = candidates.FirstOrDefault(File.Exists)
                    ?? throw new FileNotFoundException("Missing whisper mel filter file: whisper_mel_128_f32.bin");

                var bytes = File.ReadAllBytes(melPath);
                if (bytes.Length != 128 * 201 * sizeof(float))
                    throw new InvalidDataException($"Invalid whisper mel filter size: {bytes.Length}");

                var data = new float[128 * 201];
                Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
                _whisperMel128x201 = data;
                return data;
            }
        }

    }
}
// Equivalent Python file: cosyvoice/tools/audio_utils.py
