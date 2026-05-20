using System;
using System.IO;
using System.Text;

namespace CosyVoiceNet
{
    public static class WavHelper
    {
        // Create a PCM16 WAV file bytes from raw samples (short[] PCM16)
        public static byte[] CreateWav(short[] samples, int sampleRate, int channels = 1)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

            int byteRate = sampleRate * channels * 2;
            int blockAlign = channels * 2;
            int subChunk2Size = samples.Length * 2;
            int chunkSize = 36 + subChunk2Size;

            // RIFF header
            bw.Write(Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(chunkSize);
            bw.Write(Encoding.ASCII.GetBytes("WAVE"));

            // fmt subchunk
            bw.Write(Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16); // subchunk1 size
            bw.Write((short)1); // audio format PCM
            bw.Write((short)channels);
            bw.Write(sampleRate);
            bw.Write(byteRate);
            bw.Write((short)blockAlign);
            bw.Write((short)16); // bits per sample

            // data subchunk
            bw.Write(Encoding.ASCII.GetBytes("data"));
            bw.Write(subChunk2Size);

            // samples
            foreach (var s in samples)
            {
                bw.Write(s);
            }

            bw.Flush();
            ms.Position = 0;
            return ms.ToArray();
        }

        public static byte[] CreateSilentWav(int seconds, int sampleRate)
        {
            int totalSamples = seconds * sampleRate;
            short[] samples = new short[totalSamples];
            return CreateWav(samples, sampleRate);
        }

        public static float[] Float32BytesToSamples(byte[] bytes)
        {
            if (bytes.Length % sizeof(float) != 0)
                throw new ArgumentException("Float32 audio byte length must be divisible by 4.", nameof(bytes));

            var samples = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);
            return samples;
        }

        public static short[] FloatSamplesToPcm16(float[] samples)
        {
            var pcm = new short[samples.Length];
            for (var i = 0; i < samples.Length; i++)
            {
                var clamped = Math.Clamp(samples[i], -1.0f, 1.0f);
                pcm[i] = (short)Math.Round(clamped * short.MaxValue);
            }

            return pcm;
        }

        public static byte[] PrependSilenceToFloat32Bytes(byte[] bytes, int sampleRate, int channels = 1, int silenceMilliseconds = 0)
        {
            if (silenceMilliseconds <= 0)
                return bytes;

            if (bytes.Length % sizeof(float) != 0)
                throw new ArgumentException("Float32 audio byte length must be divisible by 4.", nameof(bytes));

            var silenceSamples = Math.Max(0, sampleRate * channels * silenceMilliseconds / 1000);
            if (silenceSamples == 0)
                return bytes;

            var result = new byte[(silenceSamples * sizeof(float)) + bytes.Length];
            Buffer.BlockCopy(bytes, 0, result, silenceSamples * sizeof(float), bytes.Length);
            return result;
        }

        public static byte[] CompressInternalSilenceInFloat32Bytes(
            byte[] bytes,
            int sampleRate,
            int channels = 1,
            int maxSilenceMilliseconds = 450,
            int windowMilliseconds = 20,
            float rmsThreshold = 0.006f)
        {
            if (maxSilenceMilliseconds <= 0)
                return bytes;
            if (channels <= 0)
                throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");
            if (bytes.Length % sizeof(float) != 0)
                throw new ArgumentException("Float32 audio byte length must be divisible by 4.", nameof(bytes));

            var samples = Float32BytesToSamples(bytes);
            if (samples.Length == 0)
                return bytes;

            var frameCount = samples.Length / channels;
            var windowFrames = Math.Max(1, sampleRate * Math.Max(1, windowMilliseconds) / 1000);
            var keepSilentWindows = Math.Max(1, (int)Math.Ceiling(maxSilenceMilliseconds / (double)Math.Max(1, windowMilliseconds)));
            var windowCount = (int)Math.Ceiling(frameCount / (double)windowFrames);
            if (windowCount <= keepSilentWindows + 2)
                return bytes;

            var silent = new bool[windowCount];
            for (var window = 0; window < windowCount; window++)
            {
                var startFrame = window * windowFrames;
                var endFrame = Math.Min(frameCount, startFrame + windowFrames);
                var sum = 0.0;
                var count = 0;
                for (var frame = startFrame; frame < endFrame; frame++)
                {
                    var sampleOffset = frame * channels;
                    for (var channel = 0; channel < channels; channel++)
                    {
                        var sample = samples[sampleOffset + channel];
                        sum += sample * sample;
                        count++;
                    }
                }

                var rms = count == 0 ? 0.0 : Math.Sqrt(sum / count);
                silent[window] = rms < rmsThreshold;
            }

            var drop = new bool[windowCount];
            var cursor = 0;
            while (cursor < windowCount)
            {
                if (!silent[cursor])
                {
                    cursor++;
                    continue;
                }

                var runStart = cursor;
                while (cursor < windowCount && silent[cursor])
                    cursor++;
                var runEnd = cursor;

                if (runStart == 0 || runEnd == windowCount)
                    continue;

                var runLength = runEnd - runStart;
                if (runLength <= keepSilentWindows)
                    continue;

                for (var window = runStart + keepSilentWindows; window < runEnd; window++)
                    drop[window] = true;
            }

            var keptFrameCount = 0;
            for (var window = 0; window < windowCount; window++)
            {
                if (drop[window])
                    continue;

                var startFrame = window * windowFrames;
                keptFrameCount += Math.Min(frameCount, startFrame + windowFrames) - startFrame;
            }

            if (keptFrameCount == frameCount)
                return bytes;

            var keptSamples = new float[keptFrameCount * channels];
            var outputOffset = 0;
            for (var window = 0; window < windowCount; window++)
            {
                if (drop[window])
                    continue;

                var startFrame = window * windowFrames;
                var endFrame = Math.Min(frameCount, startFrame + windowFrames);
                var sampleCount = (endFrame - startFrame) * channels;
                Array.Copy(samples, startFrame * channels, keptSamples, outputOffset, sampleCount);
                outputOffset += sampleCount;
            }

            var result = new byte[keptSamples.Length * sizeof(float)];
            Buffer.BlockCopy(keptSamples, 0, result, 0, result.Length);
            return result;
        }

        public static byte[] CreateWavFromFloat32Bytes(byte[] bytes, int sampleRate, int channels = 1)
        {
            return CreateWav(FloatSamplesToPcm16(Float32BytesToSamples(bytes)), sampleRate, channels);
        }
    }
}
