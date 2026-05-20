using System;
using System.Buffers;
using System.IO;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace CosyVoiceNet.TorchSharpUtils
{
    public sealed class WavHeader
    {
        public int Channels { get; private set; }
        public int SampleRate { get; private set; }
        public int BitsPerSample { get; private set; }
        public int BytesPerSample => BitsPerSample / 8;
        public long DataStart { get; private set; }
        public int NumFrames { get; private set; }

        public static WavHeader Parse(Stream stream)
        {
            var br = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

            string riff = new string(br.ReadChars(4));
            br.ReadInt32(); // file size
            string wave = new string(br.ReadChars(4));

            if (riff != "RIFF" || wave != "WAVE")
                throw new InvalidDataException("Not a WAV file");

            int channels = 0;
            int sampleRate = 0;
            int bitsPerSample = 0;
            long dataStart = 0;
            int dataSize = 0;

            while (stream.Position < stream.Length)
            {
                string chunkId = new string(br.ReadChars(4));
                int chunkSize = br.ReadInt32();

                if (chunkId == "fmt ")
                {
                    int audioFormat = br.ReadInt16();
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byte rate
                    br.ReadInt16(); // block align
                    bitsPerSample = br.ReadInt16();

                    if (chunkSize > 16)
                        br.ReadBytes(chunkSize - 16);
                }
                else if (chunkId == "data")
                {
                    dataStart = stream.Position;
                    dataSize = chunkSize;
                    stream.Seek(chunkSize, SeekOrigin.Current);
                }
                else
                {
                    stream.Seek(chunkSize, SeekOrigin.Current);
                }
            }

            if (dataStart == 0)
                throw new InvalidDataException("WAV missing data chunk");

            int bytesPerSample = bitsPerSample / 8;
            int frameSize = bytesPerSample * channels;
            int numFrames = dataSize / frameSize;

            return new WavHeader
            {
                Channels = channels,
                SampleRate = sampleRate,
                BitsPerSample = bitsPerSample,
                DataStart = dataStart,
                NumFrames = numFrames
            };
        }
    }

    public sealed class PcmReader
    {
        private readonly Stream _stream;
        private readonly int _channels;
        private readonly int _bytesPerSample;

        public PcmReader(Stream stream, int channels, int bytesPerSample)
        {
            _stream = stream;
            _channels = channels;
            _bytesPerSample = bytesPerSample;
        }

        // sf_readf_float equivalent: reads frames, returns frames read, interleaved float32
        public int ReadFramesFloat(float[] dest, int framesRequested)
        {
            int samplesRequested = framesRequested * _channels;
            int bytesRequested = samplesRequested * _bytesPerSample;

            if (bytesRequested <= 0)
                return 0;

            Span<byte> buffer = stackalloc byte[0];
            byte[] rented = null;

            if (bytesRequested <= 4096)
            {
                buffer = stackalloc byte[bytesRequested];
            }
            else
            {
                rented = ArrayPool<byte>.Shared.Rent(bytesRequested);
                buffer = rented.AsSpan(0, bytesRequested);
            }

            int bytesRead = _stream.Read(buffer);
            if (bytesRead <= 0)
            {
                if (rented != null) ArrayPool<byte>.Shared.Return(rented);
                return 0;
            }

            int samplesRead = bytesRead / _bytesPerSample;
            int framesRead = samplesRead / _channels;
            int totalSamples = framesRead * _channels;

            ConvertToFloat(buffer, dest, totalSamples, _bytesPerSample);

            if (rented != null) ArrayPool<byte>.Shared.Return(rented);
            return framesRead;
        }

        private static void ConvertToFloat(Span<byte> src, float[] dst, int samples, int bps)
        {
            switch (bps)
            {
                case 2: // PCM16
                    for (int i = 0; i < samples; i++)
                    {
                        short s = BitConverter.ToInt16(src.Slice(i * 2, 2));
                        dst[i] = Math.Clamp(s / 32768f, -1f, 1f);
                    }
                    break;

                case 3: // PCM24
                    for (int i = 0; i < samples; i++)
                    {
                        int b0 = src[i * 3 + 0];
                        int b1 = src[i * 3 + 1];
                        int b2 = src[i * 3 + 2];

                        int value = (b2 << 24) | (b1 << 16) | (b0 << 8);
                        value >>= 8; // sign extend

                        dst[i] = Math.Clamp(value / 8388608f, -1f, 1f);
                    }
                    break;

                case 4: // float32 PCM
                    for (int i = 0; i < samples; i++)
                    {
                        float f = BitConverter.ToSingle(src.Slice(i * 4, 4));
                        dst[i] = Math.Clamp(f, -1f, 1f);
                    }
                    break;

                default:
                    throw new NotSupportedException($"Unsupported sample size: {bps}");
            }
        }
    }

    public sealed class SoundFileCs : IDisposable
    {
        private readonly Stream _stream;
        private readonly bool _ownsStream;
        private readonly WavHeader _header;
        private readonly PcmReader _reader;

        public int Channels => _header.Channels;
        public int SampleRate => _header.SampleRate;
        public int Frames => _header.NumFrames;

        public SoundFileCs(string path)
        {
            _stream = File.OpenRead(path);
            _ownsStream = true;
            _header = WavHeader.Parse(_stream);
            _reader = new PcmReader(_stream, _header.Channels, _header.BytesPerSample);
        }

        public SoundFileCs(Stream stream, bool leaveOpen = false)
        {
            _stream = stream;
            _ownsStream = !leaveOpen;
            _header = WavHeader.Parse(_stream);
            _reader = new PcmReader(_stream, _header.Channels, _header.BytesPerSample);
        }

        private void SeekToFrame(int frameIndex)
        {
            long byteOffset = _header.DataStart
                            + (long)frameIndex * _header.Channels * _header.BytesPerSample;
            _stream.Seek(byteOffset, SeekOrigin.Begin);
        }

        private int PrepareRead(int start, int? stop, int framesRequested)
        {
            int totalFrames = _header.NumFrames;

            int s = Math.Clamp(start, 0, totalFrames);
            int e = stop.HasValue ? Math.Clamp(stop.Value, 0, totalFrames) : totalFrames;

            if (e < s) e = s;

            int frames = framesRequested < 0 ? (e - s) : framesRequested;

            SeekToFrame(s);
            return frames;
        }

        // Minimal SoundFile-like read:
        // - frameOffset: starting frame
        // - numFrames: frames to read (-1 or null = until end)
        // - channelsFirst: if true -> [channels, frames], else [frames, channels]
        public Tensor Read(int frameOffset = 0, int? numFrames = null, bool channelsFirst = true)
        {
            int framesToRead = PrepareRead(frameOffset, null, numFrames ?? -1);
            if (framesToRead <= 0)
                return torch.empty(new long[] { Channels, 0 }, dtype: ScalarType.Float32);

            float[] temp = new float[framesToRead * Channels];
            int framesRead = _reader.ReadFramesFloat(temp, framesToRead);

            float[,] waveform = new float[framesRead, Channels];
            for (int f = 0; f < framesRead; f++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    waveform[f, c] = temp[f * Channels + c];
                }
            }

            var tensor = torch.from_array(waveform); // [frames, channels]

            if (channelsFirst)
                tensor = tensor.transpose(0, 1); // [channels, frames]

            return tensor;
        }

        public void Dispose()
        {
            if (_ownsStream)
                _stream.Dispose();
        }
    }
}
