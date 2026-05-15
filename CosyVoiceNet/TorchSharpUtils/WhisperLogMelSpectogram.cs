using CosyVoiceApp;
using System;
using System.Linq;
using System.Text;
using System.IO.Compression;
using TorchSharp;
using static TorchSharp.torch;
using System.IO;

namespace CosyVoiceNet.TorchSharpUtils
{
    public static class WhisperLikeLogMelSpectogram
    {
        private static int loadedMels = -1;
        private static torch.Tensor? loadedMelFilter = null;
        private static torch.Tensor? loadedHannWindow = null;
        //default just like in Whisper....
        private const int NFFT = 400;
        private const int HopLength = 160;
        private const int ChunkLength = 30;
        private static torch.Tensor GetMelFilter( int nMels, DeviceType device )
        {
            if ( loadedMels != nMels )
            {
                loadedMels = nMels;
                string resource = CosyVoiceApp.AppHost.Assets["mel_filters.npz"];
                var (data, shape) = LoadNpyFromNpz(resource, $"mel_{nMels}.npy");
                loadedMelFilter = torch.tensor(data, shape.Select(x => (long)x).ToArray(), dtype: torch.float32).to(device);
            }
            return loadedMelFilter; 
        }

        private static (float[] data, int[] shape) LoadNpyFromNpz(string npzPath, string entryName)
        {
            using var archive = ZipFile.OpenRead(npzPath);
            var entry = archive.GetEntry(entryName)
                        ?? archive.Entries.FirstOrDefault(e => string.Equals(e.Name, entryName, StringComparison.OrdinalIgnoreCase))
                        ?? throw new FileNotFoundException($"Entry '{entryName}' not found in {npzPath}.");
            using var stream = entry.Open();
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: false);

            var magic = reader.ReadBytes(6);
            if (magic.Length != 6 || magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
                throw new InvalidDataException($"{entryName} is not a NumPy .npy array.");

            var major = reader.ReadByte();
            _ = reader.ReadByte();
            int headerLen = major switch
            {
                1 => reader.ReadUInt16(),
                2 or 3 => (int)reader.ReadUInt32(),
                _ => throw new InvalidDataException($"Unsupported .npy version {major}.")
            };

            var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLen));
            if (!header.Contains("'descr': '<f4'", StringComparison.Ordinal) &&
                !header.Contains("\"descr\": \"<f4\"", StringComparison.Ordinal))
                throw new InvalidDataException($"{entryName} must contain little-endian float32 data.");
            if (header.Contains("True", StringComparison.Ordinal))
                throw new InvalidDataException($"{entryName} uses Fortran order, which is not supported.");

            var shapeStart = header.IndexOf('(');
            var shapeEnd = header.IndexOf(')', shapeStart + 1);
            if (shapeStart < 0 || shapeEnd < 0)
                throw new InvalidDataException($"{entryName} does not contain a valid shape header.");

            var shape = header.Substring(shapeStart + 1, shapeEnd - shapeStart - 1)
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => int.Parse(s.Trim()))
                .ToArray();
            var count = shape.Aggregate(1, (acc, dim) => acc * dim);
            var bytes = reader.ReadBytes(count * sizeof(float));
            if (bytes.Length != count * sizeof(float))
                throw new EndOfStreamException($"{entryName} ended before all float32 data could be read.");

            var data = new float[count];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return (data, shape);
        }
        private static torch.Tensor GetHannWindow( DeviceType device)
        {
            bool isLoaded = loadedHannWindow is not null;
            if (!isLoaded)
            {
                loadedHannWindow = torch.hann_window(NFFT).to( device );
            }
            return loadedHannWindow;
        }
        public static torch.Tensor WhisperLogMelSpectogram( torch.Tensor waveform, int nMels = 128 )
        {
            if( nMels  != 128 && nMels != 80 ) 
                throw new Exception("Invalid nMels, only 80 and 128 are supported");

            var device = waveform.device_type; 
            var window = GetHannWindow(device);
            var melFilter = GetMelFilter(nMels, device);

            using var scope = torch.NewDisposeScope();

            var stft = torch.stft(waveform, NFFT, HopLength, window: window, return_complex: true).to(device);

            var magnitutes = stft[.., .., ..^1].abs().pow(2);

            var melSpectrogram = torch.matmul(melFilter, magnitutes);

            var logSpec = torch.clamp( melSpectrogram, min: 1e-10f).log10();
            var compressed = torch.maximum(logSpec, logSpec.max() - 8.0);
            var final = ( compressed + 4.0) / 4.0;

            return scope.MoveToOuter(final.detach().to(device));
        }
    }
}
