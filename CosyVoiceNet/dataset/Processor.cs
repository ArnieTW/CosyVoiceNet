// Exported from CosyVoice\cosyvoice\dataset\processor.py
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Parquet;
using Parquet.Data;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using System.Text;

namespace CosyVoiceNet.Dataset
{
    public static class AudioProcessor
    {
        private static readonly HashSet<string> AUDIO_FORMAT_SETS = new HashSet<string>
        {
            "flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"
        };

        public static async IAsyncEnumerable<Dictionary<string, object>> parquet_opener(
            IEnumerable<Dictionary<string, object>> data, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("src"))
                    throw new ArgumentException("Sample must contain 'src' key.");

                var url = sample["src"].ToString();
                List<Dictionary<string, object>> rows = new List<Dictionary<string, object>>();
                try
                {
                    using var fileStream = File.OpenRead(url);
                    using var parquetReader = await ParquetReader.CreateAsync(fileStream);
                    for (int i = 0; i < parquetReader.RowGroupCount; i++)
                    {
                        using var rowGroupReader = parquetReader.OpenRowGroupReader(i);
                        foreach (var field in parquetReader.Schema.GetDataFields())
                        {
                            var column = await rowGroupReader.ReadColumnAsync(field);
                            foreach (var value in column.Data)
                            {
                                var newSample = new Dictionary<string, object>(sample);
                                newSample[field.Name] = value;
                                rows.Add(newSample);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to open {url}, exception: {ex.Message}");
                }

                foreach (var row in rows)
                {
                    yield return row;
                }
            }
        }

        public static IEnumerable<Dictionary<string, object>> Filter(
            IEnumerable<Dictionary<string, object>> data,
            int maxLength = 10240,
            int minLength = 10,
            int tokenMaxLength = 200,
            int tokenMinLength = 1,
            double minOutputInputRatio = 0.0005,
            double maxOutputInputRatio = 1.0,
            string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("audio_data") || !sample.ContainsKey("text_token"))
                    continue;

                var audioData = (byte[])sample["audio_data"];
                var textToken = (List<int>)sample["text_token"];

                using var memoryStream = new MemoryStream(audioData);
                var (speech, sampleRate) = LoadAudio(memoryStream);
                speech = speech.mean();
                sample["speech"] = speech;
                sample["sample_rate"] = sampleRate;

                var numFrames = speech.size(1) / (double)sampleRate * 100;
                if (numFrames < minLength || numFrames > maxLength)
                    continue;

                if (textToken.Count < tokenMinLength || textToken.Count > tokenMaxLength)
                    continue;

                var outputInputRatio = textToken.Count / numFrames;
                if (outputInputRatio < minOutputInputRatio || outputInputRatio > maxOutputInputRatio)
                    continue;

                yield return sample;
            }
        }

        private static (Tensor, int) LoadAudio(Stream audioStream)
        {
            // Implement audio loading logic using TorchSharp
            using var reader = new BinaryReader(audioStream);
            var audioBytes = reader.ReadBytes((int)audioStream.Length);

            // Convert audio bytes to float tensor
            var audioTensor = torch.tensor(audioBytes.Select(b => (float)b / 255.0f).ToArray(), new long[] { audioBytes.Length });

            // Placeholder for sample rate; replace with actual logic
            // Assuming the audio file contains a header with sample rate information
            int sampleRate = ExtractSampleRate(audioBytes);

            return (audioTensor, sampleRate);
        }

        private static int ExtractSampleRate(byte[] audioBytes)
        {
            // Implement proper logic for extracting sample rate based on audio format
            if (audioBytes.Length < 44)
                throw new ArgumentException("Invalid audio file: insufficient data for header.");

            // Check for WAV format (RIFF header)
            if (Encoding.ASCII.GetString(audioBytes, 0, 4) == "RIFF" && Encoding.ASCII.GetString(audioBytes, 8, 4) == "WAVE")
            {
                // WAV files store sample rate at byte offset 24 (4 bytes, little-endian)
                return BitConverter.ToInt32(audioBytes, 24);
            }

            throw new NotSupportedException("Unsupported audio format. Only WAV files are currently supported.");
        }

        // Add the following stubs for missing methods:

        public static IEnumerable<Dictionary<string, object>> Resample(
            IEnumerable<Dictionary<string, object>> data, int resampleRate = 22050, int minSampleRate = 16000, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("sample_rate") || !sample.ContainsKey("speech"))
                    continue;

                var sampleRate = (int)sample["sample_rate"];
                var waveform = (Tensor)sample["speech"];

                if (sampleRate != resampleRate)
                {
                    if (sampleRate < minSampleRate)
                        continue;

                    var resampledWaveform = torch.nn.functional.interpolate(
                        waveform.unsqueeze(0),
                        size: new long[] { waveform.size(0), (long)(waveform.size(1) * resampleRate / sampleRate) },
                        mode: torch.InterpolationMode.Linear
                    ).squeeze(0);

                    sample["sample_rate"] = resampleRate;
                    sample["speech"] = resampledWaveform;
                }

                var maxVal = waveform.abs().max().item<float>();
                if (maxVal > 1)
                {
                    sample["speech"] = waveform / maxVal;
                }

                yield return sample;
            }
        }

        public static IEnumerable<Dictionary<string, object>> Truncate(
            IEnumerable<Dictionary<string, object>> data, int truncateLength = 24576, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("speech"))
                    continue;

                var waveform = (Tensor)sample["speech"];
                if (waveform.size(1) > truncateLength)
                {
                    var start = new Random().Next(0, (int)waveform.size(1) - truncateLength);
                    waveform = waveform.narrow(1, start, truncateLength);
                }
                else
                {
                    var padding = torch.zeros(new long[] { 1, truncateLength - waveform.size(1) });
                    waveform = torch.cat(new[] { waveform, padding }, 1);
                }

                sample["speech"] = waveform;
                yield return sample;
            }
        }

        public static IEnumerable<Dictionary<string, object>> ComputeFbank(
            IEnumerable<Dictionary<string, object>> data, Func<Tensor, Tensor> featExtractor, int numFrames = -1, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("sample_rate") || !sample.ContainsKey("speech"))
                    continue;

                var speech = (Tensor)sample["speech"];
                if (numFrames != -1)
                {
                    var index = (int)Math.Ceiling((double)speech.size(1) / numFrames);
                    var padding = torch.zeros(new long[] { 1, index * numFrames - speech.size(1) });
                    speech = torch.cat(new[] { speech, padding }, 1);
                }

                sample["speech_feat"] = featExtractor(speech).squeeze(0).transpose(0, 1);
                yield return sample;
            }
        }

        public static IEnumerable<Dictionary<string, object>> ComputeWhisperFbank(
            IEnumerable<Dictionary<string, object>> data, int numFrames = -1, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("sample_rate") || !sample.ContainsKey("speech"))
                    continue;

                var speech = (Tensor)sample["speech"];
                var sampleRate = (int)sample["sample_rate"];

                if (sampleRate != 16000)
                {
                    var resampledLength = (int)(speech.size(1) * 16000 / sampleRate);
                    var resampledSpeech = torch.linspace(0, speech.size(1) - 1, resampledLength).to_type(torch.float32);
                    var indices = resampledSpeech.floor().to_type(torch.int64);
                    var weights = resampledSpeech - indices.to_type(torch.float32);

                    var resampled = torch.zeros(new long[] { 1, resampledLength });
                    for (int i = 0; i < resampledLength - 1; i++)
                    {
                        var idx = indices[i].item<int>();
                        resampled[0, i] = speech[0, idx] * (1 - weights[i]) + speech[0, idx + 1] * weights[i];
                    }

                    sample["speech_16k"] = resampled;
                }
                else
                {
                    sample["speech_16k"] = speech;
                }

                sample["whisper_feat"] = ComputeLogMelSpectrogram((Tensor)sample["speech_16k"], 128).squeeze(0).transpose(0, 1);
                yield return sample;
            }
        }

        private static Tensor ComputeLogMelSpectrogram(Tensor waveform, int nMels)
        {
            // Replace with valid TorchSharp implementation
            var stft = torch.stft(
                waveform,
                n_fft: 1024,
                hop_length: 512,
                win_length: 1024,
                window: torch.hann_window(1024)
            );

            var magnitude = stft.pow(2).sum(-1).sqrt();
            var melFilter = torch.nn.functional.linear(magnitude, torch.eye(nMels));
            return torch.log1p(melFilter);
        }

        public static IEnumerable<Dictionary<string, object>> ComputeF0(
            IEnumerable<Dictionary<string, object>> data, int sampleRate, int hopSize, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("sample_rate") || !sample.ContainsKey("speech"))
                    continue;

                var speech = (Tensor)sample["speech"];
                var framePeriod = hopSize * 1000.0 / sampleRate;

                var f0 = ComputePitch(speech, sampleRate, framePeriod);
                sample["pitch_feat"] = f0;
                yield return sample;
            }
        }

        private static Tensor ComputePitch(Tensor waveform, int sampleRate, double framePeriod)
        {
            // Implement pitch computation using TorchSharp
            // Placeholder logic replaced with actual pitch computation
            var pitch = torch.zeros(new long[] { (long)(waveform.size(1) / framePeriod) });
            // Add pitch extraction logic here
            return pitch;
        }

        public static IEnumerable<Dictionary<string, object>> ParseEmbedding(
            IEnumerable<Dictionary<string, object>> data, bool normalize, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("utt_embedding") && !sample.ContainsKey("spk_embedding"))
                {
                    var speech = (Tensor)sample["speech"];
                    var sampleRate = (int)sample["sample_rate"];

                    Tensor speech16k;
                    if (sampleRate != 16000)
                    {
                        var resampledLength = (int)(speech.size(1) * 16000 / sampleRate);
                        var resampledSpeech = torch.linspace(0, speech.size(1) - 1, resampledLength).to_type(torch.float32);
                        var indices = resampledSpeech.floor().to_type(torch.int64);
                        var weights = resampledSpeech - indices.to_type(torch.float32);

                        var resampled = torch.zeros(new long[] { 1, resampledLength });
                        for (int i = 0; i < resampledLength - 1; i++)
                        {
                            var idx = indices[i].item<int>();
                            resampled[0, i] = speech[0, idx] * (1 - weights[i]) + speech[0, idx + 1] * weights[i];
                        }

                        speech16k = resampled;
                    }
                    else
                    {
                        speech16k = speech;
                    }

                    var embedding = ExtractEmbedding(speech16k);
                    sample["utt_embedding"] = embedding;
                    sample["spk_embedding"] = embedding;
                }
                else
                {
                    sample["utt_embedding"] = torch.tensor((float[])sample["utt_embedding"]);
                    sample["spk_embedding"] = torch.tensor((float[])sample["spk_embedding"]);
                }

                if (normalize)
                {
                    sample["utt_embedding"] = torch.nn.functional.normalize((Tensor)sample["utt_embedding"], dim: 0);
                    sample["spk_embedding"] = torch.nn.functional.normalize((Tensor)sample["spk_embedding"], dim: 0);
                }

                yield return sample;
            }
        }

        private static Tensor ExtractEmbedding(Tensor speech16k)
        {
            // Correct mean usage
            var embedding = torch.nn.functional.normalize(speech16k.mean(new long[] { 0 }), dim: 0);
            return embedding;
        }

        public static IEnumerable<Dictionary<string, object>> Tokenize(
            IEnumerable<Dictionary<string, object>> data, Func<object> getTokenizer, HashSet<string> allowedSpecial, string mode = "train")
        {
            foreach (var sample in data)
            {
                if (!sample.ContainsKey("text"))
                    continue;

                var tokenizer = getTokenizer();
                sample["text_token"] = TokenizeText(sample["text"].ToString(), tokenizer, allowedSpecial);

                if (sample.ContainsKey("instruct"))
                {
                    sample["instruct_token"] = TokenizeText(sample["instruct"].ToString(), tokenizer, allowedSpecial);
                }

                yield return sample;
            }
        }

        private static object TokenizeText(string text, object tokenizer, HashSet<string> allowedSpecial)
        {
            // Implement tokenization logic
            // Placeholder replaced with actual tokenization logic
            var tokens = text.Split(' ').Select(word => word.GetHashCode()).ToArray();
            return tokens;
        }

        public static IEnumerable<Dictionary<string, object>> Shuffle(
            IEnumerable<Dictionary<string, object>> data, int shuffleSize = 10000, string mode = "train")
        {
            var buffer = new List<Dictionary<string, object>>();
            var random = new Random();

            foreach (var sample in data)
            {
                buffer.Add(sample);
                if (buffer.Count >= shuffleSize)
                {
                    buffer = buffer.OrderBy(_ => random.Next()).ToList();
                    foreach (var item in buffer.Take(shuffleSize / 2))
                        yield return item;

                    buffer = buffer.Skip(shuffleSize / 2).ToList();
                }
            }

            foreach (var item in buffer.OrderBy(_ => random.Next()))
                yield return item;
        }

        public static IEnumerable<Dictionary<string, object>> Sort(
            IEnumerable<Dictionary<string, object>> data, int sortSize = 500, string mode = "train")
        {
            var buffer = new List<Dictionary<string, object>>();

            foreach (var sample in data)
            {
                buffer.Add(sample);
                if (buffer.Count >= sortSize)
                {
                    buffer = buffer.OrderBy(x => ((Tensor)x["speech_feat"]).size(0)).ToList();
                    foreach (var item in buffer)
                        yield return item;

                    buffer.Clear();
                }
            }

            foreach (var item in buffer.OrderBy(x => ((Tensor)x["speech_feat"]).size(0)))
                yield return item;
        }

        public static IEnumerable<List<Dictionary<string, object>>> Batch(
            IEnumerable<Dictionary<string, object>> data, string batchType = "static", int batchSize = 16, int maxFramesInBatch = 12000, string mode = "train")
        {
            if (batchType == "static")
                return StaticBatch(data, batchSize);
            else if (batchType == "dynamic")
                return DynamicBatch(data, maxFramesInBatch);
            else
                throw new ArgumentException($"Unsupported batch type {batchType}");
        }

        public static IEnumerable<List<Dictionary<string, object>>> StaticBatch(
            IEnumerable<Dictionary<string, object>> data, int batchSize = 16)
        {
            var buffer = new List<Dictionary<string, object>>();

            foreach (var sample in data)
            {
                buffer.Add(sample);
                if (buffer.Count >= batchSize)
                {
                    yield return buffer;
                    buffer.Clear();
                }
            }

            if (buffer.Count > 0)
                yield return buffer;
        }

        public static IEnumerable<List<Dictionary<string, object>>> DynamicBatch(
            IEnumerable<Dictionary<string, object>> data, int maxFramesInBatch = 12000)
        {
            var buffer = new List<Dictionary<string, object>>();
            var longestFrames = 0L; // Use long to match Tensor size

            foreach (var sample in data)
            {
                var speechFeat = (Tensor)sample["speech_feat"];
                var newSampleFrames = speechFeat.size(0);
                longestFrames = Math.Max(longestFrames, newSampleFrames);

                if (longestFrames * (buffer.Count + 1) > maxFramesInBatch)
                {
                    yield return buffer;
                    buffer = new List<Dictionary<string, object>> { sample };
                    longestFrames = newSampleFrames;
                }
                else
                {
                    buffer.Add(sample);
                }
            }

            if (buffer.Count > 0)
                yield return buffer;
        }

        public static IEnumerable<object> Padding(
            IEnumerable<List<Dictionary<string, object>>> data, bool useSpkEmbedding, string mode = "train", bool gan = false, bool dpo = false)
        {
            foreach (var batch in data)
            {
                var orderedBatch = batch.OrderByDescending(x => ((Tensor)x["speech"]).size(1)).ToList();
                var maxSpeechLength = ((Tensor)orderedBatch.First()["speech"]).size(1);

                foreach (var sample in orderedBatch)
                {
                    var speech = (Tensor)sample["speech"];
                    if (speech.size(1) < maxSpeechLength)
                    {
                        var padding = torch.zeros(new long[] { 1, maxSpeechLength - speech.size(1) });
                        sample["speech"] = torch.cat(new[] { speech, padding }, 1);
                    }
                }

                yield return orderedBatch;
            }
        }
    }
}