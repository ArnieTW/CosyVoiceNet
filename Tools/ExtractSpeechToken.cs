using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using CosyVoiceNet.Tools;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CosyVoiceNet.Tools
{
    // Port of CosyVoice/tools/extract_speech_token.py
    public class ExtractSpeechToken
    {
        private readonly InferenceSession _session;
        private readonly int _numThreads;

        public ExtractSpeechToken(string onnxPath, int numThreads = 8)
        {
            if (!File.Exists(onnxPath)) throw new FileNotFoundException("ONNX model not found", onnxPath);
            var opts = new SessionOptions();
            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            opts.IntraOpNumThreads = 1;
            // prefer default providers; OnnxRuntime will select CUDA if available via environment
            _session = new InferenceSession(onnxPath, opts);
            _numThreads = Math.Max(1, numThreads);
        }

        public Dictionary<string, string> LoadWavScp(string dir)
        {
            var path = Path.Combine(dir, "wav.scp");
            var dict = new Dictionary<string, string>();
            foreach (var l in File.ReadAllLines(path))
            {
                var parts = l.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2) dict[parts[0]] = parts[1];
            }
            return dict;
        }

        //public Dictionary<string, float[]> Run(string dir)
        //{
        //    var utt2wav = LoadWavScp(dir);
        //    var results = new ConcurrentDictionary<string, float[]>();
        //    var bag = new ConcurrentBag<string>(utt2wav.Keys);
        //    var tasks = new List<Task>();
        //    for (int t = 0; t < _numThreads; t++)
        //    {
        //        tasks.Add(Task.Run(() =>
        //        {
        //            while (bag.TryTake(out var utt))
        //            {
        //                try
        //                {
        //                    var wavPath = utt2wav[utt];
        //                    var speech = AudioUtils.load_wav(wavPath, 16000);
        //                    var (samples, sr, channels) = AudioUtils.ReadWav(wavPath);
        //                    if (sr != 16000) samples = AudioUtils.ResampleLinear(samples, sr, 16000, channels);
        //                    var mono = AudioUtils.ToMono(samples, channels);
        //                    if (mono.Length / 16000.0 > 30.0) { results[utt] = new float[0]; continue; }
        //                    var mel = AudioUtils.LogMelSpectrogram(mono, 16000, nMels: 128);
        //                    int nMels = mel.GetLength(0); int T = mel.GetLength(1);
        //                    // construct tensor as [1, n_mels, T]
        //                    float[] data = new float[nMels * T];
        //                    for (int i = 0; i < nMels; i++) for (int j = 0; j < T; j++) data[i * T + j] = mel[i, j];
        //                    var namedInputs = new List<NamedOnnxValue>
        //                    {
        //                        NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(data, new[] {1, nMels, T})),
        //                        NamedOnnxValue.CreateFromTensor("input_len", new DenseTensor<int>(new int[]{T}, new[] {1}))
        //                    };
        //                    using var outputs = _session.Run(namedInputs);
        //                    var first = outputs.FirstOrDefault();
        //                    if (first != null)
        //                    {
        //                        var tensor = first.AsTensor<float>();
        //                        results[utt] = tensor.ToArray();
        //                    }
        //                }
        //                catch { }
        //            }
        //        }));
        //    }
        //    Task.WaitAll(tasks.ToArray());

        //    // save
        //    File.WriteAllText(Path.Combine(dir, "utt2speech_token.pt.json"), System.Text.Json.JsonSerializer.Serialize(results));
        //    return results.ToDictionary(k => k.Key, v => v.Value);
        //}
    }
}
// Equivalent Python file: cosyvoice/tools/extract_speech_token.py
