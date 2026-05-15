using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using CosyVoiceNet.Tools;

namespace CosyVoiceNet.Tools
{
    // C# port of CosyVoice/tools/extract_embedding.py
    // - Reads wav.scp and utt2spk
    // - Computes kaldi-like fbank via AudioUtils.Fbank
    // - Runs ONNX model to extract embeddings
    public class ExtractEmbedding
    {
        private readonly InferenceSession? _session;
        private readonly dynamic? _jitModule;
        private readonly int _numThreads;

        public ExtractEmbedding(string modelPath, int numThreads = 8)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException("model not found", modelPath);
            _session = null;
            _jitModule = null;
            var ext = Path.GetExtension(modelPath)?.ToLowerInvariant();
            // if TorchScript image provided (.pt), prefer torch.jit load
            if (ext == ".pt")
            {
                try
                {
                    // load TorchScript module
                    _jitModule = TorchSharp.torch.jit.load(modelPath);
                }
                catch
                {
                    _jitModule = null;
                }
            }

            if (_jitModule == null)
            {
                var opts = new SessionOptions();
                opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                opts.IntraOpNumThreads = 1;
                // prefer CUDA provider if available
                try
                {
                    if (TorchSharp.torch.cuda.is_available())
                    {
                        _session = new InferenceSession(modelPath, opts);
                        // InferenceSession will pick providers based on environment; this is sufficient
                    }
                    else
                    {
                        _session = new InferenceSession(modelPath, opts);
                    }
                }
                catch
                {
                    _session = null;
                }
            }
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

        public Dictionary<string, string> LoadUtt2Spk(string dir)
        {
            var path = Path.Combine(dir, "utt2spk");
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
        //    var utt2spk = LoadUtt2Spk(dir);
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
        //                    var audio = AudioUtils.load_wav(wavPath, 16000);
        //                    var feat = AudioUtils.fbank(audio, 16000, numMels: 80);
        //                    // convert feat [T, M] to float[] flat row-major
        //                    //int T = feat.GetLength(0); int M = feat.GetLength(1);
        //                    //var data = new float[T * M];
        //                    //for (int i = 0; i < T; i++) for (int j = 0; j < M; j++) data[i * M + j] = feat[i, j];

        //                    // create tensor input: assuming model expects [1, T, M]
        //                    if ((_jitModule as object) != null)
        //                    {
        //                        try
        //                        {
        //                            //var dims = new long[] { 1, T, M };
        //                            //var t = TorchSharp.torch.tensor(data, dtype: TorchSharp.torch.ScalarType.Float32).view(dims);
        //                            var outObj = _jitModule.forward(t);
        //                            try
        //                            {
        //                                var outTensor = (TorchSharp.torch.Tensor)outObj;
        //                                results[utt] = outTensor.data<float>().ToArray();
        //                            }
        //                            catch
        //                            {
        //                                // try to handle tuple outputs
        //                                try
        //                                {
        //                                    var firstOut = outObj[0];
        //                                    var outTensor = (TorchSharp.torch.Tensor)firstOut;
        //                                    results[utt] = outTensor.data<float>().ToArray();   
        //                                }
        //                                catch { }
        //                            }
        //                        }
        //                        catch { }
        //                    }
        //                    else if ((_session as object) != null)
        //                    {
        //                        try
        //                        {
        //                            var namedInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(data, new[] { 1, T, M })) };
        //                            using var outputs = _session.Run(namedInputs);
        //                            var first = outputs.FirstOrDefault();
        //                            if (first != null)
        //                            {
        //                                var tensor = first.AsTensor<float>();
        //                                results[utt] = tensor.ToArray();
        //                            }
        //                        }
        //                        catch { }
        //                    }
        //                }
        //                catch { }
        //            }
        //        }));
        //    }
        //    Task.WaitAll(tasks.ToArray());

        //    // aggregate by speaker mean
        //    var spkMap = new Dictionary<string, List<float[]>>();
        //    foreach (var kv in results) {
        //        var spk = utt2spk.ContainsKey(kv.Key) ? utt2spk[kv.Key] : "default";
        //        if (!spkMap.ContainsKey(spk)) spkMap[spk] = new List<float[]>();
        //        spkMap[spk].Add(kv.Value);
        //    }
        //    var spk2embedding = new Dictionary<string, float[]>();
        //    foreach (var kv in spkMap)
        //    {
        //        var list = kv.Value;
        //        int dim = list[0].Length;
        //        var mean = new float[dim];
        //        foreach (var v in list) for (int i = 0; i < dim; i++) mean[i] += v[i];
        //        for (int i = 0; i < dim; i++) mean[i] /= list.Count;
        //        spk2embedding[kv.Key] = mean;
        //    }

        //    // save files as JSON mapping for interoperability with C# tools
        //    var uttOut = Path.Combine(dir, "utt2embedding.pt.json");
        //    var spkOut = Path.Combine(dir, "spk2embedding.pt.json");
        //    File.WriteAllText(uttOut, System.Text.Json.JsonSerializer.Serialize(results));
        //    File.WriteAllText(spkOut, System.Text.Json.JsonSerializer.Serialize(spk2embedding));

        //    return results.ToDictionary(k => k.Key, v => v.Value);
        //}
    }
}

// Equivalent Python file: cosyvoice/tools/extract_embedding.py
