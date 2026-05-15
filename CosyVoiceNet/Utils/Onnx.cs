// Equivalent Python file: cosyvoice/utils/onnx.py
using System;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TorchSharp;
using System.Collections.Generic;
using System.Linq;

namespace CosyVoiceNet.Utils
{
    public class SpeechTokenExtractor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _maxLen = 10 * 16000;

        public SpeechTokenExtractor(string modelPath, bool useCuda = false, int deviceId = 0)
        {
            var opts = new SessionOptions();
            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            opts.IntraOpNumThreads = 1;
            if (useCuda)
            {
                opts.AppendExecutionProvider_CUDA(deviceId);
            }
            _session = new InferenceSession(modelPath, opts);
        }

        public (TorchSharp.torch.Tensor tokens, TorchSharp.torch.Tensor lengths) Inference(TorchSharp.torch.Tensor feat, TorchSharp.torch.Tensor featLengths)
        {
            if ((long)feat.shape[1] * (long)feat.shape[2] > _maxLen)
            {
                var maxSamples = _maxLen;
                var start = new Random().Next(0, Math.Max(1, (int)feat.shape[1] - maxSamples));
                feat = feat.narrow(1, start, maxSamples);
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", ConvertToDenseTensor<float>(feat.transpose(1, 2))),
                NamedOnnxValue.CreateFromTensor("lengths", ConvertToDenseTensor<int>(featLengths))
            };

            using var results = _session.Run(inputs);
            var tokens = torch.tensor(results.First().AsEnumerable<int>().ToArray(), dtype: torch.int32);
            var lengths = featLengths.div(4).to(torch.int32);
            return (tokens, lengths);
        }

        private DenseTensor<T> ConvertToDenseTensor<T>(TorchSharp.torch.Tensor tensor) where T : struct
        {
            var data = tensor.cpu().to(torch.float32).data<float>().ToArray();
            return new DenseTensor<T>(data.Cast<T>().ToArray(), tensor.shape.Select(dim => (int)dim).ToArray());
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }

    public class EmbeddingExtractor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _maxLen = 10 * 16000;

        public EmbeddingExtractor(string modelPath)
        {
            var opts = new SessionOptions();
            opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            opts.IntraOpNumThreads = 1;
            _session = new InferenceSession(modelPath, opts);
        }

        public TorchSharp.torch.Tensor Inference(TorchSharp.torch.Tensor speech)
        {
            if (speech.shape[1] > _maxLen)
            {
                var startIndex = new Random().Next(0, (int)speech.shape[1] - _maxLen);
                speech = speech.narrow(1, startIndex, _maxLen);
            }

            var feat = ProcessInput(speech);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", ConvertToDenseTensor<float>(feat.unsqueeze(0)))
            };

            using var results = _session.Run(inputs);
            var embedding = torch.tensor(results.First().AsEnumerable<float>().ToArray(), dtype: torch.float32);
            return embedding;
        }

        public TorchSharp.torch.Tensor ProcessInput(TorchSharp.torch.Tensor speech)
        {
            var feat = speech - speech.mean(new long[] { 0L }, keepdim: true);
            return feat;
        }

        private DenseTensor<T> ConvertToDenseTensor<T>(TorchSharp.torch.Tensor tensor) where T : struct
        {
            var data = tensor.cpu().to(torch.float32).data<float>().ToArray();
            return new DenseTensor<T>(data.Cast<T>().ToArray(), tensor.shape.Select(dim => (int)dim).ToArray());
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
