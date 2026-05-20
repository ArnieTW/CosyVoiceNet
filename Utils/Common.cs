using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Utils
{
    public enum CosyVoiceSamplingBackend
    {
        LogitsDevice = 0,
        Cpu = 1,
        Cuda = 2,
        Deterministic = 3
    }

    public static class Common
    {
        public const int IGNORE_ID = -1;
        private static readonly object SamplingRandomLock = new();
        private static ulong SamplingRandomState = MixSeed(0);
        private static bool? CudaSamplingAvailable;
        public static CosyVoiceSamplingBackend SamplingBackend { get; set; } = CosyVoiceSamplingBackend.LogitsDevice;

        public static readonly List<string> InstructList = new List<string>
        {
            "You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用东北话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用甘肃话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用贵州话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用河南话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用湖北话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用湖南话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用江西话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用闽南话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用宁夏话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用山西话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用陕西话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用山东话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用上海话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用四川话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用天津话表达。<|endofprompt|>",
            "You are a helpful assistant. 请用云南话表达。<|endofprompt|>",
            "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
            "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
            "You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>",
            "You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>",
            "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
            "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
            "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
            "You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>",
            "You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>"
        };

        public static Tensor PadList(List<Tensor> xs, int padValue)
        {
            if (xs == null || xs.Count == 0) return torch.zeros(new long[] { 0 });
            var maxLen = xs.Max(x => x.shape[0]);
            var B = xs.Count;
            var ndim = xs[0].shape.Length;
            Tensor padRes;
            var dtype = xs[0].dtype;
            var device = xs[0].device;
            if (ndim == 1)
            {
                padRes = torch.zeros(new long[] { B, maxLen }, dtype: dtype).to(device);
            }
            else if (ndim == 2)
            {
                padRes = torch.zeros(new long[] { B, maxLen, xs[0].shape[1] }, dtype: dtype).to(device);
            }
            else if (ndim == 3)
            {
                padRes = torch.zeros(new long[] { B, maxLen, xs[0].shape[1], xs[0].shape[2] }, dtype: dtype).to(device);
            }
            else throw new ArgumentException($"Unsupported ndim: {ndim}");
            padRes = padRes.fill_(padValue);
            for (int i = 0; i < B; i++)
            {
                var len = xs[i].shape[0];
                if (len == 0) continue;
                padRes[i].slice(0, 0, len, 1).copy_(xs[i]);
            }
            return padRes;
        }

        public static void SetAllRandomSeed(int seed)
        {
            ApplyTorchRandomSeed(seed);

            lock (SamplingRandomLock)
                SamplingRandomState = MixSeed(seed);
        }

        private static void ApplyTorchRandomSeed(int seed)
        {
            try
            {
                torch.InitializeDeviceType(DeviceType.CPU);
                torch.manual_seed(seed);
            }
            catch
            {
                // Keep YAML loading resilient; the model load path will report TorchSharp initialization failures.
            }

            try
            {
                if (!torch.TryInitializeDeviceType(DeviceType.CUDA) || !torch.cuda.is_available())
                    return;

                torch.cuda.manual_seed(seed);
                torch.cuda.manual_seed_all(seed);
            }
            catch
            {
                // CUDA seeding is best-effort because CPU fallback is valid when CUDA is unavailable.
            }
        }

        public static Tensor ThAccuracy(Tensor padOutputs, Tensor padTargets, int ignoreLabel)
        {
            var padPred = padOutputs.view(new long[] { padTargets.shape[0], padTargets.shape[1], padOutputs.shape[1] }).argmax(2);
            var mask = padTargets != ignoreLabel;
            var numerator = padPred.masked_select(mask).eq(padTargets.masked_select(mask)).sum();
            var denominator = mask.sum();
            return (numerator / denominator).detach();
        }

        public static int NucleusSampling(Tensor weightedScores, double topP = 0.8, int topK = 25)
        {
            var scores = weightedScores.detach().to(DeviceType.CPU).to_type(ScalarType.Float32).contiguous().data<float>().ToArray();
            var probabilities = Softmax(scores);
            var sorted = probabilities
                .Select((probability, index) => (probability, index))
                .OrderByDescending(item => item.probability)
                .ThenBy(item => item.index)
                .ToArray();

            var cumulativeProb = 0.0;
            var selectedProbabilities = new List<double>();
            var selectedIndices = new List<int>();
            foreach (var item in sorted)
            {
                if (cumulativeProb < topP && selectedIndices.Count < topK)
                {
                    var probability = item.probability;
                    cumulativeProb += probability;
                    selectedProbabilities.Add(probability);
                    selectedIndices.Add(item.index);
                }
                else
                {
                    break;
                }
            }

            if (selectedIndices.Count == 0)
                return ArgMax(scores);

            var sampledOffset = TorchMultinomial(selectedProbabilities, weightedScores.device);
            return selectedIndices[sampledOffset];
        }

        public static Tensor FadeInOut(Tensor fadeInMel, Tensor fadeOutMel, Tensor window)
        {
            var melOverlapLen = (int)(window.shape[0] / 2);
            var fadeInHead = fadeInMel.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(0, melOverlapLen) });
            var fadeOutTail = fadeOutMel.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(-melOverlapLen, null) });
            fadeInHead.copy_(fadeInHead * window[TensorIndex.Slice(0, melOverlapLen)] + fadeOutTail * window[TensorIndex.Slice(melOverlapLen, null)]);
            return fadeInMel;
        }

        public static Tensor MaskToBias(Tensor mask, ScalarType dtype)
        {
            if (mask.dtype != ScalarType.Bool) throw new ArgumentException("Mask must be of type Bool.");
            if (dtype != ScalarType.Float32 && dtype != ScalarType.BFloat16 && dtype != ScalarType.Float16)
                throw new ArgumentException("Invalid dtype for mask bias.");

            mask = mask.to(dtype);
            mask = (1.0 - mask) * -1.0e+10;
            return mask;
        }

        public static int GetPadding(int kernelSize, int dilation = 1)
        {
            return (kernelSize * dilation - dilation) / 2;
        }

        public static void InitWeights(torch.nn.Module module, double mean = 0.0, double std = 0.01)
        {
            if (module.GetType().Name.Contains("Conv"))
            {
                module.parameters().First().normal_(mean, std);
            }
        }

        public static int RasSampling(Tensor weightedScores, List<int> decodedTokens, bool sampling, double topP = 0.8, int topK = 25, int winSize = 10, double tauR = 0.1)
        {
            decodedTokens ??= new List<int>();
            int topId = NucleusSampling(weightedScores, topP, topK);
            int repNum = decodedTokens.TakeLast(winSize).Count(token => token == topId);

            if (repNum >= winSize * tauR)
            {
                weightedScores[topId] = double.NegativeInfinity;
                topId = RandomSampling(weightedScores, decodedTokens, sampling);
            }

            return topId;
        }

        public static int RandomSampling(Tensor weightedScores, List<int> decodedTokens, bool sampling)
        {
            var scores = weightedScores.detach().to(DeviceType.CPU).to_type(ScalarType.Float32).contiguous().data<float>().ToArray();
            var probabilities = Softmax(scores);
            return TorchMultinomial(probabilities, weightedScores.device);
        }

        private static double[] Softmax(float[] scores)
        {
            var max = double.NegativeInfinity;
            foreach (var score in scores)
            {
                if (!float.IsNaN(score) && score > max)
                    max = score;
            }

            if (double.IsNegativeInfinity(max))
                return Enumerable.Repeat(0.0, scores.Length).ToArray();

            var probabilities = new double[scores.Length];
            var sum = 0.0;
            for (var i = 0; i < scores.Length; i++)
            {
                var score = scores[i];
                if (float.IsNaN(score) || float.IsNegativeInfinity(score))
                {
                    probabilities[i] = 0.0;
                    continue;
                }

                var probability = Math.Exp(score - max);
                probabilities[i] = probability;
                sum += probability;
            }

            if (sum <= 0.0 || double.IsNaN(sum))
                return probabilities;

            for (var i = 0; i < probabilities.Length; i++)
                probabilities[i] /= sum;
            return probabilities;
        }

        private static int ArgMax(float[] scores)
        {
            var bestIndex = 0;
            var bestScore = float.NegativeInfinity;
            for (var i = 0; i < scores.Length; i++)
            {
                if (scores[i] > bestScore)
                {
                    bestScore = scores[i];
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static int DeterministicMultinomial(IReadOnlyList<double> probabilities)
        {
            var total = 0.0;
            for (var i = 0; i < probabilities.Count; i++)
                total += Math.Max(0.0, probabilities[i]);
            if (total <= 0.0 || double.IsNaN(total))
                return 0;

            var threshold = NextSamplingDouble() * total;
            var cumulative = 0.0;
            for (var i = 0; i < probabilities.Count; i++)
            {
                cumulative += Math.Max(0.0, probabilities[i]);
                if (threshold <= cumulative)
                    return i;
            }

            return probabilities.Count - 1;
        }

        private static int TorchMultinomial(IReadOnlyList<double> probabilities, Device device)
        {
            if (SamplingBackend == CosyVoiceSamplingBackend.Deterministic)
                return DeterministicMultinomial(probabilities);

            var values = new float[probabilities.Count];
            var total = 0.0;
            for (var i = 0; i < probabilities.Count; i++)
            {
                var probability = Math.Max(0.0, probabilities[i]);
                values[i] = (float)probability;
                total += probability;
            }

            if (total <= 0.0 || double.IsNaN(total))
                return 0;

            try
            {
                var sampleDevice = ResolveTorchSamplingDevice(device);
                using var probabilityTensor = torch.tensor(values, dtype: ScalarType.Float32, device: sampleDevice);
                using var sampled = probabilityTensor.multinomial(1, replacement: true);
                using var sampledCpu = sampled.to(DeviceType.CPU);
                return ScalarToInt(sampledCpu);
            }
            catch
            {
                return DeterministicMultinomial(probabilities);
            }
        }

        private static Device ResolveTorchSamplingDevice(Device logitsDevice)
        {
            return SamplingBackend switch
            {
                CosyVoiceSamplingBackend.Cpu => torch.CPU,
                CosyVoiceSamplingBackend.Cuda => CanUseCudaForSampling() ? torch.CUDA : torch.CPU,
                _ => logitsDevice.type == DeviceType.CUDA ? logitsDevice : torch.CPU
            };
        }

        private static bool CanUseCudaForSampling()
        {
            if (CudaSamplingAvailable.HasValue)
                return CudaSamplingAvailable.Value;

            lock (SamplingRandomLock)
            {
                if (CudaSamplingAvailable.HasValue)
                    return CudaSamplingAvailable.Value;

                try
                {
                    CudaSamplingAvailable = torch.TryInitializeDeviceType(DeviceType.CUDA) && torch.cuda.is_available();
                }
                catch
                {
                    CudaSamplingAvailable = false;
                }

                return CudaSamplingAvailable.Value;
            }
        }

        private static double NextSamplingDouble()
        {
            lock (SamplingRandomLock)
            {
                SamplingRandomState += 0x9E3779B97F4A7C15UL;
                var z = SamplingRandomState;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                z ^= z >> 31;
                return (z >> 11) * (1.0 / (1UL << 53));
            }
        }

        private static ulong MixSeed(int seed)
        {
            unchecked
            {
                var value = (ulong)(uint)seed + 0x9E3779B97F4A7C15UL;
                value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
                value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
                return value ^ (value >> 31);
            }
        }

        private static int ScalarToInt(Tensor value)
        {
            if (value is null || value.numel() == 0)
                return 0;

            var scalar = value.flatten()[0];
            return scalar.dtype switch
            {
                ScalarType.Int64 => checked((int)scalar.item<long>()),
                ScalarType.Int32 => scalar.item<int>(),
                ScalarType.Int16 => scalar.item<short>(),
                ScalarType.Byte => scalar.item<byte>(),
                ScalarType.Float32 => checked((int)scalar.item<float>()),
                ScalarType.Float64 => checked((int)scalar.item<double>()),
                _ => checked((int)scalar.to_type(ScalarType.Int64).item<long>())
            };
        }
    }

    public class TrtContextWrapper
    {
        private readonly ConcurrentQueue<object> _pool;
        private readonly dynamic _engine;

        public TrtContextWrapper(dynamic trtEngine, int trtConcurrent = 1, string device = "cuda:0")
        {
            _engine = trtEngine;
            _pool = new ConcurrentQueue<object>();
            for (int i = 0; i < trtConcurrent; i++)
            {
                var trtContext = trtEngine.create_execution_context();
                if (trtContext == null) throw new InvalidOperationException("Failed to create TRT context. Try reducing TRT concurrency.");
                _pool.Enqueue(trtContext);
            }
            if (_pool.IsEmpty) throw new InvalidOperationException("No available estimator context.");
        }

        public object AcquireEstimator()
        {
            if (_pool.TryDequeue(out var context)) return context;
            throw new InvalidOperationException("No available TRT context.");
        }

        public void ReleaseEstimator(object context)
        {
            _pool.Enqueue(context);
        }
    }
}

// Equivalent Python file: cosyvoice/utils/common.py

// This file is aligned with 'D:\Dev\NekoBot_LLM\CosyVoice\cosyvoice\utils\common.py'.
