// Exported from HuggingFace Qwen2ForCausalLM reference
using System;
using System.Collections.Generic;
using System.Diagnostics;
using CosyVoiceNet.cli;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.TorchSharpUtils
{
    public sealed class Qwen2LayerCache
    {
        public Tensor Key { get; init; }
        public Tensor Value { get; init; }
    }

    public sealed class Qwen2KvCache
    {
        public List<Qwen2LayerCache> Layers { get; } = new();
    }

    public sealed class Qwen2PreallocatedLayerCache
    {
        public Tensor Key { get; set; }
        public Tensor Value { get; set; }
        public long Length { get; set; }
    }

    public sealed class Qwen2PreallocatedKvCache
    {
        public Qwen2PreallocatedKvCache(long maxLength)
        {
            if (maxLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, "Qwen preallocated KV cache length must be positive.");
            MaxLength = maxLength;
        }

        public long MaxLength { get; }
        public List<Qwen2PreallocatedLayerCache> Layers { get; } = new();
        public long Length => Layers.Count > 0 ? Layers[0].Length : 0;

        public Qwen2PreallocatedLayerCache GetOrCreateLayer(int index)
        {
            while (Layers.Count <= index)
                Layers.Add(new Qwen2PreallocatedLayerCache());
            return Layers[index];
        }
    }

    // RMSNorm: weight * x / sqrt(mean(x^2) + eps)
    public class RMSNorm : nn.Module<Tensor, Tensor>
    {
        public Parameter weight;
        private readonly float eps;

        public RMSNorm(long hiddenSize, float eps = 1e-6f) : base("RMSNorm")
        {
            this.eps = eps;
            weight = new Parameter(torch.ones(hiddenSize));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var floatX = x.to(ScalarType.Float32);
            var variance = floatX.pow(2).mean(new long[] { -1 }, keepdim: true);
            var normed = floatX * (variance + eps).rsqrt();
            return (weight * normed).to(x.dtype);
        }
    }

    // Qwen2 Grouped-Query Attention with RoPE
    public class Qwen2Attention : nn.Module<Tensor, Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> q_proj;
        public readonly nn.Module<Tensor, Tensor> k_proj;
        public readonly nn.Module<Tensor, Tensor> v_proj;
        public readonly nn.Module<Tensor, Tensor> o_proj;

        private readonly int num_heads;
        private readonly int num_kv_heads;
        private readonly int head_dim;
        private readonly int hidden_size;
        private readonly float rope_theta;
        private static readonly object RotaryCacheLock = new();
        private static readonly Dictionary<RotaryCacheKey, RotaryCacheEntry> RotaryCache = new();
        public ICosyVoiceProfiler? Profiler { get; set; }
        public QwenAttentionBackend AttentionBackend { get; set; } = QwenAttentionBackend.Auto;
        public int LayerIndex { get; set; } = -1;

        public Qwen2Attention(int hiddenSize, int numHeads, int numKvHeads, float rmsNormEps, float ropeTheta = 10000.0f, bool qkvBias = true)
            : base("Qwen2Attention")
        {
            hidden_size = hiddenSize;
            num_heads = numHeads;
            num_kv_heads = numKvHeads;
            head_dim = hiddenSize / numHeads;
            rope_theta = ropeTheta;

            q_proj = nn.Linear(hiddenSize, numHeads * head_dim, hasBias: qkvBias);
            k_proj = nn.Linear(hiddenSize, numKvHeads * head_dim, hasBias: qkvBias);
            v_proj = nn.Linear(hiddenSize, numKvHeads * head_dim, hasBias: qkvBias);
            o_proj = nn.Linear(hiddenSize, hiddenSize, hasBias: false);
            RegisterComponents();
        }

        private static (Tensor, Tensor) ApplyRotaryEmb(Tensor q, Tensor k, long seqLen, Device device, float ropeTheta, long positionOffset = 0)
        {
            int headDim = (int)q.shape[^1];
            var (cosBase, sinBase) = GetRotaryCache(device, headDim, ropeTheta, positionOffset + seqLen);
            var cos = cosBase.narrow(2, positionOffset, seqLen);
            var sin = sinBase.narrow(2, positionOffset, seqLen);

            static Tensor RotateHalf(Tensor x)
            {
                int h = (int)(x.shape[^1] / 2);
                var x1 = x[.., .., .., ..h];
                var x2 = x[.., .., .., h..];
                return torch.cat(new[] { -x2, x1 }, dim: -1);
            }

            var qRot = (q * cos) + (RotateHalf(q) * sin);
            var kRot = (k * cos) + (RotateHalf(k) * sin);
            return (qRot.to(q.dtype), kRot.to(k.dtype));
        }

        private static (Tensor cos, Tensor sin) GetRotaryCache(Device device, int headDim, float ropeTheta, long requiredLength)
        {
            var key = new RotaryCacheKey(device.ToString(), headDim, ropeTheta);
            lock (RotaryCacheLock)
            {
                if (RotaryCache.TryGetValue(key, out var existing) && existing.Length >= requiredLength)
                    return (existing.Cos, existing.Sin);

                var newLength = Math.Max(requiredLength, existing?.Length * 2 ?? 128);
                var invFreq = 1f / torch.pow(
                    ropeTheta,
                    torch.arange(0, headDim, 2, dtype: ScalarType.Float32, device: device) / headDim);
                var pos = torch.arange(0, newLength, dtype: ScalarType.Float32, device: device);
                var freqs = torch.outer(pos, invFreq);
                var emb = torch.cat(new[] { freqs, freqs }, dim: -1);
                var cos = emb.cos().unsqueeze(0).unsqueeze(0).DetachFromDisposeScope();
                var sin = emb.sin().unsqueeze(0).unsqueeze(0).DetachFromDisposeScope();
                invFreq.Dispose();
                pos.Dispose();
                freqs.Dispose();
                emb.Dispose();

                existing?.Cos.Dispose();
                existing?.Sin.Dispose();
                RotaryCache[key] = new RotaryCacheEntry(cos, sin, newLength);
                return (cos, sin);
            }
        }

        private readonly record struct RotaryCacheKey(string Device, int HeadDim, float RopeTheta);

        private sealed record RotaryCacheEntry(Tensor Cos, Tensor Sin, long Length);

        // forward(hidden_states, attn_mask) → output
        public override Tensor forward(Tensor hidden_states, Tensor attention_mask)
        {
            long bsz = hidden_states.shape[0];
            long seqLen = hidden_states.shape[1];

            var q = q_proj.forward(hidden_states).view(bsz, seqLen, num_heads, head_dim).transpose(1, 2);
            var k = k_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);
            var v = v_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);

            (q, k) = ApplyRotaryEmb(q, k, seqLen, hidden_states.device, rope_theta);

            // Expand KV heads to match Q heads if using GQA
            if (num_kv_heads != num_heads)
            {
                int reps = num_heads / num_kv_heads;
                k = k.repeat_interleave(reps, dim: 1);
                v = v.repeat_interleave(reps, dim: 1);
            }

            Tensor attn_output;
            if (UseScaledDotProductAttention(hidden_states))
            {
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask: attention_mask,
                    p: 0.0,
                    is_casual: false);
            }
            else
            {
                var scale = Math.Sqrt(head_dim);
                var attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (float)scale;
                if (attention_mask is not null)
                    attn_weights = attn_weights + attention_mask;
                attn_weights = attn_weights.softmax(-1).to(q.dtype);
                attn_output = torch.matmul(attn_weights, v);
            }
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqLen, hidden_size);
            return o_proj.forward(attn_output);
        }

        public (Tensor output, Qwen2LayerCache cache) ForwardWithCache(Tensor hidden_states, Tensor attention_mask, Qwen2LayerCache pastCache = null)
        {
            long bsz = hidden_states.shape[0];
            long seqLen = hidden_states.shape[1];
            long pastLen = pastCache?.Key?.shape[2] ?? 0;

            var q = q_proj.forward(hidden_states).view(bsz, seqLen, num_heads, head_dim).transpose(1, 2);
            var k = k_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);
            var v = v_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);

            (q, k) = ApplyRotaryEmb(q, k, seqLen, hidden_states.device, rope_theta, pastLen);

            if (pastCache is not null)
            {
                k = torch.cat(new[] { pastCache.Key, k }, dim: 2);
                v = torch.cat(new[] { pastCache.Value, v }, dim: 2);
            }

            var nextCache = new Qwen2LayerCache
            {
                Key = k.detach(),
                Value = v.detach()
            };

            if (num_kv_heads != num_heads)
            {
                int reps = num_heads / num_kv_heads;
                k = k.repeat_interleave(reps, dim: 1);
                v = v.repeat_interleave(reps, dim: 1);
            }

            Tensor attn_output;
            if (UseScaledDotProductAttention(hidden_states))
            {
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask: attention_mask,
                    p: 0.0,
                    is_casual: false);
            }
            else
            {
                var scale = Math.Sqrt(head_dim);
                var attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (float)scale;
                if (attention_mask is not null)
                    attn_weights = attn_weights + attention_mask;
                attn_weights = attn_weights.softmax(-1).to(q.dtype);
                attn_output = torch.matmul(attn_weights, v);
            }
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqLen, hidden_size);
            return (o_proj.forward(attn_output), nextCache);
        }

        public Tensor ForwardWithPreallocatedCache(Tensor hidden_states, Tensor attention_mask, Qwen2PreallocatedLayerCache layerCache, long maxCacheLength)
        {
            long bsz = hidden_states.shape[0];
            long seqLen = hidden_states.shape[1];
            long pastLen = layerCache.Length;
            long nextLen = pastLen + seqLen;
            if (nextLen > maxCacheLength)
                throw new InvalidOperationException($"Qwen preallocated KV cache exhausted: required {nextLen}, allocated {maxCacheLength}.");

            var sw = StartProfile(hidden_states);
            var q = q_proj.forward(hidden_states).view(bsz, seqLen, num_heads, head_dim).transpose(1, 2);
            var k = k_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);
            var v = v_proj.forward(hidden_states).view(bsz, seqLen, num_kv_heads, head_dim).transpose(1, 2);
            StopProfile("qwen.attn.qkv_proj", sw, hidden_states, pastLen, nextLen);

            sw = StartProfile(hidden_states);
            (q, k) = ApplyRotaryEmb(q, k, seqLen, hidden_states.device, rope_theta, pastLen);
            StopProfile("qwen.attn.rope", sw, hidden_states, pastLen, nextLen);

            sw = StartProfile(hidden_states);
            EnsurePreallocatedStorage(layerCache, bsz, num_kv_heads, maxCacheLength, head_dim, k.dtype, k.device);
            layerCache.Key.narrow(2, pastLen, seqLen).copy_(k);
            layerCache.Value.narrow(2, pastLen, seqLen).copy_(v);
            layerCache.Length = nextLen;

            k = layerCache.Key.narrow(2, 0, nextLen);
            v = layerCache.Value.narrow(2, 0, nextLen);
            StopProfile("qwen.attn.kv_cache", sw, hidden_states, pastLen, nextLen);

            if (num_kv_heads != num_heads)
            {
                sw = StartProfile(hidden_states);
                int reps = num_heads / num_kv_heads;
                k = k.repeat_interleave(reps, dim: 1);
                v = v.repeat_interleave(reps, dim: 1);
                StopProfile("qwen.attn.repeat_kv", sw, hidden_states, pastLen, nextLen);
            }

            Tensor attn_output;
            sw = StartProfile(hidden_states);
            if (UseScaledDotProductAttention(hidden_states))
            {
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask: attention_mask,
                    p: 0.0,
                    is_casual: false);
            }
            else
            {
                var scale = Math.Sqrt(head_dim);
                var attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (float)scale;
                if (attention_mask is not null)
                    attn_weights = attn_weights + attention_mask;
                attn_weights = attn_weights.softmax(-1).to(q.dtype);
                attn_output = torch.matmul(attn_weights, v);
            }
            StopProfile("qwen.attn.core", sw, hidden_states, pastLen, nextLen);
            sw = StartProfile(hidden_states);
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqLen, hidden_size);
            var output = o_proj.forward(attn_output);
            StopProfile("qwen.attn.output_proj", sw, hidden_states, pastLen, nextLen);
            return output;
        }

        private Stopwatch? StartProfile(Tensor tensor)
        {
            if (Profiler is null)
                return null;
            SynchronizeIfCuda(tensor);
            return Stopwatch.StartNew();
        }

        private void StopProfile(string name, Stopwatch? stopwatch, Tensor tensor, long pastLen, long nextLen)
        {
            if (stopwatch is null || Profiler is null)
                return;
            SynchronizeIfCuda(tensor);
            stopwatch.Stop();
            Profiler.Record(name, stopwatch.Elapsed.TotalMilliseconds, new Dictionary<string, string>
            {
                ["layer"] = LayerIndex.ToString(),
                ["past_len"] = pastLen.ToString(),
                ["next_len"] = nextLen.ToString()
            });
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }

        private bool UseScaledDotProductAttention(Tensor tensor)
        {
            return AttentionBackend == QwenAttentionBackend.ScaledDotProductAttention ||
                   (AttentionBackend == QwenAttentionBackend.Auto && tensor.device.type == DeviceType.CUDA);
        }

        private static void EnsurePreallocatedStorage(
            Qwen2PreallocatedLayerCache layerCache,
            long batchSize,
            int kvHeads,
            long maxLength,
            int headDim,
            ScalarType dtype,
            Device device)
        {
            if (layerCache.Key is not null && layerCache.Value is not null)
                return;

            var shape = new long[] { batchSize, kvHeads, maxLength, headDim };
            layerCache.Key = torch.empty(shape, dtype: dtype, device: device);
            layerCache.Value = torch.empty(shape, dtype: dtype, device: device);
        }
    }

    // Qwen2 SwiGLU MLP
    public class Qwen2MLP : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> gate_proj;
        public readonly nn.Module<Tensor, Tensor> up_proj;
        public readonly nn.Module<Tensor, Tensor> down_proj;
        public ICosyVoiceProfiler? Profiler { get; set; }
        public QwenMlpBackend MlpBackend { get; set; } = QwenMlpBackend.Auto;
        public int LayerIndex { get; set; } = -1;
        private Tensor? _fusedGateUpWeightT;
        private string? _fusedGateUpKey;

        public Qwen2MLP(int hiddenSize, int intermediateSize) : base("Qwen2MLP")
        {
            gate_proj = nn.Linear(hiddenSize, intermediateSize, hasBias: false);
            up_proj   = nn.Linear(hiddenSize, intermediateSize, hasBias: false);
            down_proj = nn.Linear(intermediateSize, hiddenSize, hasBias: false);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            Stopwatch? sw = null;
            if (Profiler is not null)
            {
                SynchronizeIfCuda(x);
                sw = Stopwatch.StartNew();
            }
            var output = UseFusedGateUp(x)
                ? ForwardFusedGateUp(x)
                : down_proj.forward(nn.functional.silu(gate_proj.forward(x)) * up_proj.forward(x));
            if (sw is not null && Profiler is not null)
            {
                SynchronizeIfCuda(output);
                sw.Stop();
                Profiler.Record("qwen.layer.mlp", sw.Elapsed.TotalMilliseconds, new Dictionary<string, string>
                {
                    ["layer"] = LayerIndex.ToString()
                });
            }
            return output;
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }

        private bool UseFusedGateUp(Tensor x)
        {
            if (MlpBackend == QwenMlpBackend.SeparateProjections)
                return false;
            if (MlpBackend == QwenMlpBackend.FusedGateUpProjection)
                return true;

            return false;
        }

        private Tensor ForwardFusedGateUp(Tensor x)
        {
            var fusedWeightT = GetFusedGateUpWeightT();
            var projected = x.matmul(fusedWeightT);
            var width = projected.shape[^1] / 2;
            var gate = projected.narrow(-1, 0, width);
            var up = projected.narrow(-1, width, width);
            return down_proj.forward(nn.functional.silu(gate) * up);
        }

        private Tensor GetFusedGateUpWeightT()
        {
            dynamic gate = gate_proj;
            dynamic up = up_proj;
            Tensor gateWeight = gate.weight;
            Tensor upWeight = up.weight;
            var key = $"{gateWeight.device}|{gateWeight.dtype}|{gateWeight.Handle}|{upWeight.Handle}";
            if (_fusedGateUpWeightT is not null && _fusedGateUpKey == key)
                return _fusedGateUpWeightT;

            _fusedGateUpWeightT?.Dispose();
            _fusedGateUpWeightT = torch.cat(new[] { gateWeight, upWeight }, dim: 0)
                .transpose(0, 1)
                .contiguous()
                .DetachFromDisposeScope();
            _fusedGateUpKey = key;
            return _fusedGateUpWeightT;
        }
    }

    // Single Qwen2 decoder layer
    public class Qwen2DecoderLayer : nn.Module<Tensor, Tensor, Tensor>
    {
        public readonly Qwen2Attention self_attn;
        public readonly Qwen2MLP mlp;
        public readonly RMSNorm input_layernorm;
        public readonly RMSNorm post_attention_layernorm;
        private ICosyVoiceProfiler? _profiler;
        public ICosyVoiceProfiler? Profiler
        {
            get => _profiler;
            set
            {
                _profiler = value;
                self_attn.Profiler = value;
                mlp.Profiler = value;
            }
        }

        public QwenAttentionBackend AttentionBackend
        {
            get => self_attn.AttentionBackend;
            set => self_attn.AttentionBackend = value;
        }

        public QwenMlpBackend MlpBackend
        {
            get => mlp.MlpBackend;
            set => mlp.MlpBackend = value;
        }

        public int LayerIndex
        {
            get => self_attn.LayerIndex;
            set
            {
                self_attn.LayerIndex = value;
                mlp.LayerIndex = value;
            }
        }

        public Qwen2DecoderLayer(int hiddenSize, int numHeads, int numKvHeads, int intermediateSize, float rmsNormEps, float ropeTheta)
            : base("Qwen2DecoderLayer")
        {
            self_attn = new Qwen2Attention(hiddenSize, numHeads, numKvHeads, rmsNormEps, ropeTheta);
            mlp = new Qwen2MLP(hiddenSize, intermediateSize);
            input_layernorm = new RMSNorm(hiddenSize, rmsNormEps);
            post_attention_layernorm = new RMSNorm(hiddenSize, rmsNormEps);
            RegisterComponents();
        }

        public override Tensor forward(Tensor hidden_states, Tensor attention_mask)
        {
            var residual = hidden_states;
            hidden_states = self_attn.forward(input_layernorm.forward(hidden_states), attention_mask);
            hidden_states = residual + hidden_states;
            residual = hidden_states;
            hidden_states = mlp.forward(post_attention_layernorm.forward(hidden_states));
            return residual + hidden_states;
        }

        public (Tensor hiddenStates, Qwen2LayerCache cache) ForwardWithCache(Tensor hidden_states, Tensor attention_mask, Qwen2LayerCache pastCache = null)
        {
            var residual = hidden_states;
            var attn = self_attn.ForwardWithCache(input_layernorm.forward(hidden_states), attention_mask, pastCache);
            hidden_states = residual + attn.output;
            residual = hidden_states;
            hidden_states = mlp.forward(post_attention_layernorm.forward(hidden_states));
            return (residual + hidden_states, attn.cache);
        }

        public Tensor ForwardWithPreallocatedCache(Tensor hidden_states, Tensor attention_mask, Qwen2PreallocatedLayerCache layerCache, long maxCacheLength)
        {
            var residual = hidden_states;
            var normSw = StartProfile(hidden_states);
            var normed = input_layernorm.forward(hidden_states);
            StopProfile("qwen.layer.input_norm", normSw, normed);
            hidden_states = self_attn.ForwardWithPreallocatedCache(normed, attention_mask, layerCache, maxCacheLength);
            hidden_states = residual + hidden_states;
            residual = hidden_states;
            normSw = StartProfile(hidden_states);
            normed = post_attention_layernorm.forward(hidden_states);
            StopProfile("qwen.layer.post_norm", normSw, normed);
            hidden_states = mlp.forward(normed);
            return residual + hidden_states;
        }

        private Stopwatch? StartProfile(Tensor tensor)
        {
            if (Profiler is null)
                return null;
            SynchronizeIfCuda(tensor);
            return Stopwatch.StartNew();
        }

        private void StopProfile(string name, Stopwatch? stopwatch, Tensor tensor)
        {
            if (stopwatch is null || Profiler is null)
                return;
            SynchronizeIfCuda(tensor);
            stopwatch.Stop();
            Profiler.Record(name, stopwatch.Elapsed.TotalMilliseconds, new Dictionary<string, string>
            {
                ["layer"] = LayerIndex.ToString()
            });
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }
    }

    // Inner Qwen2Model (registered as sub-module "model" inside Qwen2ForCausalLM)
    public class Qwen2Model : nn.Module<Tensor, Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> embed_tokens;
        public readonly ModuleList<Qwen2DecoderLayer> layers;
        public readonly RMSNorm norm;
        private ICosyVoiceProfiler? _profiler;
        public ICosyVoiceProfiler? Profiler
        {
            get => _profiler;
            set
            {
                _profiler = value;
                for (var i = 0; i < layers.Count; i++)
                    layers[i].Profiler = value;
            }
        }

        public QwenAttentionBackend AttentionBackend
        {
            get => layers.Count > 0 ? layers[0].AttentionBackend : QwenAttentionBackend.Auto;
            set
            {
                for (var i = 0; i < layers.Count; i++)
                    layers[i].AttentionBackend = value;
            }
        }

        public QwenMlpBackend MlpBackend
        {
            get => layers.Count > 0 ? layers[0].MlpBackend : QwenMlpBackend.Auto;
            set
            {
                for (var i = 0; i < layers.Count; i++)
                    layers[i].MlpBackend = value;
            }
        }

        public Qwen2Model(int vocabSize, int hiddenSize, int numLayers, int numHeads, int numKvHeads, int intermediateSize, float rmsNormEps, float ropeTheta)
            : base("Qwen2Model")
        {
            embed_tokens = nn.Embedding(vocabSize, hiddenSize);
            layers = new ModuleList<Qwen2DecoderLayer>();
            for (int i = 0; i < numLayers; i++)
            {
                var layer = new Qwen2DecoderLayer(hiddenSize, numHeads, numKvHeads, intermediateSize, rmsNormEps, ropeTheta)
                {
                    LayerIndex = i
                };
                layers.Add(layer);
            }
            norm = new RMSNorm(hiddenSize, rmsNormEps);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input_ids, Tensor attention_mask)
        {
            var hidden_states = embed_tokens.forward(input_ids);
            return ForwardEmbeds(hidden_states, attention_mask);
        }

        // Returns the final normalized hidden state.
        // Mirrors HuggingFace Qwen2: outs.hidden_states[-1] is appended after model.norm.
        public Tensor ForwardEmbeds(Tensor hidden_states, Tensor attention_mask)
        {
            foreach (var layer in layers)
                hidden_states = layer.forward(hidden_states, attention_mask);
            return norm.forward(hidden_states);
        }

        public (Tensor hiddenStates, Qwen2KvCache cache) ForwardEmbedsWithCache(Tensor hidden_states, Tensor attention_mask, Qwen2KvCache pastCache = null)
        {
            var nextCache = new Qwen2KvCache();
            for (var i = 0; i < layers.Count; i++)
            {
                var layerCache = pastCache is not null && i < pastCache.Layers.Count ? pastCache.Layers[i] : null;
                var result = layers[i].ForwardWithCache(hidden_states, attention_mask, layerCache);
                hidden_states = result.hiddenStates;
                nextCache.Layers.Add(result.cache);
            }
            return (norm.forward(hidden_states), nextCache);
        }

        public (Tensor hiddenStates, Qwen2PreallocatedKvCache cache) ForwardEmbedsWithPreallocatedCache(
            Tensor hidden_states,
            Tensor attention_mask,
            Qwen2PreallocatedKvCache pastCache)
        {
            for (var i = 0; i < layers.Count; i++)
            {
                var layerCache = pastCache.GetOrCreateLayer(i);
                hidden_states = layers[i].ForwardWithPreallocatedCache(hidden_states, attention_mask, layerCache, pastCache.MaxLength);
            }
            Stopwatch? sw = null;
            if (Profiler is not null)
            {
                SynchronizeIfCuda(hidden_states);
                sw = Stopwatch.StartNew();
            }
            var output = norm.forward(hidden_states);
            if (sw is not null && Profiler is not null)
            {
                SynchronizeIfCuda(output);
                sw.Stop();
                Profiler.Record("qwen.model.final_norm", sw.Elapsed.TotalMilliseconds);
            }
            return (output, pastCache);
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }
    }

    // Top-level Qwen2ForCausalLM: model sub-module + lm_head
    public class Qwen2ForCausalLM : nn.Module<Tensor, Tensor, Tensor>
    {
        public readonly Qwen2Model model;
        public readonly nn.Module<Tensor, Tensor> lm_head;
        public ICosyVoiceProfiler? Profiler
        {
            get => model.Profiler;
            set => model.Profiler = value;
        }

        public QwenAttentionBackend AttentionBackend
        {
            get => model.AttentionBackend;
            set => model.AttentionBackend = value;
        }

        public QwenMlpBackend MlpBackend
        {
            get => model.MlpBackend;
            set => model.MlpBackend = value;
        }

        public readonly int vocab_size;
        public readonly int hidden_size;
        public readonly int num_attention_heads;
        public readonly int num_key_value_heads;
        public readonly int num_hidden_layers;
        public readonly int intermediate_size;
        public readonly float rms_norm_eps;
        public readonly float rope_theta;

        public Qwen2ForCausalLM(
            int vocabSize        = 151936,
            int hiddenSize       = 896,
            int numAttentionHeads = 14,
            int numKeyValueHeads = 2,
            int numHiddenLayers  = 24,
            int intermediateSize = 4864,
            float rmsNormEps     = 1e-6f,
            float ropeTheta      = 10000.0f
        ) : base("Qwen2ForCausalLM")
        {
            vocab_size = vocabSize;
            hidden_size = hiddenSize;
            num_attention_heads = numAttentionHeads;
            num_key_value_heads = numKeyValueHeads;
            num_hidden_layers = numHiddenLayers;
            intermediate_size = intermediateSize;
            rms_norm_eps = rmsNormEps;
            rope_theta = ropeTheta;

            model = new Qwen2Model(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, numKeyValueHeads, intermediateSize, rmsNormEps, ropeTheta);
            lm_head = nn.Linear(hiddenSize, vocabSize, hasBias: false);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input_ids, Tensor attention_mask)
        {
            var hidden = model.forward(input_ids, attention_mask);
            return lm_head.forward(hidden);
        }

        // Forward from pre-computed embeddings (used by Qwen2Encoder.RunWithEmbeds)
        public Tensor ForwardEmbeds(Tensor inputs_embeds, Tensor attention_mask)
        {
            return model.ForwardEmbeds(inputs_embeds, attention_mask);
        }

        public (Tensor hiddenStates, Qwen2KvCache cache) ForwardEmbedsWithCache(Tensor inputs_embeds, Tensor attention_mask, Qwen2KvCache pastCache = null)
        {
            return model.ForwardEmbedsWithCache(inputs_embeds, attention_mask, pastCache);
        }

        public (Tensor hiddenStates, Qwen2PreallocatedKvCache cache) ForwardEmbedsWithPreallocatedCache(
            Tensor inputs_embeds,
            Tensor attention_mask,
            Qwen2PreallocatedKvCache pastCache)
        {
            return model.ForwardEmbedsWithPreallocatedCache(inputs_embeds, attention_mask, pastCache);
        }
    }
}
