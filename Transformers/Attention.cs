using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public sealed class PreallocatedAttentionCache : IDisposable
    {
        public Tensor? KeyValue { get; set; }
        public long Length { get; set; }

        public void Dispose()
        {
            KeyValue?.Dispose();
            KeyValue = null;
            Length = 0;
        }
    }

    // Port of cosyvoice.transformer.attention.MultiHeadedAttention and
    // RelPositionMultiHeadedAttention. Preserves Python API: forward returns
    // (output, new_cache).
    // Equivalent Python file: cosyvoice/transformer/attention.py
    public class MultiHeadedAttention : torch.nn.Module<(Tensor, Tensor, Tensor), (Tensor, Tensor)>
    {
        protected readonly int d_k;
        protected readonly int h;
        public readonly nn.Module<Tensor, Tensor> linear_q;
        public readonly nn.Module<Tensor, Tensor> linear_k;
        public readonly nn.Module<Tensor, Tensor> linear_v;
        public readonly nn.Module<Tensor, Tensor> linear_out;
        public readonly nn.Module<Tensor, Tensor> dropout;

        public MultiHeadedAttention(int n_head, int n_feat, double dropout_rate, bool key_bias = true, bool deferRegisterComponents = false) : base("MultiHeadedAttention")
        {
            if (n_feat % n_head != 0) throw new ArgumentException("n_feat must be divisible by n_head");
            d_k = n_feat / n_head;
            h = n_head;
            linear_q = torch.nn.Linear(n_feat, n_feat);
            linear_k = torch.nn.Linear(n_feat, n_feat, key_bias);
            linear_v = torch.nn.Linear(n_feat, n_feat);
            linear_out = torch.nn.Linear(n_feat, n_feat);
            dropout = torch.nn.Dropout(dropout_rate);
            if (!deferRegisterComponents)
                RegisterComponents();
        }

        protected (Tensor q, Tensor k, Tensor v) ForwardQKV(Tensor query, Tensor key, Tensor value)
        {
            var n_batch = (int)query.shape[0];
            var q = linear_q.forward(query).view(n_batch, -1, h, d_k);
            var k = linear_k.forward(key).view(n_batch, -1, h, d_k);
            var v = linear_v.forward(value).view(n_batch, -1, h, d_k);
            q = q.transpose(1, 2);
            k = k.transpose(1, 2);
            v = v.transpose(1, 2);
            return (q, k, v);
        }

        protected Tensor ForwardAttention(Tensor value, Tensor scores, Tensor mask)
        {
            var n_batch = (int)value.shape[0];
            if (mask.shape.Length > 2 && mask.shape[2] > 0)
            {
                mask = mask.unsqueeze(1).eq(0);
                mask = mask.slice(3, 0, scores.shape[scores.shape.Length - 1], 1);
                scores = scores.masked_fill(mask, double.NegativeInfinity);
                var attn = torch.softmax(scores, -1).masked_fill(mask, 0.0);
                var p_attn = dropout.forward(attn);
                var x = torch.matmul(p_attn, value);
                x = x.transpose(1, 2).contiguous().view(n_batch, -1, h * d_k);
                return linear_out.forward(x);
            }
            else
            {
                var attn = torch.softmax(scores, -1);
                var p_attn = dropout.forward(attn);
                var x = torch.matmul(p_attn, value);
                x = x.transpose(1, 2).contiguous().view(n_batch, -1, h * d_k);
                return linear_out.forward(x);
            }
        }

        public (Tensor, Tensor) Forward(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor? mask = null,
            Tensor? pos_emb = null,
            Tensor? cache = null)
        {
            mask ??= torch.ones(new long[] { 0, 0, 0 }, dtype: ScalarType.Bool, device: query.device);
            cache ??= torch.zeros(new long[] { 0, 0, 0, 0 }, dtype: query.dtype, device: query.device);

            var (q, k, v) = ForwardQKV(query, key, value);

            if (cache.shape[0] > 0)
            {
                var lastDim = cache.shape.Length - 1;
                var half = cache.shape[lastDim] / 2;
                var key_cache = cache.narrow(lastDim, 0, half);
                var value_cache = cache.narrow(lastDim, half, half);
                k = torch.cat(new Tensor[] { key_cache, k }, 2);
                v = torch.cat(new Tensor[] { value_cache, v }, 2);
            }
            var new_cache = torch.cat(new Tensor[] { k, v }, -1);

            var scores = torch.matmul(q, k.transpose(-2, -1)) / Math.Sqrt((double)d_k);
            var outTensor = ForwardAttention(v, scores, mask);
            return (outTensor, new_cache);
        }

        public (Tensor output, long nextLength) ForwardWithPreallocatedCache(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            Tensor? pos_emb,
            PreallocatedAttentionCache cache,
            long maxCacheLength)
        {
            var (q, k, v) = ForwardQKV(query, key, value);
            var nextLength = WritePreallocatedCache(cache, k, v, maxCacheLength);

            var lastDim = cache.KeyValue!.shape.Length - 1;
            var half = cache.KeyValue.shape[lastDim] / 2;
            var keyCache = cache.KeyValue.narrow(lastDim, 0, half).narrow(2, 0, nextLength);
            var valueCache = cache.KeyValue.narrow(lastDim, half, half).narrow(2, 0, nextLength);

            var scores = torch.matmul(q, keyCache.transpose(-2, -1)) / Math.Sqrt((double)d_k);
            var outTensor = ForwardAttention(valueCache, scores, mask);
            return (outTensor, nextLength);
        }

        public override (Tensor, Tensor) forward((Tensor, Tensor, Tensor) inputs)
        {
            var (query, key, value) = inputs;
            return Forward(query, key, value);
        }

        protected long WritePreallocatedCache(PreallocatedAttentionCache cache, Tensor k, Tensor v, long maxCacheLength)
        {
            var pastLength = cache.Length;
            var sequenceLength = k.shape[2];
            var nextLength = pastLength + sequenceLength;
            if (nextLength > maxCacheLength)
                throw new InvalidOperationException($"Legacy transformer attention cache exhausted: required {nextLength}, allocated {maxCacheLength}.");

            EnsurePreallocatedCache(cache, k.shape[0], k.shape[1], maxCacheLength, k.shape[3], k.dtype, k.device);
            var lastDim = cache.KeyValue!.shape.Length - 1;
            var half = cache.KeyValue.shape[lastDim] / 2;
            cache.KeyValue.narrow(lastDim, 0, half).narrow(2, pastLength, sequenceLength).copy_(k);
            cache.KeyValue.narrow(lastDim, half, half).narrow(2, pastLength, sequenceLength).copy_(v);
            cache.Length = nextLength;
            return nextLength;
        }

        private static void EnsurePreallocatedCache(
            PreallocatedAttentionCache cache,
            long batchSize,
            long heads,
            long maxCacheLength,
            long headDim,
            ScalarType dtype,
            Device device)
        {
            if (cache.KeyValue is not null)
                return;

            cache.KeyValue = torch.empty(
                new[] { batchSize, heads, maxCacheLength, headDim * 2 },
                dtype: dtype,
                device: device).DetachFromDisposeScope();
        }
    }

    public class RelPositionMultiHeadedAttention : MultiHeadedAttention
    {
        public readonly nn.Module<Tensor, Tensor> linear_pos;
        public readonly TorchSharp.Modules.Parameter pos_bias_u;
        public readonly TorchSharp.Modules.Parameter pos_bias_v;

        public RelPositionMultiHeadedAttention(int n_head, int n_feat, double dropout_rate, bool key_bias = true)
            : base(n_head, n_feat, dropout_rate, key_bias, deferRegisterComponents: true)
        {
            linear_pos = torch.nn.Linear(n_feat, n_feat, false);
            pos_bias_u = torch.nn.Parameter(torch.empty(new long[] { h, d_k }));
            pos_bias_v = torch.nn.Parameter(torch.empty(new long[] { h, d_k }));
            torch.nn.init.xavier_uniform_(pos_bias_u);
            torch.nn.init.xavier_uniform_(pos_bias_v);
            RegisterComponents();
        }

        private Tensor RelShift(Tensor x)
        {
            var zero_pad = torch.zeros(new long[] { x.shape[0], x.shape[1], x.shape[2], 1 }, dtype: x.dtype, device: x.device);
            var x_padded = torch.cat(new Tensor[] { zero_pad, x }, -1);
            x_padded = x_padded.view(new long[] { x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2] });
            var res = x_padded.slice(2, 1, x_padded.shape[2], 1).view_as(x);
            return res.narrow(3, 0, x.shape[3] / 2 + 1);
        }

        public new (Tensor, Tensor) Forward(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor? mask = null,
            Tensor? pos_emb = null,
            Tensor? cache = null)
        {
            mask ??= torch.ones(new long[] { 0, 0, 0 }, dtype: ScalarType.Bool, device: query.device);
            pos_emb ??= torch.empty(new long[] { 0 }, dtype: query.dtype, device: query.device);
            cache ??= torch.zeros(new long[] { 0, 0, 0, 0 }, dtype: query.dtype, device: query.device);

            var (q, k, v) = ForwardQKV(query, key, value);
            q = q.transpose(1, 2);

            if (cache.shape[0] > 0)
            {
                var lastDim = cache.shape.Length - 1;
                var half = cache.shape[lastDim] / 2;
                var key_cache = cache.narrow(lastDim, 0, half);
                var value_cache = cache.narrow(lastDim, half, half);
                k = torch.cat(new Tensor[] { key_cache, k }, 2);
                v = torch.cat(new Tensor[] { value_cache, v }, 2);
            }
            var new_cache = torch.cat(new Tensor[] { k, v }, -1);

            var p = linear_pos.forward(pos_emb).view(pos_emb.shape[0], -1, h, d_k).transpose(1, 2);
            var q_with_bias_u = (q + pos_bias_u).transpose(1, 2);
            var q_with_bias_v = (q + pos_bias_v).transpose(1, 2);

            var matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1));
            var matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1));
            if (!matrix_ac.shape.SequenceEqual(matrix_bd.shape))
            {
                matrix_bd = RelShift(matrix_bd);
            }
            var scores = (matrix_ac + matrix_bd) / Math.Sqrt((double)d_k);
            var outTensor2 = ForwardAttention(v, scores, mask);
            return (outTensor2, new_cache);
        }

        public new (Tensor output, long nextLength) ForwardWithPreallocatedCache(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            Tensor? pos_emb,
            PreallocatedAttentionCache cache,
            long maxCacheLength)
        {
            pos_emb ??= torch.empty(new long[] { 0 }, dtype: query.dtype, device: query.device);

            var (q, k, v) = ForwardQKV(query, key, value);
            q = q.transpose(1, 2);
            var nextLength = WritePreallocatedCache(cache, k, v, maxCacheLength);

            var lastDim = cache.KeyValue!.shape.Length - 1;
            var half = cache.KeyValue.shape[lastDim] / 2;
            var keyCache = cache.KeyValue.narrow(lastDim, 0, half).narrow(2, 0, nextLength);
            var valueCache = cache.KeyValue.narrow(lastDim, half, half).narrow(2, 0, nextLength);

            var p = linear_pos.forward(pos_emb).view(pos_emb.shape[0], -1, h, d_k).transpose(1, 2);
            var q_with_bias_u = (q + pos_bias_u).transpose(1, 2);
            var q_with_bias_v = (q + pos_bias_v).transpose(1, 2);

            var matrix_ac = torch.matmul(q_with_bias_u, keyCache.transpose(-2, -1));
            var matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1));
            if (!matrix_ac.shape.SequenceEqual(matrix_bd.shape))
            {
                matrix_bd = RelShift(matrix_bd);
            }
            var scores = (matrix_ac + matrix_bd) / Math.Sqrt((double)d_k);
            var outTensor = ForwardAttention(valueCache, scores, mask);
            return (outTensor, nextLength);
        }

        public override (Tensor, Tensor) forward((Tensor, Tensor, Tensor) inputs)
        {
            var (query, key, value) = inputs;
            return Forward(query, key, value);
        }
    }
}
