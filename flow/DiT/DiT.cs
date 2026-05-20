// Exported from CosyVoice\cosyvoice\flow\DiT\dit.py
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using CosyVoiceNet.Utils;

namespace CosyVoiceNet.flow.DiT
{
    // InputEmbedding: forward(x, cond, textEmbed, spks) → x
    public class InputEmbedding : nn.Module
    {
        public readonly nn.Module<Tensor, Tensor> proj;
        public readonly CausalConvPositionEmbedding conv_pos_embed;
        private readonly int spk_dim;

        public InputEmbedding(int melDim, int textDim, int outDim, int? spkDim = null) : base("InputEmbedding")
        {
            spk_dim = spkDim ?? 0;
            proj = nn.Linear(melDim * 2 + textDim + spk_dim, outDim);
            conv_pos_embed = new CausalConvPositionEmbedding(outDim);
            RegisterComponents();
        }

        public Tensor Forward(Tensor x, Tensor cond, Tensor textEmbed, Tensor spks)
        {
            var toCat = new List<Tensor> { x, cond, textEmbed };
            if (spk_dim > 0 && spks is not null && spks.shape.Length == 2 && spks.shape[1] == spk_dim)
            {
                var spksExp = spks.unsqueeze(1).expand(x.shape[0], x.shape[1], spk_dim);
                toCat.Add(spksExp);
            }

            var concatenated = torch.cat(toCat.ToArray(), dim: -1);
            x = proj.forward(concatenated);
            x = conv_pos_embed.forward(x) + x;
            return x;
        }
    }

    // DiT backbone — state-dict keys align to checkpoint decoder.estimator.*
    public class DiT : nn.Module
    {
        public readonly TimestepEmbedding time_embed;
        public readonly InputEmbedding input_embed;
        public readonly RotaryEmbedding rotary_embed;
        public readonly ModuleList<DiTBlock> transformer_blocks;
        public readonly AdaLayerNormZero_Final norm_out;
        public readonly nn.Module<Tensor, Tensor> proj_out;
        public readonly nn.Module<Tensor, Tensor> long_skip_connection;  // null when unused
        public readonly int static_chunk_size;
        public readonly int num_decoding_left_chunks;

        public DiT(int dim, int depth = 8, int heads = 8, int dimHead = 64, float dropout = 0.1f, float ffMult = 4,
                   int melDim = 80, int? muDim = null, bool longSkipConnection = false, int? spkDim = null,
                   int? outChannels = null, int staticChunkSize = 50, int numDecodingLeftChunks = 2) : base("DiT")
        {
            time_embed = new TimestepEmbedding(dim);
            input_embed = new InputEmbedding(melDim, muDim ?? melDim, dim, spkDim);
            rotary_embed = new RotaryEmbedding(dimHead);

            transformer_blocks = new ModuleList<DiTBlock>();
            for (int i = 0; i < depth; i++)
                transformer_blocks.Add(new DiTBlock(dim, heads, dimHead, (int)ffMult, dropout));

            // Only register long_skip_connection when actually used (null = not in state dict)
            long_skip_connection = longSkipConnection ? nn.Linear(dim * 2, dim, hasBias: false) : null;

            norm_out = new AdaLayerNormZero_Final(dim);
            proj_out = nn.Linear(dim, outChannels ?? melDim);
            static_chunk_size = staticChunkSize;
            num_decoding_left_chunks = numDecodingLeftChunks;
            RegisterComponents();
        }

        // forward: x/mu/cond are (batch, n_feats, seq_len); spks is (batch, spk_dim)
        public Tensor Forward(Tensor x, Tensor mask, Tensor mu, Tensor t, Tensor spks = null, Tensor cond = null, bool streaming = false)
        {
            return Forward(x, mask, mu, t, spks, cond, streaming, null, null);
        }

        public (Tensor attnMask, (Tensor freqs, Tensor scale) rope) PrepareAttention(Tensor mask, long seqLen, bool streaming, Device device)
        {
            var (ropeFreqs, ropeScale) = rotary_embed.ForwardFromSeqLen((int)seqLen);
            using var dummy = torch.empty(new long[] { mask.shape[0], seqLen, 1 }, device: device, dtype: ScalarType.Float32);

            Tensor attnMask;
            var boolMask = mask.to_type(ScalarType.Bool);
            if (streaming)
            {
                attnMask = Mask.AddOptionalChunkMask(dummy, boolMask, false, false, 0, static_chunk_size, -1)
                               .unsqueeze(1);
            }
            else
            {
                attnMask = Mask.AddOptionalChunkMask(dummy, boolMask, false, false, 0, 0, -1)
                               .repeat(new long[] { 1, seqLen, 1 })
                               .unsqueeze(1);
            }

            return (attnMask.to_type(ScalarType.Bool), (ropeFreqs, ropeScale));
        }

        public Tensor Forward(
            Tensor x,
            Tensor mask,
            Tensor mu,
            Tensor t,
            Tensor spks = null,
            Tensor cond = null,
            bool streaming = false,
            Tensor preparedAttnMask = null,
            (Tensor freqs, Tensor scale)? preparedRope = null)
        {
            // Python: x, mu, cond all transposed from (b, d, n) → (b, n, d)
            x = x.transpose(1, 2);
            mu = mu.transpose(1, 2);
            if (cond is not null) cond = cond.transpose(1, 2);

            long bsz = x.shape[0], seqLen = x.shape[1];

            if (t.ndim == 0)
                t = t.expand(bsz);

            t = time_embed.forward(t);

            // spks: (batch, spk_dim) — InputEmbedding expands to (batch, seq_len, spk_dim)
            var spksArg = (spks is not null && spks.shape.Length == 2 && spks.shape[1] > 0) ? spks : null;
            var condArg = cond ?? torch.zeros_like(x);
            x = input_embed.Forward(x, condArg, mu, spksArg);

            var rope = preparedRope ?? rotary_embed.ForwardFromSeqLen((int)seqLen);

            Tensor residual = null;
            if (long_skip_connection is not null)
                residual = x;

            var attnMask = preparedAttnMask ?? (mask is null ? null : PrepareAttention(mask, seqLen, streaming, x.device).attnMask);

            var perfDit = false;
            var traceShapes = false;
            if (traceShapes)
                Console.WriteLine($"[CosyVoiceShapes.DiT] batch={bsz} seq_len={seqLen} x={Shape(x)} mu={Shape(mu)} cond={Shape(cond)} spks={Shape(spks)} attn_mask={Shape(attnMask)} device={x.device} dtype={x.dtype}");
            var sw = perfDit ? System.Diagnostics.Stopwatch.StartNew() : null;
            for (int i = 0; i < transformer_blocks.Count; i++)
            {
                var block = transformer_blocks[i];
                x = block.forward(x, t, attnMask, rope);
                if (perfDit)
                {
                    SynchronizeIfCuda(x);
                    Console.WriteLine($"[CosyVoicePerf.DiT] block{i}_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                    sw.Restart();
                }
            }

            if (long_skip_connection is not null)
                x = long_skip_connection.forward(torch.cat(new[] { x, residual }, dim: -1));

            x = norm_out.forward(x, t);
            return proj_out.forward(x).transpose(1, 2);
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }

        private static string Shape(Tensor tensor)
        {
            return tensor is null ? "null" : "[" + string.Join(",", tensor.shape) + "]";
        }
    }
}
