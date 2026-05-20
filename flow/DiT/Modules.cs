// Exported from CosyVoice\cosyvoice\flow\DiT\modules.py
// Deep alignment confirmed with modules.py
// Class definitions, constructors, forward methods, tensor operations, and layer initialization verified.
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.Transforms;
using static TorchSharp.torch;
using static TorchSharp.torchaudio;

namespace CosyVoiceNet.flow.DiT
{
    public static class RotaryEmbeddings
    {
        public static Tensor PrecomputeFreqsCis(int dim, int end, float theta = 10000.0f, float thetaRescaleFactor = 1.0f)
        {
            theta *= (float)Math.Pow(thetaRescaleFactor, dim / (dim - 2.0f));

            var freqs = 1.0f / torch.pow(
                torch.tensor(theta, dtype: ScalarType.Float32),
                torch.arange(0, dim, 2, dtype: ScalarType.Float32) / dim
            );

            var t = torch.arange(end, device: freqs.device, dtype: ScalarType.Float32);
            freqs = torch.outer(t, freqs).to_type(ScalarType.Float32);

            var freqsCos = freqs.cos();
            var freqsSin = freqs.sin();
            return torch.cat(new[] { freqsCos, freqsSin }, dim: -1);
        }

        public static Tensor GetPosEmbedIndices(Tensor start, long length, long maxPos, float scale = 1.0f)
        {
            var scaleTensor = scale * torch.ones_like(start, dtype: ScalarType.Float32);
            var positions = start.unsqueeze(1) +
                            (torch.arange(length, device: start.device, dtype: ScalarType.Float32)
                                .unsqueeze(0) * scaleTensor.unsqueeze(1)).to_type(ScalarType.Int64);

            positions = torch.where(positions < maxPos, positions, torch.full_like(positions, maxPos - 1));
            return positions;
        }

        public static Tensor ApplyRotaryPosEmb(Tensor x, Tensor freqs, float scale)
        {
            var seqLen = x.size(-2);
            if (freqs.dim() == 2)
                freqs = freqs.unsqueeze(0);
            freqs = freqs.narrow(-2, freqs.size(-2) - seqLen, seqLen);
            if (x.dim() == 4 && freqs.dim() == 3)
                freqs = freqs.unsqueeze(1);
            freqs = freqs.to(x.device);

            var rotDim = freqs.size(-1);
            var rotated = x.narrow(-1, 0, rotDim);
            var unrotated = x.size(-1) > rotDim
                ? x.narrow(-1, rotDim, x.size(-1) - rotDim)
                : null;

            var cos = freqs.cos().to_type(x.dtype);
            var sin = freqs.sin().to_type(x.dtype);
            var outRot = (rotated * cos + RotateHalf(rotated) * sin) * scale;
            return unrotated is null ? outRot : torch.cat(new[] { outRot, unrotated }, dim: -1);
        }

        private static Tensor RotateHalf(Tensor x)
        {
            var last = x.size(-1);
            var x1 = x.slice(-1, 0, last, 2);
            var x2 = x.slice(-1, 1, last, 2);
            return torch.stack(new[] { -x2, x1 }, dim: -1).reshape(x.shape);
        }

    }

    public class MelSpec : nn.Module<Tensor, Tensor>
    {
        private readonly MelSpectrogram melStft;
        private Tensor dummy;

        public MelSpec(
            int filterLength = 1024,
            int hopLength = 256,
            int winLength = 1024,
            int nMelChannels = 100,
            int targetSampleRate = 24000,
            bool normalize = false,
            int power = 1,
            MelNorm norm = MelNorm.none,
            bool center = true
        ) : base("MelSpec")
        {
            melStft = torchaudio.transforms.MelSpectrogram(
                sample_rate: targetSampleRate,
                n_fft: filterLength,
                win_length: winLength,
                hop_length: hopLength,
                n_mels: nMelChannels,
                power: power,
                center: center,
                normalized: normalize,
                norm: norm
            );

            // TorchSharp has no RegisterBuffer — just store it
            dummy = torch.tensor(0);

            RegisterComponents();
        }

        public override Tensor forward(Tensor inp)
        {
            if (inp.dim() == 3)
                inp = inp.squeeze(1);

            if (dummy.device != inp.device)
                this.to(inp.device);

            var mel = melStft.forward(inp);
            mel = mel.clamp(min: 1e-5).log();
            return mel;
        }
    }

    public class SinusPositionEmbedding : nn.Module<Tensor, Tensor>
    {
        private readonly int dim;

        public SinusPositionEmbedding(int dim) : base("SinusPositionEmbedding")
        {
            this.dim = dim;
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return forward(x, 1000f);
        }

        public Tensor forward(Tensor x, float scale)
        {
            var device = x.device;
            int halfDim = dim / 2;

            // emb = log(10000) / (half_dim - 1)
            float divTerm = (float)(Math.Log(10000.0) / (halfDim - 1));

            // torch.arange(half_dim, device=device).float() * -emb
            var arange = torch.arange(halfDim, device: device, dtype: torch.float32);
            var freq = torch.exp(arange * -divTerm);

            // scale * x.unsqueeze(1) * freq.unsqueeze(0)
            var xExpanded = x.unsqueeze(1);      // [B, 1]
            var freqExpanded = freq.unsqueeze(0); // [1, halfDim]

            var emb = scale * xExpanded * freqExpanded; // [B, halfDim]

            // concat(sin, cos)
            var sin = emb.sin();
            var cos = emb.cos();

            return torch.cat(new[] { sin, cos }, dim: -1); // [B, dim]
        }
    }


    public class ConvPositionEmbedding : nn.Module<Tensor, Tensor>
    {
        private readonly Sequential conv1d;

        public ConvPositionEmbedding(int dim, int kernelSize = 31, int groups = 16)
            : base("ConvPositionEmbedding")
        {
            if (kernelSize % 2 == 0)
                throw new ArgumentException("kernel_size must be odd");

            conv1d = nn.Sequential(
                nn.Conv1d(dim, dim, kernelSize, stride: 1, padding: kernelSize / 2, groups: groups),
                nn.Mish(),
                nn.Conv1d(dim, dim, kernelSize, stride: 1, padding: kernelSize / 2, groups: groups),
                nn.Mish()
            );

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return forward(x, null);
        }

        public Tensor forward(Tensor x, Tensor mask)
        {
            // mask: [B, N] → [B, N, 1]
            if (mask is not null)
            {
                var m = mask.unsqueeze(-1);
                x = x.masked_fill(m.logical_not(), 0.0f);
            }

            // x: [B, N, D] → [B, D, N]
            x = x.permute(0, 2, 1);

            x = conv1d.forward(x);

            // back to [B, N, D]
            var outTensor = x.permute(0, 2, 1);

            if (mask is not null)
            {
                var m = mask.unsqueeze(-1);
                outTensor = outTensor.masked_fill(m.logical_not(), 0.0f);
            }

            return outTensor;
        }
    }
    public class CausalConvPositionEmbedding : nn.Module<Tensor, Tensor>
    {
        private readonly int kernelSize;
        private readonly Sequential conv1;
        private readonly Sequential conv2;

        public CausalConvPositionEmbedding(int dim, int kernelSize = 31, int groups = 16)
            : base("CausalConvPositionEmbedding")
        {
            if (kernelSize % 2 == 0)
                throw new ArgumentException("kernel_size must be odd");

            this.kernelSize = kernelSize;

            conv1 = nn.Sequential(
                nn.Conv1d( in_channels: dim, out_channels: dim, kernel_size: kernelSize, stride: 1, padding: 0L, dilation: 1, padding_mode: PaddingModes.Zeros, groups: groups, bias: true ),
                nn.Mish()
            );

            conv2 = nn.Sequential(
                nn.Conv1d( in_channels: dim, out_channels: dim, kernel_size: kernelSize, stride: 1, padding: 0L, dilation: 1, padding_mode: PaddingModes.Zeros, groups: groups, bias: true ),
                nn.Mish()
            );

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return forward(x, null);
        }

        public Tensor forward(Tensor x, Tensor mask)
        {
            // mask: [B, N] → [B, N, 1]
            if (mask is not null)
            {
                var m = mask.unsqueeze(-1);
                x = x.masked_fill(m.logical_not(), 0.0f);
            }

            // x: [B, N, D] → [B, D, N]
            x = x.permute(0, 2, 1);

            // F.pad(x, (kernel_size - 1, 0, 0, 0))
            x = torch.nn.functional.pad(
                x,
                pad: new long[] { kernelSize - 1, 0, 0, 0 }
            );

            x = conv1.forward(x);

            // second causal pad
            x = torch.nn.functional.pad(
                x,
                pad: new long[] { kernelSize - 1, 0, 0, 0 }
            );

            x = conv2.forward(x);

            // back to [B, N, D]
            var outTensor = x.permute(0, 2, 1);

            if (mask is not null)
            {
                var m = mask.unsqueeze(-1);
                outTensor = outTensor.masked_fill(m.logical_not(), 0.0f);
            }

            return outTensor;
        }
    }

    public class GRN : nn.Module<Tensor, Tensor>
    {
        private readonly Parameter gamma;
        private readonly Parameter beta;

        public GRN(int dim) : base("GRN")
        {
            gamma = nn.Parameter(torch.zeros(new long[] { 1, 1, dim }));
            beta = nn.Parameter(torch.zeros(new long[] { 1, 1, dim }));

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Gx = ||x||_2 over dim=1 (sequence dimension)
            var Gx = torch.norm(x, p: 2, dimension: 1, keepdim: true);

            // Nx = Gx / (mean(Gx) + eps)
            var Nx = Gx / (Gx.mean(dimensions: new long[] { -1 }, keepdim: true) + 1e-6f);

            // return gamma * (x * Nx) + beta + x
            return gamma * (x * Nx) + beta + x;
        }
    }

    // ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
    // ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108
    public class ConvNeXtV2Block : nn.Module<Tensor, Tensor>
    {
        private readonly Conv1d dwconv;
        private readonly LayerNorm norm;
        private readonly Linear pwconv1;
        private readonly GELU act;
        private readonly GRN grn;
        private readonly Linear pwconv2;

        public ConvNeXtV2Block(int dim, int intermediateDim, int dilation = 1)
            : base("ConvNeXtV2Block")
        {
            int padding = (dilation * (7 - 1)) / 2;

            dwconv = nn.Conv1d(
                in_channels: dim,
                out_channels: dim,
                kernel_size: 7,
                stride: 1,
                padding: padding,      // safe: long literal
                dilation: dilation,
                padding_mode: PaddingModes.Zeros,
                groups: dim,
                bias: true
            );

            norm = nn.LayerNorm(dim, eps: 1e-6);
            pwconv1 = nn.Linear(dim, intermediateDim);
            act = torch.nn.GELU(false);   // matches nn.GELU() default
            grn = new GRN(intermediateDim);
            pwconv2 = nn.Linear(intermediateDim, dim);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var residual = x;

            // b n d -> b d n
            x = x.transpose(1, 2);

            x = dwconv.forward(x);

            // b d n -> b n d
            x = x.transpose(1, 2);

            x = norm.forward(x);
            x = pwconv1.forward(x);
            x = act.forward(x);
            x = grn.forward(x);
            x = pwconv2.forward(x);

            return residual + x;
        }
    }

    // AdaLayerNormZero
    // return with modulated x for attn input, and params for later mlp modulation
    public class AdaLayerNormZero : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> silu;
        private readonly Linear linear;
        private readonly LayerNorm norm;

        public AdaLayerNormZero(int dim) : base("AdaLayerNormZero")
        {
            silu = nn.SiLU();
            linear = nn.Linear(dim, dim * 6);
            norm = nn.LayerNorm(dim, elementwise_affine: false, eps: 1e-6);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            throw new InvalidOperationException("Use forward(x, emb)");
        }

        public (Tensor normOut, Tensor gateMsa, Tensor shiftMlp, Tensor scaleMlp, Tensor gateMlp)
            forward(Tensor x, Tensor emb)
        {
            // emb = linear(silu(emb))
            emb = linear.forward(silu.forward(emb));

            // chunk into 6 parts along dim=1
            var chunks = emb.chunk(6, dim: 1);
            var shiftMsa = chunks[0];
            var scaleMsa = chunks[1];
            var gateMsa = chunks[2];
            var shiftMlp = chunks[3];
            var scaleMlp = chunks[4];
            var gateMlp = chunks[5];

            // x = norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            var normX = norm.forward(x);
            var scale = 1 + scaleMsa.unsqueeze(1);
            var shift = shiftMsa.unsqueeze(1);

            var outX = normX * scale + shift;

            return (outX, gateMsa, shiftMlp, scaleMlp, gateMlp);
        }
    }

    // AdaLayerNormZero for final layer
    // return only with modulated x for attn input, cuz no more mlp modulation
    public class AdaLayerNormZero_Final : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> silu;
        private readonly Linear linear;
        private readonly LayerNorm norm;

        public AdaLayerNormZero_Final(int dim) : base("AdaLayerNormZero_Final")
        {
            silu = nn.SiLU();
            linear = nn.Linear(dim, dim * 2);
            norm = nn.LayerNorm(dim, elementwise_affine: false, eps: 1e-6);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            throw new InvalidOperationException("Use forward(x, emb)");
        }

        public Tensor forward(Tensor x, Tensor emb)
        {
            // emb = linear(silu(emb))
            emb = linear.forward(silu.forward(emb));

            // scale, shift = chunk(emb, 2, dim=1)
            var chunks = emb.chunk(2, dim: 1);
            var scale = chunks[0];
            var shift = chunks[1];

            // x = norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
            var normX = norm.forward(x);

            var scaleB = (1 + scale).unsqueeze(1); // [B, 1, D]
            var shiftB = shift.unsqueeze(1);       // [B, 1, D]

            return normX * scaleB + shiftB;
        }
    }

    public class FeedForward : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> ff;

        public FeedForward(
            int dim,
            int? dimOut = null,
            int mult = 4,
            float dropout = 0.0f,
            string approximate = "none"
        ) : base("FeedForward")
        {
            int innerDim = dim * mult;
            int outDim = dimOut ?? dim;

            var projectIn = nn.Sequential(
                nn.Linear(dim, innerDim),
                torch.nn.GELU(approximate == "tanh")
            );

            ff = nn.Sequential(
                projectIn,
                nn.Dropout(dropout),
                nn.Linear(innerDim, outDim)
            );

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return ff.forward(x);
        }
    }

    public class Attention : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> to_q;
        public readonly nn.Module<Tensor, Tensor> to_k;
        public readonly nn.Module<Tensor, Tensor> to_v;

        public readonly nn.Module<Tensor, Tensor> to_k_c;
        public readonly nn.Module<Tensor, Tensor> to_v_c;
        public readonly nn.Module<Tensor, Tensor> to_q_c;

        public readonly ModuleList<nn.Module<Tensor, Tensor>> to_out;
        public readonly nn.Module<Tensor, Tensor> to_out_c;

        public readonly int dim;
        public readonly int heads;
        public readonly int innerDim;
        public readonly float dropout;

        public readonly int? contextDim;
        public readonly bool? contextPreOnly;

        public readonly AttnProcessor processor;

        public Attention(
            AttnProcessor processor,
            int dim,
            int heads = 8,
            int dimHead = 64,
            float dropout = 0.0f,
            int? contextDim = null,
            bool? contextPreOnly = null
        ) : base("Attention")
        {
            this.processor = processor;

            this.dim = dim;
            this.heads = heads;
            this.innerDim = dimHead * heads;
            this.dropout = dropout;

            this.contextDim = contextDim;
            this.contextPreOnly = contextPreOnly;

            to_q = nn.Linear(dim, innerDim);
            to_k = nn.Linear(dim, innerDim);
            to_v = nn.Linear(dim, innerDim);

            if (contextDim is not null)
            {
                to_k_c = nn.Linear(contextDim.Value, innerDim);
                to_v_c = nn.Linear(contextDim.Value, innerDim);
                if (contextPreOnly is not null)
                    to_q_c = nn.Linear(contextDim.Value, innerDim);
            }

            to_out = new ModuleList<nn.Module<Tensor, Tensor>>();
            to_out.append(nn.Linear(innerDim, dim));
            to_out.append(nn.Dropout(dropout));

            if (contextPreOnly is not null && contextPreOnly == false)
                to_out_c = nn.Linear(innerDim, dim);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
            => forward(x, mask: null, rope: null);

        public Tensor forward(
            Tensor x,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null)
        {
            return processor.forward(this, x, mask: mask, rope: rope);
        }

        public (Tensor xOut, Tensor cOut) forward(
            Tensor x,
            Tensor c,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null,
            (Tensor freqs, Tensor xposScale)? cRope = null
        )
        {
            if (processor is not JointAttnProcessor joint)
                throw new InvalidOperationException("Joint attention requires JointAttnProcessor.");

            return joint.forward(this, x, c, mask, rope, cRope);
        }
    }
    public class AttnProcessor : nn.Module<Tensor, Tensor>
    {
        public AttnProcessor() : base("AttnProcessor") { }

        public override Tensor forward(Tensor x)
            => throw new InvalidOperationException("Use forward(attn, x, mask, rope)");
        public Tensor forward(
            Attention attn,
            Tensor x,
            Tensor c,
            Tensor mask,
            (Tensor freqs, Tensor xposScale)? rope,
            (Tensor freqs, Tensor xposScale)? cRope
        )
        {
            // This method is for joint attention and should be overridden by JointAttnProcessor
            throw new InvalidOperationException("Use JointAttnProcessor for joint attention.");
        }

        public Tensor forward(
            Attention attn,
            Tensor x,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null)
        {
            var batch = x.shape[0];

            // projections
            var query = attn.to_q.forward(x);
            var key = attn.to_k.forward(x);
            var value = attn.to_v.forward(x);

            // RoPE / XPOS matches x-transformers: apply before reshaping into heads.
            if (rope.HasValue)
            {
                var (freqs, scaleTensor) = rope.Value;

                float qScale = scaleTensor is not null ? scaleTensor.item<float>() : 1.0f;
                float kScale = scaleTensor is not null ? 1.0f / scaleTensor.item<float>() : 1.0f;

                query = RotaryEmbeddings.ApplyRotaryPosEmb(query, freqs, qScale);
                key = RotaryEmbeddings.ApplyRotaryPosEmb(key, freqs, kScale);
            }

            // reshape into heads
            long innerDim = key.shape[^1];
            long headDim = innerDim / attn.heads;

            query = query.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            key = key.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            value = value.view(batch, -1, attn.heads, headDim).transpose(1, 2);

            // mask broadcast
            Tensor attnMask = null;
            if (mask is not null)
            {
                attnMask = mask;

                if (attnMask.dim() == 2)
                {
                    attnMask = attnMask.unsqueeze(1).unsqueeze(1);
                    attnMask = attnMask.expand(batch, attn.heads, query.size(-2), key.size(-2)).contiguous();
                }
            }

            var ctx = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask: attnMask,
                p: 0.0,
                is_casual: false);

            ctx = ctx.transpose(1, 2).reshape(batch, -1, attn.heads * headDim);
            ctx = ctx.to_type(query.dtype);

            // output projection
            foreach (var m in attn.to_out)
            {
                ctx = m.forward(ctx);
            }
            // post-mask
            if (mask is not null)
            {
                Tensor m = mask;

                if (m.dim() == 2)
                {
                    m = m.unsqueeze(-1);
                }
                else
                {
                    // Python: mask[:, 0, -1]
                    m = m.index(new TensorIndex[] {
                    TensorIndex.Colon,
                    TensorIndex.Single(0),
                    TensorIndex.Single(-1)
                }).unsqueeze(-1);
                }

                ctx = ctx.masked_fill(m.logical_not(), 0.0f);
            }

            return ctx;
        }
    }
    public class JointAttnProcessor : AttnProcessor
    {
        public JointAttnProcessor() : base() { }

        public override Tensor forward(Tensor x)
            => throw new InvalidOperationException("Use forward(attn, x, c, mask, rope, cRope)");

        public (Tensor xOut, Tensor cOut) forward(
            Attention attn,
            Tensor x,
            Tensor c,
            Tensor mask,
            (Tensor freqs, Tensor xposScale)? rope,
            (Tensor freqs, Tensor xposScale)? cRope
        )
        {
            var residual = x;
            var batch = c.shape[0];

            // sample projections
            var query = attn.to_q.forward(x);
            var key = attn.to_k.forward(x);
            var value = attn.to_v.forward(x);

            // context projections
            var cQuery = attn.to_q_c.forward(c);
            var cKey = attn.to_k_c.forward(c);
            var cValue = attn.to_v_c.forward(c);

            // RoPE for x — matches x-transformers: apply before reshaping into heads.
            if (rope.HasValue)
            {
                var (freqs, scaleTensor) = rope.Value;
                float qScale = scaleTensor is not null ? scaleTensor.item<float>() : 1.0f;
                float kScale = scaleTensor is not null ? 1.0f / scaleTensor.item<float>() : 1.0f;

                query = RotaryEmbeddings.ApplyRotaryPosEmb(query, freqs, qScale);
                key   = RotaryEmbeddings.ApplyRotaryPosEmb(key,   freqs, kScale);
            }

            // RoPE for c — matches x-transformers.
            if (cRope.HasValue)
            {
                var (freqs, scaleTensor) = cRope.Value;
                float qScale = scaleTensor is not null ? scaleTensor.item<float>() : 1.0f;
                float kScale = scaleTensor is not null ? 1.0f / scaleTensor.item<float>() : 1.0f;

                cQuery = RotaryEmbeddings.ApplyRotaryPosEmb(cQuery, freqs, qScale);
                cKey   = RotaryEmbeddings.ApplyRotaryPosEmb(cKey,   freqs, kScale);
            }

            // reshape into heads
            long innerDim = key.shape[^1];
            long headDim = innerDim / attn.heads;

            query  = query.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            key    = key.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            value  = value.view(batch, -1, attn.heads, headDim).transpose(1, 2);

            cQuery = cQuery.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            cKey   = cKey.view(batch, -1, attn.heads, headDim).transpose(1, 2);
            cValue = cValue.view(batch, -1, attn.heads, headDim).transpose(1, 2);

            // concat x + c along sequence dim (dim=2 after transpose)
            query = torch.cat(new[] { query, cQuery }, dim: 2);
            key   = torch.cat(new[] { key,   cKey   }, dim: 2);
            value = torch.cat(new[] { value, cValue }, dim: 2);

            // mask: pad mask with True for context tokens
            Tensor attnMask = null;
            if (mask is not null)
            {
                long cLen = c.shape[1];

                attnMask = torch.nn.functional.pad(
                    mask,
                    pad: new long[] { 0, cLen },
                    value: 1f
                );

                attnMask = attnMask.unsqueeze(1).unsqueeze(1);
                attnMask = attnMask.expand(batch, attn.heads, query.size(-2), key.size(-2)).contiguous();
            }

            var outTensor = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask: attnMask,
                p: 0.0,
                is_casual: false);

            // [B, heads, seq, headDim] -> [B, seq, heads*headDim]
            outTensor = outTensor.transpose(1, 2).reshape(batch, -1, attn.heads * headDim);
            outTensor = outTensor.to_type(query.dtype);

            // split back into x and c
            long xLen = residual.shape[1];

            var xOut = outTensor.index(new TensorIndex[] {
                    TensorIndex.Colon,
                    TensorIndex.Slice(0, xLen),
                    TensorIndex.Colon
                });

            var cOut = outTensor.index(new TensorIndex[] {
                    TensorIndex.Colon,
                    TensorIndex.Slice(xLen, outTensor.shape[1]),
                    TensorIndex.Colon
                });

            // linear proj
            var proj = (nn.Module<Tensor, Tensor>)attn.to_out[0];
            var drop = (nn.Module<Tensor, Tensor>)attn.to_out[1];

            xOut = proj.forward(xOut);
            xOut = drop.forward(xOut);

            if (attn.contextPreOnly is not null && attn.contextPreOnly == false)
                cOut = attn.to_out_c.forward(cOut);

            // post-mask for x only
            if (mask is not null)
            {
                var m = mask.unsqueeze(-1);
                xOut = xOut.masked_fill(m.logical_not(), 0.0f);
            }

            return (xOut, cOut);
        }
    }

    // DiTBlock — forward(x, t) → x, with full AdaLN-Zero gating
    public class DiTBlock : nn.Module
    {
        private readonly AdaLayerNormZero attn_norm;
        private readonly Attention attn;

        private readonly LayerNorm ff_norm;
        private readonly FeedForward ff;

        public DiTBlock(int dim, int heads, int dimHead, int ffMult = 4, float dropout = 0.1f)
            : base("DiTBlock")
        {
            attn_norm = new AdaLayerNormZero(dim);

            attn = new Attention(
                processor: new AttnProcessor(),
                dim: dim,
                heads: heads,
                dimHead: dimHead,
                dropout: dropout
            );

            ff_norm = nn.LayerNorm(dim, elementwise_affine: false, eps: 1e-6);
            ff = new FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh");

            RegisterComponents();
        }
        public (Tensor cOut, Tensor xOut) ForwardWithContext(
            Tensor x,
            Tensor c,
            Tensor t,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null,
            (Tensor freqs, Tensor xposScale)? c_rope = null
        )
        {
            throw new InvalidOperationException("DiTBlock does not support context. Use MMDiTBlock instead.");
        }

        public Tensor forward(
            Tensor x,
            Tensor t,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null
        )
        {
            var perf = false;
            var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            // AdaLNZero → returns (normed, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            var (norm, gateMsa, shiftMlp, scaleMlp, gateMlp) = attn_norm.forward(x, t);
            if (perf)
            {
                SynchronizeIfCuda(norm);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] attn_norm_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // Attention
            var attnOut = attn.forward(norm, mask: mask, rope: rope);
            if (perf)
            {
                SynchronizeIfCuda(attnOut);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] attn_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // x = x + gate_msa.unsqueeze(1) * attn_output
            x = x + gateMsa.unsqueeze(1) * attnOut;
            if (perf)
            {
                SynchronizeIfCuda(x);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] attn_residual_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // ff_norm = ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            var ffNorm = ff_norm.forward(x);
            ffNorm = ffNorm * (1 + scaleMlp.unsqueeze(1)) + shiftMlp.unsqueeze(1);
            if (perf)
            {
                SynchronizeIfCuda(ffNorm);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] ff_norm_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // FeedForward
            var ffOut = ff.forward(ffNorm);
            if (perf)
            {
                SynchronizeIfCuda(ffOut);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] ff_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // x = x + gate_mlp.unsqueeze(1) * ff_output
            x = x + gateMlp.unsqueeze(1) * ffOut;
            if (perf)
            {
                SynchronizeIfCuda(x);
                Console.WriteLine($"[CosyVoicePerf.DiTBlock] ff_residual_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
            }

            return x;
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }
    }
    public class MMDiTBlock : nn.Module
    {
        private readonly bool context_pre_only;

        private readonly AdaLayerNormZero attn_norm_x;
        private readonly LayerNorm ff_norm_x;
        private readonly FeedForward ff_x;

        private readonly LayerNorm ff_norm_c;
        private readonly FeedForward ff_c;
        private readonly nn.Module attn_norm_c;

        private readonly Attention attn;

        public MMDiTBlock(
            int dim,
            int heads,
            int dimHead,
            int ffMult = 4,
            float dropout = 0.1f,
            bool context_pre_only = false
        ) : base("MMDiTBlock")
        {
            this.context_pre_only = context_pre_only;

            // context branch norm
            if (context_pre_only)
            {
                attn_norm_c = new AdaLayerNormZero_Final(dim);
            }
            else
            {
                attn_norm_c = new AdaLayerNormZero(dim);
            }

            // x branch norm
            attn_norm_x = new AdaLayerNormZero(dim);

            // joint attention
            attn = new Attention(
                processor: new JointAttnProcessor(),
                dim: dim,
                heads: heads,
                dimHead: dimHead,
                dropout: dropout,
                contextDim: dim,
                contextPreOnly: context_pre_only
            );

            // FFN for context branch
            if (!context_pre_only)
            {
                ff_norm_c = nn.LayerNorm(dim, elementwise_affine: false, eps: 1e-6);
                ff_c = new FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh");
            }

            // FFN for x branch
            ff_norm_x = nn.LayerNorm(dim, elementwise_affine: false, eps: 1e-6);
            ff_x = new FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh");

            RegisterComponents();
        }

        public (Tensor cOut, Tensor xOut) forward(
            Tensor x,
            Tensor c,
            Tensor t,
            Tensor mask = null,
            (Tensor freqs, Tensor xposScale)? rope = null,
            (Tensor freqs, Tensor xposScale)? c_rope = null
        )
        {
            Tensor norm_c = null;
            Tensor c_gate_msa = null, c_shift_mlp = null, c_scale_mlp = null, c_gate_mlp = null;

            // ----- Context branch pre-norm -----
            if (context_pre_only)
            {
                // AdaLayerNormZero_Final returns only the modulated x
                norm_c = ((AdaLayerNormZero_Final)attn_norm_c).forward(c, t);
            }
            else
            {
                var outC = ((AdaLayerNormZero)attn_norm_c).forward(c, t);
                norm_c = outC.Item1;
                c_gate_msa = outC.Item2;
                c_shift_mlp = outC.Item3;
                c_scale_mlp = outC.Item4;
                c_gate_mlp = outC.Item5;
            }

            // ----- X branch pre-norm -----
            var (norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp) =
                attn_norm_x.forward(x, t);

            // ----- Joint Attention -----
            var (xAttnOut, cAttnOut) = attn.forward(
                x: norm_x,
                c: norm_c,
                mask: mask,
                rope: rope,
                cRope: c_rope
            );

            // ----- Context branch post-attention -----
            if (context_pre_only)
            {
                c = null;
            }
            else
            {
                // c = c + gate_msa * attn_out
                c = c + c_gate_msa.unsqueeze(1) * cAttnOut;

                // FFN
                var normC = ff_norm_c.forward(c);
                normC = normC * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1);

                var cFF = ff_c.forward(normC);
                c = c + c_gate_mlp.unsqueeze(1) * cFF;
            }

            // ----- X branch post-attention -----
            x = x + x_gate_msa.unsqueeze(1) * xAttnOut;

            var normX = ff_norm_x.forward(x);
            normX = normX * (1 + x_scale_mlp.unsqueeze(1)) + x_shift_mlp.unsqueeze(1);

            var xFF = ff_x.forward(normX);
            x = x + x_gate_mlp.unsqueeze(1) * xFF;

            return (c, x);
        }
    }

    public class TimestepEmbedding : nn.Module<Tensor, Tensor>
    {
        private readonly SinusPositionEmbedding timeEmbed;
        public readonly nn.Module<Tensor, Tensor> time_mlp;

        public TimestepEmbedding(int dim, int freqEmbedDim = 256) : base("TimestepEmbedding")
        {
            timeEmbed = new SinusPositionEmbedding(freqEmbedDim);
            time_mlp = nn.Sequential(
                nn.Linear(freqEmbedDim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            );
            RegisterComponents();
        }

        public override Tensor forward(Tensor timestep)
        {
            var timeHidden = timeEmbed.forward(timestep);
            timeHidden = timeHidden.to_type(timestep.dtype);
            return time_mlp.forward(timeHidden);
        }
    }

    // RotaryEmbedding — holds inv_freq buffer matching checkpoint key rotary_embed.inv_freq
    public class RotaryEmbedding : nn.Module
    {
        public Tensor inv_freq { get; private set; }

        public RotaryEmbedding(int dim) : base("RotaryEmbedding")
        {
            inv_freq = 1.0f / torch.pow(10000.0f,
                torch.arange(0, dim, 2, dtype: ScalarType.Float32) / dim);
            register_buffer("inv_freq", inv_freq);
            RegisterComponents();
        }

        public (Tensor freqs, Tensor scale) ForwardFromSeqLen(int seqLen)
        {
            var t = torch.arange(seqLen, dtype: ScalarType.Float32, device: inv_freq.device);
            var freqs = torch.outer(t, inv_freq);
            freqs = torch.stack(new[] { freqs, freqs }, dim: -1).reshape(seqLen, inv_freq.shape[0] * 2);
            return (freqs, null);
        }
    }
}
