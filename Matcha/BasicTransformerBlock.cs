using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class GeluProjection : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> proj;

        public GeluProjection(int dim, int dimOut) : base("GELU")
        {
            proj = nn.Linear(dim, dimOut);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = proj.forward(x);
            return nn.functional.gelu(x);
        }
    }

    public class MatchaFeedForward : nn.Module<Tensor, Tensor>
    {
        public readonly ModuleList<nn.Module> net;

        public MatchaFeedForward(int dim, int dimOut, float dropout, string activationFn) : base("FeedForward")
        {
            net = new ModuleList<nn.Module>();
            net.append(new GeluProjection(dim, dimOut));
            net.append(nn.Dropout(dropout));
            net.append(nn.Linear(dimOut, dim));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            foreach (dynamic module in net)
                x = module.forward(x);
            return x;
        }
    }

    public class MatchaAttention : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> to_q;
        public readonly nn.Module<Tensor, Tensor> to_k;
        public readonly nn.Module<Tensor, Tensor> to_v;
        public readonly ModuleList<nn.Module<Tensor, Tensor>> to_out;
        private readonly int heads;

        public MatchaAttention(int dim, int numHeads, int headDim, float dropout) : base("Attention")
        {
            heads = numHeads;
            var innerDim = numHeads * headDim;
            to_q = nn.Linear(dim, innerDim, hasBias: false);
            to_k = nn.Linear(dim, innerDim, hasBias: false);
            to_v = nn.Linear(dim, innerDim, hasBias: false);
            to_out = new ModuleList<nn.Module<Tensor, Tensor>>();
            to_out.append(nn.Linear(innerDim, dim));
            to_out.append(nn.Dropout(dropout));
            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => Forward(x, null);

        public Tensor Forward(Tensor x, Tensor attentionMask)
        {
            var batch = x.shape[0];
            var q = to_q.forward(x);
            var k = to_k.forward(x);
            var v = to_v.forward(x);
            var headDim = q.shape[^1] / heads;

            q = q.view(batch, -1, heads, headDim).transpose(1, 2);
            k = k.view(batch, -1, heads, headDim).transpose(1, 2);
            v = v.view(batch, -1, heads, headDim).transpose(1, 2);

            Tensor attnMask = null;
            if (attentionMask is not null)
            {
                attnMask = attentionMask;
                if (attnMask.dim() == 2)
                    attnMask = attnMask.unsqueeze(1).unsqueeze(1);
                else if (attnMask.dim() == 3)
                    attnMask = attnMask.unsqueeze(1);
                attnMask = attnMask.to(q.device).to(q.dtype);
            }

            var output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask: attnMask, p: 0.0, is_casual: false);
            output = output.transpose(1, 2).reshape(batch, -1, heads * headDim);
            foreach (var module in to_out)
                output = module.forward(output);
            return output;
        }
    }

    public class BasicTransformerBlock : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> norm1;
        public readonly MatchaAttention attn1;
        public readonly nn.Module<Tensor, Tensor> norm3;
        public readonly MatchaFeedForward ff;

        public BasicTransformerBlock(int dim, int numHeads, int headDim, float dropout, string activationFn) : base("BasicTransformerBlock")
        {
            norm1 = nn.LayerNorm(dim);
            attn1 = new MatchaAttention(dim, numHeads, headDim, dropout);
            norm3 = nn.LayerNorm(dim);
            ff = new MatchaFeedForward(dim, dim * 4, dropout, activationFn);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
            => Forward(x, null, null);

        public Tensor Forward(Tensor hiddenStates, Tensor attentionMask = null, Tensor timestep = null)
        {
            var normHidden = norm1.forward(hiddenStates);
            var attnOutput = attn1.Forward(normHidden, attentionMask);
            hiddenStates = attnOutput + hiddenStates;
            normHidden = norm3.forward(hiddenStates);
            var ffOutput = ff.forward(normHidden);
            return ffOutput + hiddenStates;
        }
    }
}
