// Equivalent Python file: cosyvoice/transformer/positionwise_feed_forward.py
using System;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Transformers
{
    public class PositionwiseFeedForward : torch.nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> w_1;
        public readonly nn.Module<Tensor, Tensor> w_2;
        public readonly nn.Module<Tensor, Tensor> dropout;
        private readonly Func<Tensor, Tensor> activation;

        public PositionwiseFeedForward(int idim, int hiddenUnits, double dropoutRate, Func<Tensor, Tensor> activationFn = null) : base("PositionwiseFeedForward")
        {
            w_1 = torch.nn.Linear(idim, hiddenUnits);
            activation = activationFn ?? (x => torch.nn.functional.relu(x));
            dropout = torch.nn.Dropout(dropoutRate);
            w_2 = torch.nn.Linear(hiddenUnits, idim);

            RegisterComponents();
        }

        public override Tensor forward(Tensor xs)
        {
            return w_2.forward(dropout.forward(activation(w_1.forward(xs))));
        }
    }

    public class MoEFFNLayer : torch.nn.Module<Tensor, Tensor>
    {
        private readonly dynamic gate;
        private readonly dynamic experts;
        private readonly int nExpertPerToken;

        public MoEFFNLayer(int nExpert, int nExpertPerToken, int idim, int hiddenUnits, double dropoutRate, dynamic activationFn = null) : base("MoEFFNLayer")
        {
            gate = torch.nn.Linear(idim, nExpert, false);
            experts = torch.nn.ModuleList();
            for (int i = 0; i < nExpert; i++)
            {
                experts.append(new PositionwiseFeedForward(idim, hiddenUnits, dropoutRate, activationFn));
            }
            this.nExpertPerToken = nExpertPerToken;

            RegisterComponents();
        }

        public override Tensor forward(Tensor xs)
        {
            var shape = xs.shape;
            var B = shape[0];
            var L = shape[1];
            var D = shape[2];
            xs = xs.view(new long[] { -1, D });
            var router = gate.forward(xs);
            var topk = torch.topk(router, nExpertPerToken);
            var logits = topk.values;
            var indices = topk.indices;
            var weights = torch.nn.functional.softmax(logits, dim: 1, dtype: xs.dtype);
            var output = torch.zeros_like(xs);

            for (int i = 0; i < experts.Count; i++)
            {
                var mask = indices == i;
                var batchIdx = torch.where(mask)[0];
                if (batchIdx.numel() > 0)
                {
                    var selectedWeights = weights.index_select(0, batchIdx).unsqueeze(1);
                    var selectedInputs = xs.index_select(0, batchIdx);
                    var expertOutput = experts[i].forward(selectedInputs);

                    // Manually update the output tensor
                    var updatedOutput = output.index_select(0, batchIdx) + selectedWeights * expertOutput;
                    output = output.index_copy(0, batchIdx, updatedOutput);
                }
            }

            return output.view(new long[] { B, L, D });
        }
    }
}
