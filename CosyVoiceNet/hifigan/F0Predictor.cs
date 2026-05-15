// Equivalent Python file: cosyvoice/hifigan/f0_predictor.py
using System;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using CosyVoiceNet.TorchSharpUtils;

namespace CosyVoiceNet.hifigan
{
    public class ConvRNNF0Predictor : nn.Module<Tensor, Tensor>
    {
        public readonly ModuleList<nn.Module> condnet;
        public readonly Linear classifier;

        public ConvRNNF0Predictor(int numClass = 1, int inChannels = 80, int condChannels = 512)
            : base("ConvRNNF0Predictor")
        {
            condnet = nn.ModuleList<nn.Module>();
            condnet.append(new WeightNormedConv1d(inChannels, condChannels, 3, padding: 1));  // [0]
            condnet.append(nn.ELU());                                                          // [1]
            condnet.append(new WeightNormedConv1d(condChannels, condChannels, 3, padding: 1)); // [2]
            condnet.append(nn.ELU());                                                          // [3]
            condnet.append(new WeightNormedConv1d(condChannels, condChannels, 3, padding: 1)); // [4]
            condnet.append(nn.ELU());                                                          // [5]
            condnet.append(new WeightNormedConv1d(condChannels, condChannels, 3, padding: 1)); // [6]
            condnet.append(nn.ELU());                                                          // [7]
            condnet.append(new WeightNormedConv1d(condChannels, condChannels, 3, padding: 1)); // [8]
            condnet.append(nn.ELU());                                                          // [9]
            classifier = nn.Linear(condChannels, numClass);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            foreach (var m in condnet)
                x = ((nn.Module<Tensor, Tensor>)m).forward(x);
            x = x.transpose(1, 2);
            return torch.abs(classifier.forward(x).squeeze(-1));
        }
    }

    public class CausalConvRNNF0Predictor : nn.Module<Tensor, Tensor>
    {
        public readonly ModuleList<nn.Module> condnet;
        public readonly Linear classifier;

        public CausalConvRNNF0Predictor(int numClass = 1, int inChannels = 80, int condChannels = 512)
            : base("CausalConvRNNF0Predictor")
        {
            condnet = nn.ModuleList<nn.Module>();
            condnet.append(new WeightNormedCausalConv1d(inChannels, condChannels, 4, causalType: "right")); // [0]
            condnet.append(nn.ELU());                                                                        // [1]
            condnet.append(new WeightNormedCausalConv1d(condChannels, condChannels, 3, causalType: "left")); // [2]
            condnet.append(nn.ELU());                                                                        // [3]
            condnet.append(new WeightNormedCausalConv1d(condChannels, condChannels, 3, causalType: "left")); // [4]
            condnet.append(nn.ELU());                                                                        // [5]
            condnet.append(new WeightNormedCausalConv1d(condChannels, condChannels, 3, causalType: "left")); // [6]
            condnet.append(nn.ELU());                                                                        // [7]
            condnet.append(new WeightNormedCausalConv1d(condChannels, condChannels, 3, causalType: "left")); // [8]
            condnet.append(nn.ELU());                                                                        // [9]
            classifier = nn.Linear(condChannels, numClass);
            RegisterComponents();
        }

        public int CausalPadding => ((WeightNormedCausalConv1d)condnet[0]).CausalPadding;

        public override Tensor forward(Tensor x) => forward(x, finalize: true);

        public Tensor forward(Tensor x, bool finalize)
        {
            var conv0 = (WeightNormedCausalConv1d)condnet[0];
            Tensor cur;
            if (finalize)
            {
                cur = conv0.forward(x);
            }
            else
            {
                long T = x.shape[2];
                var xMain = x.narrow(2, 0, T - conv0.CausalPadding);
                var xCache = x.narrow(2, T - conv0.CausalPadding, conv0.CausalPadding);
                cur = conv0.forward(xMain, xCache);
            }
            for (int i = 1; i < condnet.Count; i++)
                cur = ((nn.Module<Tensor, Tensor>)condnet[i]).forward(cur);
            cur = cur.transpose(1, 2);
            return torch.abs(classifier.forward(cur).squeeze(-1));
        }
    }
}
