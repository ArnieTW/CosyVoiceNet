using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class TimestepEmbedding : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> linear_1;
        public readonly nn.Module<Tensor, Tensor> linear_2;
        private readonly nn.Module<Tensor, Tensor> act;

        public TimestepEmbedding(int inChannels, int timeEmbedDim, string actFn) : base("TimestepEmbedding")
        {
            linear_1 = nn.Linear(inChannels, timeEmbedDim);
            act = actFn.Equals("silu", System.StringComparison.OrdinalIgnoreCase) ? nn.SiLU() : nn.Mish();
            linear_2 = nn.Linear(timeEmbedDim, timeEmbedDim);
            RegisterComponents();
        }

        public override Tensor forward(Tensor t)
        {
            t = linear_1.forward(t);
            t = act.forward(t);
            return linear_2.forward(t);
        }
    }
}
