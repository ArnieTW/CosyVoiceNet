using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class Downsample1D : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> conv;

        public Downsample1D(int channels) : base("Downsample1D")
        {
            conv = nn.Conv1d(channels, channels, 3, 2, 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return conv.forward(x);
        }
    }
}
