using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class Upsample1D : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module<Tensor, Tensor> conv;

        public Upsample1D(int channels, bool useConvTranspose = true) : base("Upsample1D")
        {
            if (useConvTranspose)
            {
                conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1);
            }
            else
            {
                conv = nn.Conv1d(channels, channels, 3, 1, 1);
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return conv.forward(x);
        }
    }
}
