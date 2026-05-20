using System;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class SinusoidalPosEmb : nn.Module<Tensor, Tensor>
    {
        private readonly int dim;

        public SinusoidalPosEmb(int dim) : base("SinusoidalPosEmb")
        {
            this.dim = dim;
        }

        public override Tensor forward(Tensor x)
            => Forward(x);

        public Tensor Forward(Tensor x, float scale = 1000.0f)
        {
            if (x.dim() < 1)
                x = x.unsqueeze(0);
            var halfDim = dim / 2;
            var emb = torch.arange(0, halfDim, dtype: ScalarType.Float32, device: x.device);
            emb = torch.exp(emb * (float)(-Math.Log(10000) / (halfDim - 1)));
            emb = scale * x.unsqueeze(1) * emb.unsqueeze(0);
            return torch.cat(new[] { torch.sin(emb), torch.cos(emb) }, dim: -1);
        }
    }
}
