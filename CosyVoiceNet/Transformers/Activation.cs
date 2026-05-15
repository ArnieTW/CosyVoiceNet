using System;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Transformers
{
    // Port of cosyvoice.transformer.activation

    // Equivalent Python file: cosyvoice/transformers/activation.py
    public static class Activation
    {
        public static Tensor Swish(Tensor x)
        {
            return x * torch.sigmoid(x);
        }

        // Snake activation as in the original repo
        public class Snake : torch.nn.Module<Tensor,Tensor>
        {
            private readonly TorchSharp.Modules.Parameter alpha;
            private readonly bool alphaLogScale;
            private const double NoDivByZero = 1e-9;

            public Snake(int inFeatures, double alphaInit = 1.0, bool alphaTrainable = true, bool alphaLogScale = false) : base("Snake")
            {
                this.alphaLogScale = alphaLogScale;

                // Initialize alpha
                if (alphaLogScale)
                {
                    this.alpha = new TorchSharp.Modules.Parameter(torch.zeros(new long[] { inFeatures }) * alphaInit);
                }
                else
                {
                    this.alpha = new TorchSharp.Modules.Parameter(torch.ones(new long[] { inFeatures }) * alphaInit);
                }

                this.alpha.requires_grad = alphaTrainable;
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                // Reshape alpha to match input dimensions
                var reshapedAlpha = alpha.unsqueeze(0).unsqueeze(-1); // Align with [B, C, T] input shape
                if (alphaLogScale)
                {
                    reshapedAlpha = reshapedAlpha.exp();
                }

                // Compute Snake activation
                return x + (1.0 / (reshapedAlpha + NoDivByZero)) * torch.pow(torch.sin(x * reshapedAlpha), 2);
            }
        }
    }
}
