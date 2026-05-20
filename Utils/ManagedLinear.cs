using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

// Equivalent Python file: cosyvoice/utils/managed_linear.py

namespace CosyVoiceNet.Utils
{
    public class ManagedLinear : torch.nn.Module<Tensor, Tensor>
    {
        private Parameter weight;
        private Parameter bias;
        public bool HasBias { get; }

        public ManagedLinear(int inputSize, int outputSize, bool hasBias = true) : base("ManagedLinear")
        {
            HasBias = hasBias;
            var w = torch.empty(outputSize, inputSize);
            XavierUniform(w);
            weight = new Parameter(w);
            if (hasBias)
                bias = new Parameter(torch.zeros(outputSize));
            RegisterComponents();
        }

        private void XavierUniform(Tensor tensor)
        {
            var fanIn = tensor.size(1);
            var fanOut = tensor.size(0);
            var std = Math.Sqrt(2.0 / (fanIn + fanOut));
            var bound = Math.Sqrt(3.0) * std;
            tensor.uniform_(-bound, bound);
        }

        public override Tensor forward(Tensor input)
        {
            var outputSize = weight.shape[0];
            if (input.shape.Length == 2)
            {
                if (HasBias)
                    return torch.addmm(bias, input, weight.transpose(0, 1), beta: 1.0f, alpha: 1.0f);
                return torch.mm(input, weight.transpose(0, 1));
            }

            var leadingShape = input.shape.Take(input.shape.Length - 1).ToArray();
            var flatInput = input.reshape(-1, input.shape[^1]);
            var flatOutput = HasBias
                ? torch.addmm(bias, flatInput, weight.transpose(0, 1), beta: 1.0f, alpha: 1.0f)
                : torch.mm(flatInput, weight.transpose(0, 1));
            return flatOutput.reshape(leadingShape.Concat(new[] { outputSize }).ToArray());
        }
    }
}
