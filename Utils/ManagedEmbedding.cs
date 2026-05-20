using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

// Equivalent Python file: cosyvoice/utils/managed_embedding.py

namespace CosyVoiceNet.Utils
{
    public class ManagedEmbedding : torch.nn.Module<Tensor, Tensor>
    {
        public Parameter weight;

        public ManagedEmbedding(int vocabSize, int embeddingSize) : base("ManagedEmbedding")
        {
            var w = torch.empty(vocabSize, embeddingSize);
            XavierUniform(w);
            weight = new Parameter(w);
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
            // Handle both 1D and 2D input tensors (like PyTorch Embedding)
            var originalShape = input.shape;
            var flatInput = input.reshape(-1).to(torch.int64);
            var embedded = weight.index_select(0, flatInput);

            // Reshape back to [*originalShape, embeddingSize]
            if (originalShape.Length > 1)
            {
                var newShape = originalShape.Concat(new long[] { embedded.shape[1] }).ToArray();
                return embedded.reshape(newShape);
            }
            return embedded;
        }

        public Tensor GetWeight() => weight;
    }
}