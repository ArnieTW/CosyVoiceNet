using System;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Transformers
{
    /// <summary>
    /// Label-smoothing loss implementation.
    /// </summary>
    public class LabelSmoothingLoss : nn.Module<(Tensor, Tensor), Tensor>
    {
        private readonly int _paddingIdx;
        private readonly float _confidence;
        private readonly float _smoothing;
        private readonly int _size;
        private readonly bool _normalizeLength;
        private readonly dynamic _criterion;

        public LabelSmoothingLoss(int size, int paddingIdx, float smoothing, bool normalizeLength = false)
            : base("LabelSmoothingLoss")
        {
            _paddingIdx = paddingIdx;
            _confidence = 1.0f - smoothing;
            _smoothing = smoothing;
            _size = size;
            _normalizeLength = normalizeLength;
            _criterion = torch.nn.KLDivLoss(reduction: torch.nn.Reduction.None);

            RegisterComponents();
        }

        public override Tensor forward((Tensor, Tensor) inputs)
        {
            var (x, target) = inputs;

            if (x.size(2) != _size)
            {
                throw new ArgumentException("Input tensor size does not match the number of classes.");
            }

            var batchSize = x.size(0);
            x = x.view(-1, _size);
            target = target.view(-1);

            // Create true distribution tensor
            var trueDist = torch.zeros_like(x);
            trueDist.fill_(_smoothing / (_size - 1));

            // Mask padding indices
            var ignore = target == _paddingIdx;
            var total = target.numel() - ignore.sum().item<long>();
            target = target.masked_fill(ignore, 0);

            // Scatter confidence values into true distribution
            trueDist.scatter_(1, target.unsqueeze(1), _confidence);

            // Compute KL divergence loss
            var kl = _criterion(torch.nn.functional.log_softmax(x, dim: 1), trueDist);

            // Normalize loss
            var denom = _normalizeLength ? total : batchSize;
            return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom;
        }
    }
}

// Equivalent Python file: cosyvoice/transformer/label_smoothing_loss.py