using System;
using TorchSharp;
using static TorchSharp.torch;

// Equivalent Python file: cosyvoice/transformer/embedding.py
namespace CosyVoiceNet.Transformers
{
    public class PositionalEncoding : torch.nn.Module<Tensor, Tensor>
    {
        protected readonly int dModel;
        protected readonly double xscale;
        protected readonly dynamic dropout;
        protected Tensor pe;
        protected readonly int maxLen;

        public PositionalEncoding(int dModel, double dropoutRate, int maxLen = 5000, bool reverse = false) : base("PositionalEncoding")
        {
            this.dModel = dModel;
            this.xscale = Math.Sqrt(dModel);
            this.dropout = torch.nn.Dropout(p: dropoutRate);
            this.maxLen = maxLen;

            pe = torch.zeros(maxLen, dModel);
            var position = torch.arange(0, maxLen, dtype: ScalarType.Float32).unsqueeze(1);
            var divTerm = torch.exp(torch.arange(0, dModel, 2, dtype: ScalarType.Float32) * -(Math.Log(10000.0) / dModel));
            pe.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(0, null, 2) }).copy_(torch.sin(position * divTerm));
            pe.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(1, null, 2) }).copy_(torch.cos(position * divTerm));
            pe = pe.unsqueeze(0);
        }

        public override Tensor forward(Tensor x)
        {
            return ForwardWithPosition(x).encoded;
        }

        public virtual (Tensor encoded, Tensor posEmb) ForwardWithPosition(Tensor x, long offset = 0)
        {
            var posEmb = position_encoding(offset, (int)x.shape[1], false, x.device);
            x = x * xscale + posEmb;
            return (dropout.forward(x), dropout.forward(posEmb));
        }

        public virtual Tensor position_encoding(long offset, int size, bool applyDropout = true, Device? device = null)
        {
            if (offset + size > maxLen)
            {
                throw new ArgumentOutOfRangeException($"Offset {offset} and size {size} exceed max length {maxLen}.");
            }

            var sourcePe = pe;
            if (device != null && sourcePe.device.ToString() != device.ToString())
            {
                sourcePe = sourcePe.to(device);
            }

            var posEmb = sourcePe.narrow(1, (int)offset, size);

            if (applyDropout)
            {
                posEmb = dropout.forward(posEmb);
            }

            return posEmb;
        }
    }

    public class RelPositionalEncoding : PositionalEncoding
    {
        public RelPositionalEncoding(int dModel, double dropoutRate, int maxLen = 5000)
            : base(dModel, dropoutRate, maxLen, reverse: true)
        {
        }

        public override (Tensor encoded, Tensor posEmb) ForwardWithPosition(Tensor x, long offset = 0)
        {
            var posEmb = position_encoding(offset, (int)x.shape[1], false, x.device);
            x = x * xscale;
            return (dropout.forward(x), dropout.forward(posEmb));
        }
    }

    public class EspnetRelPositionalEncoding : PositionalEncoding
    {
        public EspnetRelPositionalEncoding(int dModel, double dropoutRate, int maxLen = 5000)
            : base(dModel, dropoutRate, 1)
        {
            pe = null;
            ExtendPe(torch.zeros(new long[] { 1, maxLen }, dtype: ScalarType.Float32));
        }

        private void ExtendPe(Tensor x)
        {
            var requiredLen = x.shape[1] * 2 - 1;
            if (pe is not null && pe.shape[1] >= requiredLen)
            {
                if (pe.dtype != x.dtype || pe.device.ToString() != x.device.ToString())
                    pe = pe.to(x.device).to_type(x.dtype);
                return;
            }

            var len = (int)x.shape[1];
            var pePositive = torch.zeros(new long[] { len, dModel }, dtype: ScalarType.Float32, device: x.device);
            var peNegative = torch.zeros(new long[] { len, dModel }, dtype: ScalarType.Float32, device: x.device);
            var position = torch.arange(0, len, dtype: ScalarType.Float32, device: x.device).unsqueeze(1);
            var divTerm = torch.exp(torch.arange(0, dModel, 2, dtype: ScalarType.Float32, device: x.device) * -(Math.Log(10000.0) / dModel));
            pePositive.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(0, null, 2) }).copy_(torch.sin(position * divTerm));
            pePositive.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(1, null, 2) }).copy_(torch.cos(position * divTerm));
            peNegative.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(0, null, 2) }).copy_(torch.sin(-1 * position * divTerm));
            peNegative.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Slice(1, null, 2) }).copy_(torch.cos(-1 * position * divTerm));

            pePositive = torch.flip(pePositive, new long[] { 0 }).unsqueeze(0);
            peNegative = peNegative.narrow(0, 1, len - 1).unsqueeze(0);
            pe = torch.cat(new[] { pePositive, peNegative }, dim: 1).to(x.device).to_type(x.dtype);
        }

        public override (Tensor encoded, Tensor posEmb) ForwardWithPosition(Tensor x, long offset = 0)
        {
            ExtendPe(x);
            x = x * xscale;
            var posEmb = position_encoding(offset, (int)x.shape[1], false, x.device);
            return (dropout.forward(x), dropout.forward(posEmb));
        }

        public override Tensor position_encoding(long offset, int size, bool applyDropout = true, Device? device = null)
        {
            var target = pe;
            if (device != null && target.device.ToString() != device.ToString())
                target = target.to(device);

            var center = target.shape[1] / 2;
            var start = center - size - offset + 1;
            var end = center + size + offset;
            var posEmb = target.narrow(1, checked((int)start), checked((int)(end - start)));
            return applyDropout ? dropout.forward(posEmb) : posEmb;
        }
    }

    public class NoPositionalEncoding : PositionalEncoding
    {
        public NoPositionalEncoding(int dModel, double dropoutRate)
            : base(dModel, dropoutRate, 1)
        {
        }

        public override (Tensor encoded, Tensor posEmb) ForwardWithPosition(Tensor x, long offset = 0)
        {
            var posEmb = torch.zeros(new long[] { 1, x.shape[1], dModel }, dtype: x.dtype, device: x.device);
            return (dropout.forward(x), posEmb);
        }

        public override Tensor position_encoding(long offset, int size, bool applyDropout = true, Device? device = null)
        {
            return torch.zeros(new long[] { 1, size, dModel }, dtype: ScalarType.Float32, device: device ?? torch.CPU);
        }
    }
}
