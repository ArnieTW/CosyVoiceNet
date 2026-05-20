using System;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

// Equivalent Python file: cosyvoice/transformer/subsampling.py

namespace CosyVoiceNet.Transformers
{
    public abstract class BaseSubsampling : torch.nn.Module<(Tensor, Tensor, long), (Tensor, Tensor, Tensor)>
    {
        public int RightContext { get; protected set; } = 0;
        public int SubsamplingRate { get; protected set; } = 1;

        public BaseSubsampling() : base("BaseSubsampling")
        {
        }

        public virtual Tensor PositionEncoding(Tensor offset, int size)
        {
            throw new NotImplementedException("Position encoding logic should be implemented in derived classes.");
        }
    }

    public class LinearNoSubsampling : BaseSubsampling
    {
        private readonly dynamic Out;
        private readonly dynamic PosEnc;

        public LinearNoSubsampling(int idim, int odim, double dropoutRate, dynamic posEnc) : base()
        {
            Out = torch.nn.Sequential(
                torch.nn.Linear(idim, odim),
                torch.nn.LayerNorm(odim, eps: 1e-5),
                torch.nn.Dropout(dropoutRate)
            );
            PosEnc = posEnc;
            RightContext = 0;
            SubsamplingRate = 1;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            x = Out.forward(x);
            var res = PosEnc.forward((x, offset));
            return (res.Item1, res.Item2, xMask);
        }
    }

    public class EmbeddingNoSubsampling : BaseSubsampling
    {
        private readonly dynamic Embed;
        private readonly dynamic PosEnc;

        public EmbeddingNoSubsampling(int idim, int odim, dynamic posEnc) : base()
        {
            Embed = torch.nn.Embedding(idim, odim);
            PosEnc = posEnc;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            x = Embed.forward(x);
            var res = PosEnc.forward((x, offset));
            return (res.Item1, res.Item2, xMask);
        }
    }

    public class Conv1dSubsampling2 : BaseSubsampling
    {
        private readonly dynamic Conv;
        private readonly dynamic PosEnc;

        public Conv1dSubsampling2(int idim, int odim, double dropoutRate, dynamic posEnc) : base()
        {
            Conv = torch.nn.Sequential(
                torch.nn.Conv1d(idim, odim, 3, padding: 1),
                torch.nn.GELU(),
                torch.nn.Conv1d(odim, odim, 3, stride: 2, padding: 1),
                torch.nn.GELU()
            );
            PosEnc = posEnc;
            SubsamplingRate = 2;
            RightContext = 4;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            var time = (int)x.shape[1];
            x = x.transpose(1, 2);
            x = Conv.forward(x);
            x = x.transpose(1, 2);
            var res = PosEnc.forward((x, offset));
            int start = (time + 1) % 2;
            var outLen = (time - start + 1) / 2;
            var outMask = xMask.narrow(2, start, outLen);
            return (res.Item1, res.Item2, outMask);
        }
    }

    public class Conv2dSubsampling4 : BaseSubsampling
    {
        private readonly dynamic Conv;
        private readonly dynamic OutLayer;
        private readonly dynamic PosEnc;

        public Conv2dSubsampling4(int idim, int odim, dynamic posEnc) : base()
        {
            Conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU()
            );
            OutLayer = torch.nn.Linear(odim * (((idim - 1) / 2 - 1) / 2), odim);
            PosEnc = posEnc;
            SubsamplingRate = 4;
            RightContext = 6;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            x = x.unsqueeze(1);
            x = Conv.forward(x);
            var b = x.shape[0];
            var t = x.shape[2];
            var f = x.shape[3];
            x = OutLayer.forward(x.transpose(1, 2).contiguous().view(b, t, -1));
            var res = PosEnc.forward((x, offset));
            return (res.Item1, res.Item2, xMask.narrow(2, 2, xMask.shape[2] / 4));
        }
    }

    public class Conv2dSubsampling6 : BaseSubsampling
    {
        private readonly dynamic Conv;
        private readonly dynamic Linear;
        private readonly dynamic PosEnc;

        public Conv2dSubsampling6(int idim, int odim, dynamic posEnc) : base()
        {
            Conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 5, 3),
                torch.nn.ReLU()
            );
            Linear = torch.nn.Linear(odim * (((idim - 1) / 2 - 2) / 3), odim);
            PosEnc = posEnc;
            SubsamplingRate = 6;
            RightContext = 10;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            x = x.unsqueeze(1);
            x = Conv.forward(x);
            var b = x.shape[0];
            var t = x.shape[2];
            var f = x.shape[3];
            x = Linear.forward(x.transpose(1, 2).contiguous().view(b, t, -1));
            var res = PosEnc.forward((x, offset));
            return (res.Item1, res.Item2, xMask.narrow(2, 2, xMask.shape[2] / 6));
        }
    }

    public class Conv2dSubsampling8 : BaseSubsampling
    {
        private readonly dynamic Conv;
        private readonly dynamic Linear;
        private readonly dynamic PosEnc;

        public Conv2dSubsampling8(int idim, int odim, dynamic posEnc) : base()
        {
            Conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2),
                torch.nn.ReLU()
            );
            Linear = torch.nn.Linear(odim * ((((idim - 1) / 2 - 1) / 2 - 1) / 2), odim);
            PosEnc = posEnc;
            SubsamplingRate = 8;
            RightContext = 14;
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, long) inputs)
        {
            var (x, xMask, offset) = inputs;
            x = x.unsqueeze(1);
            x = Conv.forward(x);
            var b = x.shape[0];
            var t = x.shape[2];
            var f = x.shape[3];
            x = Linear.forward(x.transpose(1, 2).contiguous().view(b, t, -1));
            var res = PosEnc.forward((x, offset));
            return (res.Item1, res.Item2, xMask.narrow(2, 2, xMask.shape[2] / 8));
        }
    }
}
