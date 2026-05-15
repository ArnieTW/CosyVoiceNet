using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class Block1D : nn.Module<Tensor, Tensor>
    {
        public nn.Module<Tensor, Tensor> block = null!;

        protected Block1D(string moduleName) : base(moduleName)
        {
        }

        public Block1D(int inChannels, int outChannels, int groups = 8) : base("Block1D")
        {
            block = nn.Sequential(
                nn.Conv1d(inChannels, outChannels, 3, 1, 1),
                nn.GroupNorm(groups, outChannels),
                nn.Mish());
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
            => Forward(x, torch.ones(new long[] { x.shape[0], 1, x.shape[2] }, dtype: x.dtype, device: x.device));

        public virtual Tensor Forward(Tensor x, Tensor mask)
        {
            var output = block.forward(x * mask);
            return output * mask;
        }
    }
}
