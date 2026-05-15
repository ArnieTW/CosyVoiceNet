using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace CosyVoiceNet.Matcha
{
    public class ResnetBlock1D : nn.Module<Tensor, Tensor>
    {
        public nn.Module<Tensor, Tensor> mlp = null!;
        public Block1D block1 = null!;
        public Block1D block2 = null!;
        public nn.Module<Tensor, Tensor> res_conv = null!;
        protected int timeEmbDim;

        public ResnetBlock1D(int inChannels, int outChannels, int timeEmbDim, int groups = 8) : base("ResnetBlock1D")
        {
            this.timeEmbDim = timeEmbDim;
            mlp = nn.Sequential(nn.Mish(), nn.Linear(timeEmbDim, outChannels));
            block1 = new Block1D(inChannels, outChannels, groups);
            block2 = new Block1D(outChannels, outChannels, groups);
            res_conv = nn.Conv1d(inChannels, outChannels, 1);
            RegisterComponents();
        }

        protected ResnetBlock1D(string moduleName) : base(moduleName)
        {
        }

        public override Tensor forward(Tensor input)
            => Forward(input,
                torch.ones(new long[] { input.shape[0], 1, input.shape[2] }, dtype: input.dtype, device: input.device),
                torch.zeros(new long[] { input.shape[0], timeEmbDim }, dtype: input.dtype, device: input.device));

        public virtual Tensor Forward(Tensor x, Tensor mask, Tensor timeEmb)
        {
            var h = block1.Forward(x, mask);
            var time = mlp.forward(timeEmb).unsqueeze(-1);
            h = h + time;
            h = block2.Forward(h, mask);
            var residual = res_conv.forward(x * mask);
            return h + residual;
        }
    }
}
