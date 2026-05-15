using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.flow
{
    // Port of cosyvoice/flow/length_regulator.py InterpolateRegulator + length regulator behavior
    // Exported from #file:CosyVoice\cosyvoice\flow\length_regulator.py
    public class LengthRegulator : nn.Module<Tensor, (Tensor, Tensor)>
    {
        private readonly nn.Module<Tensor, Tensor> model;
        private readonly List<int> samplingRatios;

        public LengthRegulator(int channels, List<int> samplingRatios, int? outChannels = null, int groups = 1) : base("LengthRegulator")
        {
            this.samplingRatios = samplingRatios;
            var outChannelsValue = outChannels ?? channels;

            var modules = new List<nn.Module<Tensor, Tensor>>();
            foreach (var _ in samplingRatios)
            {
                modules.Add(nn.Conv1d(channels, channels, 3, 1, 1));
                modules.Add(nn.GroupNorm(groups, channels));
                modules.Add(nn.Mish());
            }
            modules.Add(nn.Conv1d(channels, outChannelsValue, 1, 1));
            model = nn.Sequential(modules.ToArray());
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            throw new NotImplementedException("Forward method requires ylens parameter.");
        }

        public (Tensor, Tensor) ForwardWithLengths(Tensor x, Tensor ylens)
        {
            var maxLen = ylens.max().item<long>();
            var mask = (~CosyVoiceNet.Utils.Mask.MakePadMask(ylens, (int)maxLen)).to(x).unsqueeze(-1);
            x = interpolate(x.transpose(1, 2).contiguous(), size: new long[] { maxLen }, mode: InterpolationMode.Linear);
            var output = model.forward(x).transpose(1, 2).contiguous();
            return (output * mask, ylens);
        }

        public (Tensor, int) inference(Tensor x1, Tensor x2, int melLen1, int melLen2, int inputFrameRate = 50)
        {
            if (x2.shape[1] > 40)
            {
                var overlapFrames = (int)(20 / (double)inputFrameRate * 22050 / 256);
                var x2Head = interpolate(x2.narrow(1, 0, 20).transpose(1, 2).contiguous(), size: new long[] { overlapFrames }, mode: InterpolationMode.Linear);
                var x2Mid = interpolate(x2.narrow(1, 20, x2.shape[1] - 40).transpose(1, 2).contiguous(), size: new long[] { melLen2 - overlapFrames * 2 }, mode: InterpolationMode.Linear);
                var x2Tail = interpolate(x2.narrow(1, x2.shape[1] - 20, 20).transpose(1, 2).contiguous(), size: new long[] { overlapFrames }, mode: InterpolationMode.Linear);

                x2 = torch.cat(new[] { x2Head, x2Mid, x2Tail }, 2);
            }
            else
            {
                x2 = interpolate(x2.transpose(1, 2).contiguous(), size: new long[] { melLen2 }, mode: InterpolationMode.Linear);
            }

            Tensor x;
            if (x1.shape[1] != 0)
            {
                x1 = interpolate(x1.transpose(1, 2).contiguous(), size: new long[] { melLen1 }, mode: InterpolationMode.Linear);
                x = torch.cat(new[] { x1, x2 }, 2);
            }
            else
            {
                x = x2;
            }

            var output = model.forward(x).transpose(1, 2).contiguous();
            return (output, melLen1 + melLen2);
        }
    }
}
