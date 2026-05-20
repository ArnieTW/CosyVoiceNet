// Enhancing Losses.cs to align with TorchSharp usage patterns
// Ensuring full C# managed implementation of loss functions
// Equivalent Python file: cosyvoice/utils/losses.py

using System;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.Utils
{
    public static class Losses
    {
        public static Tensor TprLoss(Tensor[] discRealOutputs, Tensor[] discGeneratedOutputs, double tau)
        {
            var loss = torch.zeros(new long[] { 1 });
            for (int i = 0; i < discRealOutputs.Length; i++)
            {
                var dr = discRealOutputs[i];
                var dg = discGeneratedOutputs[i];
                var m_DG = (dr - dg).median();
                var mask = dr.lt(dg + m_DG);
                var diff = (dr - dg) - m_DG;
                var L_rel = diff.masked_select(mask).pow(2).mean();
                loss = loss + (tau - torch.nn.functional.relu(tau - L_rel));
            }
            return loss;
        }

        public static Tensor MelLoss(Tensor realSpeech, Tensor generatedSpeech, Func<Tensor, Tensor>[] melTransforms)
        {
            var loss = torch.zeros(new long[] { 1 });
            foreach (var transform in melTransforms)
            {
                var melR = transform(realSpeech);
                var melG = transform(generatedSpeech);
                loss = loss + torch.nn.functional.l1_loss(melG, melR);
            }
            return loss;
        }

        public static Tensor GeneratorLoss(Tensor yDGs)
        {
            return nn.functional.mse_loss(yDGs, torch.zeros_like(yDGs));
        }

        public static Tensor FeatureLoss(Tensor fmapRs, Tensor fmapGs)
        {
            return nn.functional.l1_loss(fmapRs, fmapGs);
        }

        public static Tensor DiscriminatorLoss(Tensor yDRs, Tensor yDGs)
        {
            return nn.functional.mse_loss(yDRs, torch.ones_like(yDRs)) +
                   nn.functional.mse_loss(yDGs, torch.zeros_like(yDGs));
        }

        public class DPOLoss : torch.nn.Module<Tensor, Tensor>
        {
            private readonly double beta;
            private readonly double labelSmoothing;
            private readonly bool ipo;

            public DPOLoss(double beta, double labelSmoothing = 0.0, bool ipo = false) : base("DPOLoss")
            {
                this.beta = beta;
                this.labelSmoothing = labelSmoothing;
                this.ipo = ipo;
            }

            public override Tensor forward(Tensor input)
            {
                var inputs = input.chunk(4, 0);
                var policyChosenLogps = inputs[0];
                var policyRejectedLogps = inputs[1];
                var referenceChosenLogps = inputs[2];
                var referenceRejectedLogps = inputs[3];

                var piLogRatios = policyChosenLogps - policyRejectedLogps;
                var refLogRatios = referenceChosenLogps - referenceRejectedLogps;
                var logits = piLogRatios - refLogRatios;

                Tensor losses;
                if (ipo)
                {
                    losses = (logits - 1 / (2 * beta)).pow(2);
                }
                else
                {
                    losses = -torch.nn.functional.logsigmoid(beta * logits) * (1 - labelSmoothing);
                }

                return losses.mean();
            }
        }
    }
}
