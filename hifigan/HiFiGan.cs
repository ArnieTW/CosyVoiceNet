using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules; // Correct namespace for TorchSharp modules
using static CosyVoiceNet.Utils.Losses; // Use static import for Losses methods

// Equivalent Python file: cosyvoice/hifigan/hifigan.py

namespace CosyVoiceNet.hifigan
{
    public class HiFiGan : nn.Module
    {
        private readonly dynamic generator;
        private readonly dynamic discriminator;
        private readonly dynamic mel_spec_transform;

        private readonly float multi_mel_spectral_recon_loss_weight;
        private readonly float feat_match_loss_weight;
        private readonly float tpr_loss_weight;
        private readonly float tpr_loss_tau;

        public HiFiGan(dynamic generator, dynamic discriminator, dynamic mel_spec_transform,
                       float multi_mel_spectral_recon_loss_weight = 45f,
                       float feat_match_loss_weight = 2.0f,
                       float tpr_loss_weight = 1.0f,
                       float tpr_loss_tau = 0.04f) : base("HiFiGan")
        {
            this.generator = generator;
            this.discriminator = discriminator;
            this.mel_spec_transform = mel_spec_transform;
            this.multi_mel_spectral_recon_loss_weight = multi_mel_spectral_recon_loss_weight;
            this.feat_match_loss_weight = feat_match_loss_weight;
            this.tpr_loss_weight = tpr_loss_weight;
            this.tpr_loss_tau = tpr_loss_tau;
        }

        public Dictionary<string, Tensor> forward(Dictionary<string, Tensor> batch, Device device)
        {
            if (batch["turn"].ToString() == "generator")
            {
                return ForwardGenerator(batch, device);
            }
            else
            {
                return ForwardDiscriminator(batch, device);
            }
        }

        private Dictionary<string, Tensor> ForwardGenerator(Dictionary<string, Tensor> batch, Device device)
        {
            var real_speech = batch["speech"].to(device);
            var pitch_feat = batch["pitch_feat"].to(device);

            // Generator outputs
            var generatorOutput = (Tuple<Tensor, Tensor>)generator.forward(batch, device);
            var generated_speech = generatorOutput.Item1;
            var generated_f0 = generatorOutput.Item2;

            // Discriminator outputs
            var discriminatorOutput = (Tuple<Tensor, Tensor, Tensor, Tensor>)discriminator.forward(real_speech, generated_speech.detach());
            var y_d_rs = discriminatorOutput.Item1;
            var y_d_gs = discriminatorOutput.Item2;
            var fmap_rs = discriminatorOutput.Item3;
            var fmap_gs = discriminatorOutput.Item4;

            // Loss calculations
            var loss_gen = GeneratorLoss(y_d_gs);
            var loss_fm = FeatureLoss(fmap_rs, fmap_gs);
            var loss_mel = MelLoss(real_speech, generated_speech, mel_spec_transform);
            var loss_tpr = tpr_loss_weight != 0 ? TprLoss(new Tensor[] { y_d_gs }, new Tensor[] { y_d_rs }, tpr_loss_tau) : torch.zeros(1).to(device);
            var loss_f0 = nn.functional.l1_loss(generated_f0, pitch_feat);

            var loss = loss_gen + feat_match_loss_weight * loss_fm +
                       multi_mel_spectral_recon_loss_weight * loss_mel +
                       tpr_loss_weight * loss_tpr + loss_f0;

            return new Dictionary<string, Tensor>
            {
                { "loss", loss },
                { "loss_gen", loss_gen },
                { "loss_fm", loss_fm },
                { "loss_mel", loss_mel },
                { "loss_tpr", loss_tpr },
                { "loss_f0", loss_f0 }
            };
        }

        private Dictionary<string, Tensor> ForwardDiscriminator(Dictionary<string, Tensor> batch, Device device)
        {
            var real_speech = batch["speech"].to(device);

            // Generator outputs
            Tensor generated_speech;
            Tensor generated_f0;
            using (torch.no_grad())
            {
                var generatorOutput = (Tuple<Tensor, Tensor>)generator.forward(batch, device);
                generated_speech = generatorOutput.Item1;
                generated_f0 = generatorOutput.Item2;
            }

            // Discriminator outputs
            var discriminatorOutput = (Tuple<Tensor, Tensor, Tensor, Tensor>)discriminator.forward(real_speech, generated_speech.detach());
            var y_d_rs = discriminatorOutput.Item1;
            var y_d_gs = discriminatorOutput.Item2;

            // Loss calculations
            var loss_disc = DiscriminatorLoss(y_d_rs, y_d_gs);
            var loss_tpr = tpr_loss_weight != 0 ? TprLoss(new Tensor[] { y_d_rs }, new Tensor[] { y_d_gs }, tpr_loss_tau) : torch.zeros(1).to(device);

            var loss = loss_disc + tpr_loss_weight * loss_tpr;

            return new Dictionary<string, Tensor>
            {
                { "loss", loss },
                { "loss_disc", loss_disc },
                { "loss_tpr", loss_tpr }
            };
        }

        public Tensor Synthesize(Tensor mel)
        {
            // Ensure the generator is properly initialized
            if (generator == null)
            {
                throw new InvalidOperationException("Generator is not initialized.");
            }

            // Forward pass through the generator
            var audio = generator.forward(mel);

            // Post-process the audio tensor if necessary
            return audio;
        }
    }
}
