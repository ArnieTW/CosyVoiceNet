// Equivalent Python file: cosyvoice/hifigan/discriminator.py

using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

namespace CosyVoiceNet.hifigan
{
    public class MultipleDiscriminator : torch.nn.Module<(Tensor, Tensor), (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<List<Tensor>>)>
    {
        public torch.nn.Module mpd;
        public torch.nn.Module mrd;

        public MultipleDiscriminator(torch.nn.Module mpd = null, torch.nn.Module mrd = null) : base("MultipleDiscriminator")
        {
            this.mpd = mpd;
            this.mrd = mrd;
        }

        public override (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<List<Tensor>>) forward((Tensor, Tensor) inputs)
        {
            var (y, y_hat) = inputs;
            var y_d_rs = new List<Tensor>();
            var y_d_gs = new List<Tensor>();
            var fmap_rs = new List<List<Tensor>>();
            var fmap_gs = new List<List<Tensor>>();

            if (mpd != null)
            {
                var result = ((dynamic)mpd).forward(y.unsqueeze(1), y_hat.unsqueeze(1));
                var this_y_d_rs = (List<Tensor>)result[0];
                var this_y_d_gs = (List<Tensor>)result[1];
                var this_fmap_rs = (List<List<Tensor>>)result[2];
                var this_fmap_gs = (List<List<Tensor>>)result[3];
                y_d_rs.AddRange(this_y_d_rs);
                y_d_gs.AddRange(this_y_d_gs);
                fmap_rs.AddRange(this_fmap_rs);
                fmap_gs.AddRange(this_fmap_gs);
            }

            if (mrd != null)
            {
                var result = ((dynamic)mrd).forward(y, y_hat);
                var this_y_d_rs = (List<Tensor>)result[0];
                var this_y_d_gs = (List<Tensor>)result[1];
                var this_fmap_rs = (List<List<Tensor>>)result[2];
                var this_fmap_gs = (List<List<Tensor>>)result[3];
                y_d_rs.AddRange(this_y_d_rs);
                y_d_gs.AddRange(this_y_d_gs);
                fmap_rs.AddRange(this_fmap_rs);
                fmap_gs.AddRange(this_fmap_gs);
            }

            return (y_d_rs, y_d_gs, fmap_rs, fmap_gs);
        }

        public void LoadStateDict(Dictionary<string, Tensor> weights)
        {
            // Implement logic to load weights into MultipleDiscriminator
            foreach (var weight in weights)
            {
                // Example: Assign weights to layers
                if (weight.Key.Contains("discriminator"))
                {
                    // Assign to discriminator layers
                }
            }
        }
    }

    public class DiscriminatorR : torch.nn.Module<Tensor, (Tensor, List<Tensor>)>
    {
        public int window_length;
        public float hop_factor;
        public torch.nn.Module spec_fn;
        public List<List<torch.nn.Module>> band_convs;
        public torch.nn.Module emb;
        public torch.nn.Module conv_post;
        public List<(int, int)> bands;

        public DiscriminatorR(int window_length, int? num_embeddings = null, int channels = 32, float hop_factor = 0.25f) : base("DiscriminatorR")
        {
            this.window_length = window_length;
            this.hop_factor = hop_factor;
            spec_fn = null; // Expected to be assigned externally
            bands = new List<(int, int)>();
            band_convs = new List<List<torch.nn.Module>>();

            if (num_embeddings.HasValue)
            {
                emb = torch.nn.Embedding(num_embeddings.Value, channels);
                emb.parameters().First().zero_();
            }

            conv_post = torch.nn.Conv2d(channels, 1, (3, 3), stride: (1, 1), padding: (1, 1));
        }

        public List<Tensor> spectrogram(Tensor x)
        {
            if (spec_fn == null)
                throw new InvalidOperationException("Spectrogram function not assigned.");

            x = x - x.mean(new long[] { -1 }, keepdim: true);
            x = 0.8 * x / (x.abs().max(-1, keepdim: true).values + 1e-9);
            var spec = ((dynamic)spec_fn).forward(x);
            var x_real = torch.view_as_real(spec);
            var x_bands = new List<Tensor>();
            foreach (var (start, end) in bands)
            {
                x_bands.Add(x_real.slice(-1, start, end, 1));
            }
            return x_bands;
        }

        public override (Tensor, List<Tensor>) forward(Tensor x)
        {
            var x_bands = spectrogram(x);
            var fmap = new List<Tensor>();
            var outs = new List<Tensor>();

            for (int i = 0; i < band_convs.Count; i++)
            {
                var band = x_bands[i];
                foreach (var layer in band_convs[i])
                {
                    band = ((dynamic)layer).forward(band);
                    band = torch.nn.functional.leaky_relu(band, 0.1);
                }
                outs.Add(band);
            }

            var xcat = torch.cat(outs.ToArray(), -1);
            Tensor h = null;
            if (emb != null)
            {
                var e = ((dynamic)emb).forward(x);
                h = (e.unsqueeze(-1).unsqueeze(-1) * xcat).sum(1, keepdim: true);
            }

            var outTensor = ((dynamic)conv_post).forward(xcat);
            fmap.Add(outTensor);

            // Further refining the logical operation to ensure compatibility
            // Refining the null check for TorchSharp compatibility
            if (h is not null)
            {
                if (h.numel() > 0)
                {
                    if (h.item<bool>())
                    {
                        outTensor += h;
                    }
                }
            }

            return (outTensor, fmap);
        }
    }

    // Fixing dynamic deconstruction issues in MultiResolutionDiscriminator
    public class MultiResolutionDiscriminator : torch.nn.Module<(Tensor, Tensor), (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<List<Tensor>>)> {
        public List<torch.nn.Module<Tensor, (Tensor, List<Tensor>)>> discriminators;

        public MultiResolutionDiscriminator(int[] fft_sizes = null, int? num_embeddings = null) : base("MultiResolutionDiscriminator") {
            fft_sizes ??= new[] { 2048, 1024, 512 };
            discriminators = fft_sizes.Select(w => new DiscriminatorR(w, num_embeddings)).Cast<torch.nn.Module<Tensor, (Tensor, List<Tensor>)>>().ToList();
        }

        public override (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<List<Tensor>>) forward((Tensor, Tensor) inputs) {
            var (y, y_hat) = inputs;
            var y_d_rs = new List<Tensor>();
            var y_d_gs = new List<Tensor>();
            var fmap_rs = new List<List<Tensor>>();
            var fmap_gs = new List<List<Tensor>>();

            foreach (var d in discriminators) {
                var result_r = ((torch.nn.Module<Tensor, (Tensor, List<Tensor>)>)d).forward(y);
                var result_g = ((torch.nn.Module<Tensor, (Tensor, List<Tensor>)>)d).forward(y_hat);

                y_d_rs.Add(result_r.Item1);
                fmap_rs.Add(result_r.Item2);
                y_d_gs.Add(result_g.Item1);
                fmap_gs.Add(result_g.Item2);
            }

            return (y_d_rs, y_d_gs, fmap_rs, fmap_gs);
        }
    }
}
