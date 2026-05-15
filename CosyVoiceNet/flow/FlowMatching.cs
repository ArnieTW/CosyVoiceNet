// Exported from CosyVoice\cosyvoice\flow\flow_matching.py
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;
using CosyVoiceNet.flow.DiT;
using CosyVoiceNet.Utils;

namespace CosyVoiceNet.flow
{
    public class ConditionalCFM : nn.Module<Tensor, Tensor>
    {
        public readonly nn.Module estimator;
        protected readonly string t_scheduler;
        private readonly float training_cfg_rate;
        protected readonly float inference_cfg_rate;

        public ConditionalCFM(int inChannels, Dictionary<string, object> cfmParams, int nSpks = 1, int spkEmbDim = 64, nn.Module estimator = null) : base("ConditionalCFM")
        {
            t_scheduler = (string)cfmParams["t_scheduler"];
            training_cfg_rate = (float)(double)cfmParams["training_cfg_rate"];
            inference_cfg_rate = (float)(double)cfmParams["inference_cfg_rate"];
            this.estimator = estimator ?? throw new ArgumentNullException(nameof(estimator), "Estimator module cannot be null.");
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            throw new NotImplementedException("Forward method requires additional parameters.");
        }

        public (Tensor, Tensor) Forward(Tensor mu, Tensor mask, int nTimesteps, float temperature = 1.0f, Tensor spks = null, Tensor cond = null, int promptLen = 0, Tensor cache = null)
        {
            var z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature;
            var cacheSize = cache?.shape[2] ?? 0;

            if (cacheSize != 0)
            {
                z[.., .., ..(int)cacheSize]  = cache[.., .., .., 0];
                mu[.., .., ..(int)cacheSize] = cache[.., .., .., 1];
            }
            var zCache  = torch.cat(new[] { z[.., .., ..promptLen],  z[.., .., ^34..] }, dim: 2);
            var muCache = torch.cat(new[] { mu[.., .., ..promptLen], mu[.., .., ^34..] }, dim: 2);
            cache = torch.stack(new[] { zCache, muCache }, dim: -1);

            var tSpan = torch.linspace(0, 1, nTimesteps + 1, device: mu.device, dtype: mu.dtype);
            if (t_scheduler == "cosine")
                tSpan = 1 - torch.cos(tSpan * 0.5f * (float)Math.PI);

            return (SolveEuler(z, tSpan, mu, mask, spks, cond), cache);
        }

        protected virtual Tensor SolveEuler(Tensor x, Tensor tSpan, Tensor mu, Tensor mask, Tensor spks, Tensor cond, bool streaming = false)
        {
            var perf = false;
            var traceShapes = false;
            var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            var t   = tSpan[0].unsqueeze(0);
            var dt  = tSpan[1] - tSpan[0];
            long bsz = x.shape[0];
            var xIn = torch.zeros(new long[] { 2 * bsz, x.shape[1], x.shape[2] }, device: x.device, dtype: spks.dtype);
            Tensor maskIn = null;
            var muIn = torch.zeros(new long[] { 2 * bsz, mu.shape[1], mu.shape[2] }, device: x.device, dtype: spks.dtype);
            var tIn = torch.zeros(new long[] { 2 * bsz }, device: x.device, dtype: spks.dtype);
            var spksIn = torch.zeros(new long[] { 2 * bsz, spks.shape[1] }, device: x.device, dtype: spks.dtype);
            var condIn = torch.zeros(new long[] { 2 * bsz, cond.shape[1], cond.shape[2] }, device: x.device, dtype: spks.dtype);

            if (mask is not null)
            {
                maskIn = torch.zeros(new long[] { 2 * bsz, mask.shape[1], mask.shape[2] }, device: x.device, dtype: spks.dtype);
                maskIn[..(int)bsz].copy_(mask);
                maskIn[(int)bsz..].copy_(mask);
            }
            muIn[..(int)bsz].copy_(mu);
            spksIn[..(int)bsz].copy_(spks);
            condIn[..(int)bsz].copy_(cond);

            Tensor preparedAttnMask = null;
            (Tensor freqs, Tensor scale)? preparedRope = null;
            if (estimator is DiT.DiT dit && maskIn is not null)
            {
                var prepared = dit.PrepareAttention(maskIn, x.shape[2], streaming, x.device);
                preparedAttnMask = prepared.attnMask;
                preparedRope = prepared.rope;
            }
            if (traceShapes)
                Console.WriteLine($"[CosyVoiceShapes.Flow] solve x={Shape(x)} mu={Shape(mu)} cond={Shape(cond)} spks={Shape(spks)} mask={Shape(mask)} prepared_mask={Shape(preparedAttnMask)} steps={tSpan.shape[0] - 1}");
            if (perf)
            {
                SynchronizeIfCuda(x);
                Console.WriteLine($"[CosyVoicePerf.Flow] solve_prepare_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            for (int step = 1; step < tSpan.shape[0]; step++)
            {
                using (torch.NewDisposeScope())
                {
                    // Classifier-Free Guidance: double the batch, second half has zero conditioning
                    xIn[..(int)bsz].copy_(x);
                    xIn[(int)bsz..].copy_(x);
                    tIn.copy_(t.expand(2 * bsz));

                    var dphi = ForwardEstimator(xIn, maskIn, muIn, tIn, spksIn, condIn, streaming, preparedAttnMask, preparedRope);
                    if (perf)
                    {
                        SynchronizeIfCuda(dphi);
                        Console.WriteLine($"[CosyVoicePerf.Flow] solve_step{step}_estimator_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                        sw.Restart();
                    }
                    var dphi0 = dphi[..(int)bsz];
                    var dphi1 = dphi[(int)bsz..];
                    var dphiCfg = (1.0f + inference_cfg_rate) * dphi0 - inference_cfg_rate * dphi1;

                    x = (x + dt * dphiCfg).MoveToOuterDisposeScope();
                    t = (t + dt).MoveToOuterDisposeScope();
                    if (step < tSpan.shape[0] - 1)
                        dt = (tSpan[step + 1] - t).MoveToOuterDisposeScope();
                    if (perf)
                    {
                        SynchronizeIfCuda(x);
                        Console.WriteLine($"[CosyVoicePerf.Flow] solve_step{step}_update_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                        sw.Restart();
                    }
                }
            }
            return x.to(ScalarType.Float32);
        }

        private static void SynchronizeIfCuda(Tensor tensor)
        {
            if (tensor.device.type == DeviceType.CUDA)
                torch.cuda.synchronize(tensor.device);
        }

        private static string Shape(Tensor tensor)
        {
            return tensor is null ? "null" : "[" + string.Join(",", tensor.shape) + "]";
        }

        protected virtual Tensor ForwardEstimator(
            Tensor x,
            Tensor mask,
            Tensor mu,
            Tensor t,
            Tensor spks,
            Tensor cond,
            bool streaming = false,
            Tensor preparedAttnMask = null,
            (Tensor freqs, Tensor scale)? preparedRope = null)
        {
            if (estimator is DiT.DiT dit)
                return dit.Forward(x, mask, mu, t, spks, cond, streaming, preparedAttnMask, preparedRope);
            if (estimator is ConditionalDecoder decoder)
                return decoder.Forward(x, mask, mu, t, spks, cond, streaming);
            throw new NotImplementedException($"Estimator type '{estimator?.GetType().Name}' is not supported.");
        }

        public Tensor ComputeLoss(Tensor x1, Tensor mask, Tensor mu, Tensor spks = null, Tensor cond = null, bool streaming = false)
        {
            var b = mu.shape[0];
            var t = torch.rand(new long[] { b, 1, 1 }, device: mu.device, dtype: mu.dtype);
            var z = torch.randn_like(x1);

            var y = (1 - (1 - 1e-6f) * t) * z + t * x1;
            var u = x1 - (1 - 1e-6f) * z;

            if (training_cfg_rate > 0)
            {
                var cfgMask = torch.rand(new long[] { b }, device: x1.device) > training_cfg_rate;
                mu   = mu   * cfgMask.unsqueeze(-1).unsqueeze(-1);
                if (spks is not null) spks = spks * cfgMask.unsqueeze(-1);
                if (cond is not null) cond = cond * cfgMask.unsqueeze(-1).unsqueeze(-1);
            }

            var pred = ForwardEstimator(y, mask, mu, t.squeeze(), spks, cond, streaming);
            var loss = nn.functional.mse_loss(pred * mask, u * mask, reduction: torch.nn.Reduction.Sum) / (torch.sum(mask) * u.shape[1]);
            return loss;
        }
    }

    public class CausalConditionalCFM : ConditionalCFM
    {
        private readonly Tensor randNoise;

        public CausalConditionalCFM(int inChannels, Dictionary<string, object> cfmParams, int nSpks = 1, int spkEmbDim = 64, nn.Module estimator = null)
            : base(inChannels, cfmParams, nSpks, spkEmbDim, estimator)
        {
            Common.SetAllRandomSeed(0);
            randNoise = torch.randn(new long[] { 1, 80, 50 * 300 });
        }

        public (Tensor, Tensor) Forward(Tensor mu, Tensor mask, int nTimesteps, float temperature = 1.0f, Tensor spks = null, Tensor cond = null, bool streaming = false)
        {
            var z = randNoise.narrow(2, 0, mu.shape[2]).to(mu.device).to(mu.dtype) * temperature;
            var tSpan = torch.linspace(0, 1, nTimesteps + 1, device: mu.device, dtype: mu.dtype);
            if (t_scheduler == "cosine")
                tSpan = 1 - torch.cos(tSpan * 0.5f * (float)Math.PI);
            return (SolveEuler(z, tSpan, mu, mask, spks, cond, streaming), null);
        }
    }
}
