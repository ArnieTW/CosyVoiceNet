// Equivalent Python file: cosyvoice/flow/flow.py
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;
using CosyVoiceNet.Utils;
using CosyVoiceNet.flow;
using CosyVoiceNet.Transformers;

namespace CosyVoiceNet.flow
{
    public class MaskedDiffWithXvec : nn.Module, IFlowInference
    {
        private int input_size;
        private int output_size;
        private int spk_embed_dim;
        private string output_type;
        private int vocab_size;
        private int input_frame_rate;
        private bool only_mask_loss;

        public nn.Module input_embedding;
        public nn.Module<Tensor, Tensor> spk_embed_affine_layer;
        public nn.Module encoder;
        public nn.Module<Tensor, Tensor> encoder_proj;
        public nn.Module decoder;
        public LengthRegulator length_regulator;
        public int token_mel_ratio => 2;
        public int InputFrameRate => input_frame_rate;
        public int PreLookaheadLen => 0;

        public MaskedDiffWithXvec(int input_size = 512, int output_size = 80, int spk_embed_dim = 192,
                                  string output_type = "mel", int vocab_size = 4096, int input_frame_rate = 50,
                                  bool only_mask_loss = true, nn.Module encoder = null,
                                  LengthRegulator length_regulator = null, nn.Module decoder = null)
            : base("MaskedDiffWithXvec")
        {
            this.input_size = input_size;
            this.output_size = output_size;
            this.spk_embed_dim = spk_embed_dim;
            this.output_type = output_type;
            this.vocab_size = vocab_size;
            this.input_frame_rate = input_frame_rate;
            this.only_mask_loss = only_mask_loss;

            // embedding
            this.input_embedding = nn.Embedding(vocab_size, input_size);
            this.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size);
            this.encoder = encoder;
            this.encoder_proj = nn.Linear(this.encoder == null ? 512 : ((dynamic)this.encoder).output_size(), output_size);
            this.decoder = decoder;
            this.length_regulator = length_regulator ?? new LengthRegulator(input_size, new List<int> { 1, 2, 3 }, output_size);
            
            // RegisterComponents() discovers public nn.Module fields for state_dict
            RegisterComponents();
        }

        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token, Tensor token_len,
            Tensor prompt_token, Tensor prompt_token_len,
            Tensor prompt_feat, Tensor prompt_feat_len,
            Tensor embedding, bool streaming, bool finalize)
        {
            var cache = torch.zeros(new long[] { 1, output_size, 0, 2 }, device: token.device, dtype: embedding.dtype);
            return Inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, cache);
        }

        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token, Tensor token_len,
            Tensor prompt_token, Tensor prompt_token_len,
            Tensor prompt_feat, Tensor prompt_feat_len,
            Tensor embedding, Tensor cache)
        {
            if (token.shape[0] != 1)
                throw new ArgumentException("MaskedDiffWithXvec inference expects batch size 1.", nameof(token));

            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);

            var tokenLen1 = (int)prompt_token.shape[1];
            var tokenLen2 = (int)token.shape[1];
            token = torch.cat(new[] { prompt_token, token }, dim: 1);
            token_len = prompt_token_len + token_len;
            var mask = (~Mask.MakePadMask(token_len, checked((int)token_len.max().item<long>())))
                .unsqueeze(-1)
                .to(embedding);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;

            var encRes = ((dynamic)encoder).Forward(tokenEmb, token_len);
            Tensor h = encRes.Item1;
            h = encoder_proj.forward(h);

            var melLen1 = (int)prompt_feat.shape[1];
            var melLen2 = (int)(tokenLen2 / (double)input_frame_rate * 22050.0 / 256.0);
            var lrRes = length_regulator.inference(
                h.narrow(1, 0, tokenLen1),
                h.narrow(1, tokenLen1, h.shape[1] - tokenLen1),
                melLen1,
                melLen2,
                input_frame_rate);
            h = lrRes.Item1;

            var conds = torch.zeros(new long[] { 1, melLen1 + melLen2, output_size }, device: token.device, dtype: h.dtype);
            if (melLen1 > 0)
                conds[TensorIndex.Colon, TensorIndex.Slice(0, melLen1), TensorIndex.Colon] = prompt_feat.to(h.device).to(h.dtype);
            conds = conds.transpose(1, 2);

            var totalLen = torch.tensor(new long[] { melLen1 + melLen2 }, dtype: ScalarType.Int64, device: h.device);
            var featMask = (~Mask.MakePadMask(totalLen, melLen1 + melLen2)).to(h).unsqueeze(1);
            cache = cache?.to(h.device).to(h.dtype);
            var decoderResult = ((ConditionalCFM)decoder).Forward(
                h.transpose(1, 2).contiguous(),
                featMask,
                nTimesteps: 10,
                spks: embedding,
                cond: conds,
                promptLen: melLen1,
                cache: cache);
            var feat = decoderResult.Item1[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(melLen1, null)];
            if (feat.shape[2] != melLen2)
                throw new InvalidOperationException($"Flow output length mismatch. Expected {melLen2}, got {feat.shape[2]}.");

            return (feat.to(ScalarType.Float32), decoderResult.Item2);
        }

        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> batch, torch.Device device)
        {
            // Convert inputs similar to Python forward
            var token = batch["speech_token"].to(device);
            var token_len = batch["speech_token_len"].to(device);
            var feat = batch["speech_feat"].to(device);
            var feat_len = batch["speech_feat_len"].to(device);
            var embedding = batch["embedding"].to(device);

            // xvec projection
            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);

            // concat text and prompt_text
            var mask = (~Mask.MakePadMask(token_len, token_len.max().item<int>())).to(device).unsqueeze(-1);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;

            // text encode
            dynamic encRes = null;
            if (encoder != null)
            {
                encRes = ((dynamic)encoder).Forward(tokenEmb, token_len);
            }
            else
            {
                // fallback: use tokenEmb directly
                encRes = (tokenEmb, token_len);
            }
            Tensor h = encRes.Item1;
            Tensor h_lengths = encRes.Item2;

            h = encoder_proj.forward(h);
            var lrRes = length_regulator.ForwardWithLengths(h, token_len);
            var expanded = lrRes.Item1;
            var melTotal = lrRes.Item2;

            // decoder compute_loss
            var conds = torch.zeros(new long[] { feat.shape[0], feat.shape[1], feat.shape[2] }, device: token.device).to(h.dtype);
            var rand = new Random();
            for (int i = 0; i < feat.shape[0]; i++)
            {
                if (rand.NextDouble() < 0.5) continue;
                int j = (int)feat_len.data<int>()[i];
                int index = rand.Next(0, (int)(0.3 * j));
                // copy first index frames
                if (index > 0)
                {
                    var slice = feat[i].slice(0, 0, index, 1);
                    conds[i].narrow(0, 0, index).copy_(slice);
                }
            }
            conds = conds.transpose(1, 2);
            var mask2 = (~Mask.MakePadMask(feat_len, feat_len.max().item<int>())).to(h);
            var lossRes = ((dynamic)decoder).compute_loss(feat.transpose(1, 2).contiguous(), mask2.unsqueeze(1), h.transpose(1, 2).contiguous(), embedding, conds);
            return new Dictionary<string, Tensor> { { "loss", (Tensor)lossRes.Item1 } };
        }
    }

    public class CausalMaskedDiffWithXvec : nn.Module, IFlowInference
    {
        private readonly int input_size;
        private readonly int output_size;
        private readonly int spk_embed_dim;
        private readonly string output_type;
        private readonly int vocab_size;
        private readonly bool only_mask_loss;

        public readonly nn.Module input_embedding;
        public readonly nn.Module<Tensor, Tensor> spk_embed_affine_layer;
        public readonly nn.Module encoder;
        public readonly nn.Module<Tensor, Tensor> encoder_proj;
        public readonly nn.Module decoder;

        public int token_mel_ratio { get; }
        public int InputFrameRate { get; }
        public int PreLookaheadLen { get; }

        public CausalMaskedDiffWithXvec(
            int input_size = 512,
            int output_size = 80,
            int spk_embed_dim = 192,
            string output_type = "mel",
            int vocab_size = 4096,
            int input_frame_rate = 50,
            bool only_mask_loss = true,
            int token_mel_ratio = 2,
            int pre_lookahead_len = 3,
            nn.Module encoder = null,
            nn.Module decoder = null,
            Dictionary<string, object> decoder_conf = null)
            : base("CausalMaskedDiffWithXvec")
        {
            this.input_size = input_size;
            this.output_size = output_size;
            this.spk_embed_dim = spk_embed_dim;
            this.output_type = output_type;
            this.vocab_size = vocab_size;
            this.only_mask_loss = only_mask_loss;
            this.token_mel_ratio = token_mel_ratio;
            InputFrameRate = input_frame_rate;
            PreLookaheadLen = pre_lookahead_len;

            input_embedding = nn.Embedding(vocab_size, input_size);
            spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size);
            this.encoder = encoder;
            encoder_proj = nn.Linear(this.encoder == null ? input_size : ((dynamic)this.encoder).output_size(), output_size);
            this.decoder = decoder;

            RegisterComponents();
        }

        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> batch, Device device)
        {
            var token = batch["speech_token"].to(device);
            var tokenLen = batch["speech_token_len"].to(device);
            var feat = batch["speech_feat"].to(device);
            var featLen = batch["speech_feat_len"].to(device);
            var embedding = batch["embedding"].to(device);

            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);

            var mask = (~Mask.MakePadMask(tokenLen, checked((int)tokenLen.max().item<long>())))
                .to(ScalarType.Float32).unsqueeze(-1).to(device);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;

            var encRes = ((dynamic)encoder).Forward(tokenEmb, tokenLen, null, 0, -1, streaming: false);
            Tensor h = encRes.Item1;
            Tensor hLengths = encRes.Item2;
            h = encoder_proj.forward(h);

            var conds = torch.zeros(feat.shape, device: token.device, dtype: h.dtype);
            conds = conds.transpose(1, 2);
            var frameLengths = hLengths.sum(dim: -1).squeeze(dim: 1);
            var featMask = (~Mask.MakePadMask(frameLengths, checked((int)frameLengths.max().item<long>()))).to(h);
            var lossRes = ((dynamic)decoder).ComputeLoss(
                feat.transpose(1, 2).contiguous(),
                featMask.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                embedding,
                conds,
                streaming: false);
            Tensor loss = lossRes is Tensor t ? t : lossRes.Item1;
            return new Dictionary<string, Tensor> { ["loss"] = loss };
        }

        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token, Tensor token_len,
            Tensor prompt_token, Tensor prompt_token_len,
            Tensor prompt_feat, Tensor prompt_feat_len,
            Tensor embedding, Tensor cache)
        {
            return Inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len,
                embedding, streaming: false, finalize: true);
        }

        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token,
            Tensor token_len,
            Tensor prompt_token,
            Tensor prompt_token_len,
            Tensor prompt_feat,
            Tensor prompt_feat_len,
            Tensor embedding,
            bool streaming,
            bool finalize)
        {
            if (token.shape[0] != 1)
                throw new ArgumentException("CausalMaskedDiffWithXvec inference expects batch size 1.", nameof(token));

            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);

            token = torch.cat(new[] { prompt_token, token }, dim: 1);
            token_len = prompt_token_len + token_len;
            var mask = (~Mask.MakePadMask(token_len, checked((int)token_len.max().item<long>())))
                .unsqueeze(-1)
                .to(embedding);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;

            Tensor context = null;
            if (!finalize)
            {
                context = tokenEmb[TensorIndex.Colon, TensorIndex.Slice(tokenEmb.shape[1] - PreLookaheadLen, null), TensorIndex.Colon];
                tokenEmb = tokenEmb[TensorIndex.Colon, TensorIndex.Slice(null, tokenEmb.shape[1] - PreLookaheadLen), TensorIndex.Colon];
            }

            var encRes = ((dynamic)encoder).Forward(tokenEmb, token_len, context, 0, -1, streaming);
            Tensor h = encRes.Item1;

            var melLen1 = (int)prompt_feat.shape[1];
            var melLen2 = (int)h.shape[1] - melLen1;
            h = encoder_proj.forward(h);

            var totalLen = melLen1 + melLen2;
            var conds = torch.zeros(new long[] { 1, totalLen, output_size }, device: token.device, dtype: h.dtype);
            if (melLen1 > 0)
                conds[TensorIndex.Colon, TensorIndex.Slice(0, melLen1), TensorIndex.Colon] = prompt_feat.to(h.device).to(h.dtype);
            conds = conds.transpose(1, 2);

            var totalLenTensor = torch.tensor(new long[] { totalLen }, dtype: ScalarType.Int64, device: h.device);
            var featMask = (~Mask.MakePadMask(totalLenTensor, totalLen)).to(h);
            var decoderResult = ((CausalConditionalCFM)decoder).Forward(
                h.transpose(1, 2).contiguous(),
                featMask.unsqueeze(1),
                nTimesteps: 10,
                spks: embedding,
                cond: conds,
                streaming: streaming);
            var feat = decoderResult.Item1[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(melLen1, null)];
            if (feat.shape[2] != melLen2)
                throw new InvalidOperationException($"Causal flow output length mismatch. Expected {melLen2}, got {feat.shape[2]}.");

            return (feat.to(ScalarType.Float32), null);
        }
    }

    // Equivalent Python: cosyvoice/flow/flow.py CausalMaskedDiffWithDiT
    public class CausalMaskedDiffWithDiT : nn.Module, IFlowInference
    {
        public readonly nn.Module<Tensor, Tensor> input_embedding;
        public readonly nn.Module<Tensor, Tensor> spk_embed_affine_layer;
        public readonly PreLookaheadLayer pre_lookahead_layer;
        public readonly CausalConditionalCFM decoder;

        public readonly int input_size;
        public readonly int output_size;
        public int token_mel_ratio { get; }
        public int InputFrameRate { get; }
        public int PreLookaheadLen { get; }
        public readonly bool only_mask_loss;

        public CausalMaskedDiffWithDiT(
            int inputSize = 512,
            int outputSize = 80,
            int spkEmbedDim = 192,
            string outputType = "mel",
            int vocabSize = 4096,
            int inputFrameRate = 50,
            bool onlyMaskLoss = true,
            int tokenMelRatio = 2,
            int preLookaheadLen = 3,
            PreLookaheadLayer preLookaheadLayer = null,
            CausalConditionalCFM decoder = null
        ) : base("CausalMaskedDiffWithDiT")
        {
            input_size       = inputSize;
            output_size      = outputSize;
            token_mel_ratio  = tokenMelRatio;
            PreLookaheadLen = preLookaheadLen;
            only_mask_loss   = onlyMaskLoss;
            InputFrameRate = inputFrameRate;

            input_embedding        = nn.Embedding(vocabSize, inputSize);
            spk_embed_affine_layer = nn.Linear(spkEmbedDim, outputSize);
            pre_lookahead_layer    = preLookaheadLayer;
            this.decoder           = decoder;
            RegisterComponents();
        }

        public Dictionary<string, Tensor> Forward(Dictionary<string, Tensor> batch, Device device)
        {
            var token     = batch["speech_token"].to(device);
            var token_len = batch["speech_token_len"].to(device);
            var feat      = batch["speech_feat"].to(device);
            var feat_len  = batch["speech_feat_len"].to(device);
            var embedding = batch["embedding"].to(device);

            bool streaming = new Random().NextDouble() < 0.5;

            // xvec projection
            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);

            // token embedding with mask
            var mask = (~Mask.MakePadMask(token_len, token_len.max().item<int>()))
                          .to(ScalarType.Float32).unsqueeze(-1).to(device);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;

            // text encode: pre_lookahead_layer → repeat_interleave
            var h = pre_lookahead_layer.Forward(tokenEmb);
            h = h.repeat_interleave(token_mel_ratio, dim: 1);
            var maskRepeat = mask.repeat_interleave(token_mel_ratio, dim: 1).squeeze(-1);

            // build conditions
            var conds = torch.zeros(feat.shape, device: token.device);
            var rand  = new Random();
            for (int i = 0; i < feat.shape[0]; i++)
            {
                if (rand.NextDouble() < 0.5) continue;
                int j     = (int)feat_len[i].item<int>();
                int index = rand.Next(0, (int)(0.3 * j));
                if (index > 0) conds[i].narrow(0, 0, index).copy_(feat[i].narrow(0, 0, index));
            }
            conds = conds.transpose(1, 2);

            var loss = decoder.ComputeLoss(
                feat.transpose(1, 2).contiguous(),
                maskRepeat.unsqueeze(1),
                h.transpose(1, 2).contiguous(),
                embedding, conds, streaming);

            return new Dictionary<string, Tensor> { { "loss", loss } };
        }
        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token, Tensor token_len,
            Tensor prompt_token, Tensor prompt_token_len,
            Tensor prompt_feat, Tensor prompt_feat_len,
            Tensor embedding, Tensor cache)
        {
            // For simplicity, we ignore the cache in this example. In a real implementation, you would use it to speed up inference.
            return Inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, streaming: false, finalize: true);
        }

        public (Tensor feat, Tensor flow_cache) Inference(
            Tensor token,         Tensor token_len,
            Tensor prompt_token,  Tensor prompt_token_len,
            Tensor prompt_feat,   Tensor prompt_feat_len,
            Tensor embedding,     bool streaming, bool finalize)
        {
            var perf = false;
            var traceShapes = false;
            var sw = perf ? System.Diagnostics.Stopwatch.StartNew() : null;
            // xvec projection
            embedding = nn.functional.normalize(embedding, dim: 1);
            embedding = spk_embed_affine_layer.forward(embedding);
            if (perf)
            {
                SynchronizeIfCuda(embedding);
                Console.WriteLine($"[CosyVoicePerf.Flow] embedding_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // concat prompt + current token
            token     = torch.cat(new[] { prompt_token, token }, dim: 1);
            token_len = prompt_token_len + token_len;
            var mask     = torch.ones(new long[] { token.shape[0], token.shape[1], 1 }, device: token.device, dtype: embedding.dtype);
            var tokenEmb = ((dynamic)input_embedding).forward(token.clamp_min(0)) * mask;
            if (traceShapes)
                Console.WriteLine($"[CosyVoiceShapes.Flow] token={Shape(token)} prompt_feat={Shape(prompt_feat)} token_emb={Shape(tokenEmb)} device={token.device} dtype={tokenEmb.dtype}");
            if (perf)
            {
                SynchronizeIfCuda(tokenEmb);
                Console.WriteLine($"[CosyVoicePerf.Flow] token_embed_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            // text encode
            Tensor h;
            if (finalize)
            {
                h = pre_lookahead_layer.Forward(tokenEmb);
            }
            else
            {
                // split off the lookahead context tail
                var body    = tokenEmb[.., ..^PreLookaheadLen, ..];
                var context = tokenEmb[.., ^PreLookaheadLen.., ..];
                h = pre_lookahead_layer.Forward(body, context);
            }
            h = h.repeat_interleave(token_mel_ratio, dim: 1);
            if (traceShapes)
                Console.WriteLine($"[CosyVoiceShapes.Flow] h_repeated={Shape(h)} token_mel_ratio={token_mel_ratio}");
            if (perf)
            {
                SynchronizeIfCuda(h);
                Console.WriteLine($"[CosyVoicePerf.Flow] prelookahead_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            int mel_len1 = (int)prompt_feat.shape[1];
            int mel_len2 = (int)h.shape[1] - mel_len1;

            // build conditions
            int totalLen = mel_len1 + mel_len2;
            var conds = torch.zeros(new long[] { 1, totalLen, output_size }, device: token.device).to(h.dtype);
            conds[0, ..mel_len1] = prompt_feat;
            conds = conds.transpose(1, 2);
            if (traceShapes)
                Console.WriteLine($"[CosyVoiceShapes.Flow] mel_len1={mel_len1} mel_len2={mel_len2} total_len={totalLen} conds={Shape(conds)}");

            var (feat, _) = decoder.Forward(
                h.transpose(1, 2).contiguous(),
                null,
                nTimesteps: 10,
                spks: embedding,
                cond: conds,
                streaming: streaming);
            if (perf)
            {
                SynchronizeIfCuda(feat);
                Console.WriteLine($"[CosyVoicePerf.Flow] decoder_ms={sw!.Elapsed.TotalMilliseconds:0.000}");
                sw.Restart();
            }

            feat = feat[.., .., mel_len1..];
            return (feat.to(ScalarType.Float32), null);
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
    }
}
