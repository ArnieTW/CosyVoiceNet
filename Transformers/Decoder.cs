using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;
using CosyVoiceNet.Utils;

namespace CosyVoiceNet.Transformers
{
    // Equivalent Python file: cosyvoice/transformer/decoder.py
    public class TransformerDecoder : torch.nn.Module<(Tensor, Tensor, Tensor, Tensor, Tensor?, double), (Tensor, Tensor, Tensor)>
    {
        public dynamic Embed;
        public dynamic AfterNorm;
        public bool NormalizeBefore;
        public bool UseOutputLayer;
        public dynamic OutputLayer;
        public int NumBlocks;
        public List<DecoderLayer> Decoders = new List<DecoderLayer>();
        public bool GradientCheckpointing;
        public bool TieWordEmbedding;

        public TransformerDecoder(int vocabSize, int encoderOutputSize, int attentionHeads = 4, int linearUnits = 2048, int numBlocks = 6, double dropoutRate = 0.1, double positionalDropoutRate = 0.1, double selfAttentionDropoutRate = 0.0, double srcAttentionDropoutRate = 0.0, string inputLayer = "embed", bool useOutputLayer = true, bool normalizeBefore = true, bool srcAttention = true, bool keyBias = true, string activationType = "relu", bool gradientCheckpointing = false, bool tieWordEmbedding = false) : base("TransformerDecoder")
        {
            NormalizeBefore = normalizeBefore;
            UseOutputLayer = useOutputLayer;
            if (UseOutputLayer) OutputLayer = torch.nn.Linear(encoderOutputSize, vocabSize);
            else OutputLayer = torch.nn.Identity();
            NumBlocks = numBlocks;
            GradientCheckpointing = gradientCheckpointing;
            TieWordEmbedding = tieWordEmbedding;

            Embed = torch.nn.Sequential(
                ("embedding", inputLayer == "no_pos" ? torch.nn.Identity() : torch.nn.Embedding(vocabSize, encoderOutputSize)),
                ("positional_encoding", new PositionalEncoding(encoderOutputSize, positionalDropoutRate))
            );

            AfterNorm = torch.nn.LayerNorm(encoderOutputSize, eps: 1e-5);

            for (int i = 0; i < numBlocks; i++)
            {
                Decoders.Add(new DecoderLayer(
                    encoderOutputSize,
                    new MultiHeadedAttention(attentionHeads, encoderOutputSize, selfAttentionDropoutRate, keyBias),
                    srcAttention ? new MultiHeadedAttention(attentionHeads, encoderOutputSize, srcAttentionDropoutRate, keyBias) : null,
                    new PositionwiseFeedForward(encoderOutputSize, linearUnits, dropoutRate, x => torch.nn.functional.relu(x)),
                    dropoutRate,
                    normalizeBefore
                ));
            }

            RegisterComponents();
        }

        public override (Tensor, Tensor, Tensor) forward((Tensor, Tensor, Tensor, Tensor, Tensor?, double) inputs)
        {
            var (memory, memoryMask, ysInPad, ysInLens, rYsInPad, reverseWeight) = inputs;

            var tgt = ysInPad;
            int maxlen = (int)tgt.shape[1];
            var tgtMask = torch.logical_not(CosyVoiceNet.Utils.Mask.MakePadMask(ysInLens, maxlen)).unsqueeze(1).to(tgt.device);
            var m = CosyVoiceNet.Utils.Mask.SubsequentMask((int)tgtMask.shape[2], null).unsqueeze(0);
            tgtMask = tgtMask & m;

            var embedRes = Embed.forward(tgt);
            Tensor x = embedRes is ValueTuple<Tensor, Tensor> vt ? vt.Item1 : (Tensor)embedRes.GetType().GetProperty("Item1").GetValue(embedRes);

            if (GradientCheckpointing && false)
            {
                x = ForwardLayersCheckpointed(x, tgtMask, memory, memoryMask);
            }
            else
            {
                x = ForwardLayers(x, tgtMask, memory, memoryMask);
            }

            if (NormalizeBefore)
            {
                x = AfterNorm.forward(x);
            }

            if (UseOutputLayer)
            {
                x = OutputLayer.forward(x);
            }

            var olens = tgtMask.sum(1);
            return (x, torch.tensor(0.0), olens);
        }

        public Tensor ForwardLayers(Tensor x, Tensor tgtMask, Tensor memory, Tensor memoryMask)
        {
            foreach (var layer in Decoders)
            {
                var res = layer.forward((x, tgtMask, memory, memoryMask, null));
                x = res.Item1;
                tgtMask = res.Item2;
                memory = res.Item3;
                memoryMask = res.Item4;
            }
            return x;
        }

        public Tensor ForwardLayersCheckpointed(Tensor x, Tensor tgtMask, Tensor memory, Tensor memoryMask)
        {
            foreach (var layer in Decoders)
            {
                var res = layer.forward((x, tgtMask, memory, memoryMask, null));
                x = res.Item1;
                tgtMask = res.Item2;
                memory = res.Item3;
                memoryMask = res.Item4;
            }
            return x;
        }

        public (Tensor, List<Tensor>) ForwardOneStep(Tensor memory, Tensor memoryMask, Tensor tgt, Tensor tgtMask, List<Tensor> cache = null)
        {
            var embedRes = Embed.forward(tgt);
            Tensor x = embedRes is ValueTuple<Tensor, Tensor> vt ? vt.Item1 : (Tensor)embedRes.GetType().GetProperty("Item1").GetValue(embedRes);

            var newCache = new List<Tensor>();
            for (int i = 0; i < Decoders.Count; i++)
            {
                var c = cache != null ? cache[i] : null;
                var res = Decoders[i].forward((x, tgtMask, memory, memoryMask, null));
                x = res.Item1;
                tgtMask = res.Item2;
                memory = res.Item3;
                memoryMask = res.Item4;
                newCache.Add(x);
            }

            Tensor y;
            if (NormalizeBefore)
            {
                y = AfterNorm.forward(x.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Single(-1) }));
            }
            else
            {
                y = x.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Single(-1) });
            }

            if (UseOutputLayer)
            {
                y = torch.nn.functional.log_softmax(OutputLayer.forward(y), dim: -1);
            }

            return (y, newCache);
        }

        public void TieOrCloneWeights(bool jitMode = true)
        {
            if (!UseOutputLayer) return;

            if (jitMode)
            {
                OutputLayer.weight = torch.nn.Parameter(Embed[0].weight.clone());
            }
            else
            {
                OutputLayer.weight = Embed[0].weight;
            }

            if (OutputLayer.bias != null)
            {
                var padding = OutputLayer.weight.shape[0] - OutputLayer.bias.shape[0];
                OutputLayer.bias.data = torch.nn.functional.pad(
                    OutputLayer.bias.data,
                    new long[] { 0, padding },
                    value: 0.0
                );
            }
        }
    }
}
