using System;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;

// Equivalent Python file: cosyvoice/utils/mask.py

namespace CosyVoiceNet.Utils
{
    public static class Mask
    {
        public static Tensor SubsequentMask(int size, TorchSharp.torch.Device? device)
        {
            var arange = torch.arange(size, device: device ?? torch.CPU);
            var mask = arange.expand(size, size);
            arange = arange.unsqueeze(-1);
            mask = mask.le(arange);
            return mask;
        }

        public static Tensor SubsequentChunkMask(int size, int chunkSize, int numLeftChunks = -1, TorchSharp.torch.Device? device = null)
        {
            if (numLeftChunks < 0)
            {
                var posIdx = torch.arange(size, device: device ?? torch.CPU);
                var blockValue = posIdx.div(chunkSize).floor().add(1).mul(chunkSize);
                var ret = posIdx.unsqueeze(0).lt(blockValue.unsqueeze(1));
                return ret;
            }
            else
            {
                var ret = torch.zeros(size, size, device: device ?? torch.CPU, dtype: ScalarType.Bool);
                for (int i = 0; i < size; i++)
                {
                    var start = Math.Max((i / chunkSize - numLeftChunks) * chunkSize, 0);
                    var ending = Math.Min((i / chunkSize + 1) * chunkSize, size);
                    ret[i].slice(0, start, ending, 1).fill_(true);
                }
                return ret;
            }
        }

        public static Tensor AddOptionalChunkMask(Tensor xs, Tensor masks, bool useDynamicChunk, bool useDynamicLeftChunk, int decodingChunkSize, int staticChunkSize, int numDecodingLeftChunks, bool enableFullContext = true)
        {
            Tensor chunkMasks;
            if (useDynamicChunk)
            {
                var maxLen = (int)xs.shape[1];
                int chunkSize;
                int numLeftChunks;
                if (decodingChunkSize < 0)
                {
                    chunkSize = maxLen;
                    numLeftChunks = -1;
                }
                else if (decodingChunkSize > 0)
                {
                    chunkSize = decodingChunkSize;
                    numLeftChunks = numDecodingLeftChunks;
                }
                else
                {
                    chunkSize = ScalarToInt(torch.randint(1, maxLen, 1));
                    numLeftChunks = -1;
                    if (chunkSize > maxLen / 2 && enableFullContext)
                    {
                        chunkSize = maxLen;
                    }
                    else
                    {
                        chunkSize = chunkSize % 25 + 1;
                        if (useDynamicLeftChunk)
                        {
                            var maxLeftChunks = (maxLen - 1) / chunkSize;
                            numLeftChunks = ScalarToInt(torch.randint(0, maxLeftChunks, 1));
                        }
                    }
                }
                chunkMasks = SubsequentChunkMask(maxLen, chunkSize, numLeftChunks, xs.device);
                chunkMasks = chunkMasks.unsqueeze(0);
                chunkMasks = masks.bitwise_and(chunkMasks);
            }
            else if (staticChunkSize > 0)
            {
                var numLeftChunks = numDecodingLeftChunks;
                chunkMasks = SubsequentChunkMask((int)xs.shape[1], staticChunkSize, numLeftChunks, xs.device);
                chunkMasks = chunkMasks.unsqueeze(0);
                chunkMasks = masks.bitwise_and(chunkMasks);
            }
            else
            {
                chunkMasks = masks.clone();
            }

            if (ScalarToInt(chunkMasks.sum(-1).eq(0).sum()) != 0)
            {
                chunkMasks.index_put_(chunkMasks.sum(-1).eq(0), true);
            }
            return chunkMasks;
        }

        public static Tensor MakePadMask(Tensor lengths, int maxLen)
        {
            var batchSize = lengths.shape[0];
            var range = torch.arange(maxLen, device: lengths.device).unsqueeze(0);
            var mask = range >= lengths.unsqueeze(1);
            return mask;
        }

        private static int ScalarToInt(Tensor value)
        {
            if (value is null || value.numel() == 0)
                return 0;

            var scalar = value.flatten()[0];
            return scalar.dtype switch
            {
                ScalarType.Int64 => checked((int)scalar.item<long>()),
                ScalarType.Int32 => scalar.item<int>(),
                ScalarType.Int16 => scalar.item<short>(),
                ScalarType.Byte => scalar.item<byte>(),
                ScalarType.Float32 => checked((int)scalar.item<float>()),
                ScalarType.Float64 => checked((int)scalar.item<double>()),
                _ => checked((int)scalar.to_type(ScalarType.Int64).item<long>())
            };
        }
    }
}
