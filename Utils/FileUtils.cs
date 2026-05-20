using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

// This file is aligned with 'D:\Dev\NekoBot_LLM\CosyVoice\cosyvoice\utils\file_utils.py'.

namespace CosyVoiceNet.Utils
{
    public static class FileUtils
    {
        public static List<string> ReadLists(string listFile)
        {
            var lines = File.ReadAllLines(listFile);
            return lines.Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
        }

        public static Dictionary<string, object> ReadJsonLists(string listFile)
        {
            var result = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (var fn in ReadLists(listFile))
            {
                var json = File.ReadAllText(fn);
                var dict = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
                if (dict != null)
                {
                    foreach (var kv in dict) result[kv.Key] = kv.Value;
                }
            }
            return result;
        }

        public static Tensor load_wav(string path, int targetSampleRate, int minSourceSampleRate = 16000)
        {
            var (speech, sampleRate) = CosyVoiceNet.Tools.AudioUtils.LoadWavSoundfileLike(path);

            // torchaudio: speech = speech.mean(dim=0, keepdim=True)
            speech = speech.mean(dimensions: new long[] { 0 }, keepdim: true);

            if (sampleRate != targetSampleRate)
            {
                if (sampleRate < minSourceSampleRate)
                    throw new InvalidDataException($"wav sample rate {sampleRate} must be greater than {minSourceSampleRate}");

                var resampled = CosyVoiceNet.TorchSharpUtils.TorchAudioResampler.Resample(speech, sampleRate, targetSampleRate);
                speech.Dispose(); // Clean up original tensor
                speech = resampled;
            }

            return speech;
        }

        public static void ConvertOnnxToTrt(string trtModel, IDictionary<string, object> trtKwargs, string onnxModel, bool fp16)
        {
            throw new NotImplementedException("ONNX to TensorRT conversion is not implemented.");
        }

        public static void ExportCosyVoice2Vllm(dynamic model, string modelPath, Device device)
        {
            if (File.Exists(modelPath)) return;

            var dtype = torch.bfloat16;
            model.llm.model.lm_head = model.llm_decoder;
            var embedTokens = model.llm.model.model.embed_tokens;
            model.llm.model.set_input_embeddings(model.speech_embedding);
            model.llm.model.to(device);
            model.llm.model.to(dtype);

            var tmpVocabSize = model.llm.model.config.vocab_size;
            var tmpTieEmbedding = model.llm.model.config.tie_word_embeddings;

            model.llm.model.config.vocab_size = model.speech_embedding.num_embeddings;
            model.llm.model.config.tie_word_embeddings = false;

            model.llm.model.save_pretrained(modelPath);

            model.llm.model.config.vocab_size = tmpVocabSize;
            model.llm.model.config.tie_word_embeddings = tmpTieEmbedding;
            model.llm.model.set_input_embeddings(embedTokens);
        }
    }
}

