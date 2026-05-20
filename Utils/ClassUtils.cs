using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using CosyVoiceNet.flow;
using CosyVoiceNet.hifigan;
using CosyVoiceNet.LLM;
using CosyVoiceNet.cli;

namespace CosyVoiceNet.Utils
{
    // Port of cosyvoice/utils/class_utils.py
    // Provides mapping dictionaries and model type detection helpers.
    public static class ClassUtils
    {
        public static readonly Dictionary<string, string> COSYVOICE_ACTIVATION_CLASSES = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "hardtanh", "hardtanh" },
            { "tanh", "tanh" },
            { "relu", "relu" },
            { "selu", "selu" },
            { "swish", "SiLU" }, // Updated to match Python's use of SiLU or Swish.
            { "gelu", "gelu" }
        };

        public static readonly Dictionary<string, string> COSYVOICE_SUBSAMPLE_CLASSES = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "linear", "LinearNoSubsampling" },
            { "linear_legacy", "LegacyLinearNoSubsampling" },
            { "embed", "EmbedinigNoSubsampling" },
            { "conv1d2", "Conv1dSubsampling2" },
            { "conv2d", "Conv2dSubsampling4" },
            { "conv2d6", "Conv2dSubsampling6" },
            { "conv2d8", "Conv2dSubsampling8" },
            { "paraformer_dummy", "Identity" } // Added "paraformer_dummy".
        };

        public static readonly Dictionary<string, string> COSYVOICE_EMB_CLASSES = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "embed", "PositionalEncoding" },
            { "abs_pos", "PositionalEncoding" },
            { "rel_pos", "RelPositionalEncoding" },
            { "rel_pos_espnet", "EspnetRelPositionalEncoding" },
            { "no_pos", "NoPositionalEncoding" },
            { "abs_pos_whisper", "WhisperPositionalEncoding" },
            { "embed_learnable_pe", "LearnablePositionalEncoding" }
        };

        public static readonly Dictionary<string, string> COSYVOICE_ATTENTION_CLASSES = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "selfattn", "MultiHeadedAttention" }, // Added "selfattn".
            { "rel_selfattn", "RelPositionMultiHeadedAttention" } // Added "rel_selfattn".
        };

        // Determine the CosyVoice model class based on instantiated components
        public static Type GetModelType(IDictionary<string, object> configs)
        {
            if (configs == null) throw new ArgumentNullException(nameof(configs));
            configs.TryGetValue("llm", out var llm);
            configs.TryGetValue("flow", out var flow);
            configs.TryGetValue("hift", out var hift);

            var llmIsTransformer = llm != null && llm.GetType().Name.Contains("TransformerLM", StringComparison.OrdinalIgnoreCase);
            var flowIsMaskedDiffXvec = flow != null && flow.GetType().Name.Contains("MaskedDiff", StringComparison.OrdinalIgnoreCase);
            var flowIsCausalMaskedDiff = flow != null && flow.GetType().Name.Contains("CausalMaskedDiff", StringComparison.OrdinalIgnoreCase);
            var hiftIsHiFTGenerator = hift != null && hift.GetType().Name.Contains("HiFTGenerator", StringComparison.OrdinalIgnoreCase);
            var hiftIsCausalHiFT = hift != null && hift.GetType().Name.Contains("CausalHiFT", StringComparison.OrdinalIgnoreCase);

            if (llmIsTransformer && flowIsMaskedDiffXvec && hiftIsHiFTGenerator) return typeof(CosyVoiceModel);
            if (llm != null && llm.GetType().Name.Contains("Qwen2", StringComparison.OrdinalIgnoreCase) && flowIsCausalMaskedDiff && hiftIsHiFTGenerator) return typeof(CosyVoice2Model);
            if (llm != null && llm.GetType().Name.Contains("CosyVoice3", StringComparison.OrdinalIgnoreCase) && flowIsCausalMaskedDiff && hiftIsCausalHiFT) return typeof(CosyVoice3Model);

            throw new InvalidOperationException("No valid model type found!");
        }
    }
}

// Equivalent Python file: cosyvoice/utils/class_utils.py
