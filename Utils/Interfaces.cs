using Tensor = TorchSharp.torch.Tensor;
using System.Collections.Generic;
// Define interfaces
public interface ILLMInference
{
    IEnumerable<int> inference(
        Tensor text, Tensor textLen,
        Tensor promptText, Tensor promptTextLen,
        Tensor promptSpeechToken, Tensor promptSpeechTokenLen,
        Tensor embedding,
        int sampling = 25, 
        float maxTokenTextRatio = 20.0f, 
        float minTokenTextRatio = 2.0f, 
        string uuid = "");
}

public interface IFlowInference
{
    int token_mel_ratio { get; }
    int InputFrameRate { get; }
    public int PreLookaheadLen { get; }
    (Tensor feat, Tensor flow_cache) Inference(
        Tensor token, Tensor token_len,
        Tensor prompt_token, Tensor prompt_token_len,
        Tensor prompt_feat, Tensor prompt_feat_len,
        Tensor embedding, bool streaming, bool finalize);
    (Tensor feat, Tensor flow_cache) Inference(
        Tensor token, Tensor token_len,
        Tensor prompt_token, Tensor prompt_token_len,
        Tensor prompt_feat, Tensor prompt_feat_len,
        Tensor embedding, Tensor cache );
}

public interface IHiftInference
{
    (Tensor, Tensor) Inference(Tensor speechFeat, bool finalize = true);
    (Tensor, Tensor) Inference(Tensor speechFeat, Tensor cache);
}