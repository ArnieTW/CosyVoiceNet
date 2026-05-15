using System;

namespace CosyVoiceNet
{
    public class TtsChunk
    {
        public byte[] TtsSpeech { get; }
        public int SampleRate { get; }

        public TtsChunk(byte[] ttsSpeech, int sampleRate)
        {
            TtsSpeech = ttsSpeech;
            SampleRate = sampleRate;
        }
    }
}
