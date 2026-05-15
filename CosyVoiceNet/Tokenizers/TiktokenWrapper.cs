using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Tiktoken;
using Tiktoken.Encodings;

namespace CosyVoiceNet
{
    // Wrapper for Tiktoken NuGet package (3.1.4+)
    // Uses the official Tiktoken.Encodings.EncodingLoader API
    public static class TiktokenWrapper
    {
        private static readonly Dictionary<string, object> _encodingCache = new();
        private const string WhisperPattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

        private sealed class CustomEncoding
        {
            public CustomEncoding(Encoder encoder, int vocabSize)
            {
                Encoder = encoder;
                VocabSize = vocabSize;
            }

            public Encoder Encoder { get; }
            public int VocabSize { get; }
        }

        public static object? CreateEncodingFromFile(string path, IEnumerable<string>? specialTokens = null)
        {
            if (string.IsNullOrWhiteSpace(path)) return null;

            // Check cache first
            if (_encodingCache.TryGetValue(path, out var cached))
            {
                Console.WriteLine($"[TiktokenWrapper] Using cached encoding for {path}");
                return cached;
            }

            try
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"[TiktokenWrapper] File not found: {path}");
                    return null;
                }

                Console.WriteLine($"[TiktokenWrapper] Loading tiktoken encoding from {path}");
                
                // Load the custom Whisper/CosyVoice .tiktoken ranks and let the
                // library's real BPE encoder handle segmentation. The previous
                // byte-pair shortcut produced many extra tokens for English.
                var ranks = EncodingLoader.LoadEncodingFromFile(path);
                var specialTokenMap = BuildSpecialTokenMap(ranks.Count, specialTokens);
                var encoding = new Encoding(
                    Path.GetFileName(path),
                    new[] { WhisperPattern },
                    ranks,
                    specialTokenMap);
                var custom = new CustomEncoding(new Encoder(encoding), ranks.Count + specialTokenMap.Count);

                _encodingCache[path] = custom;
                Console.WriteLine($"[TiktokenWrapper] Successfully loaded tiktoken encoding");
                return custom;

                Console.WriteLine($"[TiktokenWrapper] Failed to load encoding from {path}");
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TiktokenWrapper] Error: {ex.Message}");
                return null;
            }
        }

        public static int[]? Encode(object encoding, string text)
        {
            if (encoding == null || string.IsNullOrEmpty(text)) return null;

            try
            {
                if (encoding is CustomEncoding custom)
                {
                    return custom.Encoder.EncodeWithAllAllowedSpecial(text).ToArray();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TiktokenWrapper] Encode error: {ex.Message}");
            }

            return null;
        }

        public static string? Decode(object encoding, IEnumerable<int> ids)
        {
            if (encoding == null || ids == null) return null;

            try
            {
                if (encoding is CustomEncoding custom)
                {
                    return custom.Encoder.Decode(ids.ToArray());
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TiktokenWrapper] Decode error: {ex.Message}");
            }

            return null;
        }

        public static bool TryAddSpecialTokens(object encoding, IDictionary<string, int> specialTokens)
        {
            return true;
        }

        public static int? GetVocabularySize(object encoding)
        {
            if (encoding == null) return null;

            try
            {
                if (encoding is CustomEncoding custom)
                {
                    return custom.VocabSize;
                }
            }
            catch { }

            return null;
        }

        private static Dictionary<string, int> BuildSpecialTokenMap(int startId, IEnumerable<string>? specialTokens)
        {
            var tokens = new Dictionary<string, int>(StringComparer.Ordinal);
            var nextId = startId;

            foreach (var token in specialTokens ?? Array.Empty<string>())
            {
                if (!tokens.ContainsKey(token))
                    tokens[token] = nextId++;
            }

            return tokens;
        }
    }
}
