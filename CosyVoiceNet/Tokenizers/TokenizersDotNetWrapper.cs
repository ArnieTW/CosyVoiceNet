using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Tokenizers.DotNet;

namespace NekoBot_LLM.CosyVoiceNet.Tokenizers
{
    // Wrapper for Tokenizers.DotNet NuGet package (1.4.0)
    // Extended with managed BPE support for vocab.json + merges.txt format
    // Provides parity with Python transformers.AutoTokenizer for BPE models
    public static class TokenizersDotNetWrapper
    {
        private static Dictionary<string, object> _tokenizerCache = new();

        public static object? CreateTokenizerFromFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            {
                Console.WriteLine($"[TokenizersDotNetWrapper] Invalid path: {path}");
                return null;
            }

            try
            {
                var cacheKey = Path.GetFullPath(path);
                if (_tokenizerCache.TryGetValue(cacheKey, out var cached))
                {
                    Console.WriteLine($"[TokenizersDotNetWrapper] Using cached tokenizer");
                    return cached;
                }

                Console.WriteLine($"[TokenizersDotNetWrapper] Loading tokenizer from {path}...");

                // Try unified tokenizer.json first (HuggingFace standard)
                var tokenizerJsonPath = Path.Combine(path, "tokenizer.json");
                if (File.Exists(tokenizerJsonPath))
                {
                    try
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Found tokenizer.json, attempting to load...");
                        var tokenizer = new Tokenizer(vocabPath: tokenizerJsonPath);
                        if (tokenizer != null)
                        {
                            _tokenizerCache[cacheKey] = tokenizer;
                            Console.WriteLine($"[TokenizersDotNetWrapper] Successfully loaded tokenizer.json");
                            return tokenizer;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Failed to load tokenizer.json: {ex.Message}");
                    }
                }

                // Fall back to vocab.json + merges.txt (GPT2/BPE style tokenizers)
                var vocabPath = Path.Combine(path, "vocab.json");
                var mergesPath = Path.Combine(path, "merges.txt");
                
                if (File.Exists(vocabPath) && File.Exists(mergesPath))
                {
                    // First try native Tokenizers.DotNet constructor overloads for vocab+merges.
                    try
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Found vocab.json + merges.txt, attempting native Tokenizers.DotNet load...");

                        var ctor = typeof(Tokenizer)
                            .GetConstructors()
                            .FirstOrDefault(c =>
                            {
                                var p = c.GetParameters();
                                return p.Length >= 2
                                       && p[0].ParameterType == typeof(string)
                                       && p[1].ParameterType == typeof(string);
                            });

                        if (ctor != null)
                        {
                            var args = ctor.GetParameters().Length == 2
                                ? new object[] { vocabPath, mergesPath }
                                : new object[] { vocabPath, mergesPath, null! };

                            var tokenizer = ctor.Invoke(args) as Tokenizer;
                            if (tokenizer != null)
                            {
                                _tokenizerCache[cacheKey] = tokenizer;
                                Console.WriteLine($"[TokenizersDotNetWrapper] Successfully loaded native Tokenizers.DotNet tokenizer from vocab+merges");
                                return tokenizer;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Native vocab+merges load failed: {ex.Message}");
                    }

                    // Then fall back to managed BPE implementation.
                    try
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Loading as managed BPE tokenizer...");
                        var bpeTokenizer = ManagedBPETokenizer.FromVocabAndMerges(vocabPath, mergesPath);
                        if (bpeTokenizer != null)
                        {
                            _tokenizerCache[cacheKey] = bpeTokenizer;
                            Console.WriteLine($"[TokenizersDotNetWrapper] Successfully loaded managed BPE tokenizer");
                            return bpeTokenizer;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[TokenizersDotNetWrapper] Failed to load managed BPE tokenizer: {ex.Message}");
                    }
                }

                Console.WriteLine($"[TokenizersDotNetWrapper] No compatible tokenizer files found in {path}");
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TokenizersDotNetWrapper] Error loading tokenizer: {ex.Message}");
                return null;
            }
        }

        public static int[]? Encode(object tokenizer, string text)
        {
            if (tokenizer == null || string.IsNullOrEmpty(text)) return null;

            try
            {
                if (tokenizer is Tokenizer tok)
                {
                    // Prefer overload that allows disabling automatic special tokens,
                    // to match Python tokenizer.encode behavior used by CosyVoice.
                    var encodeWithFlag = tok.GetType().GetMethod("Encode", new[] { typeof(string), typeof(bool) });
                    if (encodeWithFlag != null)
                    {
                        var flagged = encodeWithFlag.Invoke(tok, new object[] { text, false });
                        if (flagged is IEnumerable<uint> flaggedTokens)
                        {
                            return flaggedTokens.Select(t => (int)t).ToArray();
                        }
                    }

                    var tokens = tok.Encode(text);
                    if (tokens != null)
                    {
                        return tokens.Select(t => (int)t).ToArray();
                    }
                }
                else if (tokenizer is ManagedBPETokenizer bpe)
                {
                    return bpe.Encode(text);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TokenizersDotNetWrapper] Encode error: {ex.Message}");
            }

            return null;
        }

        public static string? Decode(object tokenizer, IEnumerable<int> ids)
        {
            if (tokenizer == null || ids == null) return null;

            try
            {
                if (tokenizer is Tokenizer tok)
                {
                    var uintArray = ids.Select(i => (uint)i).ToArray();
                    return tok.Decode(uintArray);
                }
                else if (tokenizer is ManagedBPETokenizer bpe)
                {
                    return bpe.Decode(ids);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TokenizersDotNetWrapper] Decode error: {ex.Message}");
            }

            return null;
        }

        public static bool AddSpecialTokens(object tokenizer, IEnumerable<string> specialTokens)
        {
            if (tokenizer == null || specialTokens == null) return false;

            try
            {
                var tokenList = specialTokens.ToList();
                if (tokenizer is ManagedBPETokenizer bpe)
                {
                    bpe.AddSpecialTokens(tokenList);
                    Console.WriteLine($"[TokenizersDotNetWrapper] Registered {tokenList.Count} managed BPE special tokens");
                    return true;
                }

                var addSpecialTokens = tokenizer.GetType().GetMethod("AddSpecialTokens", new[] { typeof(IEnumerable<string>) })
                    ?? tokenizer.GetType().GetMethod("AddSpecialTokens", new[] { typeof(string[]) });
                if (addSpecialTokens != null)
                {
                    addSpecialTokens.Invoke(tokenizer, addSpecialTokens.GetParameters()[0].ParameterType == typeof(string[])
                        ? new object[] { tokenList.ToArray() }
                        : new object[] { tokenList });
                    Console.WriteLine($"[TokenizersDotNetWrapper] Registered {tokenList.Count} native tokenizer special tokens");
                    return true;
                }

                Console.WriteLine($"[TokenizersDotNetWrapper] Found {tokenList.Count} special tokens but tokenizer does not expose registration");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TokenizersDotNetWrapper] Error with special tokens: {ex.Message}");
                return false;
            }
        }
    }

    /// <summary>
    /// Managed C# implementation of BPE tokenizer for vocab.json + merges.txt format.
    /// Provides encode/decode parity with HuggingFace transformers AutoTokenizer.
    /// </summary>
    internal class ManagedBPETokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<int, string> _reverseVocab;
        private readonly Dictionary<(string, string), int> _mergeRanks;
        private readonly Dictionary<byte, char> _byteEncoder;
        private readonly Dictionary<char, byte> _byteDecoder;
        private readonly Dictionary<string, string> _bpeCache = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _specialTokenIds = new(StringComparer.Ordinal);
        private readonly string? _unknownToken;
        private int _nextId;

        private static readonly Regex TokenPattern = new(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private ManagedBPETokenizer(Dictionary<string, int> vocab, List<(string, string)> merges)
        {
            _vocab = vocab;
            _reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
            _unknownToken = _vocab.ContainsKey("<unk>") ? "<unk>" : null;
            _nextId = _vocab.Values.Count == 0 ? 0 : _vocab.Values.Max() + 1;

            _mergeRanks = new Dictionary<(string, string), int>();
            for (int i = 0; i < merges.Count; i++)
                _mergeRanks[merges[i]] = i;

            _byteEncoder = BuildByteEncoder();
            _byteDecoder = _byteEncoder.ToDictionary(kv => kv.Value, kv => kv.Key);
        }

        public static ManagedBPETokenizer? FromVocabAndMerges(string vocabPath, string mergesPath)
        {
            try
            {
                var vocabJson = File.ReadAllText(vocabPath);
                var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson);
                if (vocab == null)
                {
                    Console.WriteLine("[ManagedBPETokenizer] Failed to deserialize vocab.json");
                    return null;
                }

                var merges = new List<(string, string)>();
                foreach (var line in File.ReadAllLines(mergesPath))
                {
                    var trimmed = line.Trim();
                    if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#", StringComparison.Ordinal))
                        continue;

                    var parts = trimmed.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 2)
                        merges.Add((parts[0], parts[1]));
                }

                Console.WriteLine($"[ManagedBPETokenizer] Loaded {vocab.Count} vocab tokens and {merges.Count} merge rules");
            return new ManagedBPETokenizer(vocab, merges);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ManagedBPETokenizer] Error loading BPE tokenizer: {ex.Message}");
                return null;
            }
        }

        public int[]? Encode(string text)
        {
            try
            {
                if (string.IsNullOrEmpty(text))
                    return Array.Empty<int>();

                var ids = new List<int>();
                foreach (var segment in SplitSpecialTokens(text))
                {
                    if (segment.Length == 0)
                        continue;

                    if (_specialTokenIds.TryGetValue(segment, out var specialId))
                    {
                        ids.Add(specialId);
                        continue;
                    }

                    foreach (Match match in TokenPattern.Matches(segment))
                    {
                        var token = match.Value;
                        if (token.Length == 0)
                            continue;

                        var encodedToken = EncodeTokenToByteChars(token);
                        var bpeResult = Bpe(encodedToken);

                        foreach (var piece in bpeResult.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                        {
                            if (_vocab.TryGetValue(piece, out var id))
                            {
                                ids.Add(id);
                            }
                            else if (_unknownToken != null && _vocab.TryGetValue(_unknownToken, out var unkId))
                            {
                                ids.Add(unkId);
                            }
                        }
                    }
                }

                return ids.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ManagedBPETokenizer] Encode error: {ex.Message}");
                return null;
            }
        }

        public string? Decode(IEnumerable<int> ids)
        {
            try
            {
                if (ids == null)
                    return null;

                var tokenString = string.Concat(ids
                    .Where(id => _reverseVocab.ContainsKey(id))
                    .Select(id => _reverseVocab[id]));

                var bytes = new List<byte>(tokenString.Length);
                foreach (var ch in tokenString)
                {
                    if (_byteDecoder.TryGetValue(ch, out var b))
                        bytes.Add(b);
                }

                return Encoding.UTF8.GetString(bytes.ToArray());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ManagedBPETokenizer] Decode error: {ex.Message}");
                return null;
            }
        }

        public void AddSpecialTokens(IEnumerable<string> specialTokens)
        {
            foreach (var token in specialTokens)
            {
                if (string.IsNullOrEmpty(token))
                    continue;
                if (!_vocab.TryGetValue(token, out var id))
                {
                    id = _nextId++;
                    _vocab[token] = id;
                    _reverseVocab[id] = token;
                }
                _specialTokenIds[token] = id;
            }
        }

        private IEnumerable<string> SplitSpecialTokens(string text)
        {
            if (_specialTokenIds.Count == 0)
            {
                yield return text;
                yield break;
            }

            var index = 0;
            while (index < text.Length)
            {
                string? matched = null;
                foreach (var token in _specialTokenIds.Keys.OrderByDescending(k => k.Length))
                {
                    if (index + token.Length <= text.Length &&
                        string.CompareOrdinal(text, index, token, 0, token.Length) == 0)
                    {
                        matched = token;
                        break;
                    }
                }

                if (matched is not null)
                {
                    yield return matched;
                    index += matched.Length;
                    continue;
                }

                var next = text.Length;
                foreach (var token in _specialTokenIds.Keys)
                {
                    var tokenIndex = text.IndexOf(token, index, StringComparison.Ordinal);
                    if (tokenIndex >= 0 && tokenIndex < next)
                        next = tokenIndex;
                }

                yield return text.Substring(index, next - index);
                index = next;
            }
        }

        private string EncodeTokenToByteChars(string token)
        {
            var bytes = Encoding.UTF8.GetBytes(token);
            var chars = new char[bytes.Length];
            for (int i = 0; i < bytes.Length; i++)
                chars[i] = _byteEncoder[bytes[i]];
            return new string(chars);
        }

        private string Bpe(string token)
        {
            if (_bpeCache.TryGetValue(token, out var cached))
                return cached;

            var word = token.Select(c => c.ToString()).ToList();
            if (word.Count == 1)
            {
                _bpeCache[token] = token;
                return token;
            }

            var pairs = GetPairs(word);
            while (pairs.Count > 0)
            {
                (string, string)? bestPair = null;
                var bestRank = int.MaxValue;

                foreach (var pair in pairs)
                {
                    if (_mergeRanks.TryGetValue(pair, out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestPair = pair;
                    }
                }

                if (bestPair == null)
                    break;

                var (first, second) = bestPair.Value;
                var newWord = new List<string>(word.Count);
                int i = 0;
                while (i < word.Count)
                {
                    if (i < word.Count - 1 && word[i] == first && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i++;
                    }
                }

                word = newWord;
                if (word.Count == 1)
                    break;

                pairs = GetPairs(word);
            }

            var result = string.Join(" ", word);
            _bpeCache[token] = result;
            return result;
        }

        private static HashSet<(string, string)> GetPairs(IReadOnlyList<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            for (int i = 0; i < word.Count - 1; i++)
                pairs.Add((word[i], word[i + 1]));
            return pairs;
        }

        private static Dictionary<byte, char> BuildByteEncoder()
        {
            var bs = new List<int>();
            bs.AddRange(Enumerable.Range((int)'!', (int)'~' - (int)'!' + 1));
            bs.AddRange(Enumerable.Range((int)'¡', (int)'¬' - (int)'¡' + 1));
            bs.AddRange(Enumerable.Range((int)'®', (int)'ÿ' - (int)'®' + 1));

            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            var map = new Dictionary<byte, char>(256);
            for (int i = 0; i < bs.Count; i++)
                map[(byte)bs[i]] = (char)cs[i];

            return map;
        }
    }
}
