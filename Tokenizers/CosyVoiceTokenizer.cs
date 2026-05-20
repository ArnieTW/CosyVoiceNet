using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NekoBot_LLM.CosyVoiceNet.Tokenizers;

namespace CosyVoice.Tokenizer
{
    // Managed export of CosyVoice tokenizer utilities.
    // This port aims to replicate the python `cosyvoice.tokenizer` helpers
    // in a pure-managed form using the project's `CosyVoiceNet.Tokenizer`.
    // Notes:
    // - We synthesize the same special token lists used by the original
    //   implementation and register them with the managed tokenizer.
    // - For full parity with `tiktoken` or HuggingFace AutoTokenizer, prefer
    //   configuring `TokenizersDotNetWrapper` or an HF-backed tokenizer.
    // - TODO: add optional integration with a native tiktoken binding if
    //   exact mergeable-ranks / byte-pair behavior is required.
    // Equivalent Python file: cosyvoice/tokenizers/cosyvoice_tokenizer.py

    public static class CosyVoiceTokenizerFactory
    {
        private static readonly Dictionary<string, string> LANGUAGES = new(StringComparer.OrdinalIgnoreCase)
        {
            {"en","english"},{"zh","chinese"},{"de","german"},{"es","spanish"},{"ru","russian"},{"ko","korean"},
            {"fr","french"},{"ja","japanese"},{"pt","portuguese"},{"tr","turkish"},{"pl","polish"},{"ca","catalan"},
            {"nl","dutch"},{"ar","arabic"},{"sv","swedish"},{"it","italian"},{"id","indonesian"},{"hi","hindi"},
            {"fi","finnish"},{"vi","vietnamese"},{"he","hebrew"},{"uk","ukrainian"},{"el","greek"},{"ms","malay"},
            {"cs","czech"},{"ro","romanian"},{"da","danish"},{"hu","hungarian"},{"ta","tamil"},{"no","norwegian"},
            {"th","thai"},{"ur","urdu"},{"hr","croatian"},{"bg","bulgarian"},{"lt","lithuanian"},{"la","latin"},
            {"mi","maori"},{"ml","malayalam"},{"cy","welsh"},{"sk","slovak"},{"te","telugu"},{"fa","persian"},
            {"lv","latvian"},{"bn","bengali"},{"sr","serbian"},{"az","azerbaijani"},{"sl","slovenian"},{"kn","kannada"},
            {"et","estonian"},{"mk","macedonian"},{"br","breton"},{"eu","basque"},{"is","icelandic"},{"hy","armenian"},
            {"ne","nepali"},{"mn","mongolian"},{"bs","bosnian"},{"kk","kazakh"},{"sq","albanian"},{"sw","swahili"},
            {"gl","galician"},{"mr","marathi"},{"pa","punjabi"},{"si","sinhala"},{"km","khmer"},{"sn","shona"},
            {"yo","yoruba"},{"so","somali"},{"af","afrikaans"},{"oc","occitan"},{"ka","georgian"},{"be","belarusian"},
            {"tg","tajik"},{"sd","sindhi"},{"gu","gujarati"},{"am","amharic"},{"yi","yiddish"},{"lo","lao"},
            {"uz","uzbek"},{"fo","faroese"},{"ht","haitian creole"},{"ps","pashto"},{"tk","turkmen"},{"nn","nynorsk"},
            {"mt","maltese"},{"sa","sanskrit"},{"lb","luxembourgish"},{"my","myanmar"},{"bo","tibetan"},{"tl","tagalog"},
            {"mg","malagasy"},{"as","assamese"},{"tt","tatar"},{"haw","hawaiian"},{"ln","lingala"},{"ha","hausa"},
            {"ba","bashkir"},{"jw","javanese"},{"su","sundanese"},{"yue","cantonese"},{"minnan","minnan"},{"wuyu","wuyu"},
            {"dialect","dialect"},{"zh/en","zh/en"},{"en/zh","en/zh"}
        };

        private static readonly Dictionary<string, string> TO_LANGUAGE_CODE = new(StringComparer.OrdinalIgnoreCase)
        {
            // build reverse map for alias lookup similar to python implementation
            // we include a few aliases used by the original project
            {"burmese","my"},{"valencian","ca"},{"flemish","nl"},{"haitian","ht"},
            {"letzeburgesch","lb"},{"pushto","ps"},{"panjabi","pa"},{"moldavian","ro"},
            {"moldovan","ro"},{"sinhalese","si"},{"castilian","es"},{"mandarin","zh"}
        };

        private static readonly string[] AUDIO_EVENT = new[] { "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause", "Laughter" };

        private static readonly string[] EMOTION = new[] { "HAPPY", "SAD", "ANGRY", "NEUTRAL" };

        private static readonly Dictionary<string, string> TTS_Vocal_Token = BuildTtsVocalToken();

        private static Dictionary<string, string> BuildTtsVocalToken()
        {
            var dict = new Dictionary<string, string>();
            dict["TTS/B"] = "TTS/B"; dict["TTS/O"] = "TTS/O"; dict["TTS/Q"] = "TTS/Q"; dict["TTS/A"] = "TTS/A"; dict["TTS/CO"] = "TTS/CO"; dict["TTS/CL"] = "TTS/CL"; dict["TTS/H"] = "TTS/H";
            for (int i = 1; i <= 13; i++) dict[$"TTS/SP{i:00}"] = $"TTS/SP{i:00}";
            return dict;
        }

        public class CosyVoiceWhisperTokenizer
        {
            private readonly object _encoding;
            private readonly IDictionary<string, int> _specialTokens;

            public object EncodingObject => _encoding;

            private CosyVoiceWhisperTokenizer(object encoding, IDictionary<string, int> specialTokens)
            {
                _encoding = encoding;
                _specialTokens = specialTokens;
            }

            public static CosyVoiceWhisperTokenizer? FromTiktokenAsset(
                string assetPath,
                int numLanguages = 99,
                bool cosyVoiceSpecials = true)
            {
                if (string.IsNullOrWhiteSpace(assetPath) || !File.Exists(assetPath)) return null;
                var specialTokens = cosyVoiceSpecials
                    ? BuildSpecialTokens(numLanguages)
                    : BuildOpenAIWhisperSpecialTokens(numLanguages);
                var enc = CosyVoiceNet.TiktokenWrapper.CreateEncodingFromFile(assetPath, specialTokens);
                if (enc == null) return null;
                var specials = new Dictionary<string, int>();
                int n = 0;
                foreach (var t in specialTokens) specials[t] = n++;
                // try to register special tokens with encoding if supported
                try { CosyVoiceNet.TiktokenWrapper.TryAddSpecialTokens(enc, specials); } catch { }
                return new CosyVoiceWhisperTokenizer(enc, specials);
            }

            public int[] Encode(string text)
            {
                var res = CosyVoiceNet.TiktokenWrapper.Encode(_encoding, text);
                if (res == null) return Array.Empty<int>();
                return res;
            }

            public string Decode(IEnumerable<int> ids)
            {
                var s = CosyVoiceNet.TiktokenWrapper.Decode(_encoding, ids);
                return s ?? string.Empty;
            }

            public int? VocabSize() => CosyVoiceNet.TiktokenWrapper.GetVocabularySize(_encoding);
        }

        public static object get_tokenizer(bool multilingual = true, int num_languages = 99, string? language = null, string? task = null)
        {
            const string fileName = "multilingual_zh_ja_yue_char_del.tiktoken";
            var assetPath = GetBundledAssetPath(fileName);
            if (assetPath == null)
                throw new FileNotFoundException("Bundled Whisper tokenizer asset not found.", fileName);

            return CosyVoiceWhisperTokenizer.FromTiktokenAsset(assetPath, num_languages)
                ?? throw new InvalidOperationException($"Failed to load Whisper tokenizer asset: {assetPath}");
        }

        public static object get_whisper_tokenizer(bool multilingual = true, int num_languages = 99, string? language = null, string? task = null)
        {
            var fileName = multilingual ? "multilingual.tiktoken" : "gpt2.tiktoken";
            var assetPath = GetBundledAssetPath(fileName);
            if (assetPath == null)
                throw new FileNotFoundException("Bundled OpenAI Whisper tokenizer asset not found.", fileName);

            return CosyVoiceWhisperTokenizer.FromTiktokenAsset(assetPath, num_languages, cosyVoiceSpecials: false)
                ?? throw new InvalidOperationException($"Failed to load Whisper tokenizer asset: {assetPath}");
        }

        private static string? GetBundledAssetPath(string fileName)
        {
            return CosyVoiceApp.AppHost.Assets.TryGetValue(fileName, out var path)
                ? path
                : null;
        }

        public static IEnumerable<string> BuildSpecialTokensForTiktoken(int numLanguages = 99)
            => BuildSpecialTokens(numLanguages);

        private static IEnumerable<string> BuildSpecialTokens(int numLanguages)
        {
            var specials = new List<string>
            {
                "<|endoftext|>",
                "<|startoftranscript|>",
            };

            var langKeys = LANGUAGES.Keys.Take(numLanguages).ToList();
            specials.AddRange(langKeys.Select(k => $"<|{k}|>"));
            specials.AddRange(AUDIO_EVENT.Select(a => $"<|{a}|>"));
            specials.AddRange(EMOTION.Select(e => $"<|{e}|>"));
            specials.Add("<|translate|>");
            specials.Add("<|transcribe|>");
            specials.Add("<|startoflm|>");
            specials.Add("<|startofprev|>");
            specials.Add("<|nospeech|>");
            specials.Add("<|notimestamps|>");
            for (int i = 1; i <= 30; i++) specials.Add($"<|SPECIAL_TOKEN_{i}|>");
            specials.AddRange(TTS_Vocal_Token.Keys.Select(k => $"<|{k}|>"));
            for (int i = 0; i < 1501; i++) specials.Add($"<|{i * 0.02:0.00}|>");
            return specials.Distinct();
        }

        private static IEnumerable<string> BuildOpenAIWhisperSpecialTokens(int numLanguages)
        {
            var specials = new List<string>
            {
                "<|endoftext|>",
                "<|startoftranscript|>",
            };
            specials.AddRange(LANGUAGES.Keys.Take(numLanguages).Select(k => $"<|{k}|>"));
            specials.Add("<|translate|>");
            specials.Add("<|transcribe|>");
            specials.Add("<|startoflm|>");
            specials.Add("<|startofprev|>");
            specials.Add("<|nospeech|>");
            specials.Add("<|notimestamps|>");
            for (int i = 0; i < 1501; i++) specials.Add($"<|{i * 0.02:0.00}|>");
            return specials.Distinct();
        }

        // Note: CosyVoice relies on HuggingFace/tiktoken tokenizers for exact
        // behavior. The managed regex-based tokenizer has been removed to avoid
        // silent mismatches. Use `CosyVoice2Tokenizer` / `CosyVoice3Tokenizer`
        // which require the Tokenizers.DotNet bindings and tokenizer files.

        public class CosyVoice2Tokenizer
        {
            private readonly bool _skipSpecialTokens;
            private readonly object _hfInstance;
            private readonly bool _useHf = true;

            public CosyVoice2Tokenizer(string tokenPath, bool skipSpecialTokens = true)
                : this(tokenPath, skipSpecialTokens, BuildCosyVoice2SpecialTokens())
            {
            }

            protected CosyVoice2Tokenizer(string tokenPath, bool skipSpecialTokens, IEnumerable<string> specialTokens)
            {
                _skipSpecialTokens = skipSpecialTokens;

                // Prefer the installed Tokenizers (huggingface) binding when available
                try
                {
                    var inst = TokenizersDotNetWrapper.CreateTokenizerFromFile(tokenPath ?? string.Empty);
                    if (inst == null)
                    {
                        throw new InvalidOperationException("HuggingFace Tokenizers binding not available or tokenizer files not found. Install Tokenizers.DotNet and ensure tokenizer files are present at the provided path.");
                    }
                    _hfInstance = inst;
                    _useHf = true;
                    // register special tokens via reflection wrapper
                    TokenizersDotNetWrapper.AddSpecialTokens(_hfInstance, specialTokens);
                    // no fallback: HF/tokenizers is required for exact CosyVoice behavior
                    return;
                }
                catch { /* fall back to managed tokenizer */ }
                // constructor will throw above if HF tokenizer is not available
                throw new InvalidOperationException("Unexpected fallback reached when creating CosyVoice2Tokenizer.");
            }

            public IEnumerable<int> Encode(string text)
            {
                if (!_useHf || _hfInstance == null) throw new InvalidOperationException("HF Tokenizers instance is not available. Ensure Tokenizers.DotNet is installed and tokenizer files are correct.");
                var arr = TokenizersDotNetWrapper.Encode(_hfInstance, text);
                if (arr == null) throw new InvalidOperationException("Tokenizers failed to encode text.");
                return arr;
            }

            public string Decode(IEnumerable<int> tokens)
            {
                if (!_useHf || _hfInstance == null) throw new InvalidOperationException("HF Tokenizers instance is not available. Ensure Tokenizers.DotNet is installed and tokenizer files are correct.");
                var s = TokenizersDotNetWrapper.Decode(_hfInstance, tokens);
                if (s == null) throw new InvalidOperationException("Tokenizers failed to decode ids.");
                return s;
            }

            protected static string[] BuildCosyVoice2SpecialTokens() => new[]
            {
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|endofprompt|>",
                "[breath]",
                "<strong>",
                "</strong>",
                "[noise]",
                "[laughter]",
                "[cough]",
                "[clucking]",
                "[accent]",
                "[quick_breath]",
                "<laughter>",
                "</laughter>",
                "[hissing]",
                "[sigh]",
                "[vocalized-noise]",
                "[lipsmack]",
                "[mn]"
            };
        }

        public class CosyVoice3Tokenizer : CosyVoice2Tokenizer
        {
            public CosyVoice3Tokenizer(string tokenPath, bool skipSpecialTokens = true)
                : base(tokenPath, skipSpecialTokens, BuildCosyVoice3SpecialTokens())
            {
            }

            private static string[] BuildCosyVoice3SpecialTokens() => BuildCosyVoice2SpecialTokens()
                .Concat(new[]
                {
                    "<|endofsystem|>",
                    "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
                    "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                    "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
                    "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
                    "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                    "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
                    "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
                    "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
                    "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
                    "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
                    "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                    "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
                    "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
                    "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
                    "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
                    "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                    "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
                    "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
                    "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]"
                })
                .ToArray();
        }

        public static object GetQwenTokenizer(string tokenPath, bool skipSpecialTokens, string version = "cosyvoice2")
        {
            if (version == "cosyvoice2") return new CosyVoice2Tokenizer(tokenPath, skipSpecialTokens);
            else if (version == "cosyvoice3") return new CosyVoice3Tokenizer(tokenPath, skipSpecialTokens);
            else throw new ArgumentException("unknown version");
        }
    }
}
