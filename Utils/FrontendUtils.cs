using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

// Equivalent Python file: cosyvoice/utils/frontend_utils.py

namespace CosyVoiceNet.Utils
{
    public static class FrontendUtils
    {
        private static readonly Regex ChineseCharPattern = new Regex("[\u4e00-\u9fff]+", RegexOptions.Compiled);

        public static bool ContainsChinese(string text)
        {
            if (string.IsNullOrEmpty(text)) return false;
            return ChineseCharPattern.IsMatch(text);
        }

        public static string ReplaceCornerMark(string text)
        {
            if (text == null) return null;
            text = text.Replace("²", "平方");
            text = text.Replace("³", "立方");
            return text;
        }

        public static string RemoveBracket(string text)
        {
            if (text == null) return null;
            text = text.Replace("（", "").Replace("）", "");
            text = text.Replace("【", "").Replace("】", "");
            text = text.Replace("`", "");
            text = text.Replace("——", " ");
            return text;
        }

        public static string SpellOutNumber(string text, Func<string, string> numberToWords)
        {
            if (string.IsNullOrEmpty(text) || numberToWords == null) return text;
            var result = new List<string>();
            int? st = null;
            for (int i = 0; i < text.Length; i++)
            {
                var c = text[i];
                if (!char.IsDigit(c))
                {
                    if (st.HasValue)
                    {
                        var numStr = numberToWords(text.Substring(st.Value, i - st.Value));
                        result.Add(numStr);
                        st = null;
                    }
                    result.Add(c.ToString());
                }
                else
                {
                    if (!st.HasValue) st = i;
                }
            }
            if (st.HasValue)
            {
                var numStr = numberToWords(text.Substring(st.Value));
                result.Add(numStr);
            }
            return string.Join("", result);
        }

        public static IEnumerable<string> SplitParagraph(string text, Func<string, int> tokenize, string lang, int tokenMaxN = 80, int tokenMinN = 60, int mergeLen = 20, bool commaSplit = false)
        {
            var pounc = lang == "zh"
                ? new[] { '。', '？', '！', '；', '：', '、', '.', '?', '!', ';' }
                : new[] { '.', '?', '!', ';', ':' };
            if (commaSplit)
                pounc = pounc.Concat(new[] { '，', ',' }).ToArray();
            if (!pounc.Contains(text.Last()))
                text += lang == "zh" ? "。" : ".";
            var utts = new List<string>();
            int st = 0;
            for (int i = 0; i < text.Length; i++)
            {
                if (pounc.Contains(text[i]))
                {
                    if (i > st)
                        utts.Add(text.Substring(st, i - st) + text[i]);
                    st = i + 1;
                }
            }
            // Merge logic
            var finalUtts = new List<string>();
            string curUtt = "";
            foreach (var utt in utts)
            {
                if (tokenize(curUtt + utt) > tokenMaxN && tokenize(curUtt) > tokenMinN)
                {
                    finalUtts.Add(curUtt);
                    curUtt = "";
                }
                curUtt += utt;
            }
            if (curUtt.Length > 0)
            {
                int curLen = lang == "zh" ? curUtt.Length : tokenize(curUtt);
                if (curLen < mergeLen && finalUtts.Count > 0)
                    finalUtts[finalUtts.Count - 1] += curUtt;
                else
                    finalUtts.Add(curUtt);
            }
            return finalUtts;
        }

        public static string ReplaceBlank(string text)
        {
            var result = "";
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == ' ' && i > 0 && i < text.Length - 1 && char.IsLetterOrDigit(text[i - 1]) && char.IsLetterOrDigit(text[i + 1]))
                {
                    result += text[i];
                }
                else if (text[i] != ' ')
                {
                    result += text[i];
                }
            }
            return result;
        }

        public static bool IsOnlyPunctuation(string text)
        {
            var punctuationPattern = new Regex("^[\\p{P}\\p{S}]*$", RegexOptions.Compiled);
            return punctuationPattern.IsMatch(text);
        }
    }
}
