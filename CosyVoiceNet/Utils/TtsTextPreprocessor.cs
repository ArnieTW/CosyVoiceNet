using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace CosyVoiceNet.Utils
{
    public static class TtsTextPreprocessor
    {
        private const int MaxAcronymSpellOutLength = 4;

        private static readonly IReadOnlyDictionary<string, string> PronunciationAliases =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["FullTimeSlob"] = "Fulltimeslob"
            };

        public static string PrepareForTts(string text)
        {
            if (string.IsNullOrWhiteSpace(text) || text.Contains("<|", StringComparison.Ordinal))
                return text;

            return Regex.Replace(
                text,
                @"\b[A-Za-z][A-Za-z0-9]{2,}\b",
                match => ExpandSpeakableIdentifier(match.Value),
                RegexOptions.CultureInvariant);
        }

        private static string ExpandSpeakableIdentifier(string token)
        {
            if (PronunciationAliases.TryGetValue(token, out var alias))
                return alias;

            var hasLower = token.Any(char.IsLower);
            var hasUpper = token.Any(char.IsUpper);
            var hasDigit = token.Any(char.IsDigit);
            if (!hasDigit && !(hasLower && hasUpper) && !(hasUpper && token.Length <= MaxAcronymSpellOutLength))
                return token;

            var expanded = Regex.Replace(token, @"(?<=[a-z0-9])(?=[A-Z])", " ", RegexOptions.CultureInvariant);
            expanded = Regex.Replace(expanded, @"(?<=[A-Z])(?=[A-Z][a-z])", " ", RegexOptions.CultureInvariant);
            expanded = Regex.Replace(expanded, @"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", RegexOptions.CultureInvariant);

            var parts = expanded.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            for (var i = 0; i < parts.Length; i++)
            {
                if (parts[i].Length >= 2 && parts[i].Length <= MaxAcronymSpellOutLength && parts[i].All(char.IsUpper))
                    parts[i] = string.Join(" ", parts[i].ToCharArray());
            }

            return string.Join(" ", parts);
        }
    }
}
