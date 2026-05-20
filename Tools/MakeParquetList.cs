using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Concurrent;

namespace CosyVoiceNet.Tools
{
    // Port of make_parquet_list.py; writes tar files containing raw audio bytes and associated metadata files
    public static class MakeParquetList
    {
        public static void Run(string srcDir, string desDir, int numUttsPerParquet = 1000, int numProcesses = 1, bool dpo = false)
        {
            Directory.CreateDirectory(desDir);
            var utt2wav = LoadMap(Path.Combine(srcDir, "wav.scp"));
            var utt2text = LoadMapText(Path.Combine(srcDir, "text"));
            var utt2spk = LoadMap(Path.Combine(srcDir, "utt2spk"));
            var utts = utt2wav.Keys.ToList();

            var utt2embedding = File.Exists(Path.Combine(srcDir, "utt2embedding.pt")) ? JsonSerializer.Deserialize<Dictionary<string, float[]>>(File.ReadAllText(Path.Combine(srcDir, "utt2embedding.pt.json"))) : null;
            var spk2embedding = File.Exists(Path.Combine(srcDir, "spk2embedding.pt")) ? JsonSerializer.Deserialize<Dictionary<string, float[]>>(File.ReadAllText(Path.Combine(srcDir, "spk2embedding.pt.json"))) : null;
            var utt2speech_token = File.Exists(Path.Combine(srcDir, "utt2speech_token.pt.json")) ? JsonSerializer.Deserialize<Dictionary<string, float[]>>(File.ReadAllText(Path.Combine(srcDir, "utt2speech_token.pt.json"))) : null;
            var utt2instruct = File.Exists(Path.Combine(srcDir, "instruct")) ? LoadMapText(Path.Combine(srcDir, "instruct")) : null;
            var utt2reject_speech_token = dpo && File.Exists(Path.Combine(srcDir + "_reject", "utt2speech_token.pt")) ? JsonSerializer.Deserialize<Dictionary<string, float[]>>(File.ReadAllText(Path.Combine(srcDir + "_reject", "utt2speech_token.pt.json"))) : null;

            var parquetList = new List<string>();
            var utt2parquetList = new List<string>();
            var spk2parquetList = new List<string>();

            for (int i = 0, j = 0; j < utts.Count; i++, j += numUttsPerParquet)
            {
                var slice = utts.Skip(j).Take(numUttsPerParquet).ToList();
                var parquetFile = Path.Combine(desDir, $"parquet_{i:000000000}.tar");
                var utt2parquetFile = Path.Combine(desDir, $"utt2parquet_{i:000000000}.json");
                var spk2parquetFile = Path.Combine(desDir, $"spk2parquet_{i:000000000}.json");

                // create simple TAR-like file by concatenating raw bytes with a small index
                using (var fs = File.Create(parquetFile))
                using (var bw = new BinaryWriter(fs))
                {
                    var index = new Dictionary<string, long>();
                    foreach (var utt in slice)
                    {
                        var data = File.ReadAllBytes(utt2wav[utt]);
                        index[utt] = fs.Position;
                        bw.Write(data.Length);
                        bw.Write(data);
                    }
                }

                File.WriteAllText(utt2parquetFile, JsonSerializer.Serialize(slice.ToDictionary(u => u, u => parquetFile)));
                var spkList = slice.Select(u => utt2spk[u]).Distinct().ToList();
                File.WriteAllText(spk2parquetFile, JsonSerializer.Serialize(spkList.ToDictionary(s => s, s => parquetFile)));

                parquetList.Add(parquetFile);
                utt2parquetList.Add(utt2parquetFile);
                spk2parquetList.Add(spk2parquetFile);
            }

            File.WriteAllLines(Path.Combine(desDir, "data.list"), parquetList);
            File.WriteAllLines(Path.Combine(desDir, "utt2data.list"), utt2parquetList);
            File.WriteAllLines(Path.Combine(desDir, "spk2data.list"), spk2parquetList);
        }

        private static Dictionary<string, string> LoadMap(string path)
        {
            var dict = new Dictionary<string, string>();
            foreach (var l in File.ReadAllLines(path))
            {
                var parts = l.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2) dict[parts[0]] = parts[1];
            }
            return dict;
        }

        private static Dictionary<string, string> LoadMapText(string path)
        {
            var dict = new Dictionary<string, string>();
            foreach (var l in File.ReadAllLines(path))
            {
                var parts = l.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2) dict[parts[0]] = string.Join(' ', parts.Skip(1));
            }
            return dict;
        }
    }
}

// Equivalent Python file: cosyvoice/tools/make_parquet_list.py
