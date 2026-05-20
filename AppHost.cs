using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;

namespace CosyVoiceApp
{
    public static class AppHost
    {
        public static readonly Dictionary<string, string> Assets = new();

        static AppHost()
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach (var resourceName in assembly.GetManifestResourceNames())
            {
                if (resourceName.Contains(".asset.") || resourceName.EndsWith(".asset"))
                {
                    var key = resourceName.Substring(resourceName.IndexOf(".asset.") + 7);
                    var tempDirectory = Path.Combine(Path.GetTempPath(), "CosyVoiceNet", "asset");
                    Directory.CreateDirectory(tempDirectory);
                    var safeKey = key.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
                    var tempFile = Path.Combine(tempDirectory, safeKey);
                    var tempFileDirectory = Path.GetDirectoryName(tempFile);
                    if (!string.IsNullOrWhiteSpace(tempFileDirectory))
                        Directory.CreateDirectory(tempFileDirectory);

                    using (var stream = assembly.GetManifestResourceStream(resourceName))
                    {
                        if (stream == null)
                            continue;

                        if (File.Exists(tempFile) && IsSameContent(tempFile, stream))
                        {
                            Assets[key] = tempFile;
                            continue;
                        }

                        stream.Position = 0;
                        using var file = File.Create(tempFile);
                        stream.CopyTo(file);
                    }
                    Assets[key] = tempFile;
                }
            }
        }

        private static bool IsSameContent(string path, Stream stream)
        {
            try
            {
                var originalPosition = stream.CanSeek ? stream.Position : 0;
                using var existing = File.OpenRead(path);
                var existingHash = SHA256.HashData(existing);
                if (stream.CanSeek)
                    stream.Position = originalPosition;

                var embeddedHash = SHA256.HashData(stream);
                if (stream.CanSeek)
                    stream.Position = originalPosition;

                return existingHash.SequenceEqual(embeddedHash);
            }
            catch
            {
                if (stream.CanSeek)
                    stream.Position = 0;
                return false;
            }
        }
    }
}
