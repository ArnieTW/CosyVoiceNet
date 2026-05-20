// CosyVoice.cs
// Exported from CosyVoice/cosyvoice/cli/cosyvoice.py
using HuggingfaceHub;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using CosyVoiceNet.Utils;
using TorchSharp;
using CosyVoiceNet.Utilities;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using static CosyVoiceNet.cli.CosyVoiceModel;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;

namespace CosyVoiceNet.cli
{
    public class CosyVoice : IDisposable
    {
        protected readonly string _modelDir;
        protected readonly bool _fp16;
        protected CosyVoiceFrontEnd _frontend;
        protected int _sampleRate;
        protected CosyVoiceModel _model;
        protected Dictionary<string, object> Configs;
        private readonly Dictionary<string, SavedVoiceMetadata> _savedVoiceMetadata = new(StringComparer.OrdinalIgnoreCase);
        protected virtual string SpeechTokenizerOnnx => "speech_tokenizer_v1.onnx";
        protected virtual Type ModelType => typeof(CosyVoiceModel);
        private const int SavedVoicePromptProcessingVersion = 5;
        public bool NormalizeTextForTts { get; set; }
        private static readonly object BackendLock = new object();
        private static CosyVoiceBackend _globalBackend = CosyVoiceBackend.Auto;
        private readonly CosyVoiceBackend _requestedBackend;
        private readonly CosyVoiceRuntimeOptions _runtimeOptions;
        private readonly ICosyVoiceLogger? _logger;

        public static CosyVoiceBackend GlobalBackend
        {
            get
            {
                lock (BackendLock)
                    return _globalBackend;
            }
        }

        public static void SetGlobalBackend(CosyVoiceBackend backend)
        {
            CosyVoiceBackendResolver.Resolve(backend);
            lock (BackendLock)
                _globalBackend = backend;
        }

        public CosyVoiceBackend RequestedBackend => _model?.RequestedBackend ?? _requestedBackend;
        public CosyVoiceBackend ActiveBackend => _model?.ActiveBackend ?? CosyVoiceBackend.Cpu;
        public string Device => _model?.Device ?? "cpu";
        public string LlmDevice => _model?.LlmDevice ?? Device;
        public string FlowDevice => _model?.FlowDevice ?? Device;
        public string HiftDevice => _model?.HiftDevice ?? Device;
        public CosyVoiceBackend CampPlusOnnxBackend => _frontend?.CampPlusOnnxBackend ?? CosyVoiceBackend.Cpu;
        public CosyVoiceBackend SpeechTokenizerOnnxBackend => _frontend?.SpeechTokenizerOnnxBackend ?? CosyVoiceBackend.Cpu;
        public int SampleRate => _sampleRate;
        public int SamplingTopK
        {
            get => _model?.SamplingTopK ?? 25;
            set
            {
                if (_model is null)
                    throw new InvalidOperationException("CosyVoice model is not initialized.");
                _model.SamplingTopK = value;
            }
        }

        public CosyVoice(string modelDir, bool loadJit = false, bool loadTrt = false, bool fp16 = false, int trtConcurrent = 1, CosyVoiceBackend? backend = null, CosyVoiceRuntimeOptions runtimeOptions = null)
        {
            _requestedBackend = backend ?? GlobalBackend;
            _runtimeOptions = runtimeOptions?.Clone() ?? new CosyVoiceRuntimeOptions();
            _logger = _runtimeOptions.Logger;
            NormalizeTextForTts = _runtimeOptions.NormalizeTextForTts;
            var backendSelection = CosyVoiceBackendResolver.Resolve(_requestedBackend);
            ApplyTorchRuntimeOptions(backendSelection.ActiveBackend, _runtimeOptions);
            var modelLocalDir = ResolveLocalModelDir(modelDir);
            _modelDir = modelLocalDir;
            _fp16 = fp16;
            EnsureModelDir(modelDir);
            Configs = LoadYamlConfig();
            var onnxBackend = _runtimeOptions.OnnxBackend ?? backendSelection.ActiveBackend;
            InitFrontend(Configs, modelLocalDir, SpeechTokenizerOnnx, onnxBackend);
            LoadSavedVoices();
            _sampleRate = Configs.TryGetValue("sample_rate", out var sr) ? Convert.ToInt32(sr) : 16000;
            InitModel(Configs, fp16, modelLocalDir, _requestedBackend);
            if (loadJit) LoadJitModules(fp16, modelLocalDir);
            if (loadTrt) LoadTrtModules(fp16, modelLocalDir, trtConcurrent);
        }

        public CosyVoiceRuntimeOptions RuntimeOptions => _runtimeOptions.Clone();

        public void ApplyRuntimeOptions()
        {
            ApplyTorchRuntimeOptions(ActiveBackend, _runtimeOptions);
        }

        private static void ApplyTorchRuntimeOptions(CosyVoiceBackend activeBackend, CosyVoiceRuntimeOptions options)
        {
            options ??= new CosyVoiceRuntimeOptions();
            Common.SamplingBackend = activeBackend == CosyVoiceBackend.Cpu && options.SamplingBackend == CosyVoiceSamplingBackend.Cuda
                ? CosyVoiceSamplingBackend.Cpu
                : options.SamplingBackend;

            if (activeBackend != CosyVoiceBackend.Cpu)
                return;

            if (options.CpuThreads.HasValue)
                TrySetTorchThreads(torch.set_num_threads, options.CpuThreads.Value, "CPU intra-op", options.Logger);

            if (options.CpuInteropThreads.HasValue)
            {
                TrySetTorchThreads(torch.set_num_interop_threads, options.CpuInteropThreads.Value, "CPU inter-op", options.Logger);
            }
            else if (torch.get_num_interop_threads() > 1)
            {
                TrySetTorchThreads(torch.set_num_interop_threads, 1, "CPU inter-op", options.Logger);
            }
        }

        private static void TrySetTorchThreads(Action<int> setter, int count, string label, ICosyVoiceLogger? logger)
        {
            if (count <= 0)
                return;

            try
            {
                setter(count);
            }
            catch (Exception ex)
            {
                CosyVoiceLog.Write(logger, CosyVoiceLogLevel.Warning, $"Unable to set {label} thread count to {count}.", ex);
            }
        }

        // Instance config loader, can be overridden in derived classes
        protected virtual Dictionary<string, object> LoadYamlConfig()
        {
            return LoadYamlConfigWithOverride(Path.Combine(_modelDir, "cosyvoice.yaml"), null);
        }

        private static readonly HttpClient ModelDownloadHttpClient = new()
        {
            Timeout = TimeSpan.FromHours(6)
        };

        public static void EnsureModelDir(string modelVer, IProgress<CosyVoiceDownloadProgress>? progress = null)
        {
            var modelLocalDir = ResolveLocalModelDir(modelVer);
            ReportDownloadProgress(
                progress,
                modelVer,
                modelLocalDir,
                repositoryId: null,
                fileName: null,
                fileIndex: 0,
                fileCount: 0,
                fileBytesDownloaded: 0,
                fileTotalBytes: null,
                modelBytesDownloaded: 0,
                modelTotalBytes: null,
                CosyVoiceDownloadStage.CheckingLocal,
                $"Checking local model directory: {modelLocalDir}");

            if (IsUsableModelDir(modelLocalDir))
            {
                ReportDownloadProgress(
                    progress,
                    modelVer,
                    modelLocalDir,
                    repositoryId: null,
                    fileName: null,
                    fileIndex: 0,
                    fileCount: 0,
                    fileBytesDownloaded: 0,
                    fileTotalBytes: null,
                    modelBytesDownloaded: 0,
                    modelTotalBytes: null,
                    CosyVoiceDownloadStage.CompletedModel,
                    "Model is already available locally.");
                return;
            }

            Directory.CreateDirectory(modelLocalDir);

            var errors = new List<string>();
            var triedRepos = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var repoId in ResolveRemoteRepoCandidates(modelVer))
            {
                if (!triedRepos.Add(repoId))
                    continue;

                try
                {
                    DownloadModelFilesAsync(modelVer, modelLocalDir, repoId, progress).GetAwaiter().GetResult();

                    if (IsUsableModelDir(modelLocalDir))
                    {
                        ReportDownloadProgress(
                            progress,
                            modelVer,
                            modelLocalDir,
                            repoId,
                            fileName: null,
                            fileIndex: 0,
                            fileCount: 0,
                            fileBytesDownloaded: 0,
                            fileTotalBytes: null,
                            modelBytesDownloaded: 0,
                            modelTotalBytes: null,
                            CosyVoiceDownloadStage.CompletedModel,
                            "Model download completed.");
                        return;
                    }

                    errors.Add($"{repoId}: download completed but required files are still missing ({DescribeMissingRequiredFiles(modelLocalDir)}).");
                }
                catch (Exception ex)
                {
                    ReportDownloadProgress(
                        progress,
                        modelVer,
                        modelLocalDir,
                        repoId,
                        fileName: null,
                        fileIndex: 0,
                        fileCount: 0,
                        fileBytesDownloaded: 0,
                        fileTotalBytes: null,
                        modelBytesDownloaded: 0,
                        modelTotalBytes: null,
                        CosyVoiceDownloadStage.Failed,
                        ex.Message);
                    errors.Add($"{repoId}: {ex.Message}");
                }
            }

            throw new DirectoryNotFoundException(
                $"Model directory is incomplete and auto-download failed: {modelLocalDir}. Tried: {string.Join("; ", errors)}");
        }

        private static async Task DownloadModelFilesAsync(
            string modelVer,
            string modelLocalDir,
            string repoId,
            IProgress<CosyVoiceDownloadProgress>? progress)
        {
            ReportDownloadProgress(
                progress,
                modelVer,
                modelLocalDir,
                repoId,
                fileName: null,
                fileIndex: 0,
                fileCount: 0,
                fileBytesDownloaded: 0,
                fileTotalBytes: null,
                modelBytesDownloaded: 0,
                modelTotalBytes: null,
                CosyVoiceDownloadStage.ResolvingRepository,
                $"Resolving Hugging Face repository: {repoId}");

            var info = await HuggingfaceHub.HFDownloader.GetModelInfoAsync(repoId, filesMetadata: true);
            var files = ((IEnumerable<dynamic>)info.Siblings)
                .Where(ShouldDownloadModelFile)
                .ToList();
            var fileSizes = files.Select(TryGetRemoteFileSize).ToArray();
            long? modelTotalBytes = fileSizes.All(size => size.HasValue)
                ? fileSizes.Sum(size => size!.Value)
                : null;
            long modelBytesDownloaded = 0;

            ReportDownloadProgress(
                progress,
                modelVer,
                modelLocalDir,
                repoId,
                fileName: null,
                fileIndex: 0,
                fileCount: files.Count,
                fileBytesDownloaded: 0,
                fileTotalBytes: null,
                modelBytesDownloaded,
                modelTotalBytes,
                CosyVoiceDownloadStage.ListingFiles,
                $"Found {files.Count} files to check.");

            for (var i = 0; i < files.Count; i++)
            {
                var file = files[i];
                var fileName = GetModelFileName(file);
                var fileSize = fileSizes[i];
                var localPath = Path.Combine(modelLocalDir, fileName.Replace('/', Path.DirectorySeparatorChar));
                var parent = Path.GetDirectoryName(localPath);
                if (!string.IsNullOrEmpty(parent))
                    Directory.CreateDirectory(parent);

                if (IsExistingDownloadComplete(localPath, fileSize))
                {
                    var existingBytes = fileSize ?? new FileInfo(localPath).Length;
                    modelBytesDownloaded += existingBytes;
                    ReportDownloadProgress(
                        progress,
                        modelVer,
                        modelLocalDir,
                        repoId,
                        fileName,
                        i + 1,
                        files.Count,
                        existingBytes,
                        fileSize,
                        modelBytesDownloaded,
                        modelTotalBytes,
                        CosyVoiceDownloadStage.SkippedFile,
                        $"Already downloaded: {fileName}");
                    continue;
                }

                var downloadedForFile = await DownloadModelFileWithProgressAsync(
                    modelVer,
                    modelLocalDir,
                    repoId,
                    fileName,
                    localPath,
                    i + 1,
                    files.Count,
                    fileSize,
                    modelBytesDownloaded,
                    modelTotalBytes,
                    progress);
                modelBytesDownloaded += downloadedForFile;
            }
        }

        private static async Task<long> DownloadModelFileWithProgressAsync(
            string modelVer,
            string modelLocalDir,
            string repoId,
            string fileName,
            string localPath,
            int fileIndex,
            int fileCount,
            long? expectedFileBytes,
            long completedModelBytes,
            long? modelTotalBytes,
            IProgress<CosyVoiceDownloadProgress>? progress)
        {
            var tmpPath = localPath + ".download";
            if (File.Exists(tmpPath))
                File.Delete(tmpPath);

            var url = BuildHuggingFaceResolveUrl(repoId, fileName);
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.UserAgent.ParseAdd("CosyVoiceNet/1.0");
            using var response = await ModelDownloadHttpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var totalFileBytes = expectedFileBytes ?? response.Content.Headers.ContentLength;
            await using var responseStream = await response.Content.ReadAsStreamAsync();
            await using var fileStream = new FileStream(tmpPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, useAsync: true);
            var buffer = new byte[1024 * 1024];
            long fileBytesDownloaded = 0;
            long lastReportBytes = -1;
            var lastReportTime = DateTimeOffset.MinValue;

            ReportDownloadProgress(
                progress,
                modelVer,
                modelLocalDir,
                repoId,
                fileName,
                fileIndex,
                fileCount,
                0,
                totalFileBytes,
                completedModelBytes,
                modelTotalBytes,
                CosyVoiceDownloadStage.DownloadingFile,
                $"Downloading: {fileName}");

            while (true)
            {
                var read = await responseStream.ReadAsync(buffer);
                if (read == 0)
                    break;

                await fileStream.WriteAsync(buffer.AsMemory(0, read));
                fileBytesDownloaded += read;

                var now = DateTimeOffset.UtcNow;
                if (fileBytesDownloaded == totalFileBytes ||
                    fileBytesDownloaded - lastReportBytes >= 8L * 1024 * 1024 ||
                    now - lastReportTime >= TimeSpan.FromMilliseconds(500))
                {
                    lastReportBytes = fileBytesDownloaded;
                    lastReportTime = now;
                    ReportDownloadProgress(
                        progress,
                        modelVer,
                        modelLocalDir,
                        repoId,
                        fileName,
                        fileIndex,
                        fileCount,
                        fileBytesDownloaded,
                        totalFileBytes,
                        completedModelBytes + fileBytesDownloaded,
                        modelTotalBytes,
                        CosyVoiceDownloadStage.DownloadingFile,
                        $"Downloading: {fileName}");
                }
            }

            await fileStream.FlushAsync();
            fileStream.Close();

            if (totalFileBytes.HasValue && fileBytesDownloaded != totalFileBytes.Value)
                throw new IOException($"Downloaded {fileBytesDownloaded} bytes for {fileName}, expected {totalFileBytes.Value} bytes.");

            if (File.Exists(localPath))
                File.Delete(localPath);
            File.Move(tmpPath, localPath);

            ReportDownloadProgress(
                progress,
                modelVer,
                modelLocalDir,
                repoId,
                fileName,
                fileIndex,
                fileCount,
                fileBytesDownloaded,
                totalFileBytes,
                completedModelBytes + fileBytesDownloaded,
                modelTotalBytes,
                CosyVoiceDownloadStage.CompletedFile,
                $"Downloaded: {fileName}");

            return fileBytesDownloaded;
        }

        private static bool IsExistingDownloadComplete(string localPath, long? expectedSize)
        {
            if (!File.Exists(localPath))
                return false;

            return !expectedSize.HasValue || new FileInfo(localPath).Length == expectedSize.Value;
        }

        private static string BuildHuggingFaceResolveUrl(string repoId, string fileName)
        {
            var escapedPath = string.Join("/", fileName.Replace('\\', '/').Split('/').Select(Uri.EscapeDataString));
            return $"https://huggingface.co/{repoId}/resolve/main/{escapedPath}";
        }

        private static void ReportDownloadProgress(
            IProgress<CosyVoiceDownloadProgress>? progress,
            string model,
            string localDirectory,
            string? repositoryId,
            string? fileName,
            int fileIndex,
            int fileCount,
            long fileBytesDownloaded,
            long? fileTotalBytes,
            long modelBytesDownloaded,
            long? modelTotalBytes,
            CosyVoiceDownloadStage stage,
            string message)
        {
            progress?.Report(new CosyVoiceDownloadProgress(
                model,
                Path.GetFullPath(localDirectory),
                repositoryId,
                fileName,
                fileIndex,
                fileCount,
                fileBytesDownloaded,
                fileTotalBytes,
                modelBytesDownloaded,
                modelTotalBytes,
                stage,
                message));
        }

        public static string GetLocalModelDir(string modelVer)
        {
            return ResolveLocalModelDir(modelVer);
        }

        public static bool IsModelAvailable(string modelVer)
        {
            return IsUsableModelDir(ResolveLocalModelDir(modelVer));
        }

        private static string ResolveLocalModelDir(string modelVer)
        {
            if (Path.IsPathRooted(modelVer))
                return modelVer;

            if (modelVer.Contains('/') || modelVer.Contains('\\'))
            {
                var normalized = modelVer.Replace('\\', '/').TrimEnd('/');
                if (RemoteToLocalName.TryGetValue(normalized, out var localName))
                    return ResolveLocalModelName(localName);

                return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, modelVer));
            }

            return ResolveLocalModelName(modelVer);
        }

        private static string ResolveLocalModelName(string localName)
        {
            return Path.Combine(AppContext.BaseDirectory, "models", localName);
        }

        private static readonly Dictionary<string, string> RemoteToLocalName = new(StringComparer.OrdinalIgnoreCase)
        {
            ["FunAudioLLM/Fun-CosyVoice3-0.5B-2512"] = "Fun-CosyVoice3-0.5B",
            ["FunAudioLLM/CosyVoice2-0.5B"] = "CosyVoice2-0.5B",
            ["FunAudioLLM/CosyVoice-300M"] = "CosyVoice-300M",
            ["FunAudioLLM/CosyVoice-300M-SFT"] = "CosyVoice-300M-SFT",
            ["FunAudioLLM/CosyVoice-300M-Instruct"] = "CosyVoice-300M-Instruct"
        };

        private static readonly Dictionary<string, string[]> LocalNameToRemoteCandidates = new(StringComparer.OrdinalIgnoreCase)
        {
            ["Fun-CosyVoice3-0.5B"] = new[] { "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" },
            ["Fun-CosyVoice3-0.5B-2512"] = new[] { "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" },
            ["CosyVoice2-0.5B"] = new[] { "FunAudioLLM/CosyVoice2-0.5B" },
            ["CosyVoice-300M"] = new[] { "FunAudioLLM/CosyVoice-300M" },
            ["CosyVoice-300M-SFT"] = new[] { "FunAudioLLM/CosyVoice-300M-SFT" },
            ["CosyVoice-300M-Instruct"] = new[] { "FunAudioLLM/CosyVoice-300M-Instruct" }
        };

        private static IEnumerable<string> ResolveRemoteRepoCandidates(string modelVer)
        {
            var normalized = modelVer.Replace('\\', '/').TrimEnd('/');
            var isRooted = Path.IsPathRooted(modelVer);
            if (!isRooted && normalized.Contains('/'))
                yield return normalized;

            var localName = Path.GetFileName(normalized);
            if (LocalNameToRemoteCandidates.TryGetValue(localName, out var aliases))
            {
                foreach (var alias in aliases)
                    yield return alias;
            }

            if (!isRooted && !normalized.Contains('/'))
                yield return normalized;
        }

        private static bool ShouldDownloadModelFile(dynamic file)
        {
            string name = GetModelFileName(file);
            if (string.IsNullOrWhiteSpace(name))
                return false;

            name = name.Replace('\\', '/');
            return true;
        }

        private static string GetModelFileName(object file)
        {
            if (TryGetObjectValue(file, "Filename", out var filename) ||
                TryGetObjectValue(file, "filename", out filename) ||
                TryGetObjectValue(file, "Rfilename", out filename) ||
                TryGetObjectValue(file, "rfilename", out filename))
            {
                return Convert.ToString(filename) ?? string.Empty;
            }

            return string.Empty;
        }

        private static long? TryGetRemoteFileSize(object file)
        {
            if (TryGetObjectValue(file, "Size", out var size) ||
                TryGetObjectValue(file, "size", out size))
            {
                return TryConvertToInt64(size);
            }

            if (TryGetObjectValue(file, "Lfs", out var lfs) ||
                TryGetObjectValue(file, "lfs", out lfs))
            {
                if (TryGetObjectValue(lfs, "Size", out size) ||
                    TryGetObjectValue(lfs, "size", out size))
                {
                    return TryConvertToInt64(size);
                }
            }

            return null;
        }

        private static bool TryGetObjectValue(object? source, string name, out object? value)
        {
            value = null;
            if (source is null)
                return false;

            if (source is IDictionary<string, object> dictionary && dictionary.TryGetValue(name, out value))
                return true;

            var property = source.GetType().GetProperty(name);
            if (property is null)
                return false;

            value = property.GetValue(source);
            return true;
        }

        private static long? TryConvertToInt64(object? value)
        {
            if (value is null)
                return null;

            try
            {
                return Convert.ToInt64(value);
            }
            catch
            {
                return null;
            }
        }

        private static bool IsUsableModelDir(string modelLocalDir)
        {
            if (!Directory.Exists(modelLocalDir))
                return false;

            var config = GetExistingConfig(modelLocalDir);
            if (config == null)
                return false;

            return GetRequiredFilesForConfig(config)
                .All(path => File.Exists(Path.Combine(modelLocalDir, path)));
        }

        private static string DescribeMissingRequiredFiles(string modelLocalDir)
        {
            var config = GetExistingConfig(modelLocalDir);
            if (config == null)
                return "no cosyvoice*.yaml config found";

            var missing = GetRequiredFilesForConfig(config)
                .Where(path => !File.Exists(Path.Combine(modelLocalDir, path)))
                .ToArray();
            return missing.Length == 0 ? "none" : string.Join(", ", missing);
        }

        private static string? GetExistingConfig(string modelLocalDir)
        {
            foreach (var name in new[] { "cosyvoice3.yaml", "cosyvoice2.yaml", "cosyvoice.yaml" })
            {
                var path = Path.Combine(modelLocalDir, name);
                if (File.Exists(path))
                    return name;
            }
            return null;
        }

        private static IEnumerable<string> GetRequiredFilesForConfig(string configName)
        {
            yield return configName;
            yield return "campplus.onnx";
            yield return "llm.pt";
            yield return "flow.pt";
            yield return "hift.pt";

            if (configName.Equals("cosyvoice3.yaml", StringComparison.OrdinalIgnoreCase))
            {
                yield return "speech_tokenizer_v3.onnx";
                yield return Path.Combine("CosyVoice-BlankEN", "config.json");
                yield return Path.Combine("CosyVoice-BlankEN", "tokenizer_config.json");
                yield return Path.Combine("CosyVoice-BlankEN", "vocab.json");
                yield return Path.Combine("CosyVoice-BlankEN", "merges.txt");
            }
            else if (configName.Equals("cosyvoice2.yaml", StringComparison.OrdinalIgnoreCase))
            {
                yield return "speech_tokenizer_v2.onnx";
                yield return Path.Combine("CosyVoice-BlankEN", "config.json");
                yield return Path.Combine("CosyVoice-BlankEN", "tokenizer_config.json");
                yield return Path.Combine("CosyVoice-BlankEN", "vocab.json");
                yield return Path.Combine("CosyVoice-BlankEN", "merges.txt");
            }
            else
            {
                yield return "speech_tokenizer_v1.onnx";
            }
        }

        public static Dictionary<string, object> LoadYamlConfigWithOverride(string yamlPath, Dictionary<string, object> overrides = null)
        {
            if (!File.Exists(yamlPath))
                throw new FileNotFoundException($"{yamlPath} not found!");
            return CosyVoiceNet.Utils.YamlLoader.Load(yamlPath, overrides);
        }

        protected void InitFrontend(Dictionary<string, object> configs, string modelDir, string speechTokenizerOnnx, CosyVoiceBackend onnxBackend)
        {
            _frontend = new CosyVoiceFrontEnd(
                configs["get_tokenizer"] as Func<object>,
                configs["feat_extractor"] as Func<torch.Tensor, torch.Tensor>,
                Path.Combine(modelDir, "campplus.onnx"),
                Path.Combine(modelDir, speechTokenizerOnnx),
                Path.Combine(modelDir, "spk2info.pt"),
                configs.TryGetValue("allowed_special", out var allowed) ? allowed.ToString() : "all",
                onnxBackend,
                _runtimeOptions.Logger
            );
            _frontend.Logger = _runtimeOptions.Logger;
            _frontend.TracePromptTrim = _runtimeOptions.TracePromptTrim;
        }

        protected virtual void InitModel(Dictionary<string, object> configs, bool fp16, string modelDir, CosyVoiceBackend backend)
        {
            // Python: CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
            var llm = configs["llm"] as torch.nn.Module;
            var flow = configs["flow"] as torch.nn.Module;
            var hift = configs["hift"] as torch.nn.Module;
            if (flow == null || hift == null)
                throw new InvalidCastException("flow and hift must be torch.nn.Module instances loaded from YAML.");
            //_model = new CosyVoiceModel(llm, flow, hift, fp16);
            _model = Activator.CreateInstance(ModelType, llm, flow, hift, fp16, backend) as CosyVoiceModel; // Use version-specific model class if overridden
            if (_model is not null)
            {
                var resolvedOptions = _runtimeOptions.Resolve(backend);
                _model.LegacyTransformerCacheBackend = resolvedOptions.LegacyTransformerCacheBackend;
                _model.QwenKvCacheBackend = resolvedOptions.QwenKvCacheBackend;
                _model.QwenAttentionBackend = resolvedOptions.QwenAttentionBackend;
                _model.QwenMlpBackend = resolvedOptions.QwenMlpBackend;
                _model.Logger = resolvedOptions.Logger;
                _model.Profiler = resolvedOptions.Profiler;
                _model.TraceLlmInputShapes = resolvedOptions.TraceLlmInputShapes;
                _model.TraceGeneratedTokens = resolvedOptions.TraceGeneratedTokens;
            }
            _model.Load(
                Path.Combine(modelDir, "llm.pt"),
                Path.Combine(modelDir, "flow.pt"),
                Path.Combine(modelDir, "hift.pt")
            );
        }

        public void SetBackend(CosyVoiceBackend backend)
        {
            _model.SetBackend(backend);
        }

        public void SetComponentBackends(CosyVoiceBackend llmBackend, CosyVoiceBackend flowBackend, CosyVoiceBackend hiftBackend)
        {
            _model.SetComponentBackends(llmBackend, flowBackend, hiftBackend);
        }

        protected void LoadJitModules(bool fp16, string modelDir)
        {
            _model.LoadJit(
                Path.Combine(modelDir, $"llm.text_encoder.{(fp16 ? "fp16" : "fp32")}.zip"),
                Path.Combine(modelDir, $"llm.llm.{(fp16 ? "fp16" : "fp32")}.zip"),
                Path.Combine(modelDir, $"flow.encoder.{(fp16 ? "fp16" : "fp32")}.zip")
            );
        }
        protected void LoadTrtModules(bool fp16, string modelDir, int trtConcurrent)
        {
            _model.LoadTrt(
                Path.Combine(modelDir, $"flow.decoder.estimator.{(fp16 ? "fp16" : "fp32")}.mygpu.plan"),
                Path.Combine(modelDir, "flow.decoder.estimator.fp32.onnx"),
                trtConcurrent,
                fp16
            );
        }

        public List<string> ListAvailableSpks()
        {
            return ListAvailableSpks(_modelDir);
        }

        public List<string> ListAvailableSpks(IEnumerable<string>? providedWavs)
        {
            return ListAvailableSpks(_modelDir, providedWavs)
                .ToList();
        }

        public static List<string> ListAvailableSpks(string modelDirOrName, IEnumerable<string>? providedWavs = null, bool ensureDownloaded = false)
        {
            return ListAvailableVoiceOptions(modelDirOrName, providedWavs, ensureDownloaded)
                .Select(option => option.Id)
                .ToList();
        }

        public List<CosyVoiceVoiceOption> ListAvailableVoiceOptions(IEnumerable<string>? providedWavs = null)
        {
            var options = new List<CosyVoiceVoiceOption>();
            AddRuntimeSpk2InfoVoiceOptions(options);
            AddStaticSavedVoiceOptions(options, _modelDir);

            if (SupportsSavedVoiceInference)
                AddProvidedWavVoiceOptions(options, providedWavs);

            return NormalizeVoiceOptions(options);
        }

        public static List<CosyVoiceVoiceOption> ListAvailableVoiceOptions(string modelDirOrName, IEnumerable<string>? providedWavs = null, bool ensureDownloaded = false)
        {
            var options = new List<CosyVoiceVoiceOption>();
            var modelLocalDir = ResolveLocalModelDir(modelDirOrName);
            if (ensureDownloaded)
                EnsureModelDir(modelDirOrName);

            AddStaticSpk2InfoVoiceOptions(options, modelLocalDir);
            AddStaticSavedVoiceOptions(options, modelLocalDir);

            if (SupportsSavedVoiceInferenceForModel(modelLocalDir))
                AddProvidedWavVoiceOptions(options, providedWavs);

            return NormalizeVoiceOptions(options);
        }

        private static List<CosyVoiceVoiceOption> NormalizeVoiceOptions(IEnumerable<CosyVoiceVoiceOption> options)
        {
            return options
                .GroupBy(option => option.Id, StringComparer.OrdinalIgnoreCase)
                .Select(group => group
                    .OrderByDescending(option => VoiceKindPriority(option.Kind))
                    .First())
                .OrderBy(option => option.Kind, StringComparer.OrdinalIgnoreCase)
                .ThenBy(option => option.DisplayName, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        private static int VoiceKindPriority(string kind)
        {
            return kind switch
            {
                "saved" => 4,
                "built_in_cloned" => 3,
                "provided_wav" => 2,
                "built_in" => 1,
                _ => 0
            };
        }

        private void AddRuntimeSpk2InfoVoiceOptions(List<CosyVoiceVoiceOption> options)
        {
            foreach (var entry in _frontend.Spk2Info)
            {
                if (string.IsNullOrWhiteSpace(entry.Key))
                    continue;

                if (entry.Value is IDictionary<string, object> typed && IsZeroShotVoicePackage(typed))
                {
                    options.Add(CosyVoiceVoiceOption.BuiltInCloned(entry.Key));
                    continue;
                }

                if (entry.Value is IDictionary dictionary)
                {
                    var keys = dictionary.Keys
                        .Cast<object?>()
                        .Select(key => key?.ToString())
                        .Where(key => !string.IsNullOrWhiteSpace(key))
                        .ToHashSet(StringComparer.Ordinal);

                    if (keys.Contains("flow_embedding")
                        && keys.Contains("prompt_speech_feat")
                        && keys.Contains("flow_prompt_speech_token"))
                    {
                        options.Add(CosyVoiceVoiceOption.BuiltInCloned(entry.Key));
                        continue;
                    }
                }

                options.Add(CosyVoiceVoiceOption.BuiltIn(entry.Key));
            }
        }

        private static void AddStaticSpk2InfoVoiceOptions(List<CosyVoiceVoiceOption> options, string modelLocalDir)
        {
            var spk2InfoPath = Path.Combine(modelLocalDir, "spk2info.pt");
            if (!File.Exists(spk2InfoPath))
                return;

            try
            {
                var loaded = PickleUnpickler.Unpickle(spk2InfoPath);
                if (loaded is IDictionary dictionary)
                {
                    foreach (DictionaryEntry entry in dictionary)
                    {
                        var name = entry.Key?.ToString();
                        if (!string.IsNullOrWhiteSpace(name))
                            options.Add(IsStaticZeroShotVoicePackage(entry.Value)
                                ? CosyVoiceVoiceOption.BuiltInCloned(name)
                                : CosyVoiceVoiceOption.BuiltIn(name));
                    }
                }
                else if (loaded is IDictionary<string, object> typed)
                {
                    foreach (var (key, value) in typed)
                    {
                        if (!string.IsNullOrWhiteSpace(key))
                            options.Add(IsStaticZeroShotVoicePackage(value)
                                ? CosyVoiceVoiceOption.BuiltInCloned(key)
                                : CosyVoiceVoiceOption.BuiltIn(key));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CosyVoice] Failed to statically read spk2info.pt from {spk2InfoPath}: {ex.Message}");
            }
        }

        private static bool IsStaticZeroShotVoicePackage(object? value)
        {
            if (value is IDictionary<string, object> typed)
                return IsZeroShotVoicePackage(typed);

            if (value is not IDictionary dictionary)
                return false;

            var keys = dictionary.Keys
                .Cast<object?>()
                .Select(key => key?.ToString())
                .Where(key => !string.IsNullOrWhiteSpace(key))
                .ToHashSet(StringComparer.Ordinal);

            return keys.Contains("flow_embedding")
                && keys.Contains("prompt_speech_feat")
                && keys.Contains("flow_prompt_speech_token");
        }

        private static void AddStaticSavedVoiceOptions(List<CosyVoiceVoiceOption> options, string modelLocalDir)
        {
            var savedVoicesDir = GetSavedVoicesDir(modelLocalDir);
            if (!Directory.Exists(savedVoicesDir))
                return;

            foreach (var path in Directory.EnumerateFiles(savedVoicesDir, "*.json"))
            {
                try
                {
                    using var stream = File.OpenRead(path);
                    using var doc = System.Text.Json.JsonDocument.Parse(stream);
                    if (doc.RootElement.TryGetProperty("Name", out var nameProperty))
                    {
                        var name = nameProperty.GetString();
                        if (!string.IsNullOrWhiteSpace(name))
                            options.Add(CosyVoiceVoiceOption.Saved(name));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[CosyVoice] Failed to statically read saved voice {path}: {ex.Message}");
                }
            }
        }

        private static void AddProvidedWavVoiceOptions(List<CosyVoiceVoiceOption> options, IEnumerable<string>? providedWavs)
        {
            if (providedWavs == null)
                return;

            var existing = new HashSet<string>(
                options.Select(option => option.Id),
                StringComparer.OrdinalIgnoreCase);

            foreach (var wav in providedWavs)
            {
                if (string.IsNullOrWhiteSpace(wav))
                    continue;

                var fullPath = Path.GetFullPath(wav);
                if (!File.Exists(fullPath))
                    continue;

                if (TryGetIntegratedPromptVoiceName(fullPath, out var integratedVoiceName) &&
                    existing.Contains(integratedVoiceName))
                {
                    continue;
                }

                var option = CosyVoiceVoiceOption.ProvidedWav(fullPath);
                if (existing.Add(option.Id))
                    options.Add(option);
            }
        }

        public List<string> ListSavedVoices()
        {
            return _frontend.Spk2Info
                .Where(kv => kv.Value is IDictionary<string, object> dict && IsZeroShotVoicePackage(dict))
                .Select(kv => kv.Key)
                .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        public string CloneAndSaveVoice(string voiceName, string promptText, string promptWav, bool overwrite = true, bool textFrontend = true)
        {
            if (!SupportsSavedVoiceInference)
                throw new NotSupportedException($"{GetType().Name} does not support WAV-based cloned voices.");
            if (string.IsNullOrWhiteSpace(voiceName))
                throw new ArgumentException("Voice name cannot be empty.", nameof(voiceName));
            if (string.IsNullOrWhiteSpace(promptWav))
                throw new ArgumentException("Prompt WAV path cannot be empty.", nameof(promptWav));
            if (!File.Exists(promptWav))
                throw new FileNotFoundException("Prompt WAV was not found.", promptWav);
            if (!overwrite && _frontend.Spk2Info.ContainsKey(voiceName))
                throw new InvalidOperationException($"Voice '{voiceName}' already exists.");

            var resolvedPromptText = ResolvePromptTranscript(promptText, promptWav);
            var normalizedPromptText = NormalizeZeroShotPromptText(resolvedPromptText, textFrontend);
            var modelInput = _frontend.FrontendZeroShot(string.Empty, normalizedPromptText, promptWav, _sampleRate, string.Empty);
            DisposeOwnedInputTensors(modelInput, new[] { "text", "text_len" });
            modelInput.Remove("text");
            modelInput.Remove("text_len");
            var voicePackage = CloneTensorPackage(modelInput);
            if (_frontend.Spk2Info.TryGetValue(voiceName, out var existingVoice) &&
                existingVoice is IDictionary<string, object> existingVoiceDict &&
                IsZeroShotVoicePackage(existingVoiceDict))
            {
                CosyVoiceFrontEnd.DisposeTensorDictionary(existingVoiceDict);
            }

            _frontend.Spk2Info[voiceName] = voicePackage;

            var metadata = CreateSavedVoiceMetadata(resolvedPromptText, normalizedPromptText, promptWav);
            _savedVoiceMetadata[voiceName] = metadata;
            var path = SaveVoicePackage(voiceName, voicePackage, metadata);
            return path;
        }

        public string CloneProvidedWavIfNeeded(string promptWav, string promptText, string? voiceName = null, bool textFrontend = true)
        {
            if (!SupportsSavedVoiceInference)
                throw new NotSupportedException($"{GetType().Name} does not support WAV-based cloned voices.");
            if (string.IsNullOrWhiteSpace(promptWav))
                throw new ArgumentException("Prompt WAV path cannot be empty.", nameof(promptWav));

            var fullPath = Path.GetFullPath(promptWav);
            if (!File.Exists(fullPath))
                throw new FileNotFoundException("Prompt WAV was not found.", fullPath);

            var resolvedPromptText = ResolvePromptTranscript(promptText, fullPath);
            var resolvedVoiceName = string.IsNullOrWhiteSpace(voiceName)
                ? BuildProvidedWavVoiceId(fullPath, resolvedPromptText)
                : voiceName.Trim();

            var normalizedPromptText = NormalizeZeroShotPromptText(resolvedPromptText, textFrontend);
            var requestedMetadata = CreateSavedVoiceMetadata(resolvedPromptText, normalizedPromptText, fullPath);

            if (_frontend.Spk2Info.TryGetValue(resolvedVoiceName, out var voiceObj) &&
                voiceObj is IDictionary<string, object> voice &&
                IsZeroShotVoicePackage(voice) &&
                SavedVoiceMetadataMatches(resolvedVoiceName, requestedMetadata))
            {
                return resolvedVoiceName;
            }

            CloneAndSaveVoice(resolvedVoiceName, resolvedPromptText, fullPath, overwrite: true, textFrontend: textFrontend);
            return resolvedVoiceName;
        }

        public bool AddZeroShotSpk(string promptText, string promptWav, string zeroShotSpkId)
        {
            CloneAndSaveVoice(zeroShotSpkId, promptText, promptWav);
            return true;
        }

        public void SaveSpkInfo()
        {
            foreach (var kv in _frontend.Spk2Info)
            {
                if (kv.Value is IDictionary<string, object> voice && IsZeroShotVoicePackage(voice))
                    SaveVoicePackage(kv.Key, voice);
            }
        }

        private static Dictionary<string, object> ConvertToTensorDict(Dictionary<string, object> dict)
        {
            var result = new Dictionary<string, object>();
            foreach (var kv in dict)
            {
                if (kv.Value is TorchSharp.torch.Tensor tensor)
                    result[kv.Key] = tensor;
                else if (kv.Value is float[] arr)
                    result[kv.Key] = torch.tensor(arr);
                else if (kv.Value is double[] darr)
                    result[kv.Key] = torch.tensor(darr);
                else if (kv.Value is int[] iarr)
                    result[kv.Key] = torch.tensor(iarr);
                else if (kv.Value is Dictionary<string, object> nestedDict)
                    result[kv.Key] = ConvertToTensorDict(nestedDict);
                else
                    result[kv.Key] = kv.Value; // fallback: keep as is
            }
            return result;
        }

        public class InferenceResult
        {
            public byte[] TtsSpeech { get; set; }
        }

        protected string PrepareTtsTextForInference(string text)
        {
            return NormalizeTextForTts
                ? TtsTextPreprocessor.PrepareForTts(text)
                : text;
        }

        private void TraceInferenceText(string mode, string text)
        {
            if (!_runtimeOptions.TraceTextInput || _logger is null)
                return;

            try
            {
                var (tokens, len) = _frontend.ExtractTextToken(text);
                _logger.Log(
                    CosyVoiceLogLevel.Trace,
                    "Text input tokenized.",
                    tags: new Dictionary<string, string>
                    {
                        ["mode"] = mode,
                        ["text_length"] = text.Length.ToString(),
                        ["token_length"] = len.ToString(),
                        ["text"] = text,
                        ["tokens_head"] = string.Join(", ", tokens.Take(Math.Min(64, tokens.Length)))
                    });
            }
            catch (Exception ex)
            {
                CosyVoiceLog.Write(_logger, CosyVoiceLogLevel.Warning, "Text input trace failed.", ex);
            }
        }

        public IEnumerable<InferenceResult> InferenceSavedVoice(
            string ttsText,
            string voiceName,
            string instructText = "",
            bool crossLingual = false,
            bool stream = false,
            double speed = 1.0,
            bool textFrontend = true)
        {
            if (string.IsNullOrWhiteSpace(voiceName))
                throw new ArgumentException("Voice name cannot be empty.", nameof(voiceName));
            if (!_frontend.Spk2Info.TryGetValue(voiceName, out var voiceObj) ||
                voiceObj is not IDictionary<string, object> voice ||
                !IsZeroShotVoicePackage(voice))
            {
                throw new KeyNotFoundException($"Saved cloned voice '{voiceName}' was not found.");
            }

            if (TryRefreshSavedVoiceIfOutdated(voiceName))
            {
                voiceObj = _frontend.Spk2Info[voiceName];
                voice = (IDictionary<string, object>)voiceObj;
            }

            var normalizedInstructText = string.IsNullOrWhiteSpace(instructText)
                ? string.Empty
                : NormalizeInstruct2PromptText(instructText, textFrontend);

            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                TraceInferenceText("saved_voice", text);
                Dictionary<string, object> modelInput;
                if (!string.IsNullOrWhiteSpace(normalizedInstructText))
                {
                    modelInput = _frontend.FrontendInstruct2(text, normalizedInstructText, string.Empty, _sampleRate, voiceName);
                }
                else if (crossLingual)
                {
                    modelInput = _frontend.FrontendCrossLingual(text, string.Empty, _sampleRate, voiceName);
                    if (this is CosyVoice3)
                        AddCosyVoice3PromptBoundary(modelInput);
                }
                else
                {
                    modelInput = _frontend.FrontendZeroShot(text, string.Empty, string.Empty, _sampleRate, voiceName);
                }

                var ownedInputKeys = new List<string> { "text", "text_len" };
                if (!string.IsNullOrWhiteSpace(normalizedInstructText) ||
                    (crossLingual && this is CosyVoice3))
                {
                    ownedInputKeys.Add("prompt_text");
                    ownedInputKeys.Add("prompt_text_len");
                }

                foreach (var result in RunModelTts(modelInput, stream, speed, ownedInputKeys.ToArray()))
                    yield return result;
            }
        }

        public IEnumerable<InferenceResult> InferenceWithVoice(
            string ttsText,
            string voiceNameOrPromptWav,
            string promptText = "",
            string instructText = "",
            bool crossLingual = false,
            bool stream = false,
            double speed = 1.0,
            bool textFrontend = true)
        {
            if (string.IsNullOrWhiteSpace(voiceNameOrPromptWav))
                throw new ArgumentException("Voice name or prompt WAV path cannot be empty.", nameof(voiceNameOrPromptWav));

            if (_frontend.Spk2Info.TryGetValue(voiceNameOrPromptWav, out var voiceObj))
            {
                if (voiceObj is IDictionary<string, object> voice && IsZeroShotVoicePackage(voice))
                {
                    foreach (var result in InferenceSavedVoice(ttsText, voiceNameOrPromptWav, instructText, crossLingual, stream, speed, textFrontend))
                        yield return result;
                    yield break;
                }

                foreach (var result in InferenceSft(ttsText, voiceNameOrPromptWav, stream, speed, textFrontend))
                    yield return result;
                yield break;
            }

            var promptWav = voiceNameOrPromptWav;
            if (voiceNameOrPromptWav.StartsWith(ProvidedWavVoicePrefix, StringComparison.OrdinalIgnoreCase))
                promptWav = voiceNameOrPromptWav.Substring(ProvidedWavVoicePrefix.Length);

            if (File.Exists(promptWav))
            {
                var resolvedPromptText = ResolvePromptTranscript(promptText, promptWav);
                if (!crossLingual &&
                    string.IsNullOrWhiteSpace(instructText) &&
                    string.IsNullOrWhiteSpace(resolvedPromptText))
                {
                    throw new ArgumentException(
                        "Prompt transcript is required the first time a provided WAV is used for zero-shot voice cloning.",
                        nameof(promptText));
                }

                var clonedVoiceName = CloneProvidedWavIfNeeded(promptWav, resolvedPromptText, textFrontend: textFrontend);
                foreach (var result in InferenceSavedVoice(ttsText, clonedVoiceName, instructText, crossLingual, stream, speed, textFrontend))
                    yield return result;
                yield break;
            }

            throw new KeyNotFoundException($"Voice '{voiceNameOrPromptWav}' was not found and is not an existing WAV file.");
        }

        // Update Inference methods to return the correct type
        public IEnumerable<InferenceResult> InferenceSft(string ttsText, string spkId, bool stream = false, double speed = 1.0, bool textFrontend = true)
        {
            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                var modelInput = _frontend.FrontendSft(text, spkId);
                foreach (var result in RunModelTts(modelInput, stream, speed, "text", "text_len"))
                    yield return result;
            }
        }

        public IEnumerable<InferenceResult> InferenceZeroShot(string ttsText, string promptText, string promptWav, string zeroShotSpkId = "", bool stream = false, double speed = 1.0, bool textFrontend = true)
        {
            var resolvedPromptText = ResolvePromptTranscript(promptText, promptWav);
            var normalizedPromptText = NormalizeZeroShotPromptText(resolvedPromptText, textFrontend);
            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                TraceInferenceText("zero_shot", text);
                var modelInput = _frontend.FrontendZeroShot(
                    text,
                    normalizedPromptText,
                    promptWav,
                    _sampleRate,
                    zeroShotSpkId
                );
                foreach (var result in RunModelTts(modelInput, stream, speed, "text", "text_len"))
                    yield return result;
            }
        }

        public IEnumerable<InferenceResult> InferenceCrossLingual(string ttsText, string promptWav, string zeroShotSpkId = "", bool stream = false, double speed = 1.0, bool textFrontend = true)
        {
            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                var modelInput = _frontend.FrontendCrossLingual(text, promptWav, _sampleRate, zeroShotSpkId);
                if (this is CosyVoice3)
                    AddCosyVoice3PromptBoundary(modelInput);
                var ownedInputKeys = this is CosyVoice3
                    ? new[] { "text", "text_len", "prompt_text", "prompt_text_len" }
                    : new[] { "text", "text_len" };
                foreach (var result in RunModelTts(modelInput, stream, speed, ownedInputKeys))
                    yield return result;
            }
        }

        public IEnumerable<InferenceResult> InferenceInstruct(string ttsText, string spkId, string instructText, bool stream = false, double speed = 1.0, bool textFrontend = true)
        {
            var normalizedInstructText = NormalizeInstructPromptText(instructText, textFrontend);
            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                var modelInput = _frontend.FrontendInstruct(
                    text,
                    spkId,
                    normalizedInstructText
                );
                foreach (var result in RunModelTts(modelInput, stream, speed, "text", "text_len", "prompt_text", "prompt_text_len"))
                    yield return result;
            }
        }

        public IEnumerable<InferenceResult> InferenceVc(string sourceWav, string promptWav, bool stream = false, double speed = 1.0)
        {
            var modelInput = _frontend.FrontendVc(sourceWav, promptWav, _sampleRate);
            foreach (var result in RunModelTts(
                modelInput,
                stream,
                speed,
                "source_speech_token",
                "source_speech_token_len",
                "flow_prompt_speech_token",
                "flow_prompt_speech_token_len",
                "prompt_speech_feat",
                "prompt_speech_feat_len",
                "flow_embedding"))
            {
                yield return result;
            }
        }

        // Helper method to convert tensor to byte array
        protected static byte[] TensorToBytes(Tensor tensor)
        {
            // tensor is float32, shape [1, num_samples] or [num_samples]
            var floatData = tensor.data<float>().ToArray();
            var bytes = new byte[floatData.Length * sizeof(float)];
            Buffer.BlockCopy(floatData, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        protected static byte[] SpeechOutputToBytesAndDispose(IDictionary<string, Tensor> modelOutput)
        {
            try
            {
                return TensorToBytes(modelOutput["tts_speech"]);
            }
            finally
            {
                foreach (var tensor in modelOutput.Values)
                    tensor?.Dispose();
            }
        }

        protected IEnumerable<InferenceResult> RunModelTts(Dictionary<string, object> modelInput, bool stream, double speed, params string[] ownedInputTensorKeys)
        {
            try
            {
                foreach (var modelOutput in _model.Tts(modelInput, stream, speed))
                {
                    var speechBytes = SpeechOutputToBytesAndDispose(modelOutput);
                    yield return new InferenceResult { TtsSpeech = speechBytes };
                }
            }
            finally
            {
                DisposeOwnedInputTensors(modelInput, ownedInputTensorKeys);
            }
        }

        protected static void DisposeOwnedInputTensors(IDictionary<string, object> modelInput, IEnumerable<string> keys)
        {
            var disposed = new List<Tensor>();
            foreach (var key in keys)
            {
                if (!modelInput.TryGetValue(key, out var value) || value is not Tensor tensor)
                    continue;

                if (disposed.Any(existing => ReferenceEquals(existing, tensor)))
                    continue;

                tensor.Dispose();
                disposed.Add(tensor);
            }
        }

        private static Dictionary<string, object> CloneTensorPackage(IDictionary<string, object> source)
        {
            var result = new Dictionary<string, object>(StringComparer.Ordinal);
            var cloned = new List<(Tensor Original, Tensor Clone)>();

            foreach (var pair in source)
            {
                if (pair.Value is Tensor tensor)
                {
                    var existing = cloned.FirstOrDefault(item => ReferenceEquals(item.Original, tensor));
                    if (existing.Clone is not null)
                    {
                        result[pair.Key] = existing.Clone;
                        continue;
                    }

                    var clone = tensor.clone();
                    cloned.Add((tensor, clone));
                    result[pair.Key] = clone;
                    continue;
                }

                result[pair.Key] = pair.Value;
            }

            return result;
        }

        public virtual void Dispose()
        {
            _frontend?.Dispose();
            _model?.Dispose();
        }

        private const string ProvidedWavVoicePrefix = "wav:";
        private string SavedVoicesDir => GetSavedVoicesDir(_modelDir);
        private bool SupportsSavedVoiceInference => SupportsSavedVoiceInferenceForModel(_modelDir);

        private static string GetSavedVoicesDir(string modelLocalDir)
        {
            return Path.Combine(modelLocalDir, "cosyvoice-net-voices");
        }

        private static bool SupportsSavedVoiceInferenceForModel(string modelDirOrName)
        {
            var modelLocalDir = ResolveLocalModelDir(modelDirOrName);
            var localName = Path.GetFileName(modelLocalDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (localName.Equals("CosyVoice-300M", StringComparison.OrdinalIgnoreCase) ||
                localName.Equals("CosyVoice2-0.5B", StringComparison.OrdinalIgnoreCase) ||
                localName.Equals("Fun-CosyVoice3-0.5B", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            var configName = GetExistingConfig(modelLocalDir);
            return configName != null &&
                   (configName.Equals("cosyvoice2.yaml", StringComparison.OrdinalIgnoreCase) ||
                    configName.Equals("cosyvoice3.yaml", StringComparison.OrdinalIgnoreCase));
        }

        protected static string EnsureCosyVoicePromptBoundary(string instructText)
        {
            const string EndOfPrompt = "<|endofprompt|>";
            if (string.IsNullOrWhiteSpace(instructText) || instructText.Contains(EndOfPrompt, StringComparison.Ordinal))
                return instructText;

            return $"You are a helpful assistant. {instructText.Trim()}{EndOfPrompt}";
        }

        private static string ResolvePromptTranscript(string promptText, string promptWav)
        {
            if (!string.IsNullOrWhiteSpace(promptText))
                return promptText;

            if (TryReadPromptTranscriptSidecar(promptWav, out var sidecarTranscript))
                return sidecarTranscript;

            return TryResolveIntegratedPromptTranscript(promptWav, out var integratedTranscript)
                ? integratedTranscript
                : promptText;
        }

        private static bool TryResolveIntegratedPromptTranscript(string promptWav, out string transcript)
        {
            transcript = string.Empty;
            var fileName = Path.GetFileName(promptWav);
            if (fileName.Equals("zero_shot_prompt.wav", StringComparison.OrdinalIgnoreCase))
            {
                transcript = "希望你以后能够做的比我还好呦。";
                return true;
            }

            if (fileName.Equals("cross_lingual_prompt.wav", StringComparison.OrdinalIgnoreCase))
            {
                transcript = "在那之后完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致。这就是我们有时不买下全部的原因。";
                return true;
            }

            return false;
        }

        private static bool TryReadPromptTranscriptSidecar(string promptWav, out string transcript)
        {
            transcript = string.Empty;
            if (string.IsNullOrWhiteSpace(promptWav))
                return false;

            string fullPath;
            try
            {
                fullPath = Path.GetFullPath(promptWav);
            }
            catch
            {
                return false;
            }

            var directory = Path.GetDirectoryName(fullPath);
            var stem = Path.GetFileNameWithoutExtension(fullPath);
            if (string.IsNullOrWhiteSpace(directory) || string.IsNullOrWhiteSpace(stem))
                return false;

            var candidates = new[]
            {
                fullPath + ".txt",
                fullPath + ".transcript.txt",
                Path.Combine(directory, stem + ".txt"),
                Path.Combine(directory, stem + ".transcript.txt"),
                Path.Combine(directory, stem + ".prompt.txt"),
                Path.Combine(directory, stem + ".prompt-transcript.txt")
            };

            foreach (var path in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!File.Exists(path))
                    continue;

                var text = File.ReadAllText(path).Trim();
                if (string.IsNullOrWhiteSpace(text))
                    continue;

                transcript = text;
                return true;
            }

            return false;
        }

        private string NormalizeZeroShotPromptText(string promptText, bool textFrontend)
        {
            var normalized = string.Join(" ", _frontend.TextNormalize(promptText ?? string.Empty, false, textFrontend));
            if (this is CosyVoice3)
                return EnsureCosyVoice3ZeroShotPromptBoundary(normalized);

            return normalized;
        }

        private static string TrimTerminalAsciiSentencePunctuation(string text)
        {
            var trimmed = (text ?? string.Empty).TrimEnd();
            while (trimmed.Length > 0 && IsTerminalAsciiSentencePunctuation(trimmed[^1]))
                trimmed = trimmed[..^1].TrimEnd();
            return trimmed;
        }

        private static bool IsTerminalAsciiSentencePunctuation(char ch)
            => ch is '.' or '!' or '?' or ';' or ':';

        protected string NormalizeInstruct2PromptText(string instructText, bool textFrontend)
        {
            var normalized = string.Join(" ", _frontend.TextNormalize(instructText ?? string.Empty, false, textFrontend));
            return EnsureCosyVoicePromptBoundary(normalized);
        }

        protected string NormalizeInstructPromptText(string instructText, bool textFrontend)
        {
            var normalized = string.Join(" ", _frontend.TextNormalize(instructText ?? string.Empty, false, textFrontend));
            return EnsureCosyVoicePromptBoundary(normalized);
        }

        private static string EnsureInstructEndOfPromptBoundary(string instructText)
        {
            const string EndOfPrompt = "<|endofprompt|>";
            if (string.IsNullOrWhiteSpace(instructText) || instructText.Contains(EndOfPrompt, StringComparison.Ordinal))
                return instructText;

            return $"{instructText.Trim()}{EndOfPrompt}";
        }

        private static string EnsureCosyVoice3ZeroShotPromptBoundary(string promptText)
        {
            const string EndOfPrompt = "<|endofprompt|>";
            if (string.IsNullOrWhiteSpace(promptText) || promptText.Contains(EndOfPrompt, StringComparison.Ordinal))
                return promptText;

            return $"{EndOfPrompt}{promptText.Trim()}";
        }

        private void AddCosyVoice3PromptBoundary(Dictionary<string, object> modelInput)
        {
            const string PromptBoundary = "You are a helpful assistant.<|endofprompt|>";
            var (tokens, len) = _frontend.ExtractTextToken(PromptBoundary);
            modelInput["prompt_text"] = tensor(tokens, dtype: int64).unsqueeze(0);
            modelInput["prompt_text_len"] = tensor(new[] { len }, dtype: int32);
        }


        private void LoadSavedVoices()
        {
            if (!Directory.Exists(SavedVoicesDir))
                return;

            foreach (var path in Directory.EnumerateFiles(SavedVoicesDir, "*.json", SearchOption.TopDirectoryOnly))
            {
                try
                {
                    var json = File.ReadAllText(path);
                    var saved = System.Text.Json.JsonSerializer.Deserialize<SavedVoiceFile>(json);
                    if (saved == null || string.IsNullOrWhiteSpace(saved.Name) || saved.Tensors == null)
                        continue;

                    _frontend.Spk2Info[saved.Name] = DeserializeVoicePackage(saved.Tensors);
                    if (saved.Metadata != null)
                        _savedVoiceMetadata[saved.Name] = saved.Metadata;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[CosyVoice] Failed to load saved voice '{path}': {ex.Message}");
                }
            }
        }

        private string SaveVoicePackage(string voiceName, IDictionary<string, object> voice, SavedVoiceMetadata? metadata = null)
        {
            Directory.CreateDirectory(SavedVoicesDir);
            var saved = new SavedVoiceFile
            {
                Name = voiceName,
                CreatedUtc = DateTimeOffset.UtcNow,
                Metadata = metadata,
                Tensors = SerializeVoicePackage(voice)
            };

            var path = Path.Combine(SavedVoicesDir, SafeVoiceFileName(voiceName));
            var json = System.Text.Json.JsonSerializer.Serialize(saved, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
            return path;
        }

        private static bool IsZeroShotVoicePackage(IDictionary<string, object> voice)
        {
            return voice.ContainsKey("flow_embedding") &&
                   voice.ContainsKey("prompt_speech_feat") &&
                   voice.ContainsKey("flow_prompt_speech_token");
        }

        private static Dictionary<string, TensorPayload> SerializeVoicePackage(IDictionary<string, object> voice)
        {
            var result = new Dictionary<string, TensorPayload>(StringComparer.Ordinal);
            foreach (var kv in voice)
            {
                if (kv.Value is Tensor tensor)
                    result[kv.Key] = SerializeTensor(tensor);
            }
            return result;
        }

        private static Dictionary<string, object> DeserializeVoicePackage(Dictionary<string, TensorPayload> tensors)
        {
            var result = new Dictionary<string, object>(StringComparer.Ordinal);
            foreach (var kv in tensors)
                result[kv.Key] = DeserializeTensor(kv.Value);
            return result;
        }

        private static TensorPayload SerializeTensor(Tensor tensor)
        {
            var cpu = tensor.cpu().contiguous();
            var payload = new TensorPayload
            {
                Shape = cpu.shape.ToArray()
            };

            if (cpu.dtype == ScalarType.Float32)
            {
                payload.DType = "float32";
                payload.Data = cpu.numel() == 0 ? ToBase64(Array.Empty<float>()) : ToBase64(cpu.data<float>().ToArray());
            }
            else if (cpu.dtype == ScalarType.Int64)
            {
                payload.DType = "int64";
                payload.Data = cpu.numel() == 0 ? ToBase64(Array.Empty<long>()) : ToBase64(cpu.data<long>().ToArray());
            }
            else if (cpu.dtype == ScalarType.Int32)
            {
                payload.DType = "int32";
                payload.Data = cpu.numel() == 0 ? ToBase64(Array.Empty<int>()) : ToBase64(cpu.data<int>().ToArray());
            }
            else
            {
                throw new NotSupportedException($"Saved voices do not support tensor dtype {cpu.dtype}.");
            }

            return payload;
        }

        private static Tensor DeserializeTensor(TensorPayload payload)
        {
            if (payload == null)
                throw new ArgumentNullException(nameof(payload));
            if (payload.Shape == null)
                throw new InvalidDataException("Saved tensor is missing shape.");
            if (payload.Data == null)
                throw new InvalidDataException("Saved tensor is missing data.");

            var bytes = Convert.FromBase64String(payload.Data);
            return payload.DType switch
            {
                "float32" => torch.tensor(FromBytes<float>(bytes), dtype: ScalarType.Float32).view(payload.Shape),
                "int64" => torch.tensor(FromBytes<long>(bytes), dtype: ScalarType.Int64).view(payload.Shape),
                "int32" => torch.tensor(FromBytes<int>(bytes), dtype: ScalarType.Int32).view(payload.Shape),
                _ => throw new NotSupportedException($"Saved tensor dtype '{payload.DType}' is not supported.")
            };
        }

        private static string ToBase64<T>(T[] values) where T : struct
        {
            var bytes = new byte[Buffer.ByteLength(values)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return Convert.ToBase64String(bytes);
        }

        private static T[] FromBytes<T>(byte[] bytes) where T : struct
        {
            var size = System.Runtime.InteropServices.Marshal.SizeOf<T>();
            if (bytes.Length % size != 0)
                throw new InvalidDataException($"Saved tensor byte length {bytes.Length} is not divisible by element size {size}.");

            var values = new T[bytes.Length / size];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private static string SafeVoiceFileName(string voiceName)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var stem = new string(voiceName.Select(ch => invalid.Contains(ch) || char.IsWhiteSpace(ch) ? '_' : ch).ToArray()).Trim('_');
            if (string.IsNullOrWhiteSpace(stem))
                stem = "voice";
            if (stem.Length > 64)
                stem = stem.Substring(0, 64);

            var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(voiceName))).Substring(0, 12).ToLowerInvariant();
            return $"{stem}.{hash}.json";
        }

        private static string BuildProvidedWavVoiceId(string promptWav, string promptText = "")
        {
            var fullPath = Path.GetFullPath(promptWav);
            if (TryGetIntegratedPromptVoiceName(fullPath, out var integratedVoiceName))
                return integratedVoiceName;

            var stem = Path.GetFileNameWithoutExtension(fullPath);
            if (string.IsNullOrWhiteSpace(stem))
                stem = "wav";

            var safeStem = new string(stem.Select(ch => Path.GetInvalidFileNameChars().Contains(ch) || char.IsWhiteSpace(ch) ? '_' : ch).ToArray()).Trim('_');
            if (safeStem.Length > 48)
                safeStem = safeStem.Substring(0, 48);

            var hashInput = $"{fullPath}\n{promptText ?? string.Empty}";
            var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(hashInput))).Substring(0, 12).ToLowerInvariant();
            return $"wav_{safeStem}_{hash}";
        }

        private static bool TryGetIntegratedPromptVoiceName(string promptWav, out string voiceName)
        {
            voiceName = string.Empty;
            var fileName = Path.GetFileName(promptWav);
            if (fileName.Equals("zero_shot_prompt.wav", StringComparison.OrdinalIgnoreCase))
            {
                voiceName = "zero_shot_prompt";
                return true;
            }

            if (fileName.Equals("cross_lingual_prompt.wav", StringComparison.OrdinalIgnoreCase))
            {
                voiceName = "cross_lingual_prompt";
                return true;
            }

            return false;
        }

        private static SavedVoiceMetadata CreateSavedVoiceMetadata(string promptText, string normalizedPromptText, string promptWav)
        {
            var fullPath = Path.GetFullPath(promptWav);
            return new SavedVoiceMetadata
            {
                PromptText = promptText ?? string.Empty,
                NormalizedPromptText = normalizedPromptText ?? string.Empty,
                PromptWavPath = fullPath,
                PromptProcessingVersion = SavedVoicePromptProcessingVersion,
                PromptWavSha256 = File.Exists(fullPath)
                    ? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(fullPath))).ToLowerInvariant()
                    : string.Empty
            };
        }

        private bool SavedVoiceMetadataMatches(string voiceName, SavedVoiceMetadata requested)
        {
            if (!_savedVoiceMetadata.TryGetValue(voiceName, out var existing))
                return false;

            return string.Equals(existing.NormalizedPromptText, requested.NormalizedPromptText, StringComparison.Ordinal) &&
                   existing.PromptProcessingVersion == requested.PromptProcessingVersion &&
                   string.Equals(existing.PromptWavSha256, requested.PromptWavSha256, StringComparison.OrdinalIgnoreCase);
        }

        private bool TryRefreshSavedVoiceIfOutdated(string voiceName)
        {
            if (!_savedVoiceMetadata.TryGetValue(voiceName, out var metadata) ||
                metadata.PromptProcessingVersion >= SavedVoicePromptProcessingVersion)
            {
                return false;
            }

            if (string.IsNullOrWhiteSpace(metadata.PromptWavPath) || !File.Exists(metadata.PromptWavPath))
                return false;

            var promptText = string.IsNullOrWhiteSpace(metadata.PromptText)
                ? metadata.NormalizedPromptText
                : metadata.PromptText;
            if (string.IsNullOrWhiteSpace(promptText))
                return false;

            CosyVoiceLog.Write(_logger, CosyVoiceLogLevel.Information, $"Refreshing saved voice '{voiceName}' for prompt-processing version {SavedVoicePromptProcessingVersion}.");
            CloneAndSaveVoice(voiceName, promptText, metadata.PromptWavPath, overwrite: true, textFrontend: true);
            return true;
        }

        public sealed class CosyVoiceVoiceOption
        {
            private CosyVoiceVoiceOption(string id, string displayName, string kind, string? wavPath, bool requiresClone)
            {
                Id = id;
                DisplayName = displayName;
                Kind = kind;
                WavPath = wavPath;
                RequiresClone = requiresClone;
            }

            public string Id { get; }
            public string DisplayName { get; }
            public string Kind { get; }
            public string? WavPath { get; }
            public bool RequiresClone { get; }

            public static CosyVoiceVoiceOption BuiltIn(string name) =>
                new(name, name, "built_in", null, false);

            public static CosyVoiceVoiceOption BuiltInCloned(string name) =>
                new(name, name, "built_in_cloned", null, false);

            public static CosyVoiceVoiceOption Saved(string name) =>
                new(name, name, "saved", null, false);

            public static CosyVoiceVoiceOption ProvidedWav(string wavPath) =>
                new($"{ProvidedWavVoicePrefix}{wavPath}", Path.GetFileNameWithoutExtension(wavPath), "provided_wav", wavPath, true);
        }

        private sealed class SavedVoiceFile
        {
            public string Name { get; set; }
            public DateTimeOffset CreatedUtc { get; set; }
            public SavedVoiceMetadata Metadata { get; set; }
            public Dictionary<string, TensorPayload> Tensors { get; set; }
        }

        private sealed class SavedVoiceMetadata
        {
            public string PromptText { get; set; }
            public string NormalizedPromptText { get; set; }
            public string PromptWavPath { get; set; }
            public string PromptWavSha256 { get; set; }
            public int PromptProcessingVersion { get; set; }
        }

        private sealed class TensorPayload
        {
            public string DType { get; set; }
            public long[] Shape { get; set; }
            public string Data { get; set; }
        }

    }

    public class CosyVoice2 : CosyVoice
    {
        protected override string SpeechTokenizerOnnx => "speech_tokenizer_v2.onnx";
        protected override Type ModelType => typeof(CosyVoice2Model);

        public CosyVoice2(string modelDir, bool loadJit = false, bool loadTrt = false, bool loadVllm = false, bool fp16 = false, int trtConcurrent = 1, CosyVoiceBackend? backend = null, CosyVoiceRuntimeOptions runtimeOptions = null)
            : base(modelDir, false, false, fp16, trtConcurrent, backend, runtimeOptions) // base will not load model, we override below
        {
            _sampleRate = Configs.TryGetValue("sample_rate", out var sr) ? Convert.ToInt32(sr) : 16000;
            if (loadVllm)
                _model.LoadVllm(Path.Combine(_modelDir, "vllm"));
            if (loadJit) LoadJitModules(fp16, _modelDir);
            if (loadTrt) LoadTrtModules(fp16, _modelDir, trtConcurrent);
        }

        protected override Dictionary<string, object> LoadYamlConfig()
        {
            var overrides = new Dictionary<string, object> { { "qwen_pretrain_path", Path.Combine(_modelDir, "CosyVoice-BlankEN") } };
            return LoadYamlConfigWithOverride(Path.Combine(_modelDir, "cosyvoice2.yaml"), overrides);
        }

        // Update InferenceInstruct2 to return strongly-typed IEnumerable<CosyVoice.InferenceResult>
        public IEnumerable<CosyVoice.InferenceResult> InferenceInstruct2(string ttsText, string instructText, string promptWav, string zeroShotSpkId = "", bool stream = false, double speed = 1.0, bool textFrontend = true)
        {
            var normalizedInstructText = NormalizeInstruct2PromptText(instructText ?? string.Empty, textFrontend);
            foreach (var text in _frontend.TextNormalize(PrepareTtsTextForInference(ttsText), true, textFrontend))
            {
                var modelInput = _frontend.FrontendInstruct2(text, normalizedInstructText, promptWav, _sampleRate, zeroShotSpkId);
                var ownedInputKeys = string.IsNullOrEmpty(zeroShotSpkId)
                    ? new[] { "text", "text_len" }
                    : new[] { "text", "text_len", "prompt_text", "prompt_text_len" };
                foreach (var result in RunModelTts(modelInput, stream, speed, ownedInputKeys))
                    yield return result;
            }
        }
    }

    public class CosyVoice3 : CosyVoice2
    {
        protected override string SpeechTokenizerOnnx => "speech_tokenizer_v3.onnx";
        protected override Type ModelType => typeof(CosyVoice3Model);

        public CosyVoice3(string modelDir, bool loadTrt = false, bool loadVllm = false, bool fp16 = false, int trtConcurrent = 1, CosyVoiceBackend? backend = null, CosyVoiceRuntimeOptions runtimeOptions = null)
            : base(modelDir, false, loadTrt, loadVllm, fp16, trtConcurrent, backend, runtimeOptions) 
        {
            _sampleRate = Configs.TryGetValue("sample_rate", out var sr) ? Convert.ToInt32(sr) : 16000;
            if (loadVllm)
                _model.LoadVllm(Path.Combine(_modelDir, "vllm"));
            if (loadTrt)
            {
                if (fp16)
                {
                    // Log warning as in Python
                    Console.WriteLine("DiT tensorRT fp16 engine have some performance issue, use at caution!");
                }
                LoadTrtModules(fp16, _modelDir, trtConcurrent);
            }
        }

        protected override Dictionary<string, object> LoadYamlConfig()
        {
            var overrides = new Dictionary<string, object> { { "qwen_pretrain_path", Path.Combine(_modelDir, "CosyVoice-BlankEN") } };
            return LoadYamlConfigWithOverride(Path.Combine(_modelDir, "cosyvoice3.yaml"), overrides);
        }
    }
}
