using System;

namespace CosyVoiceNet.cli
{
    public enum CosyVoiceDownloadStage
    {
        CheckingLocal,
        ResolvingRepository,
        ListingFiles,
        SkippedFile,
        DownloadingFile,
        CompletedFile,
        CompletedModel,
        Failed
    }

    public sealed record CosyVoiceDownloadProgress(
        string Model,
        string LocalDirectory,
        string? RepositoryId,
        string? FileName,
        int FileIndex,
        int FileCount,
        long FileBytesDownloaded,
        long? FileTotalBytes,
        long ModelBytesDownloaded,
        long? ModelTotalBytes,
        CosyVoiceDownloadStage Stage,
        string Message)
    {
        public double? FilePercent => FileTotalBytes.HasValue && FileTotalBytes.Value > 0
            ? Math.Clamp(FileBytesDownloaded * 100.0 / FileTotalBytes.Value, 0.0, 100.0)
            : null;

        public double? ModelPercent => ModelTotalBytes.HasValue && ModelTotalBytes.Value > 0
            ? Math.Clamp(ModelBytesDownloaded * 100.0 / ModelTotalBytes.Value, 0.0, 100.0)
            : null;
    }
}
