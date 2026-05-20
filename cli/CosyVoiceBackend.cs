using System;
using TorchSharp;
using static TorchSharp.torch;

namespace CosyVoiceNet.cli
{
    public enum CosyVoiceBackend
    {
        Auto,
        Cpu,
        Cuda
    }

    public readonly struct CosyVoiceBackendSelection
    {
        public CosyVoiceBackendSelection(CosyVoiceBackend requestedBackend, CosyVoiceBackend activeBackend, string device)
        {
            RequestedBackend = requestedBackend;
            ActiveBackend = activeBackend;
            Device = device;
        }

        public CosyVoiceBackend RequestedBackend { get; }
        public CosyVoiceBackend ActiveBackend { get; }
        public string Device { get; }
    }

    public static class CosyVoiceBackendResolver
    {
        public static CosyVoiceBackendSelection Resolve(CosyVoiceBackend backend)
        {
            EnsureCpuInitialized();

            switch (backend)
            {
                case CosyVoiceBackend.Cpu:
                    return UseCpu(backend);

                case CosyVoiceBackend.Cuda:
                    if (TryCuda())
                        return new CosyVoiceBackendSelection(backend, CosyVoiceBackend.Cuda, "cuda");

                    Console.WriteLine("[CosyVoice] CUDA backend was requested, but TorchSharp CUDA is not available. Falling back to CPU.");
                    return UseCpu(backend);

                case CosyVoiceBackend.Auto:
                    if (TryCuda())
                        return new CosyVoiceBackendSelection(backend, CosyVoiceBackend.Cuda, "cuda");

                    return UseCpu(backend);

                default:
                    throw new ArgumentOutOfRangeException(nameof(backend), backend, "Unsupported CosyVoice backend.");
            }
        }

        private static bool TryCuda()
        {
            try
            {
                return torch.TryInitializeDeviceType(DeviceType.CUDA) && torch.cuda.is_available();
            }
            catch
            {
                return false;
            }
        }

        private static void EnsureCpuInitialized()
        {
            try
            {
                torch.InitializeDeviceType(DeviceType.CPU);
            }
            catch
            {
                // The later CPU path will surface the real error if CPU initialization truly failed.
            }
        }

        private static CosyVoiceBackendSelection UseCpu(CosyVoiceBackend requestedBackend)
        {
            torch.InitializeDeviceType(DeviceType.CPU);
            return new CosyVoiceBackendSelection(requestedBackend, CosyVoiceBackend.Cpu, "cpu");
        }
    }
}
