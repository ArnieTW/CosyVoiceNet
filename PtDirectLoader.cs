using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using Tensor = TorchSharp.torch.Tensor;
using CosyVoiceNet.Utilities;

namespace CosyVoiceNet
{
    // Loads tensors directly from a .pt file using the C# unpickler and converts
    // TorchTensorPlaceholder objects into TorchSharp tensors (float32 assumed).
    public static class PtDirectLoader
    {
        // Returns a map of parameter name -> Tensor for tensors successfully converted.
        public static Dictionary<string, Tensor> LoadTensorsFromPt(string ptPath)
        {
            var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            try
            {
                var sd = PickleUnpickler.UnpickleStateDict(ptPath, convertTensorsToFloat: false);
                if (sd == null || sd.Count == 0) return result;

                foreach (var kv in sd)
                {
                    var name = kv.Key ?? string.Empty;
                    var val = kv.Value;
                    if (val is PickleUnpickler.TorchTensorPlaceholder t)
                    {
                        try
                        {
                            var flat = t.ToFloat32Array();
                            if (flat == null) continue;
                            var shape = t.Shape?.Select(x => (long)x).ToArray() ?? new long[] { flat.Length };
                            var tensor = torch.tensor(flat, dtype: ScalarType.Float32).view(shape);
                            result[name] = tensor;
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"Failed to convert tensor {name}: {ex.Message}");
                        }
                    }
                    else if (val is PickleUnpickler.TorchParameterPlaceholder p && p.Tensor != null)
                    {
                        try
                        {
                            var tp = p.Tensor;
                            var flat = tp.ToFloat32Array();
                            if (flat == null) continue;
                            var shape = tp.Shape?.Select(x => (long)x).ToArray() ?? new long[] { flat.Length };
                            var tensor = torch.tensor(flat, dtype: ScalarType.Float32).view(shape);
                            result[name] = tensor;
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"Failed to convert parameter {name}: {ex.Message}");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("PtDirectLoader failed: " + ex.Message);
            }

            return result;
        }
    }
}

// This file is marked as NotAligned because no equivalent Python file exists.
