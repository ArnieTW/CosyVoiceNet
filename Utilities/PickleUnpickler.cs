using System;
using System.Collections.Generic;
using System.Collections;
using System.IO;
using System.Text;
using Razorvine.Pickle;
using System.Reflection;
using System.Linq;

namespace Utilities
{
    // Minimal unpickler for a focused subset of pickle opcodes used by torch.save(state_dict)
    // This does not aim to fully implement pickle but provides enough semantics to
    // reconstruct nested dict/list/tuple structures, memoization (basic), and REDUCE
    // placeholders so the state_dict layout can be inspected.
    public static class PickleUnpickler
    {
        public static object? Unpickle(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            try
            {
                // detect Zip container (torch >=1.6 sometimes writes .pt as zip archive)
                var hdr = new byte[4];
                if (fs.Read(hdr, 0, 4) == 4)
                {
                    fs.Seek(0, SeekOrigin.Begin);
                    if (hdr[0] == 0x50 && hdr[1] == 0x4B) // 'PK' zip signature
                    {
                        try
                        {
                            using var za = new System.IO.Compression.ZipArchive(fs, System.IO.Compression.ZipArchiveMode.Read, leaveOpen: true);
                            // collect all entry bytes into a map for callers that need them
                            var zipMap = new Dictionary<string, byte[]>(StringComparer.Ordinal);
                            foreach (var ent in za.Entries)
                            {
                                try
                                {
                                    if (string.IsNullOrEmpty(ent.FullName) || ent.FullName.EndsWith("/")) continue;
                                    using var esr = ent.Open();
                                    using var ms = new MemoryStream();
                                    esr.CopyTo(ms);
                                    zipMap[ent.FullName] = ms.ToArray();
                                }
                                catch { }
                            }

                            // prefer common entry name used by torch: data.pkl (or flow/data.pkl)
                            var entry = za.GetEntry("data.pkl") ?? za.GetEntry("flow/data.pkl") ?? za.Entries.FirstOrDefault(e => e.FullName.EndsWith("data.pkl", StringComparison.OrdinalIgnoreCase) || e.FullName.EndsWith(".pkl", StringComparison.OrdinalIgnoreCase) || e.FullName.Equals("data", StringComparison.OrdinalIgnoreCase));
                            if (entry != null)
                            {
                                using var es = entry.Open();
                                // read entry into memory
                                using var ms = new MemoryStream();
                                es.CopyTo(ms);
                                ms.Position = 0;

                                // use Razorvine.Pickle Unpickler with persistent_load resolver via ZipUnpickler
                                ms.Position = 0;
                                var zipUnp = new ZipUnpickler(zipMap);
                                RegisterConstructors(zipUnp);
                                ms.Position = 0;
                                var result = zipUnp.load(ms);
                                return result;
                            }
                            // fallback: try first entry
                            if (za.Entries.Count > 0)
                            {
                                using var es2 = za.Entries[0].Open();
                                using var ms2 = new MemoryStream();
                                es2.CopyTo(ms2);
                                ms2.Position = 0;
                                var zipUnp2 = new ZipUnpickler(zipMap);
                                RegisterConstructors(zipUnp2);
                                ms2.Position = 0;
                                var result2 = zipUnp2.load(ms2);
                                return result2;
                            }
                        }
                        catch (Exception zipEx)
                        {
                            if (Environment.GetEnvironmentVariable("COSYVOICE_PICKLE_DEBUG") == "1")
                                Console.Error.WriteLine($"[PickleUnpickler] Zip load failed for {path}: {zipEx}");
                        }
                        finally
                        {
                            // ensure stream position reset for fallback
                            try { fs.Seek(0, SeekOrigin.Begin); } catch { }
                        }
                    }
                }
            }
            catch { try { fs.Seek(0, SeekOrigin.Begin); } catch { } }

            // Non-zip path: use Razorvine Unpickler and register constructors
            var unp = new Razorvine.Pickle.Unpickler();
            RegisterConstructors(unp);
            return unp.load(fs);
        }

        // Register custom constructors for known torch rebuild functions so the Razorvine unpickler
        // can construct placeholder objects instead of throwing when encountering unknown classes.
        private static void RegisterConstructors(Razorvine.Pickle.Unpickler unp)
        {
            // Razorvine.Pickle.Unpickler exposes a public static registerConstructor method.
            // Call it directly to register constructors globally for the Unpickler.
            try { Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new TorchRebuildTensorConstructor()); } catch { }
            try { Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_parameter", new TorchRebuildParameterConstructor()); } catch { }
            try { Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new TorchRebuildTensorConstructor()); } catch { }
            try { Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "ClassDict", new GenericPlaceholderConstructor()); } catch { }
            try { Razorvine.Pickle.Unpickler.registerConstructor("collections", "OrderedDict", new OrderedDictConstructor()); } catch { }
        }

        // simple placeholder constructors that return inspectable objects instead of real tensors
        private class TorchRebuildTensorConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                return new TorchTensorPlaceholder(args);
            }
        }

        private class TorchRebuildParameterConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                return new TorchParameterPlaceholder(args);
            }
        }

        private class GenericPlaceholderConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                return new GenericPlaceholder(args);
            }
        }

        private class OrderedDictConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                var result = new PickleOrderedDictionary();
                if (args == null || args.Length == 0 || args[0] == null)
                    return result;

                foreach (var pair in EnumeratePairs(args[0]))
                {
                    if (pair.Length >= 2)
                        result[pair[0] ?? string.Empty] = pair[1];
                }

                return result;
            }

            private static IEnumerable<object?[]> EnumeratePairs(object value)
            {
                if (value is IEnumerable enumerable)
                {
                    foreach (var item in enumerable)
                    {
                        if (item is object?[] array)
                        {
                            yield return array;
                        }
                        else if (item is IEnumerable nested && item is not string)
                        {
                            var pair = new List<object?>();
                            foreach (var part in nested)
                                pair.Add(part);
                            yield return pair.ToArray();
                        }
                    }
                }
            }
        }

        private sealed class PickleOrderedDictionary : Hashtable
        {
            public void __setstate__(Hashtable state)
            {
                if (state == null)
                    return;

                foreach (DictionaryEntry entry in state)
                    this[entry.Key] = entry.Value;
            }
        }

        // Typed placeholder classes returned by the Razorvine constructors. These are simple
        // immutable containers that make it easy for downstream code to pattern-match on
        // tensor/parameter placeholders and inspect constructor arguments.
        public sealed class TorchTensorPlaceholder
        {
            public object[] Args { get; }
            // Raw storage bytes when available (from persistentLoad via ZipUnpickler)
            public byte[]? StorageBytes { get; }
            // Optional storage key if storage was referenced by id/string
            public string? StorageKey { get; }
            public string? StorageDType { get; }
            // Offset into storage (elements)
            public long? StorageOffset { get; }
            // Tensor shape and stride
            public int[]? Shape { get; }
            public int[]? Stride { get; }
            // requires_grad flag if present
            public bool? RequiresGrad { get; }

            public TorchTensorPlaceholder(object[] args)
            {
                Args = args ?? Array.Empty<object>();

                // common signature: (storage, storage_offset, size, stride, requires_grad, backward_hooks)
                if (Args.Length > 0)
                {
                    var storage = Args[0];
                    if (storage is byte[] bb) StorageBytes = bb;
                    else if (storage is TorchStoragePlaceholder tsp)
                    {
                        StorageBytes = tsp.Bytes;
                        StorageKey = tsp.Key;
                        StorageDType = tsp.DType;
                    }
                    else if (storage is string s) StorageKey = s;
                }

                if (Args.Length > 1 && Args[1] != null)
                {
                    try { StorageOffset = Convert.ToInt64(Args[1]); } catch { StorageOffset = null; }
                }

                if (Args.Length > 2 && Args[2] != null)
                {
                    Shape = ToIntArray(Args[2]);
                }

                if (Args.Length > 3 && Args[3] != null)
                {
                    Stride = ToIntArray(Args[3]);
                }

                if (Args.Length > 4 && Args[4] != null)
                {
                    try { RequiresGrad = Convert.ToBoolean(Args[4]); } catch { RequiresGrad = null; }
                }
            }

            private static int[]? ToIntArray(object? o)
            {
                if (o == null) return null;
                switch (o)
                {
                    case int[] ia: return ia;
                    case long[] la: return Array.ConvertAll(la, item => (int)item);
                    case object[] oa:
                        try { return Array.ConvertAll(oa, item => Convert.ToInt32(item)); } catch { return null; }
                    case IEnumerable<object> eo:
                        try
                        {
                            var list = new List<int>();
                            foreach (var v in eo)
                            {
                                if (v == null) { list.Add(0); continue; }
                                list.Add(Convert.ToInt32(v));
                            }
                            return list.ToArray();
                        }
                        catch { return null; }
                    default:
                        try { return new[] { Convert.ToInt32(o) }; } catch { return null; }
                }
            }

            public override string ToString()
            {
                var shape = Shape == null ? "?" : string.Join("x", Shape);
                return $"TorchTensorPlaceholder(shape={shape}, offset={StorageOffset}, storage={(StorageBytes != null ? StorageBytes.Length + " bytes" : StorageKey ?? "?")})";
            }

            // Attempt to interpret StorageBytes (or referenced storage) as float32 and return a flattened
            // Convert storage bytes to float32 array. If storage dtype is not known, attempt
            // to infer from storage size (supports float16/float32/float64/uint8/int8).
            public float[] ToFloat32Array()
            {
                return ToFloat32ArrayAuto();
            }

            private float[] ToFloat32ArrayAuto()
            {
                if (StorageBytes == null) throw new InvalidOperationException("No storage bytes available to convert to float array.");
                if (Shape == null || Shape.Length == 0) throw new InvalidOperationException("Shape information missing for tensor conversion.");

                long offsetElements = StorageOffset ?? 0;
                long elementCount = 1;
                foreach (var s in Shape) elementCount *= Math.Max(1, s);

                var dtype = StorageDType ?? string.Empty;

                if (dtype.Contains("HalfStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements * 2;
                    EnsureAvailable(offsetBytes, elementCount * 2);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++)
                    {
                        ushort h = BitConverter.ToUInt16(StorageBytes, (int)(offsetBytes + i * 2));
                        outArr[i] = HalfToFloat(h);
                    }
                    return outArr;
                }

                if (dtype.Contains("DoubleStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements * 8;
                    EnsureAvailable(offsetBytes, elementCount * 8);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++)
                    {
                        double d = BitConverter.ToDouble(StorageBytes, (int)(offsetBytes + i * 8));
                        outArr[i] = (float)d;
                    }
                    return outArr;
                }

                if (dtype.Contains("LongStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements * 8;
                    EnsureAvailable(offsetBytes, elementCount * 8);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++)
                        outArr[i] = BitConverter.ToInt64(StorageBytes, (int)(offsetBytes + i * 8));
                    return outArr;
                }

                if (dtype.Contains("IntStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements * 4;
                    EnsureAvailable(offsetBytes, elementCount * 4);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++)
                        outArr[i] = BitConverter.ToInt32(StorageBytes, (int)(offsetBytes + i * 4));
                    return outArr;
                }

                if (dtype.Contains("ShortStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements * 2;
                    EnsureAvailable(offsetBytes, elementCount * 2);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++)
                        outArr[i] = BitConverter.ToInt16(StorageBytes, (int)(offsetBytes + i * 2));
                    return outArr;
                }

                if (dtype.Contains("ByteStorage", StringComparison.OrdinalIgnoreCase) ||
                    dtype.Contains("CharStorage", StringComparison.OrdinalIgnoreCase) ||
                    dtype.Contains("BoolStorage", StringComparison.OrdinalIgnoreCase))
                {
                    long offsetBytes = offsetElements;
                    EnsureAvailable(offsetBytes, elementCount);
                    var outArr = new float[elementCount];
                    for (long i = 0; i < elementCount; i++) outArr[i] = StorageBytes[offsetBytes + i];
                    return outArr;
                }

                // Default to float32 when dtype is not exposed by the pickle persistent id.
                {
                    long offsetBytes = offsetElements * 4;
                    EnsureAvailable(offsetBytes, elementCount * 4);
                    var outArr = new float[elementCount];
                    Buffer.BlockCopy(StorageBytes, (int)offsetBytes, outArr, 0, (int)(elementCount * 4));
                    // handle endianness
                    if (!BitConverter.IsLittleEndian)
                    {
                        for (int i = 0; i < outArr.Length; i++)
                        {
                            var b = BitConverter.GetBytes(outArr[i]);
                            Array.Reverse(b);
                            outArr[i] = BitConverter.ToSingle(b, 0);
                        }
                    }
                    return outArr;
                }

                void EnsureAvailable(long offsetBytes, long byteCount)
                {
                    if (offsetBytes < 0 || byteCount < 0 || StorageBytes.LongLength - offsetBytes < byteCount)
                        throw new InvalidOperationException($"Storage does not contain enough bytes: offset={offsetBytes}, requested={byteCount}, total={StorageBytes.LongLength}");
                }
            }

            // Half-precision (IEEE 754) -> float32
            private static float HalfToFloat(ushort h)
            {
                uint sign = (uint)(h >> 15) & 0x00000001;
                uint exp = (uint)(h >> 10) & 0x0000001f;
                uint mant = (uint)h & 0x3ff;

                if (exp == 0)
                {
                    if (mant == 0)
                    {
                        return BitConverter.Int32BitsToSingle((int)(sign << 31));
                    }
                    while ((mant & 0x400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    exp++;
                    mant &= 0x3ff;
                }
                else if (exp == 31)
                {
                    if (mant == 0)
                    {
                        return BitConverter.Int32BitsToSingle((int)((sign << 31) | 0x7f800000));
                    }
                    return BitConverter.Int32BitsToSingle((int)((sign << 31) | 0x7f800000 | (mant << 13)));
                }

                exp = exp + (127 - 15);
                uint mantissa = mant << 13;
                uint bits = (sign << 31) | (exp << 23) | mantissa;
                return BitConverter.Int32BitsToSingle((int)bits);
            }
        }

        public sealed class TorchParameterPlaceholder
        {
            public object[] Args { get; }
            // If parameter wraps a tensor, this will be populated
            public TorchTensorPlaceholder? Tensor { get; }

            public TorchParameterPlaceholder(object[] args)
            {
                Args = args ?? Array.Empty<object>();
                if (Args.Length > 0 && Args[0] is TorchTensorPlaceholder t)
                {
                    Tensor = t;
                }
                else if (Args.Length > 0 && Args[0] is object[] oa)
                {
                    // try to parse nested tensor args
                    try { Tensor = new TorchTensorPlaceholder(oa); } catch { Tensor = null; }
                }
            }

            public override string ToString()
            {
                return Tensor != null ? $"TorchParameterPlaceholder(tensor={Tensor})" : $"TorchParameterPlaceholder(args={Args?.Length})";
            }
        }

        public sealed class GenericPlaceholder
        {
            public object[] Args { get; }
            public GenericPlaceholder(object[] args) => Args = args ?? Array.Empty<object>();
            public override string ToString() => $"GenericPlaceholder(args={Args?.Length})";
        }

        public sealed class TorchStoragePlaceholder
        {
            public string Key { get; }
            public string DType { get; }
            public byte[] Bytes { get; }

            public TorchStoragePlaceholder(string key, string dtype, byte[] bytes)
            {
                Key = key;
                DType = dtype;
                Bytes = bytes;
            }
        }

        // Convert loaded object into a state-dict-like mapping. Attempts to convert tensor
        // placeholders into float arrays when possible (assumes float32 storage by default).
        public static Dictionary<string, object?> UnpickleStateDict(string path, bool convertTensorsToFloat = true)
        {
            var root = Unpickle(path);
            var sd = FindStateDict(root);
            var result = new Dictionary<string, object?>(StringComparer.Ordinal);
            if (sd == null)
            {
                return result;
            }

            foreach (var kv in sd)
            {
                var key = kv.Key?.ToString() ?? string.Empty;
                var val = ConvertValue(kv.Value, convertTensorsToFloat);
                result[key] = val;
            }

            return result;
        }

        private static IDictionary<object, object?>? FindStateDict(object? root)
        {
            if (root is IDictionary<object, object?> d)
            {
                // common containers: {'state_dict': {...}} or direct state-dict
                if (d.TryGetValue("state_dict", out var sdObj) && sdObj is IDictionary<object, object?> sd1)
                    return sd1;
                if (d.TryGetValue("state_dict", out sdObj) && sdObj is IDictionary sd2)
                    return ToObjectDictionary(sd2);
                // sometimes keys are bytes/strings; try find any dictionary-like value that maps strings to tensors
                // Heuristic: if root itself has most string keys, treat it as state dict
                bool allStringKeys = d.Keys.OfType<object>().All(k => k != null && k.ToString() != null);
                if (allStringKeys) return d;
            }
            if (root is IDictionary nd)
            {
                if (nd.Contains("state_dict") && nd["state_dict"] is IDictionary sd)
                    return ToObjectDictionary(sd);

                var converted = ToObjectDictionary(nd);
                bool allStringKeys = converted.Keys.OfType<object>().All(k => k != null && k.ToString() != null);
                if (allStringKeys) return converted;
            }
            return null;
        }

        private static Dictionary<object, object?> ToObjectDictionary(IDictionary dictionary)
        {
            var result = new Dictionary<object, object?>();
            foreach (DictionaryEntry entry in dictionary)
            {
                result[entry.Key] = entry.Value;
            }
            return result;
        }

        private static object? ConvertValue(object? v, bool convertTensors)
        {
            if (v == null) return null;
            switch (v)
            {
                case TorchTensorPlaceholder t:
                    if (convertTensors) return TryConvertTensorToFloatArray(t) ?? (object)t;
                    return t;
                case TorchParameterPlaceholder p:
                    if (p.Tensor != null)
                    {
                        if (convertTensors) return TryConvertTensorToFloatArray(p.Tensor) ?? (object)p;
                        return p.Tensor;
                    }
                    return p;
                case GenericPlaceholder gp:
                    return gp;
                case IDictionary<object, object?> dict:
                    var map = new Dictionary<string, object?>(StringComparer.Ordinal);
                    foreach (var kv in dict)
                    {
                        var k = kv.Key?.ToString() ?? string.Empty;
                        map[k] = ConvertValue(kv.Value, convertTensors);
                    }
                    return map;
                case IDictionary dict:
                    var nonGenericMap = new Dictionary<string, object?>(StringComparer.Ordinal);
                    foreach (DictionaryEntry kv in dict)
                    {
                        var k = kv.Key?.ToString() ?? string.Empty;
                        nonGenericMap[k] = ConvertValue(kv.Value, convertTensors);
                    }
                    return nonGenericMap;
                case IEnumerable<object> list:
                    var la = new List<object?>();
                    foreach (var it in list) la.Add(ConvertValue(it, convertTensors));
                    return la.ToArray();
                default:
                    return v;
            }
        }

        private static float[]? TryConvertTensorToFloatArray(TorchTensorPlaceholder t)
        {
            try
            {
                return t.ToFloat32Array();
            }
            catch
            {
                return null;
            }
        }

    // Subclass of Razorvine.Pickle.Unpickler that resolves persistent ids from a ZIP entry map.
    // This is used when a .pt file is actually a zip archive containing a `data.pkl` plus
    // binary blobs under entries like `flow/data/<n>`; persistent ids in the pickle reference
    // those entries and must be mapped back to the raw bytes for correct inspection.
    internal class ZipUnpickler : Razorvine.Pickle.Unpickler
    {
        private readonly Dictionary<string, byte[]> zipMap;

        public ZipUnpickler(Dictionary<string, byte[]> zipMap)
        {
            this.zipMap = zipMap ?? new Dictionary<string, byte[]>(StringComparer.Ordinal);
        }

        // Match the base signature exactly. Return byte[] for matched persistent ids.
        protected override object persistentLoad(object pid)
        {
            if (pid == null) return null!;
            if (pid is byte[] bb) return bb;
            if (pid is object[] tuple && tuple.Length >= 3)
            {
                var dtype = tuple[1]?.ToString() ?? string.Empty;
                var storageKey = tuple[2]?.ToString() ?? string.Empty;
                if (TryGetStorageBytes(storageKey, out var storageBytes))
                    return new TorchStoragePlaceholder(storageKey, dtype, storageBytes);
                return storageKey;
            }
            var key = pid as string;
            if (string.IsNullOrEmpty(key)) return pid!;

            if (TryGetStorageBytes(key, out var bytes))
            {
                return bytes;
            }

            return pid!;
        }

        private bool TryGetStorageBytes(string key, out byte[] bytes)
        {
            key = key.Trim('\'', '"').TrimStart('/');
            if (zipMap.TryGetValue(key, out bytes!))
                return true;

            if (zipMap.TryGetValue("data/" + key, out bytes!))
                return true;

            var dataSuffix = "/data/" + key;
            foreach (var kv in zipMap)
            {
                if (kv.Key.EndsWith(dataSuffix, StringComparison.OrdinalIgnoreCase))
                {
                    bytes = kv.Value;
                    return true;
                }
            }

            bytes = Array.Empty<byte>();
            return false;
        }
    }

    }
}
