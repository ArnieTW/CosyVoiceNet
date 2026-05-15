using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using CosyVoiceNet;
using CosyVoiceNet.Tools;
using TorchSharp;
using YamlDotNet.RepresentationModel;
using CosyVoice.Tokenizer;

namespace CosyVoiceNet.Utils
{
    public static class YamlLoader
    {
        public static Dictionary<string, object> Load(string yamlPath, Dictionary<string, object> overrides = null)
        {
            var yaml = File.ReadAllText(yamlPath);

            var stream = new YamlStream();
            stream.Load(new StringReader(yaml));

            var root = (YamlMappingNode)stream.Documents[0].RootNode;

            var raw = new Dictionary<string, YamlNode>();
            foreach (var kv in root.Children)
            {
                var key = ((YamlScalarNode)kv.Key).Value;
                raw[key] = kv.Value;
            }

            if (overrides != null)
            {
                foreach (var kv in overrides)
                {
                    raw[kv.Key] = new YamlScalarNode(kv.Value?.ToString() ?? "null");
                }
            }

            var ctx = new EvalContext();
            foreach (var kv in raw)
            {
                ctx.Raw[kv.Key] = kv.Value;
            }
            var result = new Dictionary<string, object>();

            foreach (var kv in raw)
            {
                var value = EvaluateNode(kv.Value, ctx);
                result[kv.Key] = value;
                ctx.Parameters[kv.Key] = value;
            }

            return result;
        }

        private class EvalContext
        {
            public Dictionary<string, object> Parameters { get; } = new();
            public Dictionary<string, YamlNode> Raw { get; } = new();
            public HashSet<string> Resolving { get; } = new(StringComparer.Ordinal);
        }

        private static object EvaluateNode(YamlNode node, EvalContext ctx)
        {
            switch (node)
            {
                case YamlScalarNode scalar:
                    return EvalScalar(scalar, ctx);

                case YamlMappingNode map:
                    return EvalTaggedOrPlainMapping(map, ctx);

                case YamlSequenceNode seq:
                    return EvalTaggedOrPlainSequence(seq, ctx);

                default:
                    return null;
            }
        }

        private static object EvalTaggedOrPlainMapping(YamlMappingNode map, EvalContext ctx)
        {
            var tag = map.Tag.IsEmpty ? string.Empty : map.Tag.Value;

            if (tag.StartsWith("!new:"))
            {
                var className = tag.Substring("!new:".Length);
                var args = EvalMappingToDict(map, ctx);
                return InstantiateClass(className, args);
            }

            if (tag.StartsWith("!name:"))
            {
                var funcName = tag.Substring("!name:".Length);
                var args = EvalMappingToDict(map, ctx);
                return CreateNamedFunction(funcName, args);
            }

            if (tag.StartsWith("!apply:"))
            {
                var funcName = tag.Substring("!apply:".Length);
                var args = EvalMappingToDict(map, ctx);
                return InvokeFunction(funcName, args);
            }

            return EvalMappingToDict(map, ctx);
        }

        private static object EvalTaggedOrPlainSequence(YamlSequenceNode seq, EvalContext ctx)
        {
            var tag = seq.Tag.IsEmpty ? string.Empty : seq.Tag.Value;

            if (tag.StartsWith("!apply:"))
            {
                var funcName = tag.Substring("!apply:".Length);
                var args = seq.Children.Select(child => EvaluateNode(child, ctx)).ToList();
                return InvokeFunction(funcName, args);
            }

            var list = new List<object>();
            foreach (var item in seq.Children)
            {
                list.Add(EvaluateNode(item, ctx));
            }
            return list;
        }

        private static object EvalScalar(YamlScalarNode scalar, EvalContext ctx)
        {
            var tag = scalar.Tag.IsEmpty ? string.Empty : scalar.Tag.Value;
            var value = scalar.Value;

            if (tag == "!ref")
            {
                return EvalRef(value, ctx);
            }

            if (tag.StartsWith("!new:"))
            {
                var className = tag.Substring("!new:".Length);
                return InstantiateClass(className, new Dictionary<string, object>());
            }

            if (tag.StartsWith("!name:"))
            {
                var funcName = tag.Substring("!name:".Length);
                return CreateNamedFunction(funcName, new Dictionary<string, object>());
            }

            if (tag.StartsWith("!apply:"))
            {
                var funcName = tag.Substring("!apply:".Length);
                return InvokeFunction(funcName, new List<object> { ParsePlainScalar(value) });
            }

            return ParsePlainScalar(value);
        }

        private static object ParsePlainScalar(string value)
        {
            if (value == null || value == "null" || value == "~")
                return null;
            if (bool.TryParse(value, out var b))
                return b;
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i))
                return i;
            if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
                return d;
            return value;
        }

        private static Dictionary<string, object> EvalMappingToDict(YamlMappingNode map, EvalContext ctx)
        {
            var dict = new Dictionary<string, object>();
            foreach (var kv in map.Children)
            {
                var key = ((YamlScalarNode)kv.Key).Value;
                dict[key] = EvaluateNode(kv.Value, ctx);
            }
            return dict;
        }

        private static object CreateNamedFunction(string functionName, Dictionary<string, object> namedArgs)
        {
            if (functionName.EndsWith("get_qwen_tokenizer", StringComparison.OrdinalIgnoreCase))
            {
                var tokenPath = GetNamedArg<string>(namedArgs, "token_path") ?? string.Empty;
                var version = GetNamedArg<string>(namedArgs, "version") ?? "cosyvoice2";
                // Return a factory function that creates the appropriate tokenizer on demand
                // This matches the Python behavior: get_qwen_tokenizer returns a tokenizer object
                return new Func<object>(() => 
                {
                    if (string.IsNullOrWhiteSpace(tokenPath) || !Directory.Exists(tokenPath))
                        throw new InvalidOperationException($"Tokenizer path not found: {tokenPath}");
                    return CosyVoice.Tokenizer.CosyVoiceTokenizerFactory.GetQwenTokenizer(tokenPath, skipSpecialTokens: true, version);
                });
            }

            if (functionName.EndsWith("mel_spectrogram", StringComparison.OrdinalIgnoreCase))
            {
                var samplingRate = GetNamedArg<int>(namedArgs, "sampling_rate", 24000);
                var nMels = GetNamedArg<int>(namedArgs, "num_mels", 80);
                var nFft = GetNamedArg<int>(namedArgs, "n_fft", 512);
                var winSize = GetNamedArg<int>(namedArgs, "win_size", nFft);
                var hopSize = GetNamedArg<int>(namedArgs, "hop_size", 160);
                var fMin = GetNamedArg<int>(namedArgs, "fmin", 0);
                var fMax = GetNamedArg<int?>(namedArgs, "fmax", samplingRate / 2) ?? samplingRate / 2;
                var center = GetNamedArg<bool>(namedArgs, "center", false);

                // Change the return type to Func<Tensor, Tensor>
                return new Func<torch.Tensor, torch.Tensor>(samples =>
                {
                    using var raw = Matcha.MatchaAudio.mel_spectogram(samples, nFft, nMels, samplingRate, hopSize, winSize, fMin, fMax, center);
                    return raw.detach().clone();
                });
            }

            if (functionName.EndsWith("ras_sampling", StringComparison.OrdinalIgnoreCase))
            {
                var topP = GetNamedArg<double>(namedArgs, "top_p", 0.8);
                var topK = GetNamedArg<int>(namedArgs, "top_k", 25);
                var winSize = GetNamedArg<int>(namedArgs, "win_size", 10);
                var tauR = GetNamedArg<double>(namedArgs, "tau_r", 0.1);

                return new Func<torch.Tensor, List<int>, int, int>((scores, decodedTokens, runtimeTopK) =>
                    Common.RasSampling(scores, decodedTokens ?? new List<int>(), true, topP, runtimeTopK > 0 ? runtimeTopK : topK, winSize, tauR));
            }

            if (namedArgs == null || namedArgs.Count == 0)
            {
                try
                {
                    return ResolveFunction(functionName);
                }
                catch (MissingMethodException)
                {
                    return null;
                }
            }

            return new Func<object>(() => InvokeFunction(functionName, namedArgs));
        }

        private static object InstantiateClass(string className, Dictionary<string, object> args)
        {
            // Non-module config nodes used by YAML (e.g. omegaconf.DictConfig)
            if (className.Equals("omegaconf.DictConfig", StringComparison.OrdinalIgnoreCase))
            {
                var source = args.TryGetValue("content", out var content) && content is Dictionary<string, object> d ? d : args;
                dynamic expando = new ExpandoObject();
                var expandoDict = (IDictionary<string, object>)expando;
                foreach (var kv in source)
                    expandoDict[kv.Key] = kv.Value;
                return expando;
            }

            var clrName = NormalizeTypeName(className);
            var type = ResolveType(clrName) ?? throw new TypeLoadException($"Class not found: {className}");

            var ctors = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .OrderByDescending(c => c.GetParameters().Count(p => TryGetArg(args, p.Name, out _)))
                .ThenByDescending(c => c.GetParameters().Length)
                .ToList();

            foreach (var ctor in ctors)
            {
                var parameters = ctor.GetParameters();
                var values = new object[parameters.Length];
                var ok = true;

                for (var i = 0; i < parameters.Length; i++)
                {
                    var p = parameters[i];
                    if (TryGetArg(args, p.Name, out var raw))
                    {
                        try
                        {
                            values[i] = ConvertTo(raw, p.ParameterType);
                        }
                        catch
                        {
                            ok = false;
                            break;
                        }
                    }
                    else if (p.HasDefaultValue)
                    {
                        values[i] = p.DefaultValue;
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }

                if (ok)
                    return ctor.Invoke(values);
            }

            // Fallback: parameterless ctor + property injection
            var parameterless = type.GetConstructor(Type.EmptyTypes);
            if (parameterless != null)
            {
                var instance = parameterless.Invoke(Array.Empty<object>());
                foreach (var (k, v) in args)
                {
                    var prop = type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .FirstOrDefault(p => p.CanWrite && NormalizeName(p.Name) == NormalizeName(k));
                    if (prop != null)
                    {
                        prop.SetValue(instance, ConvertTo(v, prop.PropertyType));
                    }
                }
                return instance;
            }

            throw new InvalidOperationException($"No matching constructor for {className}");
        }

        private static MethodInfo ResolveFunction(string functionName)
        {
            var normalized = functionName.Replace("whisper.tokenizer.get_tokenizer", "CosyVoice.Tokenizer.CosyVoiceTokenizerFactory.get_whisper_tokenizer")
                                         .Replace("cosyvoice.tokenizer.tokenizer.", "CosyVoice.Tokenizer.CosyVoiceTokenizerFactory.")
                                         .Replace("cosyvoice.utils.common.", "CosyVoiceNet.Utils.Common.")
                                         .Replace("cosyvoice.", "CosyVoiceNet.");

            var lastDot = normalized.LastIndexOf('.');
            var methodShortName = lastDot >= 0 ? normalized[(lastDot + 1)..] : normalized;
            var typeNameHint = lastDot >= 0 ? normalized[..lastDot] : null;

            var assemblies = AppDomain.CurrentDomain.GetAssemblies();
            var allTypes = assemblies.SelectMany(a =>
            {
                try { return a.GetTypes(); }
                catch (ReflectionTypeLoadException ex) { return ex.Types.Where(t => t != null)!; }
            });

            if (!string.IsNullOrEmpty(typeNameHint))
            {
                var hintedType = allTypes.FirstOrDefault(t => string.Equals(t.FullName, typeNameHint, StringComparison.Ordinal));
                if (hintedType != null)
                {
                    var method = hintedType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                        .FirstOrDefault(m => m.Name.Equals(methodShortName, StringComparison.OrdinalIgnoreCase));
                    if (method != null) return method;
                }
            }

            var fallback = allTypes
                .SelectMany(t => t.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name.Equals(methodShortName, StringComparison.OrdinalIgnoreCase))
                    .Select(m => new { Method = m }))
                .FirstOrDefault();

            return fallback?.Method ?? throw new MissingMethodException($"Function not found: {functionName}");
        }

        private static object InvokeFunction(string functionName, List<object> positionalArgs)
        {
            if (IsSeedFunction(functionName))
            {
                var seed = positionalArgs.Count > 0 ? Convert.ToInt32(positionalArgs[0], CultureInfo.InvariantCulture) : 0;
                Common.SetAllRandomSeed(seed);
                return null;
            }

            var method = ResolveFunction(functionName);
            var parameters = method.GetParameters();
            if (parameters.Length != positionalArgs.Count)
                throw new InvalidOperationException($"Argument count mismatch for {functionName}: expected {parameters.Length}, got {positionalArgs.Count}");

            var callArgs = new object[parameters.Length];
            for (var i = 0; i < parameters.Length; i++)
            {
                callArgs[i] = ConvertTo(positionalArgs[i], parameters[i].ParameterType);
            }

            return method.Invoke(null, callArgs);
        }

        private static object InvokeFunction(string functionName, Dictionary<string, object> namedArgs)
        {
            if (IsSeedFunction(functionName))
            {
                var seedArg = namedArgs.Values.FirstOrDefault();
                var seed = seedArg == null ? 0 : Convert.ToInt32(seedArg, CultureInfo.InvariantCulture);
                Common.SetAllRandomSeed(seed);
                return null;
            }

            var method = ResolveFunction(functionName);
            var parameters = method.GetParameters();
            var callArgs = new object[parameters.Length];

            for (var i = 0; i < parameters.Length; i++)
            {
                var p = parameters[i];
                if (namedArgs.TryGetValue(p.Name, out var raw))
                {
                    callArgs[i] = ConvertTo(raw, p.ParameterType);
                }
                else if (p.HasDefaultValue)
                {
                    callArgs[i] = p.DefaultValue;
                }
                else
                {
                    throw new InvalidOperationException($"Missing argument '{p.Name}' for function {functionName}");
                }
            }

            return method.Invoke(null, callArgs);
        }

        private static bool IsSeedFunction(string functionName)
        {
            return functionName.Equals("random.seed", StringComparison.OrdinalIgnoreCase)
                || functionName.Equals("numpy.random.seed", StringComparison.OrdinalIgnoreCase)
                || functionName.Equals("torch.manual_seed", StringComparison.OrdinalIgnoreCase)
                || functionName.Equals("torch.cuda.manual_seed_all", StringComparison.OrdinalIgnoreCase);
        }

        private static Type ResolveType(string clrName)
        {
            var direct = Type.GetType(clrName, throwOnError: false);
            if (direct != null)
                return direct;

            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(clrName, throwOnError: false);
                if (t != null)
                    return t;
            }

            return null;
        }

        private static string NormalizeTypeName(string className)
        {
            var normalized = className
                .Replace("matcha.hifigan.models.MultiPeriodDiscriminator", "CosyVoiceNet.hifigan.MultiResolutionDiscriminator")
                .Replace("cosyvoice.hifigan.discriminator.MultiResSpecDiscriminator", "CosyVoiceNet.hifigan.MultiResolutionDiscriminator")
                .Replace("cosyvoice.llm.llm.CosyVoice3LM", "CosyVoiceNet.LLM.CosyVoice3LM")
                .Replace("cosyvoice.llm.llm.Qwen2LM", "CosyVoiceNet.LLM.Qwen2LM")
                .Replace("cosyvoice.llm.llm.TransformerLM", "CosyVoiceNet.LLM.TransformerLM")
                .Replace("cosyvoice.llm.llm.Qwen2Encoder", "CosyVoiceNet.LLM.Qwen2Encoder")
                .Replace("cosyvoice.transformer.encoder.ConformerEncoder", "CosyVoiceNet.Transformers.ConformerEncoder")
                .Replace("cosyvoice.transformer.encoder.TransformerEncoder", "CosyVoiceNet.Transformers.TransformerEncoder")
                .Replace("cosyvoice.flow.flow.MaskedDiffWithXvec", "CosyVoiceNet.flow.MaskedDiffWithXvec")
                .Replace("cosyvoice.flow.flow.CausalMaskedDiffWithXvec", "CosyVoiceNet.flow.CausalMaskedDiffWithXvec")
                .Replace("cosyvoice.flow.flow.CausalMaskedDiffWithDiT", "CosyVoiceNet.flow.CausalMaskedDiffWithDiT")
                .Replace("cosyvoice.flow.length_regulator.InterpolateRegulator", "CosyVoiceNet.flow.LengthRegulator")
                .Replace("cosyvoice.transformer.upsample_encoder.UpsampleConformerEncoder", "CosyVoiceNet.Transformers.UpsampleConformerEncoder")
                .Replace("cosyvoice.transformer.upsample_encoder.PreLookaheadLayer", "CosyVoiceNet.Transformers.PreLookaheadLayer")
                .Replace("cosyvoice.flow.flow_matching.ConditionalCFM", "CosyVoiceNet.flow.ConditionalCFM")
                .Replace("cosyvoice.flow.flow_matching.CausalConditionalCFM", "CosyVoiceNet.flow.CausalConditionalCFM")
                .Replace("cosyvoice.flow.decoder.ConditionalDecoder", "CosyVoiceNet.flow.ConditionalDecoder")
                .Replace("cosyvoice.flow.decoder.CausalConditionalDecoder", "CosyVoiceNet.flow.CausalConditionalDecoder")
                .Replace("cosyvoice.flow.DiT.dit.DiT", "CosyVoiceNet.flow.DiT.DiT")
                .Replace("cosyvoice.hifigan.generator.CausalHiFTGenerator", "CosyVoiceNet.hifigan.CausalHiFTGenerator")
                .Replace("cosyvoice.hifigan.generator.HiFTGenerator", "CosyVoiceNet.hifigan.HiFTGenerator")
                .Replace("cosyvoice.hifigan.f0_predictor.ConvRNNF0Predictor", "CosyVoiceNet.hifigan.ConvRNNF0Predictor")
                .Replace("cosyvoice.hifigan.hifigan.HiFiGan", "CosyVoiceNet.hifigan.HiFiGan")
                .Replace("cosyvoice.hifigan.discriminator.MultipleDiscriminator", "CosyVoiceNet.hifigan.MultipleDiscriminator")
                .Replace("cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor", "CosyVoiceNet.hifigan.CausalConvRNNF0Predictor")
                .Replace("cosyvoice.", "CosyVoiceNet.");

            return normalized;
        }

        private static object ConvertTo(object value, Type targetType)
        {
            if (value == null)
                return targetType.IsValueType ? Activator.CreateInstance(targetType) : null;

            if (targetType.IsInstanceOfType(value))
                return value;

            if (targetType == typeof(object))
                return value;

            if (targetType.IsEnum)
                return Enum.Parse(targetType, value.ToString(), ignoreCase: true);

            if (targetType.IsGenericType && targetType.GetGenericTypeDefinition() == typeof(List<>))
            {
                var elementType = targetType.GetGenericArguments()[0];
                if (value is System.Collections.IEnumerable enumerable)
                {
                    var list = (System.Collections.IList)Activator.CreateInstance(targetType);
                    foreach (var item in enumerable)
                    {
                        list.Add(ConvertTo(item, elementType));
                    }
                    return list;
                }
            }

            if (targetType.IsArray && value is System.Collections.IEnumerable arrEnum)
            {
                var elementType = targetType.GetElementType();
                var items = arrEnum.Cast<object>().Select(v => ConvertTo(v, elementType)).ToArray();
                var arr = Array.CreateInstance(elementType, items.Length);
                for (var i = 0; i < items.Length; i++)
                    arr.SetValue(items[i], i);
                return arr;
            }

            if (targetType == typeof(Dictionary<string, object>) && value is ExpandoObject expando)
            {
                // Convert ExpandoObject to Dictionary<string, object>
                return expando.ToDictionary(kv => kv.Key, kv => kv.Value);
            }

            if (targetType == typeof(Dictionary<string, object>) && value is Dictionary<string, object> dict)
                return dict;

            return Convert.ChangeType(value, targetType, CultureInfo.InvariantCulture);
        }

        private static T GetNamedArg<T>(Dictionary<string, object> namedArgs, string key, T defaultValue = default)
        {
            if (namedArgs != null && namedArgs.TryGetValue(key, out var raw) && raw != null)
            {
                if (raw is T t)
                    return t;

                if (typeof(T) == typeof(int?))
                {
                    if (raw is string s && (string.Equals(s, "null", StringComparison.OrdinalIgnoreCase) || s == "~"))
                        return defaultValue;
                    var convertedNullable = Convert.ToInt32(raw, CultureInfo.InvariantCulture);
                    return (T)(object)(int?)convertedNullable;
                }

                return (T)Convert.ChangeType(raw, Nullable.GetUnderlyingType(typeof(T)) ?? typeof(T), CultureInfo.InvariantCulture);
            }

            return defaultValue;
        }

        private static bool TryGetArg(Dictionary<string, object> args, string parameterName, out object value)
        {
            if (args.TryGetValue(parameterName, out value))
                return true;

            var normalizedParam = NormalizeName(parameterName);
            var match = args.FirstOrDefault(kv => NormalizeName(kv.Key) == normalizedParam);
            if (!string.IsNullOrEmpty(match.Key))
            {
                value = match.Value;
                return true;
            }

            value = null;
            return false;
        }

        private static string NormalizeName(string name)
        {
            return new string((name ?? string.Empty).Where(c => c != '_').ToArray()).ToLowerInvariant();
        }

        private static object EvalRef(string value, EvalContext ctx)
        {
            var expr = value;

            if (!expr.Contains("*"))
            {
                var key = expr.Trim('<', '>');
                if (!ctx.Parameters.TryGetValue(key, out var v))
                {
                    if (!TryResolveDeferredKey(key, ctx, out v))
                        throw new KeyNotFoundException($"Reference not found: {key}");
                }
                return v;
            }

            var parts = expr.Split('*', StringSplitOptions.RemoveEmptyEntries);
            double result = 1.0;

            foreach (var rawPart in parts)
            {
                var part = rawPart.Trim();
                object termValue;

                if (part.StartsWith("<") && part.EndsWith(">"))
                {
                    var key = part.Trim('<', '>');
                    if (!ctx.Parameters.TryGetValue(key, out termValue) && !TryResolveDeferredKey(key, ctx, out termValue))
                        throw new KeyNotFoundException($"Reference not found: {key}");
                }
                else
                {
                    if (!double.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out var num))
                        throw new FormatException($"Cannot parse term '{part}' in expression '{expr}'");
                    termValue = num;
                }

                result *= Convert.ToDouble(termValue, CultureInfo.InvariantCulture);
            }

            if (result % 1 == 0)
                return (int)result;

            return result;
        }

        private static bool TryResolveDeferredKey(string key, EvalContext ctx, out object value)
        {
            if (ctx.Parameters.TryGetValue(key, out value))
                return true;

            if (!ctx.Raw.TryGetValue(key, out var node))
            {
                value = null;
                return false;
            }

            if (!ctx.Resolving.Add(key))
                throw new InvalidOperationException($"Circular reference detected while resolving '{key}'.");

            try
            {
                value = EvaluateNode(node, ctx);
                ctx.Parameters[key] = value;
                return true;
            }
            finally
            {
                ctx.Resolving.Remove(key);
            }
        }
    }
}
