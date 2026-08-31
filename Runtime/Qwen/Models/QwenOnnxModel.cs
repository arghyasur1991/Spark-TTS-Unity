using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SparkTTS.Models;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// Qwen ONNX graph loaded through Spark's ORTModel (load policy, EP, session lifetime).
    /// Autoregressive TTS calls Run() synchronously after EnsureLoaded — do not wrap each
    /// decode step in RunDisposable/Task.Run.
    /// </summary>
    internal class QwenOnnxModel : ORTModel
    {
        private const long MaxOnnxBytes = 8_000_000_000;

        public QwenOnnxModel(string modelName, string folder, ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(modelName, folder, preAllocateOutputs: false, Precision.FP32, executionProvider, deferLoad: true)
        {
        }

        public new void EnsureLoaded()
        {
            string path = ModelFilePath;
            var info = new FileInfo(path);
            if (info.Exists && info.Length > MaxOnnxBytes)
                throw new InvalidOperationException($"ONNX file too large ({info.Length / 1e9:F2} GB): {path}");
            base.EnsureLoaded();
        }

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            EnsureLoaded();
            SetLoggingParam(ModelName);
            return Session.Run(inputs);
        }

        public string ResolveInputName(params string[] candidates)
        {
            EnsureLoaded();
            foreach (var c in candidates)
            {
                if (Session.InputMetadata.ContainsKey(c))
                    return c;
            }

            foreach (var key in Session.InputMetadata.Keys)
                return key;
            return candidates.Length > 0 ? candidates[0] : "input";
        }

        public IReadOnlyList<string> GraphInputNames
        {
            get
            {
                EnsureLoaded();
                return InputNames;
            }
        }

        public InferenceSession GetSession()
        {
            lock (NativeLifetimeGate)
            {
                if (HasLoadedSession)
                    return Session;
            }
            EnsureLoaded();
            return Session;
        }

        public static float[] CopyFloat(DisposableNamedOnnxValue value)
        {
            if (value.Value is DenseTensor<float> dense)
                return dense.Buffer.ToArray();
            return ToArray(value.AsEnumerable<float>());
        }

        public static long[] CopyLong(DisposableNamedOnnxValue value)
        {
            if (value.Value is DenseTensor<long> dense)
                return dense.Buffer.ToArray();
            return ToArray(value.AsEnumerable<long>());
        }

        public static DisposableNamedOnnxValue FindNamed(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, string name)
        {
            foreach (var o in outputs)
            {
                if (o.Name == name)
                    return o;
            }
            throw new InvalidOperationException($"ONNX output '{name}' not found.");
        }

        public static float[] LastHidden(DisposableNamedOnnxValue value, int hidden)
        {
            var all = CopyFloat(value);
            if (all.Length == hidden)
                return all;
            var last = new float[hidden];
            Array.Copy(all, all.Length - hidden, last, 0, hidden);
            return last;
        }

        private static T[] ToArray<T>(IEnumerable<T> src)
        {
            if (src is T[] arr)
                return arr;
            var list = new List<T>();
            foreach (var v in src)
                list.Add(v);
            return list.ToArray();
        }
    }
}
