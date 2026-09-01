using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using SparkTTS.Core;
using SparkTTS.Models;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// 12 Hz speech-tokenizer encoder for official Base ICL clone.
    /// Graph is traced at <see cref="GraphSamples"/> (Mimi pad/reshape
    /// freeze T; dynamo export fails). Pad/crop the wav, then keep
    /// <c>originalSamples / SamplesPerFrame</c> frames — same trim as
    /// Qwen's padding-mask slice. Tail zeros do not change prefix codes.
    /// </summary>
    internal sealed class QwenTokenizerEncoderModel : QwenOnnxModel
    {
        public const int SampleRate = 24000;
        public const int SamplesPerFrame = 1920;
        public const int Quantizers = 16;
        public const int GraphSeconds = 20;
        public const int GraphSamples = GraphSeconds * SampleRate;
        public const int GraphFrames = GraphSamples / SamplesPerFrame;

        public QwenTokenizerEncoderModel(ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(SparkTTSModelPaths.QwenTokenizerEncoder, SparkTTSModelPaths.QwenBaseFolder, executionProvider)
        {
        }

        public long[,,] Encode(float[] samples24k)
        {
            if (samples24k == null || samples24k.Length == 0)
                throw new ArgumentException("Reference audio is empty.", nameof(samples24k));

            int original = samples24k.Length;
            if (original > GraphSamples)
                original = GraphSamples;

            int keep = original / SamplesPerFrame;
            if (keep < 1)
                throw new InvalidOperationException(
                    "Reference audio is shorter than one 12 Hz codec frame (80 ms).");

            var padded = new float[GraphSamples];
            Buffer.BlockCopy(
                samples24k, 0, padded, 0,
                Math.Min(samples24k.Length, GraphSamples) * sizeof(float));

            string inputName = ResolveInputName("wav", "input_values");
            var feeds = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
            {
                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
                    inputName, new DenseTensor<float>(padded, new[] { 1, GraphSamples }))
            };

            using var results = Run(feeds);
            var flat = CopyLong(results[0]);
            if (flat.Length < keep * Quantizers)
                throw new InvalidOperationException(
                    $"tokenizer_encoder.onnx returned {flat.Length} codes, need {keep * Quantizers}.");

            var codes = new long[1, keep, Quantizers];
            for (int t = 0; t < keep; t++)
                for (int q = 0; q < Quantizers; q++)
                    codes[0, t, q] = flat[t * Quantizers + q];
            return codes;
        }
    }
}
