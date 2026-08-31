using System;
using System.Collections.Generic;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SparkTTS.Models;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// 12 Hz codec vocoder. CustomVoice uses vocoder.onnx with codes (1, 16, T);
    /// Base uses tokenizer12hz_decode.onnx with codes (1, T, 16).
    /// </summary>
    internal sealed class QwenVocoderModel : QwenOnnxModel
    {
        public const int SamplesPerFrame = 1920;
        public const int SampleRate = 24000;

        private readonly bool _timeMajor;

        public QwenVocoderModel(string folder, string modelName, bool timeMajor,
            ExecutionProvider executionProvider = ExecutionProvider.CPU)
            : base(modelName, folder, executionProvider)
        {
            _timeMajor = timeMajor;
        }

        public static QwenVocoderModel CustomVoice(ExecutionProvider ep) =>
            new QwenVocoderModel(SparkTTS.Core.SparkTTSModelPaths.QwenCustomVoiceFolder,
                SparkTTS.Core.SparkTTSModelPaths.QwenVocoder, timeMajor: false, ep);

        public static QwenVocoderModel Base(ExecutionProvider ep) =>
            new QwenVocoderModel(SparkTTS.Core.SparkTTSModelPaths.QwenBaseFolder,
                SparkTTS.Core.SparkTTSModelPaths.QwenTokenizer12HzDecode, timeMajor: true, ep);

        public float[] Decode(long[,,] codesQuantizerMajor, CancellationToken cancellationToken = default)
        {
            int batch = codesQuantizerMajor.GetLength(0);
            int quantizers = codesQuantizerMajor.GetLength(1);
            int timesteps = codesQuantizerMajor.GetLength(2);
            var flat = new long[batch * quantizers * timesteps];
            int n = 0;
            if (_timeMajor)
            {
                for (int b = 0; b < batch; b++)
                    for (int t = 0; t < timesteps; t++)
                        for (int q = 0; q < quantizers; q++)
                            flat[n++] = codesQuantizerMajor[b, q, t];
            }
            else
            {
                for (int b = 0; b < batch; b++)
                    for (int q = 0; q < quantizers; q++)
                        for (int t = 0; t < timesteps; t++)
                            flat[n++] = codesQuantizerMajor[b, q, t];
            }

            int[] shape = _timeMajor
                ? new[] { batch, timesteps, quantizers }
                : new[] { batch, quantizers, timesteps };
            return DecodeFlat(flat, shape, timesteps, cancellationToken);
        }

        public float[] Decode(long[,] codesTimeMajor, CancellationToken cancellationToken = default)
        {
            int t = codesTimeMajor.GetLength(0);
            int groups = codesTimeMajor.GetLength(1);
            if (t == 0)
                return Array.Empty<float>();

            var flat = new long[t * groups];
            int n = 0;
            for (int i = 0; i < t; i++)
                for (int g = 0; g < groups; g++)
                    flat[n++] = codesTimeMajor[i, g];

            return DecodeFlat(flat, new[] { 1, t, groups }, t, cancellationToken);
        }

        private float[] DecodeFlat(long[] flat, int[] shape, int timesteps, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string inputName = ResolveInputName("audio_codes", "codes");
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(flat, shape))
            };

            using var results = Run(feeds);
            var wav = CopyFloat(results[0]);
            // Flat buffer is the waveform. Do not use Dimensions[1] — CustomVoice
            // vocoder is [batch, 1, samples]; that dim is channels, not time.
            int wavLen = wav.Length;

            int target = timesteps * SamplesPerFrame;
            if (target > wavLen)
                target = wavLen;
            if (results.Count > 1)
            {
                var lengths = CopyLong(results[1]);
                // lengths[0] is sample count when it is in a plausible range.
                // Frame counts (T, or 1) must not trim a 24 kHz buffer to silence.
                if (lengths.Length > 0 && lengths[0] >= SamplesPerFrame / 2 && lengths[0] < wavLen)
                    target = Math.Min(target, (int)lengths[0]);
            }

            if (!_timeMajor && wav.Length != timesteps * SamplesPerFrame && wav.Length < target)
            {
                throw new InvalidOperationException(
                    $"Vocoder output mismatch: expected {timesteps * SamplesPerFrame} samples, got {wav.Length}.");
            }

            if (target >= wav.Length)
                return wav;
            var trimmed = new float[target];
            Array.Copy(wav, trimmed, target);
            return trimmed;
        }
    }
}
