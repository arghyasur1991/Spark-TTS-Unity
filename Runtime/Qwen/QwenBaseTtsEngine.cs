// Qwen3-TTS 1.7B Base: x-vector voice clone for CharacterVoice.CreateFromReference.
// Speaker encoder + talker from zukky ONNX; mel in C# (no qwen3_tts_rust.dll).

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SparkTTS.Qwen.Audio;
using SparkTTS.Qwen.Models;
using SparkTTS.Utils;
using UnityEngine;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS.Qwen
{
    internal sealed class QwenBaseTtsEngine : IDisposable
    {
        public const int NativeSampleRate = 24000;

        private readonly QwenBaseConfig _config;
        private readonly TextTokenizer _tokenizer;
        private readonly QwenBaseTalker _talker;
        private readonly Lazy<InferenceSession> _speakerEncoder;
        private readonly object _gate = new();
        private bool _disposed;

        public QwenBaseTtsEngine(string modelDir, Func<SessionOptions> sessionOptionsFactory)
        {
            if (string.IsNullOrEmpty(modelDir))
                throw new ArgumentNullException(nameof(modelDir));

            _config = QwenBaseConfig.Load(QwenBaseModelPaths.ConfigPath);
            _tokenizer = new TextTokenizer(QwenBaseModelPaths.TokenizerDir);
            _talker = new QwenBaseTalker(modelDir, _config, sessionOptionsFactory);
            _speakerEncoder = new Lazy<InferenceSession>(
                () => new InferenceSession(
                    System.IO.Path.Combine(modelDir, "speaker_encoder.onnx"),
                    sessionOptionsFactory()),
                LazyThreadSafetyMode.ExecutionAndPublication);

            TTSLogger.Log($"[QwenBaseTtsEngine] Base config loaded from {modelDir} (hidden={_config.HiddenSize})");
        }

        public float[] ExtractSpeakerEmbedding(float[] samples24k)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenBaseTtsEngine));
            if (samples24k == null || samples24k.Length == 0)
                throw new ArgumentException("Reference audio is empty.", nameof(samples24k));

            lock (_gate)
            {
                var mel = MelSpectrogram.Extract(samples24k);
                int tMel = mel.GetLength(0);
                int nMels = mel.GetLength(1);
                var flat = MelSpectrogram.FlattenTimeFirst(mel);

                var session = _speakerEncoder.Value;
                string inputName = "mels";
                if (!session.InputMetadata.ContainsKey(inputName))
                {
                    foreach (var key in session.InputMetadata.Keys)
                    {
                        inputName = key;
                        break;
                    }
                }

                var feeds = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName,
                        new DenseTensor<float>(flat, new[] { 1, tMel, nMels }))
                };

                using var results = session.Run(feeds);
                var data = results[0].AsEnumerable<float>();
                var list = new List<float>();
                foreach (var v in data)
                    list.Add(v);

                if (list.Count == 0)
                    throw new InvalidOperationException("speaker_encoder.onnx returned an empty embedding.");

            TTSLogger.LogVerbose($"[QwenBaseTtsEngine] Speaker embedding dim={list.Count}, mel frames={tMel}");
                return list.ToArray();
            }
        }

        public float[] SynthesizeClone(string text, float[] speakerEmbedding, string language,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenBaseTtsEngine));
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (speakerEmbedding == null || speakerEmbedding.Length == 0)
                throw new ArgumentException("Speaker embedding is required.", nameof(speakerEmbedding));

            lock (_gate)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokenIds = _tokenizer.BuildCustomVoicePrompt(text, speaker: null, language, instruct: null);
                if (tokenIds.Length < 8)
                    throw new InvalidOperationException("Prompt tokenization produced too few tokens.");

                var (embeds, mask, trailing, ttsPad) = BuildXVectorPrefill(tokenIds, speakerEmbedding, language);
                var codes = _talker.GenerateCodes(embeds, mask, trailing, ttsPad, 1024, cancellationToken);
                return _talker.DecodeCodes(codes, cancellationToken);
            }
        }

        public Task<float[]> SynthesizeCloneAsync(string text, float[] speakerEmbedding, string language,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(
                () => SynthesizeClone(text, speakerEmbedding, language, cancellationToken),
                cancellationToken);
        }

        public static float[] ClipToMono24k(AudioClip clip)
        {
            if (clip == null)
                throw new ArgumentNullException(nameof(clip));

            var raw = new float[clip.samples * clip.channels];
            clip.GetData(raw, 0);

            float[] mono;
            if (clip.channels <= 1)
            {
                mono = raw;
            }
            else
            {
                mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0f;
                    for (int c = 0; c < clip.channels; c++)
                        sum += raw[i * clip.channels + c];
                    mono[i] = sum / clip.channels;
                }
            }

            if (clip.frequency == NativeSampleRate)
                return mono;
            return AudioResample.Resample(mono, clip.frequency, NativeSampleRate);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _tokenizer.Dispose();
            _talker.Dispose();
            if (_speakerEncoder.IsValueCreated)
                _speakerEncoder.Value.Dispose();
        }

        private (float[,,] embeds, long[,] mask, float[,,] trailing, float[] ttsPad)
            BuildXVectorPrefill(int[] tokenIds, float[] speakerEmbedding, string language)
        {
            int h = _config.HiddenSize;
            var ids = ToLong(tokenIds);

            var ttsIds = new long[] { _config.TtsBosTokenId, _config.TtsEosTokenId, _config.TtsPadTokenId };
            var ttsEmbeds = _talker.TextProject(ttsIds);
            var ttsBos = Row(ttsEmbeds, 0);
            var ttsEos = Row(ttsEmbeds, 1);
            var ttsPad = Row(ttsEmbeds, 2);

            long[] codecPrefill;
            if (string.IsNullOrEmpty(language) ||
                string.Equals(language, "auto", StringComparison.OrdinalIgnoreCase))
            {
                codecPrefill = new long[]
                {
                    _config.CodecNothinkId,
                    _config.CodecThinkBosId,
                    _config.CodecThinkEosId
                };
            }
            else
            {
                string key = language.ToLowerInvariant();
                if (!_config.CodecLanguageId.TryGetValue(key, out int languageId))
                    throw new ArgumentException($"Unsupported language '{language}'.", nameof(language));
                codecPrefill = new long[]
                {
                    _config.CodecThinkId,
                    _config.CodecThinkBosId,
                    languageId,
                    _config.CodecThinkEosId
                };
            }

            var codec0 = Flatten(_talker.CodecEmbed(codecPrefill));
            var codec1 = Flatten(_talker.CodecEmbed(new long[] { _config.CodecPadId, _config.CodecBosId }));
            var spk = PadOrTrim(speakerEmbedding, h);
            var codecInput = ConcatRows(codec0, spk, codec1);

            var role = Flatten(_talker.TextProject(Slice(ids, 0, 3)));
            int codecT = codecInput.Count;
            int padRepeat = codecT - 2;
            var padBlock = Repeat(ttsPad, padRepeat);
            var talkerEmbed = ConcatRows(padBlock);
            talkerEmbed.Add(ttsBos);
            AddRows(talkerEmbed, codecInput, codecT - 1);

            var ttsTextFirstProj = Flatten(_talker.TextProject(Slice(ids, 3, 1)));
            var ttsTextFirst = ttsTextFirstProj[0];
            AddInPlace(ttsTextFirst, codecInput[codecT - 1]);

            var sequence = new List<float[]>();
            sequence.AddRange(role);
            sequence.AddRange(talkerEmbed);
            sequence.Add(ttsTextFirst);

            float[,,] trailing;
            int textTailStart = 4;
            int textTailEnd = ids.Length - 5;
            if (textTailEnd > textTailStart)
            {
                var tail = Flatten(_talker.TextProject(Slice(ids, textTailStart, textTailEnd - textTailStart)));
                tail.Add(ttsEos);
                trailing = To3(tail, h);
            }
            else
            {
                trailing = To3(new List<float[]> { ttsEos }, h);
            }

            var embeds = To3(sequence, h);
            int t = sequence.Count;
            var mask = new long[1, t];
            for (int i = 0; i < t; i++)
                mask[0, i] = 1;

            return (embeds, mask, trailing, ttsPad);
        }

        private static long[] ToLong(int[] ids)
        {
            var a = new long[ids.Length];
            for (int i = 0; i < ids.Length; i++)
                a[i] = ids[i];
            return a;
        }

        private static long[] Slice(long[] ids, int start, int count)
        {
            var a = new long[count];
            Array.Copy(ids, start, a, 0, count);
            return a;
        }

        private static float[] Row(float[,,] a, int t)
        {
            int h = a.GetLength(2);
            var row = new float[h];
            for (int i = 0; i < h; i++)
                row[i] = a[0, t, i];
            return row;
        }

        private static List<float[]> Flatten(float[,,] a)
        {
            int t = a.GetLength(1);
            int h = a.GetLength(2);
            var rows = new List<float[]>(t);
            for (int i = 0; i < t; i++)
            {
                var row = new float[h];
                for (int j = 0; j < h; j++)
                    row[j] = a[0, i, j];
                rows.Add(row);
            }
            return rows;
        }

        private static List<float[]> Repeat(float[] row, int n)
        {
            var list = new List<float[]>(n);
            for (int i = 0; i < n; i++)
                list.Add((float[])row.Clone());
            return list;
        }

        private static List<float[]> ConcatRows(params object[] parts)
        {
            var list = new List<float[]>();
            foreach (var part in parts)
            {
                if (part is float[] vec)
                    list.Add(vec);
                else if (part is List<float[]> rows)
                    list.AddRange(rows);
            }
            return list;
        }

        private static void AddRows(List<float[]> dst, List<float[]> src, int count)
        {
            for (int i = 0; i < count; i++)
                AddInPlace(dst[i], src[i]);
        }

        private static void AddInPlace(float[] dst, float[] src)
        {
            int n = Math.Min(dst.Length, src.Length);
            for (int i = 0; i < n; i++)
                dst[i] += src[i];
        }

        private static float[] PadOrTrim(float[] embedding, int hidden)
        {
            if (embedding.Length == hidden)
                return (float[])embedding.Clone();
            var v = new float[hidden];
            int n = Math.Min(hidden, embedding.Length);
            Array.Copy(embedding, v, n);
            return v;
        }

        private static float[,,] To3(List<float[]> rows, int hidden)
        {
            var a = new float[1, rows.Count, hidden];
            for (int t = 0; t < rows.Count; t++)
            {
                int n = Math.Min(hidden, rows[t].Length);
                for (int i = 0; i < n; i++)
                    a[0, t, i] = rows[t][i];
            }
            return a;
        }
    }
}
