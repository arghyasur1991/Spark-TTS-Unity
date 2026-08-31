// One engine for CustomVoice style TTS and Base x-vector clone.
// ONNX sessions are ORTModel (QwenOnnxModel). Graphs differ, so two generate loops stay.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using SparkTTS.Models;
using SparkTTS.Qwen.Models;
using SparkTTS.Utils;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS.Qwen
{
    internal sealed class QwenTtsEngine : IDisposable
    {
        public const int NativeSampleRate = 24000;

        private readonly object _gate = new();
        private bool _disposed;

        private readonly TextTokenizer _styleTokenizer;
        private readonly EmbeddingStore _embeddings;
        private readonly LanguageModel _languageModel;
        private readonly QwenVocoderModel _styleVocoder;

        private readonly QwenBaseConfig _baseConfig;
        private readonly TextTokenizer _cloneTokenizer;
        private readonly QwenBaseTalker _talker;
        private readonly QwenSpeakerEncoderModel _speakerEncoder;

        public bool HasCustomVoice => _languageModel != null;
        public bool HasClone => _talker != null;

        public QwenTtsEngine(ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            bool style = QwenModelPaths.IsCustomVoicePresent();
            bool clone = QwenModelPaths.IsBasePresent();
            if (!style && !clone)
            {
                throw new InvalidOperationException(
                    "Qwen3-TTS files missing. Style: " + QwenModelPaths.Root +
                    "; clone: " + QwenModelPaths.BaseRoot);
            }

            if (style)
            {
                var sw = Stopwatch.StartNew();
                var embeddingsDir = System.IO.Path.Combine(QwenModelPaths.Root, "embeddings");
                var configPath = System.IO.Path.Combine(embeddingsDir, "config.json");
                _styleTokenizer = new TextTokenizer(System.IO.Path.Combine(QwenModelPaths.Root, "tokenizer"));
                TTSLogger.Log($"[QwenTtsEngine] style tokenizer {sw.ElapsedMilliseconds}ms");
                sw.Restart();
#if UNITY_EDITOR
                var pendingEmb = NativeSessionKeepAlive.TakePendingEmbeddings();
                if (pendingEmb != null)
                {
                    _embeddings = EmbeddingStore.FromKeepAliveSlots(embeddingsDir, configPath, pendingEmb);
                    TTSLogger.Log(
                        $"[QwenTtsEngine] Wrapped native embeddings after domain reload in {sw.ElapsedMilliseconds}ms");
                }
                else
#endif
                {
                    _embeddings = new EmbeddingStore(embeddingsDir, configPath);
                    TTSLogger.Log(
                        $"[QwenTtsEngine] CustomVoice embeddings from {QwenModelPaths.Root} in {sw.ElapsedMilliseconds}ms");
                }
                sw.Restart();
                _languageModel = new LanguageModel(_embeddings, executionProvider);
                _styleVocoder = QwenVocoderModel.CustomVoice(executionProvider);
                TTSLogger.Log($"[QwenTtsEngine] style ORT wrappers {sw.ElapsedMilliseconds}ms");
            }

            if (clone)
            {
                var sw = Stopwatch.StartNew();
                _baseConfig = QwenBaseConfig.Load(QwenModelPaths.BaseConfigPath);
                _cloneTokenizer = new TextTokenizer(QwenModelPaths.BaseTokenizerDir);
                _talker = new QwenBaseTalker(_baseConfig, executionProvider);
                _speakerEncoder = new QwenSpeakerEncoderModel(executionProvider);
                TTSLogger.Log(
                    $"[QwenTtsEngine] Base clone wrappers from {QwenModelPaths.BaseRoot} in {sw.ElapsedMilliseconds}ms (hidden={_baseConfig.HiddenSize})");
            }
        }

        public float[] Synthesize(string text, string speaker, string language, string instruct,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            if (_languageModel == null)
                throw new InvalidOperationException("CustomVoice weights are not installed.");
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (text.Length > 10000)
                throw new ArgumentException("Text exceeds maximum length of 10,000 characters.", nameof(text));

            lock (_gate)
            {
                var sw = Stopwatch.StartNew();
                cancellationToken.ThrowIfCancellationRequested();
                var tokenIds = _styleTokenizer.BuildCustomVoicePrompt(text, speaker, language, instruct);
                TTSLogger.LogVerbose($"[QwenTtsEngine] Tokenized {tokenIds.Length} ids, speaker={speaker}");
                var codes = _languageModel.Generate(tokenIds, speaker, language, cancellationToken: cancellationToken);
                var pcm = _styleVocoder.Decode(codes, cancellationToken);
                TTSLogger.Log(
                    $"[QwenTtsEngine] style synth {sw.ElapsedMilliseconds}ms codes T={codes.GetLength(2)} wav={pcm.Length} @24k speaker={speaker}");
                return pcm;
            }
        }

        public Task<float[]> SynthesizeAsync(string text, string speaker, string language, string instruct,
            CancellationToken cancellationToken = default)
        {
            return BackgroundWork.Run(() => Synthesize(text, speaker, language, instruct, cancellationToken));
        }

        public float[] ExtractSpeakerEmbedding(float[] samples24k)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            if (_speakerEncoder == null)
                throw new InvalidOperationException("Base clone weights are not installed.");

            lock (_gate)
            {
                var embedding = _speakerEncoder.Encode(samples24k);
                if (embedding.Length == 0)
                    throw new InvalidOperationException("speaker_encoder.onnx returned an empty embedding.");
                TTSLogger.LogVerbose($"[QwenTtsEngine] Speaker embedding dim={embedding.Length}");
                return embedding;
            }
        }

        public float[] SynthesizeClone(string text, float[] speakerEmbedding, string language,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            if (_talker == null)
                throw new InvalidOperationException("Base clone weights are not installed.");
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (speakerEmbedding == null || speakerEmbedding.Length == 0)
                throw new ArgumentException("Speaker embedding is required.", nameof(speakerEmbedding));

            lock (_gate)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokenIds = _cloneTokenizer.BuildCustomVoicePrompt(text, speaker: null, language, instruct: null);
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
            return BackgroundWork.Run(
                () => SynthesizeClone(text, speakerEmbedding, language, cancellationToken));
        }

        /// <summary>
        /// Opens CustomVoice ONNX sessions on a worker thread. Safe to call from the main thread.
        /// </summary>
        public Task PreloadStyleAsync()
        {
            if (_languageModel == null)
                return Task.CompletedTask;
            return BackgroundWork.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                lock (_gate)
                {
                    _languageModel.PreloadSessions();
                    _styleVocoder.GetSession();
                }
                TTSLogger.Log($"[QwenTtsEngine] PreloadStyle {sw.ElapsedMilliseconds}ms");
            });
        }

        /// <summary>
        /// Opens Base clone ONNX sessions on a worker thread. Safe to call from the main thread.
        /// </summary>
        public Task PreloadCloneAsync()
        {
            if (_talker == null)
                return Task.CompletedTask;
            return BackgroundWork.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                lock (_gate)
                {
                    _talker.PreloadSessions();
                    _speakerEncoder.GetSession();
                }
                TTSLogger.Log($"[QwenTtsEngine] PreloadClone {sw.ElapsedMilliseconds}ms");
            });
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
            _styleTokenizer?.Dispose();
            _embeddings?.Dispose();
            _languageModel?.Dispose();
            _styleVocoder?.Dispose();
            _cloneTokenizer?.Dispose();
            _talker?.Dispose();
            _speakerEncoder?.Dispose();
        }

        internal NativeEmbSlot[] DetachEmbeddings() => _embeddings?.DetachNativeSlots();

        internal void CollectOnnxModels(List<ORTModel> list)
        {
            _languageModel?.CollectOnnxModels(list);
            if (_styleVocoder != null)
                list.Add(_styleVocoder);
            _talker?.CollectOnnxModels(list);
            if (_speakerEncoder != null)
                list.Add(_speakerEncoder);
        }

        internal void AdoptNativeSessions(Dictionary<string, InferenceSession> sessions)
        {
            if (sessions == null || sessions.Count == 0)
                return;
            var models = new List<ORTModel>();
            CollectOnnxModels(models);
            foreach (var model in models)
            {
                if (sessions.TryGetValue(model.SessionKeepAliveKey, out var session))
                {
                    model.AdoptSession(session);
                    sessions.Remove(model.SessionKeepAliveKey);
                    TTSLogger.Log("[QwenTtsEngine] Adopted " + model.SessionKeepAliveKey);
                }
            }
        }

        private (float[,,] embeds, long[,] mask, float[,,] trailing, float[] ttsPad)
            BuildXVectorPrefill(int[] tokenIds, float[] speakerEmbedding, string language)
        {
            int h = _baseConfig.HiddenSize;
            var ids = ToLong(tokenIds);

            var ttsIds = new long[] { _baseConfig.TtsBosTokenId, _baseConfig.TtsEosTokenId, _baseConfig.TtsPadTokenId };
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
                    _baseConfig.CodecNothinkId,
                    _baseConfig.CodecThinkBosId,
                    _baseConfig.CodecThinkEosId
                };
            }
            else
            {
                string key = language.ToLowerInvariant();
                if (!_baseConfig.CodecLanguageId.TryGetValue(key, out int languageId))
                    throw new ArgumentException($"Unsupported language '{language}'.", nameof(language));
                codecPrefill = new long[]
                {
                    _baseConfig.CodecThinkId,
                    _baseConfig.CodecThinkBosId,
                    languageId,
                    _baseConfig.CodecThinkEosId
                };
            }

            var codec0 = Flatten(_talker.CodecEmbed(codecPrefill));
            var codec1 = Flatten(_talker.CodecEmbed(new long[] { _baseConfig.CodecPadId, _baseConfig.CodecBosId }));
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
