// One engine for VoiceDesign style TTS and Base ICL clone.
// Both talkers are the ElBruno graph split (LanguageModel). Clone injects
// the speaker-encoder vector plus 12 Hz ref_code + ref_text (official ICL).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using SparkTTS.Core;
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

        private readonly ExecutionProvider _ep;
        private readonly object _gate = new();
        private bool _disposed;

        private TextTokenizer _styleTokenizer;
        private EmbeddingStore _embeddings;
        private LanguageModel _languageModel;
        private QwenVocoderModel _styleVocoder;

        private TextTokenizer _cloneTokenizer;
        private EmbeddingStore _cloneEmbeddings;
        private LanguageModel _cloneLanguageModel;
        private QwenVocoderModel _cloneVocoder;
        private QwenSpeakerEncoderModel _speakerEncoder;
        private QwenTokenizerEncoderModel _tokenizerEncoder;

        public bool HasCustomVoice => _languageModel != null;
        public bool HasClone => _cloneLanguageModel != null;
        public bool HasIclEncoder => _tokenizerEncoder != null;

        public QwenTtsEngine(ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            _ep = executionProvider;
            if (!QwenModelPaths.IsCustomVoicePresent() && !QwenModelPaths.IsBasePresent())
            {
                throw new InvalidOperationException(
                    "Qwen3-TTS files missing. Style: " + QwenModelPaths.Root +
                    "; clone: " + QwenModelPaths.BaseRoot);
            }
        }

        /// <summary>
        /// VoiceDesign npy + talker wrappers. Not Base clone. Call from VoiceDesign generate.
        /// </summary>
        internal void EnsureStyle()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            lock (_gate)
                EnsureStyleUnlocked();
        }

        void EnsureStyleUnlocked()
        {
            if (_languageModel != null)
                return;
            if (!QwenModelPaths.IsCustomVoicePresent())
                throw new InvalidOperationException("VoiceDesign weights are not installed.");

            var sw = Stopwatch.StartNew();
            var embeddingsDir = Path.Combine(QwenModelPaths.Root, "embeddings");
            var configPath = Path.Combine(embeddingsDir, "config.json");
            _styleTokenizer = new TextTokenizer(Path.Combine(QwenModelPaths.Root, "tokenizer"));
#if UNITY_EDITOR
            var pendingEmb = NativeSessionKeepAlive.TakePendingEmbeddings();
            if (pendingEmb != null)
            {
                _embeddings = EmbeddingStore.FromKeepAliveSlots(embeddingsDir, configPath, pendingEmb);
                TTSLogger.Log(
                    $"[QwenTtsEngine] Wrapped VoiceDesign embeddings after domain reload in {sw.ElapsedMilliseconds}ms");
            }
            else
#endif
            {
                _embeddings = new EmbeddingStore(embeddingsDir, configPath);
                TTSLogger.Log(
                    $"[QwenTtsEngine] VoiceDesign embeddings from {QwenModelPaths.Root} in {sw.ElapsedMilliseconds}ms");
            }
            _languageModel = new LanguageModel(_embeddings, _ep);
            _styleVocoder = QwenVocoderModel.CustomVoice(_ep);
        }

        /// <summary>
        /// Base clone npy + talker + speaker encoder + 12 Hz tokenizer encoder.
        /// Not VoiceDesign. Call from CreateFromReference / clone generate.
        /// </summary>
        internal void EnsureClone()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            lock (_gate)
                EnsureCloneUnlocked();
        }

        void EnsureCloneUnlocked()
        {
            if (_cloneLanguageModel != null)
                return;
            if (!QwenModelPaths.IsBasePresent())
                throw new InvalidOperationException("Base clone weights are not installed.");

            var sw = Stopwatch.StartNew();
            var embeddingsDir = Path.Combine(QwenModelPaths.BaseRoot, "embeddings");
            var configPath = Path.Combine(embeddingsDir, "config.json");
            _cloneTokenizer = new TextTokenizer(Path.Combine(QwenModelPaths.BaseRoot, "tokenizer"));
            _cloneEmbeddings = new EmbeddingStore(embeddingsDir, configPath);
            _cloneLanguageModel = new LanguageModel(
                _cloneEmbeddings, SparkTTSModelPaths.QwenBaseFolder, _ep);
            _cloneVocoder = QwenVocoderModel.Base(_ep);
            _speakerEncoder = new QwenSpeakerEncoderModel(_ep);
            _tokenizerEncoder = new QwenTokenizerEncoderModel(_ep);
            TTSLogger.Log(
                $"[QwenTtsEngine] Base clone embeddings from {QwenModelPaths.BaseRoot} in {sw.ElapsedMilliseconds}ms");
        }

        public float[] Synthesize(string text, string speaker, string language, string instruct,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (text.Length > 10000)
                throw new ArgumentException("Text exceeds maximum length of 10,000 characters.", nameof(text));

            lock (_gate)
            {
                EnsureStyleUnlocked();
                cancellationToken.ThrowIfCancellationRequested();
                _ = speaker;
                var assistantIds = _styleTokenizer.BuildAssistantPrompt(text);
                var instructIds = _styleTokenizer.BuildInstructTokens(instruct);
                TTSLogger.LogVerbose(
                    $"[QwenTtsEngine] VoiceDesign tokens assistant={assistantIds.Length} instruct={instructIds.Length}");
                var codes = _languageModel.GenerateVoiceDesign(
                    assistantIds, instructIds, language, cancellationToken: cancellationToken);
                var pcm = _styleVocoder.Decode(codes, cancellationToken);
                TTSLogger.Log(
                    $"[QwenTtsEngine] VoiceDesign codes T={codes.GetLength(2)} wav={pcm.Length} @24k");
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

            lock (_gate)
            {
                EnsureCloneUnlocked();
                var embedding = _speakerEncoder.Encode(samples24k);
                if (embedding.Length == 0)
                    throw new InvalidOperationException("speaker_encoder.onnx returned an empty embedding.");
                TTSLogger.LogVerbose($"[QwenTtsEngine] Speaker embedding dim={embedding.Length}");
                return embedding;
            }
        }

        public long[,,] EncodeReferenceCodes(float[] samples24k)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));

            lock (_gate)
            {
                EnsureCloneUnlocked();
                if (_tokenizerEncoder == null)
                    throw new InvalidOperationException("Base ICL tokenizer encoder is not installed.");
                var codes = _tokenizerEncoder.Encode(samples24k);
                TTSLogger.LogVerbose(
                    $"[QwenTtsEngine] ICL ref_code T={codes.GetLength(1)} Q={codes.GetLength(2)}");
                return codes;
            }
        }

        public float[] SynthesizeClone(string text, float[] speakerEmbedding, string language,
            string refText = null, long[,,] refAudioCodes = null,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(QwenTtsEngine));
            if (string.IsNullOrEmpty(text))
                throw new ArgumentException("Text cannot be empty.", nameof(text));
            if (speakerEmbedding == null || speakerEmbedding.Length == 0)
                throw new ArgumentException("Speaker embedding is required.", nameof(speakerEmbedding));

            lock (_gate)
            {
                EnsureCloneUnlocked();
                cancellationToken.ThrowIfCancellationRequested();
                var tokenIds = _cloneTokenizer.BuildCustomVoicePrompt(text, speaker: null, language, instruct: null);
                if (tokenIds.Length < 8)
                    throw new InvalidOperationException("Prompt tokenization produced too few tokens.");

                long[,,] codes;
                bool icl = !string.IsNullOrEmpty(refText) && refAudioCodes != null;
                if (icl)
                {
                    var refTokenIds = _cloneTokenizer.BuildIclRefTextTokens(refText);
                    codes = _cloneLanguageModel.GenerateWithSpeakerEmbeddingAndRefText(
                        tokenIds, speakerEmbedding, language, refTokenIds, refAudioCodes,
                        cancellationToken: cancellationToken);
                    TTSLogger.Log(
                        $"[QwenTtsEngine] Base ICL clone codes T={codes.GetLength(2)} wav pending @24k refT={refAudioCodes.GetLength(1)}");
                }
                else
                {
                    codes = _cloneLanguageModel.GenerateWithSpeakerEmbedding(
                        tokenIds, speakerEmbedding, language, cancellationToken: cancellationToken);
                    TTSLogger.Log(
                        $"[QwenTtsEngine] Base x-vector clone codes T={codes.GetLength(2)} wav pending @24k");
                }

                var pcm = _cloneVocoder.Decode(codes, cancellationToken);
                TTSLogger.Log(
                    $"[QwenTtsEngine] Base clone wav={pcm.Length} @24k icl={icl}");
                return pcm;
            }
        }

        public Task<float[]> SynthesizeCloneAsync(string text, float[] speakerEmbedding, string language,
            CancellationToken cancellationToken = default)
        {
            return BackgroundWork.Run(
                () => SynthesizeClone(text, speakerEmbedding, language, null, null, cancellationToken));
        }

        /// <summary>
        /// Opens CustomVoice ONNX sessions on a worker thread. Safe to call from the main thread.
        /// </summary>
        public Task PreloadStyleAsync()
        {
            return BackgroundWork.Run(() =>
            {
                lock (_gate)
                {
                    EnsureStyleUnlocked();
                    _languageModel.PreloadSessions();
                    _styleVocoder.GetSession();
                }
            });
        }

        /// <summary>
        /// Opens Base clone ONNX sessions on a worker thread. Safe to call from the main thread.
        /// </summary>
        public Task PreloadCloneAsync()
        {
            return BackgroundWork.Run(() =>
            {
                lock (_gate)
                {
                    EnsureCloneUnlocked();
                    _cloneLanguageModel.PreloadSessions();
                    _cloneVocoder.GetSession();
                    _speakerEncoder.GetSession();
                    _tokenizerEncoder?.GetSession();
                }
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
            _cloneEmbeddings?.Dispose();
            _cloneLanguageModel?.Dispose();
            _cloneVocoder?.Dispose();
            _speakerEncoder?.Dispose();
            _tokenizerEncoder?.Dispose();
        }

        internal NativeEmbSlot[] DetachEmbeddings() => _embeddings?.DetachNativeSlots();

        internal void CollectOnnxModels(List<ORTModel> list)
        {
            _languageModel?.CollectOnnxModels(list);
            if (_styleVocoder != null)
                list.Add(_styleVocoder);
            _cloneLanguageModel?.CollectOnnxModels(list);
            if (_cloneVocoder != null)
                list.Add(_cloneVocoder);
            if (_speakerEncoder != null)
                list.Add(_speakerEncoder);
            if (_tokenizerEncoder != null)
                list.Add(_tokenizerEncoder);
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
    }
}
