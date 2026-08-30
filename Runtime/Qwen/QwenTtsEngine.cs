// Slim TtsPipeline from ElBruno.QwenTTS (MIT). No HuggingFace download.

using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using SparkTTS.Qwen.Models;
using SparkTTS.Utils;

namespace SparkTTS.Qwen
{
    internal sealed class QwenTtsEngine : IDisposable
    {
        public const int NativeSampleRate = 24000;

        private readonly TextTokenizer _tokenizer;
        private readonly EmbeddingStore _embeddings;
        private readonly LanguageModel _languageModel;
        private readonly Vocoder _vocoder;
        private readonly object _gate = new();
        private bool _disposed;

        public QwenTtsEngine(string modelDir, Func<SessionOptions> sessionOptionsFactory = null)
        {
            if (string.IsNullOrEmpty(modelDir))
                throw new ArgumentNullException(nameof(modelDir));

            var tokenizerDir = System.IO.Path.Combine(modelDir, "tokenizer");
            var embeddingsDir = System.IO.Path.Combine(modelDir, "embeddings");
            var configPath = System.IO.Path.Combine(embeddingsDir, "config.json");

            _tokenizer = new TextTokenizer(tokenizerDir);
            _embeddings = new EmbeddingStore(embeddingsDir, configPath);
            _languageModel = new LanguageModel(modelDir, _embeddings, sessionOptionsFactory);
            _vocoder = new Vocoder(System.IO.Path.Combine(modelDir, "vocoder.onnx"), sessionOptionsFactory);
            Logger.Log($"[QwenTtsEngine] Loaded embeddings from {modelDir}");
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
                cancellationToken.ThrowIfCancellationRequested();
                var tokenIds = _tokenizer.BuildCustomVoicePrompt(text, speaker, language, instruct);
                Logger.LogVerbose($"[QwenTtsEngine] Tokenized {tokenIds.Length} ids, speaker={speaker}");
                var codes = _languageModel.Generate(tokenIds, speaker, language, cancellationToken: cancellationToken);
                return _vocoder.Decode(codes, cancellationToken);
            }
        }

        public Task<float[]> SynthesizeAsync(string text, string speaker, string language, string instruct,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(() => Synthesize(text, speaker, language, instruct, cancellationToken), cancellationToken);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _tokenizer.Dispose();
            _embeddings.Dispose();
            _languageModel.Dispose();
            _vocoder.Dispose();
        }
    }
}
