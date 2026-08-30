using System;
using System.Threading.Tasks;
using UnityEngine;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS
{
    using Models;
    using Qwen;
    using Utils;
    /// <summary>
    /// Factory class for creating CharacterVoice objects using either voice cloning or style-based generation.
    /// </summary>
    public class CharacterVoiceFactory : IDisposable
    {
        public static CharacterVoiceFactory Instance { get; private set; } = new();

        public bool LogTiming
        {
            get => SparkTTS.Core.SparkTTS.LogTiming;
            set => SparkTTS.Core.SparkTTS.LogTiming = value;
        }

        /// <summary>
        /// Gets whether the engine is initialized and ready for use.
        /// </summary>
        public static bool IsReady => Instance._initialized && !Instance._disposed;

        private QwenTtsEngine _engine;
        private ExecutionProvider _executionProvider = ExecutionProvider.CPU;
        private bool _disposed;
        private bool _initialized;

        internal CharacterVoiceFactory()
        {
            bool style = QwenModelPaths.IsCustomVoicePresent();
            bool clone = QwenModelPaths.IsBasePresent();
            _initialized = style || clone;
            if (!_initialized)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Qwen3-TTS files not found. Style needs " +
                    $"{QwenModelPaths.Root}; clone needs {QwenModelPaths.BaseRoot}. See QWEN3.md.");
            }
            else if (!style)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] CustomVoice 1.7B missing — CreateFromStyleAsync unavailable. " +
                    "Clone is available from Base at " + QwenModelPaths.BaseRoot);
            }
            else if (!clone)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Base 1.7B missing — CreateFromReference unavailable. " +
                    "Place zukky onnx_kv + tokenizer at " + QwenModelPaths.BaseRoot);
            }
        }

        /// <summary>
        /// Initializes or re-initializes the CharacterVoiceFactory with the specified settings.
        /// </summary>
        public static void Initialize(LogLevel logLevel, MemoryUsage memoryUsage, ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            TTSLogger.LogLevel = logLevel;
            ORTModel.InitializeEnvironment(logLevel);
            ORTModel.SetMemoryUsage(memoryUsage);

            Instance._executionProvider = executionProvider;
            bool style = QwenModelPaths.IsCustomVoicePresent();
            bool clone = QwenModelPaths.IsBasePresent();
            Instance._initialized = style || clone;
            if (!Instance._initialized)
            {
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] Qwen3-TTS files missing. Style: " +
                    $"{QwenModelPaths.Root}; clone: {QwenModelPaths.BaseRoot}");
                return;
            }

            TTSLogger.Log(
                $"[CharacterVoiceFactory] Initialized Qwen3-TTS (style={style}, clone={clone}), " +
                $"MemoryUsage: {memoryUsage}, ExecutionProvider: {executionProvider}");
        }

        /// <summary>
        /// Waits for models to be ready. Constructs the engine (tokenizer / embeddings).
        /// Large ONNX sessions stay deferred until the first generate or clone extract.
        /// </summary>
        public static async Task WaitForModelsLoadedAsync()
        {
            if (!Instance._initialized)
            {
                throw new InvalidOperationException("CharacterVoiceFactory is not initialized");
            }

            TTSLogger.Log("[CharacterVoiceFactory] Loading Qwen3-TTS...");
            Instance.EnsureEngine();
            TTSLogger.Log("[CharacterVoiceFactory] Qwen3-TTS ready");
            await Task.Yield();
        }

        /// <summary>
        /// Creates a character voice using style-based generation with specified voice parameters.
        /// </summary>
        public async Task<CharacterVoice> CreateFromStyleAsync(string gender, string pitch, string speed, string referenceText = "I am a character voice")
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            if (!QwenModelPaths.IsCustomVoicePresent())
            {
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] CreateFromStyleAsync needs CustomVoice at " +
                    QwenModelPaths.Root);
                return null;
            }

            if (string.IsNullOrEmpty(gender))
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Gender parameter is required for style-based voices.");
                return null;
            }

            try
            {
                EnsureEngine();
                CharacterVoice voice = new(
                    _engine,
                    gender: gender.ToLower(),
                    pitch: pitch?.ToLower() ?? "moderate",
                    speed: speed?.ToLower() ?? "moderate",
                    referenceText: referenceText
                );

                await voice.GenerateVoiceAsync(referenceText);
                return voice;
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception creating voice from style: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        public async Task<CharacterVoice> CreateFromFolderAsync(string voiceFolder)
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            if (!QwenModelPaths.IsCustomVoicePresent())
            {
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] CreateFromFolderAsync needs CustomVoice at " +
                    QwenModelPaths.Root);
                return null;
            }

            try
            {
                EnsureEngine();
                CharacterVoice voice = new(_engine);
                await voice.LoadVoiceAsync(voiceFolder);
                return voice;
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception creating voice from folder: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        public CharacterVoice CreateFromReference(AudioClip referenceClip)
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            if (!QwenModelPaths.IsBasePresent())
            {
                var missing = QwenModelPaths.GetMissingBaseFiles();
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] Voice cloning needs Qwen3-TTS 1.7B Base ONNX at " +
                    $"{QwenModelPaths.BaseRoot} (missing {missing.Count} file(s)). " +
                    "Source: https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL — copy onnx_kv/*.onnx " +
                    "and models/Qwen3-TTS-12Hz-1.7B-Base/{config.json,vocab.json,merges.txt}. " +
                    "Do not use qwen3_tts_rust.dll (Windows-only).");
                return null;
            }

            if (referenceClip == null)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Reference clip is null.");
                return null;
            }

            try
            {
                EnsureEngine();
                float[] samples = QwenTtsEngine.ClipToMono24k(referenceClip);
                float[] embedding = _engine.ExtractSpeakerEmbedding(samples);
                return new CharacterVoice(_engine, referenceClip, embedding);
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception cloning from reference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        private void EnsureEngine()
        {
            if (_engine != null)
                return;

            _engine = new QwenTtsEngine(_executionProvider);
            TTSLogger.LogVerbose(
                $"[CharacterVoiceFactory] Qwen engine ready (EP {_executionProvider}, " +
                $"style={_engine.HasCustomVoice}, clone={_engine.HasClone})");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _engine?.Dispose();
                _engine = null;
                _initialized = false;
                _disposed = true;
            }

            GC.SuppressFinalize(this);
        }

        ~CharacterVoiceFactory()
        {
            Dispose();
        }
    }
}
