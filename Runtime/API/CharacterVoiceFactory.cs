using System;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
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
            _initialized = QwenModelPaths.IsPresent();
            if (!_initialized)
            {
                var missing = QwenModelPaths.GetMissingFiles();
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Qwen3-TTS 1.7B files are not under " +
                    $"{QwenModelPaths.Root}. Missing {missing.Count} file(s). See QWEN3.md.");
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
            Instance._initialized = QwenModelPaths.IsPresent();
            if (!Instance._initialized)
            {
                var missing = QwenModelPaths.GetMissingFiles();
                TTSLogger.LogError(
                    $"[CharacterVoiceFactory] Qwen3-TTS 1.7B files missing ({missing.Count}). " +
                    $"Place them at {QwenModelPaths.Root}");
                return;
            }

            if (executionProvider != ExecutionProvider.CPU)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Qwen3-TTS on this branch uses CPU SessionOptions. " +
                    $"{executionProvider} is ignored.");
            }

            TTSLogger.Log($"[CharacterVoiceFactory] Initialized Qwen3-TTS 1.7B with MemoryUsage: {memoryUsage}, ExecutionProvider: CPU");
        }

        /// <summary>
        /// Waits for models to be ready. Constructs the Qwen engine (embeddings + tokenizer).
        /// ONNX sessions stay lazy until the first GenerateSpeechAsync.
        /// </summary>
        public static async Task WaitForModelsLoadedAsync()
        {
            if (!Instance._initialized)
            {
                throw new InvalidOperationException("CharacterVoiceFactory is not initialized");
            }

            TTSLogger.Log("[CharacterVoiceFactory] Loading Qwen3-TTS embeddings...");
            Instance.EnsureEngine();
            TTSLogger.Log("[CharacterVoiceFactory] Qwen3-TTS embeddings loaded");
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

            TTSLogger.LogError(
                "[CharacterVoiceFactory] Voice cloning is not supported on Qwen3-TTS CustomVoice 1.7B. " +
                "Use CreateFromStyleAsync. Cloning needs the Base ONNX export.");
            return null;
        }

        private void EnsureEngine()
        {
            if (_engine != null)
                return;
            if (!QwenModelPaths.IsPresent())
            {
                throw new InvalidOperationException(
                    $"Qwen3-TTS 1.7B files not found at {QwenModelPaths.Root}");
            }

            _engine = new QwenTtsEngine(QwenModelPaths.Root, CreateSessionOptions);
            TTSLogger.LogVerbose($"[CharacterVoiceFactory] Qwen engine ready (requested EP {_executionProvider}, using CPU SessionOptions)");
        }

        private SessionOptions CreateSessionOptions()
        {
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = Environment.ProcessorCount,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = true,
                EnableCpuMemArena = true
            };
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
