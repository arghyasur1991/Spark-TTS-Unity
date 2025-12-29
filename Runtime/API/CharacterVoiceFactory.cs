using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace SparkTTS
{
    using Core;
    using Models;
    using Utils;
    /// <summary>
    /// Factory class for creating CharacterVoice objects using either voice cloning or style-based generation.
    /// </summary>
    public class CharacterVoiceFactory : IDisposable
    {
        public static CharacterVoiceFactory Instance { get; private set; } = new();

        public bool LogTiming { get => SparkTTS.LogTiming; set => SparkTTS.LogTiming = value; }
        
        /// <summary>
        /// Gets whether the SparkTTS engine is initialized and ready for use.
        /// </summary>
        public static bool IsReady => Instance._initialized && !Instance._disposed;
        
        private SparkTTS _sparkTts;
        private bool _disposed = false;
        private bool _initialized = false;

        /// <summary>
        /// Initializes a new instance of the CharacterVoiceFactory.
        /// </summary>
        internal CharacterVoiceFactory()
        {
            var initConfig = new TTSInferenceConfig();
            _sparkTts = new SparkTTS(initConfig);
            _initialized = _sparkTts.IsInitialized;
        }

        /// <summary>
        /// Initializes or re-initializes the CharacterVoiceFactory with the specified settings.
        /// </summary>
        /// <param name="logLevel">The logging level.</param>
        /// <param name="memoryUsage">The memory usage mode (Performance, Balanced, or Optimal).</param>
        /// <param name="executionProvider">The execution provider to use (CPU, CUDA, or CoreML).</param>
        public static void Initialize(LogLevel logLevel, MemoryUsage memoryUsage, ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            Logger.LogLevel = logLevel;
            ORTModel.InitializeEnvironment(logLevel);
            ORTModel.SetMemoryUsage(memoryUsage);
            
            Instance._sparkTts.SetExecutionProvider(executionProvider);
            
            Logger.Log($"[CharacterVoiceFactory] Initialized with MemoryUsage: {memoryUsage}, ExecutionProvider: {executionProvider}");
        }

        /// <summary>
        /// Waits for SparkTTS models to be ready.
        /// In Performance mode, waits for all models to finish loading.
        /// In other modes, just verifies initialization.
        /// </summary>
        /// <returns>A task that completes when ready</returns>
        public static async Task WaitForModelsLoadedAsync()
        {
            if (!Instance._initialized)
            {
                throw new InvalidOperationException("CharacterVoiceFactory is not initialized");
            }
            
            // In Performance mode, wait for all models to load
            if (ORTModel.CurrentMemoryUsage == MemoryUsage.Performance)
            {
                Logger.Log("[CharacterVoiceFactory] Waiting for all models to load (Performance mode)...");
                await Instance._sparkTts.WaitForAllModelsAsync();
                Logger.Log("[CharacterVoiceFactory] All models loaded");
            }
        }
        
        /// <summary>
        /// Creates a character voice using style-based generation with specified voice parameters.
        /// </summary>
        /// <param name="gender">The gender parameter (e.g., "female", "male")</param>
        /// <param name="pitch">The pitch parameter (e.g., "low", "moderate", "high")</param>
        /// <param name="speed">The speed parameter (e.g., "low", "moderate", "high")</param>
        /// <param name="referenceText">Optional text to pre-generate for caching tokens</param>
        /// <returns>A CharacterVoice instance or null if creation fails</returns>
        public async Task<CharacterVoice> CreateFromStyleAsync(string gender, string pitch, string speed, string referenceText = "I am a character voice")
        {
            if (!_initialized || _disposed)
            {
                Logger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }
            
            if (string.IsNullOrEmpty(gender))
            {
                Logger.LogError("[CharacterVoiceFactory] Gender parameter is required for style-based voices.");
                return null;
            }
            
            try
            {
                CharacterVoice voice = new(
                    _sparkTts,
                    gender: gender.ToLower(),
                    pitch: pitch?.ToLower() ?? "moderate",
                    speed: speed?.ToLower() ?? "moderate",
                    referenceText: referenceText
                );

                await voice.GenerateVoiceAsync(referenceText);
                _sparkTts.DisposeGeneratorOnlyModels();
                return voice;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterVoiceFactory] Exception creating voice from style: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
        public async Task<CharacterVoice> CreateFromFolderAsync(string voiceFolder)
        {
            if (!_initialized || _disposed)
            {
                Logger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }
            
            try
            {
                CharacterVoice voice = new(
                    _sparkTts
                );

                await voice.LoadVoiceAsync(voiceFolder);
                return voice;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterVoiceFactory] Exception creating voice from reference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
        public CharacterVoice CreateFromReference(AudioClip referenceClip)
        {
            if (!_initialized || _disposed)
            {
                Logger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }
            
            try
            {
                CharacterVoice voice = new(
                    _sparkTts,
                    referenceClip
                );
                
                return voice;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterVoiceFactory] Exception creating voice from reference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
        public void Dispose()
        {
            if (!_disposed)
            {
                _sparkTts?.Dispose();
                _sparkTts = null;
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