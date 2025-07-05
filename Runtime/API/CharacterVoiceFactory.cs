using System;
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
        
        private SparkTTS _sparkTts;
        private bool _disposed = false;
        private bool _initialized = false;
        private Task _initializeTask;
        
        /// <summary>
        /// Initializes a new instance of the CharacterVoiceFactory.
        /// </summary>
        internal CharacterVoiceFactory()
        {
            _initializeTask = Task.Run(() => {
                var initConfig = new TTSInferenceConfig();
                _sparkTts = new SparkTTS(initConfig);
                _initialized = _sparkTts.IsInitialized;                
                if (!_initialized)
                {
                    Logger.LogError("[CharacterVoiceFactory] Failed to initialize TTSInferenceOrchestrator.");
                }
            });
        }

        public static void Initialize(LogLevel logLevel = LogLevel.INFO)
        {
            Logger.LogLevel = logLevel;
            ORTModel.InitializeEnvironment(logLevel);
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
            await _initializeTask;
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
                return voice;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterVoiceFactory] Exception creating voice from style: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
        public async Task<CharacterVoice> CreateFromReferenceAsync(AudioClip referenceClip)
        {
            await _initializeTask;
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