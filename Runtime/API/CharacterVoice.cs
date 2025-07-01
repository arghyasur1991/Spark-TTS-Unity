using UnityEngine;
using System.Threading.Tasks;
using System;

namespace SparkTTS
{
    using Core;
    using Utils;
    /// <summary>
    /// Represents a character voice with an associated output clip and/or voice style parameters.
    /// Can generate speech from text using either voice cloning or style-based generation.
    /// </summary>
    public class CharacterVoice : IDisposable
    {
        // Voice identity parameters
        public AudioClip ReferenceClip { get => GetReferenceClip(); private set => _referenceClip = value; }
        private AudioClip _referenceClip;
        public string Gender { get; private set; }
        public string Pitch { get; private set; }
        public string Speed { get; private set; }
        
        // Cached voice data for optimization
        private int[] _cachedGlobalTokenIds = null;
        private TokenizationOutput _cachedModelInputs = null;
        private AudioClip _lastGeneratedClip = null;
        
        // Orchestrator for TTS generation
        private readonly SparkTTS _sparkTts;
        private bool _disposed = false;
        private float[] _referenceWaveform = null;
        
        public static DebugLogger Logger = new();
        
        // Constructor - Private to enforce use of factory
        internal CharacterVoice(
            SparkTTS sparkTts,
            string referenceText,
            string gender, 
            string pitch, 
            string speed)
        {
            _sparkTts = sparkTts ?? throw new ArgumentNullException(nameof(sparkTts));
            Gender = gender.ToLower();
            Pitch = pitch.ToLower();
            Speed = speed.ToLower();            
        }

        internal CharacterVoice(SparkTTS sparkTts, AudioClip referenceClip)
        {
            _sparkTts = sparkTts ?? throw new ArgumentNullException(nameof(sparkTts));
            ReferenceClip = referenceClip;
            
            // Store voice parameters
            _referenceWaveform = _sparkTts.LoadAudioClip(referenceClip, 16000);
        }

        public async Task GenerateVoiceAsync(string referenceText)
        {
            var result = await Task.Run(() => _sparkTts.Inference(referenceText, null, null, Gender, Pitch, Speed));
            
            // Store voice parameters
            _referenceWaveform = result.Waveform;

            // Store pre-generated tokens if provided
            _cachedGlobalTokenIds = result.GlobalTokenIds;
            _cachedModelInputs = result.ModelInputs;
        }
        
        /// <summary>
        /// Generates speech for the given text using the character's voice.
        /// </summary>
        /// <param name="text">The text to convert to speech</param>
        /// <param name="sampleRate">Target sample rate for the generated audio</param>
        /// <returns>An AudioClip containing the generated speech</returns>
        public async Task<AudioClip> GenerateSpeechAsync(string text, int sampleRate = 16000)
        {
            if (_disposed)
            {
                Logger.LogError("[CharacterVoice.GenerateSpeech] Object has been disposed.");
                return null;
            }
            
            if (string.IsNullOrEmpty(text))
            {
                Logger.LogError("[CharacterVoice.GenerateSpeech] Input text is null or empty.");
                return null;
            }
            
            try
            {       
                Logger.Log($"[CharacterVoice.GenerateSpeech] Generating speech for text: {text}");
                
                TTSInferenceResult result;
                
                // Check if we have cached tokens for optimization
                if (_cachedModelInputs != null && _cachedGlobalTokenIds != null)
                {
                    // Use the more efficient update method if we already have tokenized inputs
                    TokenizationOutput updatedInputs = _sparkTts.UpdateTextInTokenizedInputs(
                        _cachedModelInputs,
                        text,
                        false,
                        Gender,
                        Pitch,
                        Speed);
                    
                    // Run inference with updated inputs and cached global tokens
                    result = await Task.Run(() => _sparkTts.Inference(
                        modelInputs: updatedInputs,
                        globalTokenIds: _cachedGlobalTokenIds));
                    
                    Logger.Log("[CharacterVoice.GenerateSpeech] Used optimized generation with updated tokenized inputs");
                }
                else
                {
                    // Run standard inference for first-time generation
                    result = await Task.Run(() => _sparkTts.Inference(text, _referenceWaveform, null, Gender, Pitch, Speed));
                    Logger.Log("[CharacterVoice.GenerateSpeech] Used standard generation path");
                }
                
                if (!result.Success || result.Waveform == null || result.Waveform.Length == 0)
                {
                    Logger.LogError($"[CharacterVoice.GenerateSpeech] Speech generation failed: {result.ErrorMessage}");
                    return null;
                }
                
                // Create and store the audio clip
                AudioClip clip = AudioClip.Create(
                    $"CharacterVoice_{DateTime.Now.Ticks}", 
                    result.Waveform.Length, 
                    1, // Mono
                    sampleRate, 
                    false);
                
                clip.SetData(result.Waveform, 0);
                _lastGeneratedClip = clip;
                
                // Cache tokens for future optimizations if they're not already cached
                if (_cachedGlobalTokenIds == null && result.GlobalTokenIds != null)
                {
                    _cachedGlobalTokenIds = result.GlobalTokenIds;
                    Logger.Log("[CharacterVoice.GenerateSpeech] Cached global tokens for future optimizations");
                }
                
                if (_cachedModelInputs == null && result.ModelInputs != null)
                {
                    _cachedModelInputs = result.ModelInputs;
                    Logger.Log("[CharacterVoice.GenerateSpeech] Cached model inputs for future optimizations");
                }
                
                return clip;
            }
            catch (Exception e)
            {
                Logger.LogError($"[CharacterVoice.GenerateSpeech] Exception: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
        
        /// <summary>
        /// Gets the last generated audio clip.
        /// </summary>
        public AudioClip GetLastGeneratedClip()
        {
            if (_lastGeneratedClip == null)
            {
                return ReferenceClip;
            }
            return _lastGeneratedClip;
        }

        public AudioClip GetReferenceClip()
        {
            if (_referenceClip == null && _referenceWaveform != null)
            {
                _referenceClip = AudioClip.Create(
                    $"CharacterVoice_{DateTime.Now.Ticks}", 
                    _referenceWaveform.Length, 
                    1, // Mono
                    16000, 
                    false);
                _referenceClip.SetData(_referenceWaveform, 0);
            }
            return _referenceClip;
        }
        
        public void Dispose()
        {
            if (!_disposed)
            {
                // Don't dispose the orchestrator as it might be shared
                // between multiple character voices
                
                _cachedModelInputs = null;
                _cachedGlobalTokenIds = null;
                _lastGeneratedClip = null;
                
                _disposed = true;
            }
            
            GC.SuppressFinalize(this);
        }
        
        ~CharacterVoice()
        {
            Dispose();
        }
    }
} 