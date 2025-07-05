using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

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

        internal CharacterVoice(SparkTTS sparkTts)
        {
            _sparkTts = sparkTts ?? throw new ArgumentNullException(nameof(sparkTts));
        }

        internal async Task LoadVoiceAsync(string voiceFolder)
        {
            string configPath = Path.Combine(voiceFolder, "voice_config.json");
            string configJson = File.ReadAllText(configPath);
            var voiceConfig = JsonConvert.DeserializeObject<VoiceConfig>(configJson);
            Gender = voiceConfig.gender;
            Pitch = voiceConfig.pitch;
            Speed = voiceConfig.speed;
            
            // Load the audio file
            string audioFilePath = Path.Combine(voiceFolder, voiceConfig.audioFile);
            ReferenceClip = await AudioLoaderService.LoadAudioClipAsync(audioFilePath);
            _referenceWaveform = _sparkTts.LoadAudioClip(ReferenceClip, 16000);

            // Load the global tokens
            string globalTokensPath = Path.Combine(voiceFolder, "global_tokens.bin");
            using (var stream = new MemoryStream(File.ReadAllBytes(globalTokensPath)))
            {
                using (var reader = new BinaryReader(stream))
                {
                    _cachedGlobalTokenIds = new int[reader.ReadInt32()];
                    for (int i = 0; i < _cachedGlobalTokenIds.Length; i++)
                    {
                        _cachedGlobalTokenIds[i] = reader.ReadInt32();
                    }
                }
            }

            // Load the model inputs
            string modelInputsPath = Path.Combine(voiceFolder, "model_inputs.bin");
            using (var stream = new MemoryStream(File.ReadAllBytes(modelInputsPath)))
            {
                using (var reader = new BinaryReader(stream))
                {
                    _cachedModelInputs = new TokenizationOutput();
                    _cachedModelInputs.InputIds = new List<int>();
                    _cachedModelInputs.AttentionMask = new List<int>();
                    int inputIdsCount = reader.ReadInt32();
                    for (int i = 0; i < inputIdsCount; i++)
                    {
                        _cachedModelInputs.InputIds.Add(reader.ReadInt32());
                    }
                    int attentionMaskCount = reader.ReadInt32();
                    for (int i = 0; i < attentionMaskCount; i++)
                    {
                        _cachedModelInputs.AttentionMask.Add(reader.ReadInt32());
                    }
                }
            }
        }

        public async Task GenerateVoiceAsync(string referenceText)
        {
            var result = await _sparkTts.InferenceAsync(referenceText, null, null, Gender, Pitch, Speed);
            
            // Store voice parameters
            _referenceWaveform = result.Waveform;

            // Store pre-generated tokens if provided
            _cachedGlobalTokenIds = result.GlobalTokenIds;
            _cachedModelInputs = result.ModelInputs;
        }

        public async Task SaveVoiceAsync(string voiceFolder)
        {
            if (ReferenceClip != null)
            {
                // Convert AudioClip to WAV and save
                string samplePath = Path.Combine(voiceFolder, "sample.wav");
                await AudioLoaderService.SaveAudioClipToFile(ReferenceClip, samplePath);
                Logger.LogVerbose($"[Character] Voice sample saved to: {samplePath}");

                // Also save voice config for reference
                var voiceConfig = new
                {
                    gender = Gender,
                    pitch = Pitch,
                    speed = Speed,
                    timestamp = DateTime.UtcNow,
                    audioFile = "sample.wav",
                    sampleRate = ReferenceClip.frequency,
                    channels = ReferenceClip.channels,
                    length = ReferenceClip.length
                };
                
                string configPath = Path.Combine(voiceFolder, "voice_config.json");
                string configJson = JsonConvert.SerializeObject(voiceConfig, Formatting.Indented);
                await File.WriteAllTextAsync(configPath, configJson);
            }
            if (_cachedGlobalTokenIds != null)
            {
                // Dump global tokens to a file
                string globalTokensPath = Path.Combine(voiceFolder, "global_tokens.bin");
                using (var stream = new MemoryStream())
                {
                    using (var writer = new BinaryWriter(stream))
                    {
                        writer.Write(_cachedGlobalTokenIds.Length);
                        foreach (var token in _cachedGlobalTokenIds)
                        {
                            writer.Write(token);
                        }
                    }
                    File.WriteAllBytes(globalTokensPath, stream.ToArray());
                }

                // Dump model inputs to a file
                string modelInputsPath = Path.Combine(voiceFolder, "model_inputs.bin");
                using (MemoryStream stream = new())
                {
                    using BinaryWriter writer = new(stream);
                    writer.Write(_cachedModelInputs.InputIds.Count);
                    foreach (var id in _cachedModelInputs.InputIds)
                    {
                        writer.Write(id);
                    }
                    writer.Write(_cachedModelInputs.AttentionMask.Count);
                    foreach (var mask in _cachedModelInputs.AttentionMask)
                    {
                        writer.Write(mask);
                    }
                    File.WriteAllBytes(modelInputsPath, stream.ToArray());
                }
            }
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
                bool useOptimizedGeneration = false; //TODO: fix this
                // Check if we have cached tokens for optimization
                if (useOptimizedGeneration && _cachedModelInputs != null && _cachedGlobalTokenIds != null)
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
                    result = await _sparkTts.InferenceAsync(
                        modelInputs: updatedInputs,
                        globalTokenIds: _cachedGlobalTokenIds);
                    
                    Logger.Log("[CharacterVoice.GenerateSpeech] Used optimized generation with updated tokenized inputs");
                }
                else
                {
                    // Run standard inference for first-time generation
                    result = await _sparkTts.InferenceAsync(text, _referenceWaveform, null, Gender, Pitch, Speed);
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

    internal class VoiceConfig
    {
        public string gender;
        public string pitch;
        public string speed;
        public string timestamp;
        public string audioFile;
        public int sampleRate;
        public int channels;
        public float length;
    }
}
