using System;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS
{
    using Qwen;
    using Utils;
    /// <summary>
    /// Represents a character voice with an associated output clip and/or voice style parameters.
    /// Can generate speech from text using either voice cloning or style-based generation.
    /// </summary>
    public class CharacterVoice : IDisposable
    {
        public AudioClip ReferenceClip { get => GetReferenceClip(); private set => _referenceClip = value; }
        private AudioClip _referenceClip;
        public string Gender { get; private set; }
        public string Pitch { get; private set; }
        public string Speed { get; private set; }
        public string Instruct { get; private set; }

        private AudioClip _lastGeneratedClip;
        private readonly QwenTtsEngine _engine;
        private readonly float[] _speakerEmbedding;
        private bool _disposed;
        private float[] _referenceWaveform;

        internal CharacterVoice(
            QwenTtsEngine engine,
            string referenceText,
            string gender,
            string pitch,
            string speed,
            string instruct = null)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
            Gender = gender.ToLower();
            Pitch = pitch.ToLower();
            Speed = speed.ToLower();
            Instruct = instruct;
        }

        internal CharacterVoice(QwenTtsEngine engine)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        }

        internal CharacterVoice(QwenTtsEngine engine, AudioClip referenceClip, float[] speakerEmbedding)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
            _speakerEmbedding = speakerEmbedding ?? throw new ArgumentNullException(nameof(speakerEmbedding));
            _referenceClip = referenceClip;
        }

        internal async Task LoadVoiceAsync(string voiceFolder)
        {
            string configPath = Path.Combine(voiceFolder, "voice_config.json");
            string configJson = File.ReadAllText(configPath);
            var voiceConfig = JsonConvert.DeserializeObject<VoiceConfig>(configJson);
            Gender = voiceConfig.gender;
            Pitch = voiceConfig.pitch;
            Speed = voiceConfig.speed;
            Instruct = voiceConfig.instruct;

            string audioFilePath = Path.Combine(voiceFolder, voiceConfig.audioFile ?? "sample.wav");
            if (File.Exists(audioFilePath))
            {
                ReferenceClip = await AudioLoaderService.LoadAudioClipAsync(audioFilePath);
            }

            await Task.CompletedTask;
        }

        public Task GenerateVoiceAsync(string referenceText)
        {
            // Qwen CustomVoice has no Spark-style global-token cache.
            return Task.CompletedTask;
        }

        public async Task SaveVoiceAsync(string voiceFolder)
        {
            Directory.CreateDirectory(voiceFolder);
            if (ReferenceClip != null)
            {
                string samplePath = Path.Combine(voiceFolder, "sample.wav");
                await AudioLoaderService.SaveAudioClipToFile(ReferenceClip, samplePath);
                TTSLogger.LogVerbose($"[Character] Voice sample saved to: {samplePath}");
            }

            var voiceConfig = new
            {
                gender = Gender,
                pitch = Pitch,
                speed = Speed,
                instruct = Instruct,
                clone = _speakerEmbedding != null,
                timestamp = DateTime.UtcNow,
                audioFile = "sample.wav",
                sampleRate = ReferenceClip != null ? ReferenceClip.frequency : QwenTtsEngine.NativeSampleRate,
                channels = ReferenceClip != null ? ReferenceClip.channels : 1,
                length = ReferenceClip != null ? ReferenceClip.length : 0f
            };

            string configPath = Path.Combine(voiceFolder, "voice_config.json");
            string configJson = JsonConvert.SerializeObject(voiceConfig, Formatting.Indented);
            await File.WriteAllTextAsync(configPath, configJson);
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
                TTSLogger.LogError("[CharacterVoice.GenerateSpeech] Object has been disposed.");
                return null;
            }

            if (string.IsNullOrEmpty(text))
            {
                TTSLogger.LogError("[CharacterVoice.GenerateSpeech] Input text is null or empty.");
                return null;
            }

            try
            {
                TTSLogger.Log($"[CharacterVoice.GenerateSpeech] Generating speech for text: {text}");

                if (_engine == null)
                {
                    TTSLogger.LogError("[CharacterVoice.GenerateSpeech] No TTS engine on this voice.");
                    return null;
                }

                // One worker: preload + synth. Do not await those pieces separately —
                // each await would resume on the Unity sync context and deadlock
                // EnsureLoaded().GetResult() against session construct.
                float[] pcm24;
                if (_speakerEmbedding != null)
                {
                    var embedding = _speakerEmbedding;
                    pcm24 = await BackgroundWork.Run(
                        () => _engine.SynthesizeClone(text, embedding, QwenStyleMap.DefaultLanguage));
                }
                else
                {
                    string instruct = QwenStyleMap.VoiceDesignInstruct(Gender, Pitch, Speed, Instruct);
                    pcm24 = await BackgroundWork.Run(
                        () => _engine.Synthesize(text, speaker: null, QwenStyleMap.DefaultLanguage, instruct));
                }

                if (pcm24 == null || pcm24.Length == 0)
                {
                    TTSLogger.LogError("[CharacterVoice.GenerateSpeech] Speech generation failed: empty waveform");
                    return null;
                }

                int rate = sampleRate <= 0 ? QwenTtsEngine.NativeSampleRate : sampleRate;
                float[] pcm = rate == QwenTtsEngine.NativeSampleRate
                    ? pcm24
                    : AudioResample.Resample(pcm24, QwenTtsEngine.NativeSampleRate, rate);

                AudioClip clip = AudioClip.Create(
                    $"CharacterVoice_{DateTime.Now.Ticks}",
                    pcm.Length,
                    1,
                    rate,
                    false);

                clip.SetData(pcm, 0);
                _lastGeneratedClip = clip;
                return clip;
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoice.GenerateSpeech] Exception: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

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
                    1,
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
        public string instruct;
        public bool clone;
        public string timestamp;
        public string audioFile;
        public int sampleRate;
        public int channels;
        public float length;
    }
}
