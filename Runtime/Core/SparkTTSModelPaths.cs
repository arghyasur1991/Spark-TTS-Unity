namespace SparkTTS.Core
{
    /// <summary>
    /// Centralized model path constants for SparkTTS ONNX models.
    /// Provides consistent path management for all model components.
    /// </summary>
    internal static class SparkTTSModelPaths
    {
        // Base path relative to Application.streamingAssetsPath
        public const string BaseSparkTTSPathInStreamingAssets = "SparkTTS";

        // Model folder constants
        public const string LLMFolder = "LLM";
        public const string SpeakerEncoderFolder = "SpeakerEncoder";
        public const string VocoderFolder = "Vocoder";
        public const string MelSpectrogramFolder = "MelSpectrogram";
        public const string Wav2Vec2Folder = "Wav2Vec2";
        public const string BiCodecFolder = "BiCodec";

        // Model file name constants (without extension)
        public const string LLMModelName = "model";
        public const string SpeakerEncoderModelName = "speaker_encoder_tokenizer";
        public const string VocoderModelName = "bicodec_vocoder";
        public const string MelSpectrogramModelName = "mel_spectrogram";
        public const string Wav2Vec2ModelName = "wav2vec2_model";
        public const string BiCodecEncoderQuantizerModelName = "bicodec_encoder_quantizer";
    }
} 