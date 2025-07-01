using System.IO;

namespace SparkTTS.Core
{
    internal static class SparkTTSModelPaths
    {
        // Base path relative to Application.streamingAssetsPath
        // e.g., if your models are in StreamingAssets/MyModels/SparkTTS_Subfolder, then BaseSparkTTSPath = "MyModels"
        // For now, let's assume they are directly under a "SparkTTS" folder in StreamingAssets for simplicity,
        // matching the example structure like "SparkTTS/LLM_ONNX/model.onnx"
        public const string BaseSparkTTSPathInStreamingAssets = "SparkTTS";

        /// <summary>
        /// Gets the full, platform-dependent path to a model file within the SparkTTS model structure.
        /// </summary>
        /// <param name="modelFolder">The subfolder for the specific model component (e.g., "LLM_ONNX", "SpeakerEncoder_ONNX").</param>
        /// <param name="modelFileName">The actual model file name (e.g., "model.onnx", "speaker_encoder_tokenizer.onnx").</param>
        /// <returns>The full path to the model file.</returns>
        public static string GetModelPath(string modelFolder, string modelFileName)
        {
            return Path.Combine(BaseSparkTTSPathInStreamingAssets, modelFolder, modelFileName);
        }

        // --- Define constants for your model subfolders and filenames here for easy management ---
        // Example: LLM
        public const string LLMFolder = "LLM";
        public const string LLMFile = "model.onnx"; // Assumes model.onnx and model.onnx_data are co-located by ORT

        // Example: Speaker Encoder
        public const string SpeakerEncoderFolder = ""; // You'll need to create/confirm this folder
        public const string SpeakerEncoderFile = "speaker_encoder_tokenizer.onnx";

        // Example: Vocoder (assuming a single ONNX for the BiCodec vocoder part)
        public const string VocoderFolder = ""; // You'll need to create/confirm this folder
        public const string VocoderFile = "bicodec_vocoder.onnx";

        // Example: Mel Spectrogram Generator
        public const string MelSpectrogramFolder = ""; // You'll need to create/confirm this folder
        public const string MelSpectrogramFile = "mel_spectrogram.onnx";

        // Example: Wav2Vec2
        public const string Wav2Vec2Folder = ""; // Assuming it's in the root of StreamingAssets/SparkTTS
        public const string Wav2Vec2File = "wav2vec2_model.onnx"; // Based on export script output name

        // Added for BiCodec Encoder/Quantizer
        public const string BiCodecEncoderQuantizerFolder = ""; 
        public const string BiCodecEncoderQuantizerFile = "bicodec_encoder_quantizer.onnx";

        // TODO: Add other model paths as needed (e.g., for BiCodec internal encoder/quantizer if they become separate ONNX files)
    }
} 