using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Local layout for Qwen3-TTS 1.7B CustomVoice ONNX files.
    /// Drop files under StreamingAssets/SparkTTS/Qwen3-1.7B/ — this package never downloads them.
    /// </summary>
    public static class QwenModelPaths
    {
        public const string FolderName = "Qwen3-1.7B";

        /// <summary>Relative to Application.streamingAssetsPath.</summary>
        public static string RelativeDir =>
            Path.Combine(SparkTTS.Core.SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets, FolderName);

        public static string Root =>
            Path.Combine(Application.streamingAssetsPath, RelativeDir);

        /// <summary>
        /// Files required by the 1.7B CustomVoice ONNX export
        /// (elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX).
        /// </summary>
        public static IReadOnlyList<string> ExpectedFiles
        {
            get
            {
                var files = new List<string>
                {
                    "talker_prefill.onnx",
                    "talker_prefill.onnx.data",
                    "talker_decode.onnx",
                    "talker_decode.onnx.data",
                    "code_predictor.onnx",
                    "code_predictor.onnx.data",
                    "vocoder.onnx",
                    "vocoder.onnx.data",
                    "embeddings/config.json",
                    "embeddings/talker_codec_embedding.npy",
                    "embeddings/text_embedding.npy",
                    "embeddings/text_projection_fc1_weight.npy",
                    "embeddings/text_projection_fc1_bias.npy",
                    "embeddings/text_projection_fc2_weight.npy",
                    "embeddings/text_projection_fc2_bias.npy",
                    "embeddings/codec_head_weight.npy",
                    "embeddings/speaker_ids.json",
                    "embeddings/cp_projection_weight.npy",
                    "embeddings/cp_projection_bias.npy",
                    "tokenizer/vocab.json",
                    "tokenizer/merges.txt",
                };
                for (int i = 0; i < 15; i++)
                    files.Add($"embeddings/cp_codec_embedding_{i}.npy");
                return files;
            }
        }

        public static bool IsPresent() => GetMissingFiles().Count == 0;

        public static List<string> GetMissingFiles()
        {
            var missing = new List<string>();
            string root = Root;
            foreach (var rel in ExpectedFiles)
            {
                if (!File.Exists(Path.Combine(root, rel)))
                    missing.Add(rel);
            }
            return missing;
        }
    }
}
