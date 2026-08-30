using System.Collections.Generic;
using System.IO;
using SparkTTS.Core;
using UnityEngine;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Local layouts for Qwen3-TTS ONNX under StreamingAssets/SparkTTS/.
    /// This package never downloads weights.
    /// </summary>
    public static class QwenModelPaths
    {
        public const string FolderName = SparkTTSModelPaths.QwenCustomVoiceFolder;
        public const string BaseFolderName = SparkTTSModelPaths.QwenBaseFolder;

        public static string RelativeDir =>
            Path.Combine(SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets, FolderName);

        public static string BaseRelativeDir =>
            Path.Combine(SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets, BaseFolderName);

        public static string Root =>
            Path.Combine(Application.streamingAssetsPath, RelativeDir);

        public static string BaseRoot =>
            Path.Combine(Application.streamingAssetsPath, BaseRelativeDir);

        public static IReadOnlyList<string> ExpectedCustomVoiceFiles
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

        public static IReadOnlyList<string> ExpectedBaseFiles => new[]
        {
            "talker_prefill.onnx",
            "talker_decode.onnx",
            "code_predictor.onnx",
            "code_predictor_embed.onnx",
            "codec_embed.onnx",
            "text_project.onnx",
            "speaker_encoder.onnx",
            "tokenizer12hz_decode.onnx",
            "config.json",
            "vocab.json",
            "merges.txt",
        };

        /// <summary>CustomVoice checklist (style TTS).</summary>
        public static IReadOnlyList<string> ExpectedFiles => ExpectedCustomVoiceFiles;

        public static string BaseConfigPath => Path.Combine(BaseRoot, "config.json");

        public static string BaseTokenizerDir
        {
            get
            {
                string nested = Path.Combine(BaseRoot, "tokenizer");
                if (File.Exists(Path.Combine(nested, "vocab.json")))
                    return nested;
                return BaseRoot;
            }
        }

        public static bool IsCustomVoicePresent() => MissingUnder(Root, ExpectedCustomVoiceFiles, checkVocabAtRoot: false).Count == 0;

        public static bool IsBasePresent() => GetMissingBaseFiles().Count == 0;

        public static bool IsPresent() => IsCustomVoicePresent();

        public static List<string> GetMissingFiles() => MissingUnder(Root, ExpectedCustomVoiceFiles, checkVocabAtRoot: false);

        public static List<string> GetMissingBaseFiles()
        {
            var missing = new List<string>();
            string root = BaseRoot;
            foreach (var rel in new[]
            {
                "talker_prefill.onnx",
                "talker_decode.onnx",
                "code_predictor.onnx",
                "code_predictor_embed.onnx",
                "codec_embed.onnx",
                "text_project.onnx",
                "speaker_encoder.onnx",
                "tokenizer12hz_decode.onnx",
                "config.json",
            })
            {
                if (!File.Exists(Path.Combine(root, rel)))
                    missing.Add(rel);
            }

            if (!File.Exists(Path.Combine(BaseTokenizerDir, "vocab.json")))
                missing.Add("vocab.json");
            if (!File.Exists(Path.Combine(BaseTokenizerDir, "merges.txt")))
                missing.Add("merges.txt");
            return missing;
        }

        private static List<string> MissingUnder(string root, IReadOnlyList<string> files, bool checkVocabAtRoot)
        {
            var missing = new List<string>();
            foreach (var rel in files)
            {
                if (!File.Exists(Path.Combine(root, rel)))
                    missing.Add(rel);
            }
            return missing;
        }
    }
}
