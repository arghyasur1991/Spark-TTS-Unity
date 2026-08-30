using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Local layout for Qwen3-TTS 1.7B Base ONNX (voice cloning).
    /// Single-file ONNX from zukky/Qwen3-TTS-ONNX-DLL onnx_kv/ plus the matching tokenizer/config.
    /// This package never downloads them. Do not ship the Windows DLL — Mac/Android use C# here.
    /// </summary>
    public static class QwenBaseModelPaths
    {
        public const string FolderName = "Qwen3-1.7B-Base";

        public static string RelativeDir =>
            Path.Combine(SparkTTS.Core.SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets, FolderName);

        public static string Root =>
            Path.Combine(Application.streamingAssetsPath, RelativeDir);

        /// <summary>
        /// Required files for x-vector cloning (Spark CreateFromReference).
        /// tokenizer12hz_encode.onnx is ICL-only and not required.
        /// </summary>
        public static IReadOnlyList<string> ExpectedOnnxFiles
        {
            get
            {
                return new[]
                {
                    "talker_prefill.onnx",
                    "talker_decode.onnx",
                    "code_predictor.onnx",
                    "code_predictor_embed.onnx",
                    "codec_embed.onnx",
                    "text_project.onnx",
                    "speaker_encoder.onnx",
                    "tokenizer12hz_decode.onnx",
                };
            }
        }

        public static string ConfigPath => Path.Combine(Root, "config.json");

        public static string TokenizerDir
        {
            get
            {
                string nested = Path.Combine(Root, "tokenizer");
                if (File.Exists(Path.Combine(nested, "vocab.json")))
                    return nested;
                return Root;
            }
        }

        public static bool IsPresent() => GetMissingFiles().Count == 0;

        public static List<string> GetMissingFiles()
        {
            var missing = new List<string>();
            string root = Root;
            foreach (var rel in ExpectedOnnxFiles)
            {
                if (!File.Exists(Path.Combine(root, rel)))
                    missing.Add(rel);
            }

            if (!File.Exists(ConfigPath))
                missing.Add("config.json");

            if (!File.Exists(Path.Combine(TokenizerDir, "vocab.json")))
                missing.Add("vocab.json");
            if (!File.Exists(Path.Combine(TokenizerDir, "merges.txt")))
                missing.Add("merges.txt");

            return missing;
        }
    }
}
