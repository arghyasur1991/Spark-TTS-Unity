using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;

namespace SparkTTS.Qwen
{
    internal sealed class QwenBaseConfig
    {
        public int TtsBosTokenId;
        public int TtsEosTokenId;
        public int TtsPadTokenId;
        public int HiddenSize;
        public int VocabSize;
        public int NumCodeGroups;
        public int CodecBosId;
        public int CodecEosTokenId;
        public int CodecPadId;
        public int CodecThinkId;
        public int CodecNothinkId;
        public int CodecThinkBosId;
        public int CodecThinkEosId;
        public Dictionary<string, int> CodecLanguageId;

        public static QwenBaseConfig Load(string configPath)
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException("Qwen3-TTS Base config.json not found.", configPath);

            var raw = JObject.Parse(File.ReadAllText(configPath));
            var talker = raw["talker_config"] as JObject;
            if (talker == null)
                throw new InvalidDataException("config.json is missing talker_config.");

            var cfg = new QwenBaseConfig
            {
                TtsBosTokenId = raw.Value<int?>("tts_bos_token_id") ?? 151672,
                TtsEosTokenId = raw.Value<int?>("tts_eos_token_id") ?? 151673,
                TtsPadTokenId = raw.Value<int?>("tts_pad_token_id") ?? 151671,
                HiddenSize = talker.Value<int?>("hidden_size") ?? 2048,
                VocabSize = talker.Value<int?>("vocab_size") ?? 3072,
                NumCodeGroups = talker.Value<int?>("num_code_groups") ?? 16,
                CodecBosId = talker.Value<int?>("codec_bos_id") ?? 2149,
                CodecEosTokenId = talker.Value<int?>("codec_eos_token_id") ?? 2150,
                CodecPadId = talker.Value<int?>("codec_pad_id") ?? 2148,
                CodecThinkId = talker.Value<int?>("codec_think_id") ?? 2154,
                CodecNothinkId = talker.Value<int?>("codec_nothink_id") ?? 2155,
                CodecThinkBosId = talker.Value<int?>("codec_think_bos_id") ?? 2156,
                CodecThinkEosId = talker.Value<int?>("codec_think_eos_id") ?? 2157,
                CodecLanguageId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
            };

            var lang = talker["codec_language_id"] as JObject;
            if (lang != null)
            {
                foreach (var p in lang.Properties())
                    cfg.CodecLanguageId[p.Name.ToLowerInvariant()] = p.Value.Value<int>();
            }

            return cfg;
        }
    }
}
