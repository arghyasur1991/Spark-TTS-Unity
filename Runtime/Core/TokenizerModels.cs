using Newtonsoft.Json;
using System.Collections.Generic;

namespace SparkTTS.Core
{
    internal class TokenizerDefinition
    {
        [JsonProperty("version")]
        public string Version { get; set; }

        [JsonProperty("truncation")]
        public TruncationParams Truncation { get; set; }

        [JsonProperty("padding")]
        public PaddingParams Padding { get; set; }

        [JsonProperty("added_tokens")]
        public List<AddedToken> AddedTokens { get; set; }

        [JsonProperty("normalizer")]
        public NormalizerConfig Normalizer { get; set; }

        [JsonProperty("pre_tokenizer")]
        public PreTokenizer PreTokenizer { get; set; } // Can be single or sequence

        [JsonProperty("model")]
        public TokenizerModel Model { get; set; }

        [JsonProperty("decoder")]
        public DecoderConfig Decoder { get; set; }

        [JsonProperty("post_processor")]
        public PostProcessorConfig PostProcessor { get; set; }
    }

    internal class TruncationParams
    {
        [JsonProperty("max_length")]
        public int? MaxLength { get; set; }
        [JsonProperty("strategy")]
        public string Strategy { get; set; }
        [JsonProperty("stride")]
        public int? Stride { get; set; }
    }

    internal class PaddingParams
    {
        [JsonProperty("direction")]
        public string Direction { get; set; }
        [JsonProperty("pad_to_multiple_of")]
        public int? PadToMultipleOf { get; set; }
        [JsonProperty("pad_id")]
        public int? PadId { get; set; }
        [JsonProperty("pad_token")]
        public string PadToken { get; set; }
        [JsonProperty("pad_type_id")]
        public int? PadTypeId { get; set; }
    }

    internal class AddedToken
    {
        [JsonProperty("id")]
        public int Id { get; set; }
        [JsonProperty("content")]
        public string Content { get; set; }
        [JsonProperty("single_word")]
        public bool SingleWord { get; set; }
        [JsonProperty("lstrip")]
        public bool LStrip { get; set; }
        [JsonProperty("rstrip")]
        public bool RStrip { get; set; }
        [JsonProperty("normalized")]
        public bool Normalized { get; set; }
        [JsonProperty("special")]
        public bool Special { get; set; }
    }

    internal class NormalizerConfig // Renamed from Normalizer
    {
        [JsonProperty("type")]
        public string Type { get; set; }
        [JsonProperty("lowercase")]
        public bool? Lowercase { get; set; }
        // Add other normalizer-specific fields if needed
    }

    // Represents the overall pre_tokenizer block, which could be a single pre_tokenizer or a sequence
    internal class PreTokenizer
    {
        [JsonProperty("type")]
        public string Type { get; set; } // "Sequence" or a specific type like "ByteLevel", "Split"

        // For "Sequence" type
        [JsonProperty("pretokenizers")]
        public List<PreTokenizerConfig> PreTokenizers { get; set; }

        // For single pre_tokenizer types (duplicating fields from PreTokenizerConfig for convenience if not a sequence)
        // These will be null if Type is "Sequence"
        [JsonProperty("add_prefix_space")]
        public bool? AddPrefixSpace { get; set; } // For ByteLevel
        [JsonProperty("trim_offsets")]
        public bool? TrimOffsets { get; set; } // For ByteLevel
        [JsonProperty("use_regex")]
        public bool? UseRegex { get; set; } // For ByteLevel

        [JsonProperty("pattern")]
        public PatternDefinition Pattern { get; set; } // For Split
        [JsonProperty("behavior")]
        public string Behavior { get; set; } // For Split
        [JsonProperty("invert")]
        public bool? Invert { get; set; } // For Split
    }

    // Represents the configuration of an individual pre-tokenizer in a sequence or a standalone one
    internal class PreTokenizerConfig
    {
        [JsonProperty("type")]
        public string Type { get; set; }

        // ByteLevel specific
        [JsonProperty("add_prefix_space")]
        public bool? AddPrefixSpace { get; set; }
        [JsonProperty("trim_offsets")]
        public bool? TrimOffsets { get; set; } // Though your example had it on ByteLevel, it's a field.
        [JsonProperty("use_regex")]
        public bool? UseRegex { get; set; }


        // Split specific
        [JsonProperty("pattern")]
        public PatternDefinition Pattern { get; set; }
        [JsonProperty("behavior")]
        public string Behavior { get; set; }
        [JsonProperty("invert")]
        public bool? Invert { get; set; }
    }

    internal class PatternDefinition
    {
        [JsonProperty("Regex")] // Matches your JSON
        public string Regex { get; set; }
        // Add other pattern types if they exist, e.g., "String"
    }

    internal class TokenizerModel
    {
        [JsonProperty("type")]
        public string Type { get; set; }
        [JsonProperty("dropout")]
        public float? Dropout { get; set; }
        [JsonProperty("vocab")]
        public Dictionary<string, int> Vocab { get; set; }
        [JsonProperty("merges")]
        public List<List<string>> Merges { get; set; }
        [JsonProperty("unk_token")]
        public string UnkToken { get; set; }
        [JsonProperty("continuing_subword_prefix")]
        public string ContinuingSubwordPrefix { get; set; }
        [JsonProperty("max_input_chars_per_word")]
        public int? MaxInputCharsPerWord { get; set; }
    }

    internal class DecoderConfig // Renamed from Decoder
    {
        [JsonProperty("type")]
        public string Type { get; set; }
        // Specific decoder params
    }

    internal class PostProcessorConfig // Renamed from PostProcessor
    {
        [JsonProperty("type")]
        public string Type { get; set; }
        [JsonProperty("special_tokens")]
        public Dictionary<string, AddedToken> SpecialTokens { get; set; }
        [JsonProperty("template")]
        public List<object> Template { get; set; }
    }
}