using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;
namespace SparkTTS.Core
{
    using Utils;
    internal static class SparkTTSConstants
    {
        public static readonly Dictionary<string, string> TaskTokenMap = new Dictionary<string, string>
        {
            { "vc", "<|task_vc|>" },
            { "tts", "<|task_tts|>" },
            { "asr", "<|task_asr|>" },
            { "s2s", "<|task_s2s|>" },
            { "t2s", "<|task_t2s|>" },
            { "understand", "<|task_understand|>" },
            { "caption", "<|task_cap|>" },
            { "controllable_tts", "<|task_controllable_tts|>" },
            { "prompt_tts", "<|task_prompt_tts|>" },
            { "speech_edit", "<|task_edit|>" }
        };

        public static readonly Dictionary<string, int> LevelsMap = new Dictionary<string, int>
        {
            { "very_low", 0 },
            { "low", 1 },
            { "moderate", 2 },
            { "high", 3 },
            { "very_high", 4 }
        };

        public static readonly Dictionary<int, string> LevelsMapUI = new Dictionary<int, string>
        {
            { 1, "very_low" },
            { 2, "low" },
            { 3, "moderate" },
            { 4, "high" },
            { 5, "very_high" }
        };

        public static readonly Dictionary<string, int> GenderMap = new Dictionary<string, int>
        {
            { "female", 0 },
            { "male", 1 }
        };

        public static readonly Dictionary<string, int> AgeMap = new Dictionary<string, int>
        {
            { "Child", 0 },
            { "Teenager", 1 },
            { "Youth-Adult", 2 },
            { "Middle-aged", 3 },
            { "Elderly", 4 }
        };

        public static readonly Dictionary<string, int> EmotionMap = new Dictionary<string, int>
        {
            { "UNKNOWN", 0 }, { "NEUTRAL", 1 }, { "ANGRY", 2 }, { "HAPPY", 3 }, { "SAD", 4 },
            { "FEARFUL", 5 }, { "DISGUSTED", 6 }, { "SURPRISED", 7 }, { "SARCASTIC", 8 },
            { "EXCITED", 9 }, { "SLEEPY", 10 }, { "CONFUSED", 11 }, { "EMPHASIS", 12 },
            { "LAUGHING", 13 }, { "SINGING", 14 }, { "WORRIED", 15 }, { "WHISPER", 16 },
            { "ANXIOUS", 17 }, { "NO-AGREEMENT", 18 }, { "APOLOGETIC", 19 }, { "CONCERNED", 20 },
            { "ENUNCIATED", 21 }, { "ASSERTIVE", 22 }, { "ENCOURAGING", 23 }, { "CONTEMPT", 24 }
        };
    }

    internal static class SparkTokenBuilder
    {
        public static string Task(string taskKey)
        {
            if (SparkTTSConstants.TaskTokenMap.TryGetValue(taskKey.ToLowerInvariant(), out string token))
            {
                return token;
            }
            Logger.LogWarning($"[SparkTokenBuilder.Task] Unknown task key: {taskKey}. Returning empty string.");
            return string.Empty;
        }

        public static string Age(string ageKey)
        {
            if (SparkTTSConstants.AgeMap.TryGetValue(ageKey, out int ageId))
            {
                return $"<|age_{ageId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.Age] Unknown age key: {ageKey}. Returning empty string.");
            return string.Empty;
        }

        public static string Gender(string genderKey)
        {
            if (SparkTTSConstants.GenderMap.TryGetValue(genderKey.ToLowerInvariant(), out int genderId))
            {
                return $"<|gender_{genderId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.Gender] Unknown gender key: {genderKey}. Returning empty string.");
            return string.Empty;
        }

        public static string MelValue(int mel)
        {
            mel = Mathf.Clamp(mel, 0, 1000);
            return $"<|pitch_value_{mel}|>";
        }

        public static string MelLevel(string levelKey)
        {
            if (SparkTTSConstants.LevelsMap.TryGetValue(levelKey.ToLowerInvariant(), out int levelId))
            {
                return $"<|pitch_label_{levelId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.MelLevel] Unknown level key: {levelKey}. Returning moderate as default.");
            return $"<|pitch_label_{SparkTTSConstants.LevelsMap["moderate"]}|>"; // Default
        }
        
        public static string PitchLevel(string levelKey) // Alias for MelLevel
        {
            return MelLevel(levelKey);
        }

        public static string PitchVarValue(int pitchStd)
        {
            pitchStd = Mathf.Clamp(pitchStd, 0, 10);
            return $"<|pitch_var_value_{pitchStd}|>";
        }

        public static string PitchVarLevel(string levelKey)
        {
            if (SparkTTSConstants.LevelsMap.TryGetValue(levelKey.ToLowerInvariant(), out int levelId))
            {
                return $"<|pitch_var_label_{levelId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.PitchVarLevel] Unknown level key: {levelKey}. Returning moderate as default.");
            return $"<|pitch_var_label_{SparkTTSConstants.LevelsMap["moderate"]}|>"; // Default
        }

        public static string LoudnessValue(int loudness)
        {
            loudness = Mathf.Clamp(loudness, 0, 30);
            return $"<|loudness_value_{loudness}|>";
        }

        public static string LoudnessLevel(string levelKey)
        {
            if (SparkTTSConstants.LevelsMap.TryGetValue(levelKey.ToLowerInvariant(), out int levelId))
            {
                return $"<|loudness_label_{levelId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.LoudnessLevel] Unknown level key: {levelKey}. Returning moderate as default.");
            return $"<|loudness_label_{SparkTTSConstants.LevelsMap["moderate"]}|>"; // Default
        }

        public static string SpeedValue(int speed)
        {
            speed = Mathf.Clamp(speed, 0, 10);
            return $"<|speed_value_{speed}|>";
        }

        public static string SpeedLevel(string levelKey)
        {
            if (SparkTTSConstants.LevelsMap.TryGetValue(levelKey.ToLowerInvariant(), out int levelId))
            {
                return $"<|speed_label_{levelId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.SpeedLevel] Unknown level key: {levelKey}. Returning moderate as default.");
            return $"<|speed_label_{SparkTTSConstants.LevelsMap["moderate"]}|>"; // Default
        }
        
        public static string Emotion(string emotionKey)
        {
            if (SparkTTSConstants.EmotionMap.TryGetValue(emotionKey.ToUpperInvariant(), out int emotionId))
            {
                return $"<|emotion_{emotionId}|>";
            }
            Logger.LogWarning($"[SparkTokenBuilder.Emotion] Unknown emotion key: {emotionKey}. Returning UNKNOWN as default.");
            return $"<|emotion_{SparkTTSConstants.EmotionMap["UNKNOWN"]}|>"; // Default
        }
    }
    // End of copied classes

    public class TokenizationOutput
    {
        public List<int> InputIds { get; set; }
        public List<int> AttentionMask { get; set; }
        // public List<int> TypeIds { get; set; } // If needed for sentence pairs
        // public List<int> PositionIds {get; set;} // If we need to generate these manually

        public TokenizationOutput()
        {
            InputIds = new List<int>();
            AttentionMask = new List<int>();
        }
    }

    internal class TokenizerService
    {
        private readonly TokenizerDefinition _config;
        private readonly Dictionary<string, int> _vocab;
        private readonly List<(string, string)> _merges; // Processed merges for easier lookup
        private readonly string _unkToken;
        private readonly int _unkTokenId;

        // Special token IDs - should be passed or retrieved from config
        private readonly int _bosTokenId = -1;
        private readonly int _eosTokenId = -1;
        private readonly int _padTokenId = -1;

        private readonly Dictionary<int, string> _idToTokenVocab; // For decoding
        private readonly Dictionary<string, AddedToken> _addedTokensByContent; // For checking properties like .Special during decode

        // Define known special tokens that need protection from general splitting rules
        // These are based on the Python output and typical SparkTTS prompt structure
        private static readonly HashSet<string> _protectedSpecialTokens = new HashSet<string>
        {
            // From voice cloning (assumed working)
            // "<|task_tts|>", // Will be added from SparkTTSConstants.TaskTokenMap
            "<|start_content|>",
            "<|end_content|>",
            "<|start_global_token|>",
            "<|end_global_token|>",
            "<|start_semantic_token|>",

            // Style control specific tokens (some will be generated by builder)
            // "<|task_controllable_tts|>", // Will be added from SparkTTSConstants.TaskTokenMap
            "<|start_style_label|>",
            "<|end_style_label|>",
            // Gender, Pitch Level, Speed Level tokens like <|gender_0|>, <|pitch_label_3|> are covered by regex below
        };

        // Regex to identify various special token patterns.
        // We combine specific known tokens with more general patterns.
        private static readonly Regex _combinedSpecialTokenRegex;

        // Static constructor to initialize _protectedSpecialTokens from constants and build the combined regex
        static TokenizerService()
        {
            // Add all task tokens from the constants map
            foreach (var taskTokenValue in SparkTTSConstants.TaskTokenMap.Values) // Now refers to the class in the same file
            {
                _protectedSpecialTokens.Add(taskTokenValue);
            }

            // Build the list of regex patterns for all special tokens
            List<string> allSpecialPatterns = _protectedSpecialTokens.Select(s => Regex.Escape(s)).ToList();
            
            // Add regex for patterns with numbers
            allSpecialPatterns.Add("<\\|gender_\\d+\\|>");
            allSpecialPatterns.Add("<\\|pitch_label_\\d+\\|>");
            allSpecialPatterns.Add("<\\|speed_label_\\d+\\|>");
            allSpecialPatterns.Add("<\\|age_\\d+\\|>");
            allSpecialPatterns.Add("<\\|emotion_\\d+\\|>");
            allSpecialPatterns.Add("<\\|pitch_value_\\d+\\|>");
            allSpecialPatterns.Add("<\\|pitch_var_value_\\d+\\|>");
            allSpecialPatterns.Add("<\\|pitch_var_label_\\d+\\|>"); // Added for completeness from Python TokenParser
            allSpecialPatterns.Add("<\\|loudness_value_\\d+\\|>");
            allSpecialPatterns.Add("<\\|loudness_label_\\d+\\|>"); // Added for completeness
            allSpecialPatterns.Add("<\\|speed_value_\\d+\\|>");
            allSpecialPatterns.Add("<\\|bicodec_global_\\d+\\|>");
            allSpecialPatterns.Add("<\\|bicodec_semantic_\\d+\\|>");
            allSpecialPatterns.Add("<\\|start_acoustic_token\\|>"); // From observed Python output
            allSpecialPatterns.Add("<\\|end_acoustic_token\\|>");   // From observed Python output

            string combinedPatternString = $"({string.Join("|", allSpecialPatterns)})";
            _combinedSpecialTokenRegex = new Regex(combinedPatternString);
            Logger.Log($"[TokenizerService Static Constructor] Combined Special Token Pattern for splitting: {combinedPatternString}");
        }

        // ByteLevel character mapping (simplified, extend as needed)
        // Based on common mappings, e.g., in GPT-2/Roberta tokenizers
        private static readonly Dictionary<byte, char> ByteToCharVocab = InitializeByteToCharVocab();
        private static readonly Dictionary<char, byte> CharToByteVocab = ByteToCharVocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

        private static Dictionary<byte, char> InitializeByteToCharVocab()
        {
            var byteEncoder = new Dictionary<byte, char>();
            // Create a mapping for bytes 0-255 to unique Unicode characters.
            // This is a common strategy for ByteLevel BPE.
            // Characters are chosen to be outside typical text usage if possible.
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                // Skip ASCII printable characters, map others
                if ((b >= '!' && b <= '~') || (b >= '¡' && b <= '¬') || (b >= '®' && b <= 'ÿ'))
                {
                    byteEncoder[(byte)b] = (char)b;
                }
                else
                {
                    byteEncoder[(byte)b] = (char)(256 + n);
                    n++;
                }
            }
            // Override for space, typically mapped to 'Ġ' (U+0120)
            // This is crucial for models that expect spaces to be handled this way.
            // If your model uses a different char for space, adjust here.
            if (byteEncoder.ContainsKey((byte)' '))
            {
                byteEncoder[(byte)' '] = 'Ġ'; 
            }
            else // If space wasn't in the initial range, add it.
            {
                // This case might not be hit if ASCII space (32) is already mapped
                // but as a safeguard if the loop logic changes.
                byteEncoder.Add((byte)' ', 'Ġ');
            }
            
            // Add other common control characters if needed, e.g. newline
            // byteEncoder[(byte)'\n'] = ... some char ...

            return byteEncoder;
        }

        public TokenizerDefinition Config => _config; // Public getter for _config

        public TokenizerService(TokenizerDefinition tokenizerDefinition, int bosTokenId = -1, int eosTokenId = -1, int padTokenId = -1)
        {
            if (tokenizerDefinition == null) 
            {
                throw new System.ArgumentNullException(nameof(tokenizerDefinition));
            }
            if (tokenizerDefinition.Model == null) 
            {
                throw new System.ArgumentNullException(nameof(tokenizerDefinition.Model));
            }
            if (tokenizerDefinition.Model.Vocab == null) 
            {
                throw new System.ArgumentNullException(nameof(tokenizerDefinition.Model.Vocab));
            }

            _config = tokenizerDefinition;
            _vocab = new Dictionary<string, int>(_config.Model.Vocab); // Initialize with base vocab
            _merges = new List<(string, string)>();
            _unkToken = _config.Model.UnkToken ?? "<unk>"; // Default if null

            // Augment _vocab with added_tokens
            // Giving precedence to added_tokens if there's a conflict, or simply adding if not present.
            Logger.Log($"[TokenizerService Constructor] Processing {_config.AddedTokens.Count} added_tokens.");
            foreach (var addedToken in _config.AddedTokens)
            {
                _vocab[addedToken.Content] = addedToken.Id; // Add or overwrite
            }

            // --- TEMP DEBUG ---
            if (_vocab.TryGetValue("<|task_controllable_tts|>", out int controlId))
            {
                Logger.Log($"[TokenizerService Constructor DEBUG] '<|task_controllable_tts|>' IS in _vocab. ID: {controlId} (Expected: 165143)");
            }
            else
            {
                Logger.LogError("[TokenizerService Constructor DEBUG] '<|task_controllable_tts|>' IS NOT in _vocab after merging added_tokens!");
            }
            if (_vocab.TryGetValue("<|start_content|>", out int startContentId))
            {
                Logger.Log($"[TokenizerService Constructor DEBUG] '<|start_content|>' IS in _vocab. ID: {startContentId} (Expected: 165146)");
            }
            else
            {
                Logger.LogError("[TokenizerService Constructor DEBUG] '<|start_content|>' IS NOT in _vocab after merging added_tokens!");
            }
            // You can add more checks for other critical special tokens here if needed
            // --- END TEMP DEBUG ---

            if (!_vocab.TryGetValue(_unkToken, out _unkTokenId))
            {
                Logger.LogWarning($"UNK token '{_unkToken}' not found in vocabulary. Using -1 as UNK ID.");
                _unkTokenId = -1; // Or throw error
            }

            if (_config.Model.Type == "BPE" && _config.Model.Merges != null)
            {
                foreach (var mergePairList in _config.Model.Merges)
                {
                    if (mergePairList.Count == 2)
                    {
                        _merges.Add((mergePairList[0], mergePairList[1]));
                    }
                    else
                    {
                        Logger.LogWarning($"Invalid merge entry encountered: {string.Join(", ", mergePairList)}. Expected 2 elements.");
                    }
                }
            }
            
            _bosTokenId = bosTokenId;
            _eosTokenId = eosTokenId;
            _padTokenId = padTokenId;

            // Initialize reverse vocabulary for decoding
            _idToTokenVocab = new Dictionary<int, string>();
            foreach (var pair in _vocab) // _vocab already contains added_tokens merged
            {
                if (!_idToTokenVocab.ContainsKey(pair.Value)) // Take the first content string if multiple map to the same ID
                {
                    _idToTokenVocab.Add(pair.Value, pair.Key);
                }
            }

            _addedTokensByContent = new Dictionary<string, AddedToken>();
            if (_config.AddedTokens != null)
            {
                foreach(var addedToken in _config.AddedTokens)
                {
                    if (!string.IsNullOrEmpty(addedToken.Content) && !_addedTokensByContent.ContainsKey(addedToken.Content))
                    {
                        _addedTokensByContent.Add(addedToken.Content, addedToken);
                    }
                }
            }
            // TODO: Potentially get BOS/EOS/PAD from _config.AddedTokens or _config.PostProcessor.SpecialTokens if not passed
        }

        public TokenizationOutput Encode(string text, bool addSpecialTokens = true)
        {
            if (string.IsNullOrEmpty(text)) return new TokenizationOutput();
            Logger.Log($"[TokenizerService.Encode] Input text: '{text}'");

            // 1. Normalization
            string normalizedText = Normalize(text);
            Logger.Log($"[TokenizerService.Encode] Normalized text: '{normalizedText}'");

            // 2. Pre-tokenization
            List<string> preTokenized = PreTokenize(normalizedText);
            Logger.Log($"[TokenizerService.Encode] PreTokenized output: [{string.Join(", ", preTokenized.Select(s => $"'{s}'"))}]");

            List<int> inputIds = new();

            // 3. Model-specific tokenization (BPE)
            Logger.Log($"[TokenizerService.Encode] Applying BPE and Vocab Lookup:");
            foreach (string word in preTokenized)
            {
                if (string.IsNullOrEmpty(word)) continue;
                Logger.Log($"[TokenizerService.Encode]   Processing word: '{word}'");

                if (_config.Model.Type == "BPE")
                {
                    List<string> bpeTokens = ApplyBPE(word);
                    Logger.Log($"[TokenizerService.Encode]     BPE tokens for '{word}': [{string.Join(", ", bpeTokens.Select(s => $"'{s}'"))}]");
                    foreach (string token in bpeTokens)
                    {
                        bool found = _vocab.TryGetValue(token, out int id);
                        int finalId = found ? id : _unkTokenId;
                        inputIds.Add(finalId);
                        Logger.Log($"[TokenizerService.Encode]       Token: '{token}' -> ID: {finalId}" + (found ? "" : " (UNK)"));
                    }
                }
                else
                {
                    // Placeholder for other model types like WordPiece, Unigram
                    Logger.LogWarning($"Tokenizer model type '{_config.Model.Type}' not fully supported yet. Treating word as single token.");
                    bool found = _vocab.TryGetValue(word, out int id);
                    int finalId = found ? id : _unkTokenId;
                    inputIds.Add(finalId);
                    Logger.Log($"[TokenizerService.Encode]     Token (non-BPE): '{word}' -> ID: {finalId}" + (found ? "" : " (UNK)"));
                }
            }
            
            // Removed auto-addition of EOS based on Python HF output for this tokenizer
            // if (addSpecialTokens && _eosTokenId != -1)
            // {
            //     inputIds.Add(_eosTokenId);
            // }

            // 4. TODO: Truncation & Padding (based on _config.Truncation and _config.Padding)
            // For now, we're not truncating or padding within this basic Encode call.
            // This is usually handled when batching multiple sequences.

            TokenizationOutput output = new TokenizationOutput();
            output.InputIds.AddRange(inputIds);
            output.AttentionMask.AddRange(Enumerable.Repeat(1, inputIds.Count)); // Simple attention mask (all 1s)
            
            Logger.Log($"[TokenizerService.Encode] Final Input IDs: [{string.Join(", ", output.InputIds)}]");
            Logger.Log($"[TokenizerService.Encode] Final Attention Mask: [{string.Join(", ", output.AttentionMask)}]");
            return output;
        }

        /// <summary>
        /// Encodes text specifically for LLM input, combining a prompt, main text, EOP token, and speaker tokens.
        /// </summary>
        /// <param name="text">The main input text.</param>
        /// <param name="globalSpeakerNumericIDs">Array of global speaker token IDs (numeric).</param>
        /// <param name="acousticSemanticTokensFromRefAudio">Optional array of acoustic semantic token IDs from reference audio.</param>
        /// <returns>TokenizationOutput containing combined input IDs and attention mask.</returns>
        public TokenizationOutput EncodeForLLM(string text, 
                                             IEnumerable<int> globalSpeakerNumericIDs,
                                             IEnumerable<long> acousticSemanticTokensFromRefAudio = null)
        {
            Logger.Log($"[TokenizerService.EncodeForLLM] Encoding for LLM. Text: '{text}'");

            StringBuilder sb = new();

            sb.Append("<|task_tts|>"); // Assuming TASK_TOKEN_MAP["tts"] is "<|task_tts|>"
            sb.Append("<|start_content|>");
            sb.Append(text); // The main text to synthesize
            sb.Append("<|end_content|>");

            if (globalSpeakerNumericIDs != null && globalSpeakerNumericIDs.Any())
            {
                sb.Append("<|start_global_token|>");
                foreach (int id in globalSpeakerNumericIDs)
                {
                    sb.Append($"<|bicodec_global_{id}|>");
                }
                sb.Append("<|end_global_token|>");
            }

            // Add acoustic semantic tokens from reference audio, if provided
            if (acousticSemanticTokensFromRefAudio != null && acousticSemanticTokensFromRefAudio.Any())
            {
                Logger.Log($"[TokenizerService.EncodeForLLM] Adding {acousticSemanticTokensFromRefAudio.Count()} acoustic semantic tokens from reference audio.");
                sb.Append("<|start_semantic_token|>");
                foreach (long id in acousticSemanticTokensFromRefAudio)
                {
                    sb.Append($"<|bicodec_semantic_{id}|>");
                }
                // Note: Python's SparkTTS.process_prompt doesn't add an explicit <|end_semantic_token|> here
                // when prompt_text is also present. The LLM seems to infer based on subsequent tokens or EOS.
            }

            string inputPromptString = sb.ToString();
            Logger.Log($"[TokenizerService.EncodeForLLM] ------- Constructed Input Prompt String START -------\n{inputPromptString}\n------- Constructed Input Prompt String END -------");

            // Now, tokenize this entire constructed string using the base Encode method logic
            // which handles normalization, pre-tokenization, BPE, and vocab lookup.
            // Crucially, addSpecialTokens should be false, as all special tokens are manually included.
            
            // --- Replicating relevant parts of Encode(string, bool addSpecialTokens = false) ---
            if (string.IsNullOrEmpty(inputPromptString)) return new TokenizationOutput();

            string normalizedText = Normalize(inputPromptString);
            Logger.Log($"[TokenizerService.EncodeForLLM] Normalized prompt: '{normalizedText}'");

            List<string> preTokenized = PreTokenize(normalizedText);
            Logger.Log($"[TokenizerService.EncodeForLLM] PreTokenized prompt: [{string.Join(", ", preTokenized.Select(s => $"'{s}'"))}]");

            List<int> finalInputIds = new();
            Logger.Log($"[TokenizerService.EncodeForLLM] Applying BPE and Vocab Lookup to constructed prompt:");

            foreach (string word in preTokenized)
            {
                if (string.IsNullOrEmpty(word)) continue;
                Logger.Log($"[TokenizerService.EncodeForLLM]   Processing word from prompt: '{word}'");

                if (_config.Model.Type == "BPE")
                {
                    List<string> bpeTokens = ApplyBPE(word);
                    Logger.Log($"[TokenizerService.EncodeForLLM]     BPE tokens for '{word}': [{string.Join(", ", bpeTokens.Select(s => $"'{s}'"))}]");
                    foreach (string token in bpeTokens)
                    {
                        bool found = _vocab.TryGetValue(token, out int id);
                        int finalId = found ? id : _unkTokenId;
                        finalInputIds.Add(finalId);
                        Logger.Log($"[TokenizerService.EncodeForLLM]       Token: '{token}' -> ID: {finalId}" + (found ? "" : " (UNK)"));
                    }
                }
                else
                {
                    Logger.LogWarning($"Tokenizer model type '{_config.Model.Type}' not fully supported for prompt. Treating word as single token.");
                    bool found = _vocab.TryGetValue(word, out int id);
                    int finalId = found ? id : _unkTokenId;
                    finalInputIds.Add(finalId);
                    Logger.Log($"[TokenizerService.EncodeForLLM]     Token (non-BPE): '{word}' -> ID: {finalId}" + (found ? "" : " (UNK)"));
                }
            }
            // --- End of replicated Encode logic ---
            
            TokenizationOutput output = new TokenizationOutput();
            output.InputIds.AddRange(finalInputIds);
            output.AttentionMask.AddRange(Enumerable.Repeat(1, finalInputIds.Count));

            Logger.Log($"[TokenizerService.EncodeForLLM] Final Combined Input IDs: [{string.Join(", ", output.InputIds)}]");
            Logger.Log($"[TokenizerService.EncodeForLLM] Final Combined Attention Mask: [{string.Join(", ", output.AttentionMask)}]");
            Logger.Log($"[TokenizerService.EncodeForLLM] Final Combined Output: {output}");
            
            return output;
        }

        private string Normalize(string text)
        {
            if (_config.Normalizer == null) return text;

            string currentText = text;
            // Example: Basic NFD normalization (Unicode normalization form D)
            // Python's Unidecode or specific normalizers like BertNormalizer are more complex.
            // if (_config.Normalizer.Type == "NFD" || _config.Normalizer.Type == "NFKD" || _config.Normalizer.Type == "NFC" || _config.Normalizer.Type == "NFKC")
            // {
            //    currentText = currentText.Normalize((System.Text.NormalizationForm)System.Enum.Parse(typeof(System.Text.NormalizationForm), _config.Normalizer.Type));
            // }
            // if (_config.Normalizer.Lowercase == true) // Common for BertNormalizer or simple lowercase
            // {
            //     currentText = currentText.ToLowerInvariant(); // Temporarily commenting this out
            // }
            // TODO: Implement more complex normalizers if needed based on _config.Normalizer.Type
            // (e.g., StripAccents, BertNormalizer specific rules)
            return currentText;
        }

        private List<string> PreTokenize(string text)
        {
            Logger.Log($"[TokenizerService.PreTokenize] Input text: '{text}'");
            if (string.IsNullOrEmpty(text)) return new List<string>();

            // Use the pre-built combined regex for splitting            
            Logger.Log($"[TokenizerService.PreTokenize] Using Combined Special Token Pattern: {_combinedSpecialTokenRegex.ToString()}");

            List<string> initialSegments = new();
            int lastIndex = 0;

            MatchCollection specialMatches = _combinedSpecialTokenRegex.Matches(text);
            foreach (Match match in specialMatches)
            {
                if (match.Index > lastIndex)
                {
                    initialSegments.Add(text.Substring(lastIndex, match.Index - lastIndex));
                }
                initialSegments.Add(match.Value); // This is the matched special token/pattern
                lastIndex = match.Index + match.Length;
            }
            if (lastIndex < text.Length)
            {
                initialSegments.Add(text.Substring(lastIndex));
            }
            
            Logger.Log($"[TokenizerService.PreTokenize] Initial Segments (after splitting by ALL special patterns): [{string.Join(", ", initialSegments.Select(s => $"'{s}'"))}]");

            List<string> finalPreTokens = new List<string>();

            foreach (string segment in initialSegments)
            {
                if (string.IsNullOrEmpty(segment)) continue;

                // Check if this segment IS one of the special tokens/patterns
                bool isSegmentSpecialOrPatternMatch = _combinedSpecialTokenRegex.IsMatch(segment) && 
                                                      specialMatches.Cast<Match>().Any(m => m.Value == segment && m.Length == segment.Length);

                if (isSegmentSpecialOrPatternMatch)
                {
                    Logger.Log($"[TokenizerService.PreTokenize] Segment '{segment}' IS special/pattern. It should bypass main 'Split' pretokenizer if configured.");
                    // Special tokens should generally NOT be processed by the main "Split" pretokenizer from config.
                    // They might still be processed by ByteLevel if it's later in the sequence.
                    // We pass it through the pretokenizer sequence defined in config, but ApplySinglePretokenizer
                    // needs to be smart about not re-splitting already isolated special tokens if the PT is "Split".
                    
                    List<string> processedSpecialSegment = new List<string> { segment }; // Start with the special token itself
                    if (_config.PreTokenizer?.Type == "Sequence" && _config.PreTokenizer.PreTokenizers != null)
                    {
                        foreach (var ptConfig in _config.PreTokenizer.PreTokenizers)
                        {
                            List<string> nextParts = new List<string>();
                            foreach(string partToProcess in processedSpecialSegment)
                            {
                                // If this part is one of our special tokens, and the current ptConfig is the main "Split", skip splitting it.
                                // Allow other types like "ByteLevel" to process it.
                                if (ptConfig.Type == "Split" && _combinedSpecialTokenRegex.IsMatch(partToProcess)) 
                                {
                                     Logger.Log($"[TokenizerService.PreTokenize]   Skipping configured 'Split' for already isolated special part '{partToProcess}'");
                                     nextParts.Add(partToProcess); 
                                }
                                else 
                                {
                                    Logger.Log($"[TokenizerService.PreTokenize]   Applying PT '{ptConfig.Type}' to (potentially special) part '{partToProcess}'");
                                    nextParts.AddRange(ApplySinglePreTokenizer(partToProcess, ptConfig));
                                }
                            }
                            processedSpecialSegment = nextParts;
                        }
                    }
                    else if (_config.PreTokenizer != null) // Single pretokenizer in config
                    {
                         if (_config.PreTokenizer.Type == "Split" && isSegmentSpecialOrPatternMatch)
                         {
                            Logger.Log($"[TokenizerService.PreTokenize]   Skipping single 'Split' PT for special segment '{segment}'");
                            processedSpecialSegment = new List<string> { segment };
                         }
                         else
                         {
                            Logger.Log($"[TokenizerService.PreTokenize]   Applying single PT '{_config.PreTokenizer.Type}' to special segment '{segment}'");
                            PreTokenizerConfig singlePtConfig = new PreTokenizerConfig
                            {
                                Type = _config.PreTokenizer.Type,
                                AddPrefixSpace = _config.PreTokenizer.AddPrefixSpace,
                                TrimOffsets = _config.PreTokenizer.TrimOffsets,
                                UseRegex = _config.PreTokenizer.UseRegex,
                                Pattern = _config.PreTokenizer.Pattern, // Assuming PatternConfig is assignable or constructible
                                Behavior = _config.PreTokenizer.Behavior,
                                Invert = _config.PreTokenizer.Invert
                            };
                            processedSpecialSegment = ApplySinglePreTokenizer(segment, singlePtConfig);
                         }
                    }
                    // else: no pretokenizer in config, segment remains as is.
                    finalPreTokens.AddRange(processedSpecialSegment);
                }
                else // Segment is NOT special (e.g., "Hello world" or other interstitial text)
                {
                    Logger.Log($"[TokenizerService.PreTokenize] Segment '{segment}' is NOT special. Processing with full pretokenizer sequence from config.");
                    List<string> currentNormalSegmentParts = new List<string> { segment };
                     if (_config.PreTokenizer?.Type == "Sequence" && _config.PreTokenizer.PreTokenizers != null)
                    {
                        foreach (var ptConfig in _config.PreTokenizer.PreTokenizers)
                        {
                            List<string> nextParts = new List<string>();
                            foreach(string partToProcess in currentNormalSegmentParts)
                            {
                                Logger.Log($"[TokenizerService.PreTokenize]   Applying PT '{ptConfig.Type}' to part '{partToProcess}' of non-special segment '{segment}'");
                                nextParts.AddRange(ApplySinglePreTokenizer(partToProcess, ptConfig));
                            }
                            currentNormalSegmentParts = nextParts;
                        }
                    }
                    else if (_config.PreTokenizer != null) // Single pretokenizer in config
                    {
                        Logger.Log($"[TokenizerService.PreTokenize]   Applying single PT '{_config.PreTokenizer.Type}' to non-special segment '{segment}'");
                        PreTokenizerConfig singlePtConfig = new PreTokenizerConfig
                        {
                            Type = _config.PreTokenizer.Type,
                            AddPrefixSpace = _config.PreTokenizer.AddPrefixSpace,
                            TrimOffsets = _config.PreTokenizer.TrimOffsets,
                            UseRegex = _config.PreTokenizer.UseRegex,
                            Pattern = _config.PreTokenizer.Pattern, // Assuming PatternConfig is assignable or constructible
                            Behavior = _config.PreTokenizer.Behavior,
                            Invert = _config.PreTokenizer.Invert
                        };
                        currentNormalSegmentParts = ApplySinglePreTokenizer(segment, singlePtConfig);
                    }
                    // else: no pretokenizer, segment remains as is.
                    finalPreTokens.AddRange(currentNormalSegmentParts);
                }
            }
            
            Logger.Log($"[TokenizerService.PreTokenize] Final pre-tokens before returning: [{string.Join(", ", finalPreTokens.Select(s => $"'{s}'"))}]");
            return finalPreTokens.Where(s => !string.IsNullOrEmpty(s)).ToList();
        }

        // Helper to apply a single pre-tokenizer configuration
        private List<string> ApplySinglePreTokenizer(string textSegment, PreTokenizerConfig ptConfig) // Assuming PreTokenizerConfig is the type in the list
        {
            // Minor log added here for entry, more detailed logs are in PreTokenize which calls this
            // Debug.Log($"[ApplySinglePreTokenizer] Applying '{ptConfig?.Type}' to '{textSegment}'"); 
            if (ptConfig == null || string.IsNullOrEmpty(ptConfig.Type) || string.IsNullOrEmpty(textSegment)) 
            {
                return string.IsNullOrEmpty(textSegment) ? new List<string>() : new List<string> {textSegment};
            }

            if (ptConfig.Type == "Split")
            {
                if (ptConfig.Pattern?.Regex == null)
                {
                    Logger.LogWarning("Split PreTokenizer has no Regex pattern. Returning segment as is.");
                    return new List<string> { textSegment };
                }
                // Regex.Split might not be identical to HF tokenizers' split behavior (especially with captures)
                // For "Isolated" behavior, we need to find matches, not split by them.
                MatchCollection matches = Regex.Matches(textSegment, ptConfig.Pattern.Regex);
                if (matches.Count == 0 && !string.IsNullOrEmpty(textSegment)) 
                {
                    // If regex doesn't match anything, and string is not empty, treat segment as one
                    return new List<string> { textSegment }; 
                }
                return matches.Cast<Match>().Select(m => m.Value).ToList();
            }
            else if (ptConfig.Type == "ByteLevel")
            { 
                string segmentToProcess = textSegment;
                if (ptConfig.AddPrefixSpace == true && !string.IsNullOrEmpty(textSegment) && textSegment[0] != ' ' && textSegment[0] != 'Ġ')
                {
                    segmentToProcess = " " + textSegment;
                    Logger.Log($"[TokenizerService.ApplySinglePreTokenizer ByteLevel] Prepended space due to AddPrefixSpace=true. Original: '{textSegment}', Processing: '{segmentToProcess}'");
                }

                byte[] bytes = Encoding.UTF8.GetBytes(segmentToProcess);
                StringBuilder sb = new StringBuilder();
                foreach (byte b in bytes)
                {
                    if (ByteToCharVocab.TryGetValue(b, out char c))
                    {
                        sb.Append(c);
                    }
                    else
                    {
                        // Fallback for bytes not in our simplified map (should ideally not happen with a full map)
                        sb.Append((char)b); // Or handle as error/unknown byte char
                        Logger.LogWarning($"Byte {b} not found in ByteToCharVocab. Appending as raw char.");
                    }
                }
                return new List<string> { sb.ToString() };
            }
            
            Logger.LogWarning($"PreTokenizer type '{ptConfig.Type}' not fully implemented in ApplySinglePreTokenizer. Returning segment as is.");
            return new List<string> { textSegment };
        }

        private List<string> ApplyBPE(string word)
        {
            if (string.IsNullOrEmpty(word)) return new List<string>();

            // If the entire word is already a single token in the vocab (e.g., a special token),
            // return it directly without trying to BPE-process its characters.
            if (_vocab.ContainsKey(word))
            {
                Logger.Log($"[TokenizerService.ApplyBPE] Word '{word}' found directly in vocab. Returning as single token.");
                return new List<string> { word };
            }

            // Some BPE tokenizers (e.g. from GPT-2) map spaces to 'Ġ'.
            // This logic might need to be part of pre-tokenization or configurable.
            // For now, let's assume pre-tokenizer handles this if required.

            List<string> symbols = word.Select(c => c.ToString()).ToList();

            // Handle cases where individual characters might not be in vocab (e.g. if vocab has 'ĠA' but not 'Ġ' or 'A' separately)
            // This is a fallback; ideally, vocab should cover base characters or BPE should build up from them.
            for(int i = 0; i < symbols.Count; ++i)
            {
                if (!_vocab.ContainsKey(symbols[i]))
                {
                    // If a single character is not in vocab, it might be part of a multi-char token that should have been preserved by pre-tokenizer
                    // Or, it's truly OOV at character level. Using UNK here can be problematic for BPE.
                    // Debug.LogWarning($"Symbol '{symbols[i]}' not in vocab during BPE. This might lead to suboptimal tokenization.");
                }
            }

            while (true)
            {
                (string, string) bestPair = (null, null);
                int bestPairIdx = -1;
                int minRank = int.MaxValue; // Lower rank (earlier in merges list) is better

                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    var pair = (symbols[i], symbols[i+1]);
                    int rank = _merges.IndexOf(pair);
                    if (rank != -1 && rank < minRank)
                    {
                        minRank = rank;
                        bestPair = pair;
                        bestPairIdx = i;
                    }
                }

                if (bestPairIdx == -1) break; // No more merges found

                symbols[bestPairIdx] = bestPair.Item1 + bestPair.Item2;
                symbols.RemoveAt(bestPairIdx + 1);
            }
            return symbols;
        }
        
        public string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true)
        {
            if (ids == null) return string.Empty;
            Logger.Log($"[TokenizerService.Decode] Input IDs: [{string.Join(", ", ids)}], skipSpecialTokens: {skipSpecialTokens}");

            StringBuilder rawTokenString = new StringBuilder();
            foreach (int id in ids)
            {
                if (_idToTokenVocab.TryGetValue(id, out string tokenString))
                {
                    bool isSpecial = false;
                    if (_addedTokensByContent.TryGetValue(tokenString, out AddedToken addedTokenDef))
                    {
                        isSpecial = addedTokenDef.Special;
                    }

                    if (skipSpecialTokens && isSpecial)
                    {
                        Logger.Log($"[TokenizerService.Decode] Skipping special token: '{tokenString}' (ID: {id})");
                        continue;
                    }
                    rawTokenString.Append(tokenString);
                }
                else
                {
                    Logger.LogWarning($"[TokenizerService.Decode] ID {id} not found in _idToTokenVocab. Appending UNK representation or skipping.");
                    // Optionally append UNK token string if defined and not skipping specials, or just skip.
                    // For now, just skipping if ID not found, to somewhat mimic skip_special_tokens behavior for true unknowns.
                }
            }
            
            string concatenatedString = rawTokenString.ToString();
            Logger.Log($"[TokenizerService.Decode] Concatenated token string (before ByteLevel): '{concatenatedString}'");

            // ByteLevel decoding part (similar to HF tokenizers Python code)
            // Converts special ByteLevel characters (like 'Ġ') back to bytes, then UTF-8 decodes.
            List<byte> byteList = new List<byte>();
            foreach (char c in concatenatedString)
            {
                if (CharToByteVocab.TryGetValue(c, out byte b))
                {
                    byteList.Add(b);
                }
                else
                {
                    // This case implies a character that wasn't in our ByteToCharVocab original mapping.
                    // It might be a direct character if the tokenizer includes raw chars in its vocab,
                    // or an issue if it purely relies on the byte->char map.
                    // For robustness, try to convert char to byte(s) directly via UTF-8. 
                    // This is a simplification; a full BPE decoder handles merges more intricately.
                    Logger.LogWarning($"[TokenizerService.Decode] Character '{c}' not in CharToByteVocab. Attempting direct UTF-8 encoding of char.");
                    byteList.AddRange(Encoding.UTF8.GetBytes(new char[] {c}));
                }
            }
            string decodedText = Encoding.UTF8.GetString(byteList.ToArray());
            Logger.Log($"[TokenizerService.Decode] Final decoded text: '{decodedText}'");
            return decodedText;
        }
    }
} // End of namespace SparkTTS
