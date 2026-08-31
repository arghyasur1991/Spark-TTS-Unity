// GPT-2 byte-level BPE for Qwen3-TTS. Replaces Microsoft.ML.Tokenizers (not in Unity).
// Ported from ElBruno.QwenTTS TextTokenizer + HuggingFace GPT2Tokenizer.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

namespace SparkTTS.Qwen.Models
{
    internal sealed class TextTokenizer : IDisposable
    {
        public const int EndOfTextId = 151643;
        public const int ImStartId = 151644;
        public const int ImEndId = 151645;
        public const int AudioStartId = 151669;
        public const int AudioEndId = 151670;
        public const int TtsPadId = 151671;
        public const int TtsBosId = 151672;
        public const int TtsEodId = 151673;
        public const int TtsBosSignleId = 151674;
        public const int AudioPadId = 151675;
        public const int AssistantTokenId = 77091;
        public const int NewlineTokenId = 198;

        private static readonly Dictionary<string, int> SpecialTokensMap = new()
        {
            ["<|endoftext|>"] = EndOfTextId,
            ["<|im_start|>"] = ImStartId,
            ["<|im_end|>"] = ImEndId,
            ["<|audio_start|>"] = AudioStartId,
            ["<|audio_end|>"] = AudioEndId,
            ["<tts_pad>"] = TtsPadId,
            ["<tts_text_bos>"] = TtsBosId,
            ["<tts_text_eod>"] = TtsEodId,
            ["<tts_text_bos_single>"] = TtsBosSignleId,
            ["<|audio_pad|>"] = AudioPadId,
        };

        private static readonly string[] SpecialTokenKeys;

        static TextTokenizer()
        {
            var keys = new List<string>(SpecialTokensMap.Keys);
            keys.Sort((a, b) => b.Length.CompareTo(a.Length));
            SpecialTokenKeys = keys.ToArray();
        }

        private static readonly Regex Gpt2Regex = new(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<(string, string), int> _ranks;
        private readonly Dictionary<string, string> _bpeCache = new();
        private readonly string[] _byteToUnicode;
        private readonly object _cacheLock = new();

        public TextTokenizer(string modelDir)
        {
            var vocabPath = Path.Combine(modelDir, "vocab.json");
            var mergesPath = Path.Combine(modelDir, "merges.txt");

            if (!File.Exists(vocabPath))
                throw new FileNotFoundException("vocab.json not found in tokenizer directory.", vocabPath);
            if (!File.Exists(mergesPath))
                throw new FileNotFoundException("merges.txt not found in tokenizer directory.", mergesPath);

            _vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(File.ReadAllText(vocabPath))
                     ?? throw new InvalidDataException("Failed to parse vocab.json");
            _ranks = LoadMerges(mergesPath);
            _byteToUnicode = BuildBytesToUnicode();
        }

        public int[] Encode(string text)
        {
            if (string.IsNullOrEmpty(text))
                return Array.Empty<int>();

            var ids = new List<int>();
            int i = 0;
            while (i < text.Length)
            {
                string special = MatchSpecial(text, i);
                if (special != null)
                {
                    ids.Add(SpecialTokensMap[special]);
                    i += special.Length;
                    continue;
                }

                int next = FindNextSpecial(text, i);
                string chunk = text.Substring(i, next - i);
                EncodeBpeChunk(chunk, ids);
                i = next;
            }

            return ids.ToArray();
        }

        public int[] BuildCustomVoicePrompt(string text, string speaker, string language, string instruct = null)
        {
            var tokens = new List<int>();
            if (!string.IsNullOrEmpty(instruct))
            {
                var instructWrapped = $"<|im_start|>user\n{instruct}<|im_end|>\n";
                tokens.AddRange(Encode(instructWrapped));
            }

            var assistantWrapped = $"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n";
            tokens.AddRange(Encode(assistantWrapped));
            return tokens.ToArray();
        }

        public void Dispose()
        {
        }

        private void EncodeBpeChunk(string chunk, List<int> ids)
        {
            if (chunk.Length == 0)
                return;

            foreach (Match match in Gpt2Regex.Matches(chunk))
            {
                string piece = match.Value;
                string mapped = BytesToUnicodeString(piece);
                string bpe = Bpe(mapped);
                foreach (var token in bpe.Split(' '))
                {
                    if (token.Length == 0)
                        continue;
                    if (!_vocab.TryGetValue(token, out int id))
                        throw new InvalidDataException($"BPE token not in vocab: '{token}'");
                    ids.Add(id);
                }
            }
        }

        private string Bpe(string token)
        {
            lock (_cacheLock)
            {
                if (_bpeCache.TryGetValue(token, out var cached))
                    return cached;
            }

            if (token.Length == 0)
                return token;

            var word = new List<string>(token.Length);
            foreach (char c in token)
                word.Add(c.ToString());

            var pairs = GetPairs(word);
            if (pairs.Count == 0)
                return token;

            while (true)
            {
                int bestRank = int.MaxValue;
                (string, string) bestPair = default;
                bool found = false;
                foreach (var pair in pairs)
                {
                    if (_ranks.TryGetValue(pair, out int rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestPair = pair;
                        found = true;
                    }
                }

                if (!found)
                    break;

                var newWord = new List<string>();
                int i = 0;
                while (i < word.Count)
                {
                    int j = IndexOf(word, bestPair.Item1, i);
                    if (j < 0)
                    {
                        for (int k = i; k < word.Count; k++)
                            newWord.Add(word[k]);
                        break;
                    }

                    for (int k = i; k < j; k++)
                        newWord.Add(word[k]);
                    i = j;
                    if (word[i] == bestPair.Item1 && i < word.Count - 1 && word[i + 1] == bestPair.Item2)
                    {
                        newWord.Add(bestPair.Item1 + bestPair.Item2);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                word = newWord;
                if (word.Count == 1)
                    break;
                pairs = GetPairs(word);
            }

            string joined = string.Join(" ", word);
            lock (_cacheLock)
            {
                _bpeCache[token] = joined;
            }

            return joined;
        }

        private static int IndexOf(List<string> word, string value, int start)
        {
            for (int i = start; i < word.Count; i++)
            {
                if (word[i] == value)
                    return i;
            }

            return -1;
        }

        private static HashSet<(string, string)> GetPairs(List<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            if (word.Count < 2)
                return pairs;
            for (int i = 0; i < word.Count - 1; i++)
                pairs.Add((word[i], word[i + 1]));
            return pairs;
        }

        private string BytesToUnicodeString(string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            var sb = new StringBuilder(bytes.Length);
            for (int i = 0; i < bytes.Length; i++)
                sb.Append(_byteToUnicode[bytes[i]]);
            return sb.ToString();
        }

        private static string MatchSpecial(string text, int index)
        {
            for (int s = 0; s < SpecialTokenKeys.Length; s++)
            {
                string key = SpecialTokenKeys[s];
                if (index + key.Length <= text.Length &&
                    string.CompareOrdinal(text, index, key, 0, key.Length) == 0)
                    return key;
            }

            return null;
        }

        private static int FindNextSpecial(string text, int start)
        {
            int best = text.Length;
            for (int s = 0; s < SpecialTokenKeys.Length; s++)
            {
                int idx = text.IndexOf(SpecialTokenKeys[s], start, StringComparison.Ordinal);
                if (idx >= 0 && idx < best)
                    best = idx;
            }

            return best;
        }

        private static Dictionary<(string, string), int> LoadMerges(string path)
        {
            var ranks = new Dictionary<(string, string), int>();
            int rank = 0;
            foreach (var raw in File.ReadLines(path))
            {
                var line = raw.TrimEnd();
                if (line.Length == 0 || line.StartsWith("#"))
                    continue;
                var parts = line.Split(' ');
                if (parts.Length < 2)
                    continue;
                ranks[(parts[0], parts[1])] = rank++;
            }

            return ranks;
        }

        /// <summary>
        /// HuggingFace GPT-2 bytes_to_unicode: printable latin-1 bytes keep their codepoint;
        /// the rest map into U+0100+.
        /// </summary>
        private static string[] BuildBytesToUnicode()
        {
            var bs = new List<int>();
            for (int i = (int)'!'; i <= (int)'~'; i++) bs.Add(i);
            for (int i = 0xA1; i <= 0xAC; i++) bs.Add(i);
            for (int i = 0xAE; i <= 0xFF; i++) bs.Add(i);

            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add(256 + n);
                    n++;
                }
            }

            var map = new string[256];
            for (int i = 0; i < bs.Count; i++)
                map[bs[i]] = char.ConvertFromUtf32(cs[i]);
            return map;
        }
    }
}
