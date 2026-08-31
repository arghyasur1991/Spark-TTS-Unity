// Ported from ElBruno.QwenTTS (MIT) — https://github.com/elbruno/ElBruno.QwenTTS
// Qwen3-TTS ONNX inference. Public SparkTTS CharacterVoice APIs stay unchanged.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// Embedding matrices in AllocHGlobal. Editor keep-alive stashes the
    /// pointers across domain reload so 1.5 GB npy + CP projection tables
    /// are not rebuilt.
    /// </summary>
    internal sealed class EmbeddingStore : IDisposable
    {
        public const int CpGroupCount = 15;
        public const int NativeSlotCount = 39;

        NativeFloatBuffer _textEmbedding;
        NativeFloatBuffer _fc1Weight;
        NativeFloatBuffer _fc1Bias;
        NativeFloatBuffer _fc2Weight;
        NativeFloatBuffer _fc2Bias;
        NativeFloatBuffer _talkerCodecEmbedding;
        readonly NativeFloatBuffer[] _cpCodecEmbeddings = new NativeFloatBuffer[CpGroupCount];
        Dictionary<string, int> _speakerIds;

        NativeFloatBuffer _cpProjectionWeight;
        NativeFloatBuffer _cpProjectionBias;
        NativeFloatBuffer[] _projectedCpCodecEmbeddings;
        NativeFloatBuffer _projectedTalkerCodecEmbedding;

        readonly int _textHiddenSize;
        readonly int _fc1OutSize;
        readonly int _hiddenSize;
        readonly int _cpHiddenSize;
        readonly int _cpModelHiddenSize;
        bool _ownsNative = true;

        public ModelConfig Config { get; }

        public int HiddenSize => _hiddenSize;
        public int TextHiddenSize => _textHiddenSize;
        public int CpHiddenSize => _cpHiddenSize;
        public bool HasCpProjection => _cpProjectionWeight != null && !_cpProjectionWeight.IsEmpty;
        public int CpModelHiddenSize => _cpModelHiddenSize;

        public EmbeddingStore(string embeddingsDir, string configPath)
        {
            Config = LoadConfig(configPath);

            NativeFloatBuffer text = null, fc1w = null, fc1b = null, fc2w = null, fc2b = null, talker = null;
            var cpLocal = new NativeFloatBuffer[CpGroupCount];
            Parallel.Invoke(
                () => text = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_embedding.npy")),
                () => fc1w = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_projection_fc1_weight.npy")),
                () => fc1b = NpyReader.ReadNative1D(Path.Combine(embeddingsDir, "text_projection_fc1_bias.npy")),
                () => fc2w = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "text_projection_fc2_weight.npy")),
                () => fc2b = NpyReader.ReadNative1D(Path.Combine(embeddingsDir, "text_projection_fc2_bias.npy")),
                () => talker = NpyReader.ReadNative2D(Path.Combine(embeddingsDir, "talker_codec_embedding.npy")),
                () =>
                {
                    Parallel.For(0, CpGroupCount, i =>
                    {
                        cpLocal[i] = NpyReader.ReadNative2D(
                            Path.Combine(embeddingsDir, $"cp_codec_embedding_{i}.npy"));
                    });
                });
            _textEmbedding = text;
            _fc1Weight = fc1w;
            _fc1Bias = fc1b;
            _fc2Weight = fc2w;
            _fc2Bias = fc2b;
            _talkerCodecEmbedding = talker;
            for (int i = 0; i < CpGroupCount; i++)
                _cpCodecEmbeddings[i] = cpLocal[i];

            _speakerIds = LoadSpeakerIds(Path.Combine(embeddingsDir, "speaker_ids.json"));
            _textHiddenSize = _textEmbedding.Cols;
            _fc1OutSize = _fc1Weight.Rows;
            _hiddenSize = _fc2Weight.Rows;
            _cpHiddenSize = _cpCodecEmbeddings[0].Cols;
            _cpModelHiddenSize = Config.code_predictor.hidden_size > 0
                ? Config.code_predictor.hidden_size
                : _cpHiddenSize;

            var projWeightPath = Path.Combine(embeddingsDir, "cp_projection_weight.npy");
            var projBiasPath = Path.Combine(embeddingsDir, "cp_projection_bias.npy");
            if (File.Exists(projWeightPath) && File.Exists(projBiasPath))
            {
                _cpProjectionWeight = NpyReader.ReadNative2D(projWeightPath);
                _cpProjectionBias = NpyReader.ReadNative1D(projBiasPath);
                if (_cpProjectionWeight.Rows != _cpProjectionBias.Rows)
                    throw new InvalidDataException(
                        $"CP projection dimension mismatch: weight rows ({_cpProjectionWeight.Rows}) != bias length ({_cpProjectionBias.Rows})");
                if (_cpProjectionWeight.Cols != _hiddenSize)
                    throw new InvalidDataException(
                        $"CP projection input mismatch: weight columns ({_cpProjectionWeight.Cols}) != hidden_size ({_hiddenSize})");
                PrecomputeProjected();
            }
        }

        EmbeddingStore(string embeddingsDir, string configPath, NativeEmbSlot[] slots)
        {
            Config = LoadConfig(configPath);
            _speakerIds = LoadSpeakerIds(Path.Combine(embeddingsDir, "speaker_ids.json"));
            if (slots == null || slots.Length != NativeSlotCount)
                throw new ArgumentException("Keep-alive embedding slot count is invalid.");

            _textEmbedding = Wrap(slots[0]);
            _fc1Weight = Wrap(slots[1]);
            _fc1Bias = Wrap(slots[2]);
            _fc2Weight = Wrap(slots[3]);
            _fc2Bias = Wrap(slots[4]);
            _talkerCodecEmbedding = Wrap(slots[5]);
            for (int i = 0; i < CpGroupCount; i++)
                _cpCodecEmbeddings[i] = Wrap(slots[6 + i]);

            _textHiddenSize = _textEmbedding.Cols;
            _fc1OutSize = _fc1Weight.Rows;
            _hiddenSize = _fc2Weight.Rows;
            _cpHiddenSize = _cpCodecEmbeddings[0].Cols;
            _cpModelHiddenSize = Config.code_predictor.hidden_size > 0
                ? Config.code_predictor.hidden_size
                : _cpHiddenSize;

            if (!IsEmpty(slots[21]))
            {
                _cpProjectionWeight = Wrap(slots[21]);
                _cpProjectionBias = Wrap(slots[22]);
                _projectedCpCodecEmbeddings = new NativeFloatBuffer[CpGroupCount];
                for (int i = 0; i < CpGroupCount; i++)
                    _projectedCpCodecEmbeddings[i] = Wrap(slots[23 + i]);
                _projectedTalkerCodecEmbedding = Wrap(slots[38]);
            }

            _ownsNative = true;
        }

        public static EmbeddingStore FromKeepAliveSlots(string embeddingsDir, string configPath, NativeEmbSlot[] slots)
        {
            return new EmbeddingStore(embeddingsDir, configPath, slots);
        }

        public NativeEmbSlot[] DetachNativeSlots()
        {
            var slots = new NativeEmbSlot[NativeSlotCount];
            Write(slots, 0, _textEmbedding);
            Write(slots, 1, _fc1Weight);
            Write(slots, 2, _fc1Bias);
            Write(slots, 3, _fc2Weight);
            Write(slots, 4, _fc2Bias);
            Write(slots, 5, _talkerCodecEmbedding);
            for (int i = 0; i < CpGroupCount; i++)
                Write(slots, 6 + i, _cpCodecEmbeddings[i]);
            if (HasCpProjection)
            {
                Write(slots, 21, _cpProjectionWeight);
                Write(slots, 22, _cpProjectionBias);
                for (int i = 0; i < CpGroupCount; i++)
                    Write(slots, 23 + i, _projectedCpCodecEmbeddings[i]);
                Write(slots, 38, _projectedTalkerCodecEmbedding);
            }

            _ownsNative = false;
            return slots;
        }

        static NativeFloatBuffer Wrap(NativeEmbSlot slot)
        {
            if (slot.Ptr == IntPtr.Zero)
                return null;
            return NativeFloatBuffer.Wrap(slot.Ptr, slot.Rows, slot.Cols);
        }

        static bool IsEmpty(NativeEmbSlot slot) => slot.Ptr == IntPtr.Zero || slot.Rows <= 0;

        static void Write(NativeEmbSlot[] slots, int i, NativeFloatBuffer buf)
        {
            if (buf == null || buf.IsEmpty)
                return;
            slots[i] = new NativeEmbSlot { Rows = buf.Rows, Cols = buf.Cols, Ptr = buf.Ptr };
        }

        static ModelConfig LoadConfig(string configPath)
        {
            var configJson = File.ReadAllText(configPath);
            return JsonConvert.DeserializeObject<ModelConfig>(configJson)
                ?? throw new InvalidDataException("Failed to parse config.json");
        }

        static Dictionary<string, int> LoadSpeakerIds(string path)
        {
            var speakerJson = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<Dictionary<string, int>>(speakerJson)
                ?? throw new InvalidDataException("Failed to parse speaker_ids.json");
        }

        void PrecomputeProjected()
        {
            int projOutDim = _cpProjectionWeight.Rows;
            int wRows = _cpProjectionWeight.Rows;
            int wCols = _cpProjectionWeight.Cols;
            int cpHidden = _cpHiddenSize;
            int hidden = _hiddenSize;
            var wH = _cpProjectionWeight.Ptr;
            var bH = _cpProjectionBias.Ptr;

            _projectedCpCodecEmbeddings = new NativeFloatBuffer[CpGroupCount];
            var cpSrc = new IntPtr[CpGroupCount];
            var cpDstPtr = new IntPtr[CpGroupCount];
            var cpVocab = new int[CpGroupCount];
            for (int g = 0; g < CpGroupCount; g++)
            {
                cpVocab[g] = _cpCodecEmbeddings[g].Rows;
                var dst = NativeFloatBuffer.Alloc(cpVocab[g], projOutDim);
                _projectedCpCodecEmbeddings[g] = dst;
                cpSrc[g] = _cpCodecEmbeddings[g].Ptr;
                cpDstPtr[g] = dst.Ptr;
            }

            Parallel.For(0, CpGroupCount, g =>
            {
                int vocab = cpVocab[g];
                var srcH = cpSrc[g];
                var dstH = cpDstPtr[g];
                for (int t = 0; t < vocab; t++)
                    ProjectRow(wH, bH, srcH, dstH, t, cpHidden, projOutDim, wRows, wCols);
            });

            int talkerVocab = _talkerCodecEmbedding.Rows;
            _projectedTalkerCodecEmbedding = NativeFloatBuffer.Alloc(talkerVocab, projOutDim);
            var talkerSrc = _talkerCodecEmbedding.Ptr;
            var talkerDst = _projectedTalkerCodecEmbedding.Ptr;
            Parallel.For(0, talkerVocab, t =>
            {
                ProjectRow(wH, bH, talkerSrc, talkerDst, t, hidden, projOutDim, wRows, wCols);
            });
        }

        static unsafe void ProjectRow(
            IntPtr weight, IntPtr bias, IntPtr src, IntPtr dst,
            int t, int srcDim, int projOutDim, int wRows, int wCols)
        {
            float* inRow = (float*)src + (long)t * srcDim;
            float* outRow = (float*)dst + (long)t * projOutDim;
            float* w = (float*)weight;
            float* b = (float*)bias;
            for (int i = 0; i < wRows; i++)
            {
                float sum = 0;
                float* wrow = w + (long)i * wCols;
                for (int j = 0; j < wCols; j++)
                    sum += wrow[j] * inRow[j];
                outRow[i] = sum + b[i];
            }
        }

        public void TextEmbedding(int tokenId, Span<float> output)
        {
            if (output.Length != _textHiddenSize)
                throw new ArgumentException($"Output must be length {_textHiddenSize}");
            _textEmbedding.CopyRow(tokenId, output);
        }

        public void TextProjection(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != _textHiddenSize)
                throw new ArgumentException($"Input must be length {_textHiddenSize}");
            if (output.Length != _hiddenSize)
                throw new ArgumentException($"Output must be length {_hiddenSize}");

            var hidden = new float[_fc1OutSize];
            MatMul(_fc1Weight, input, hidden);
            for (int i = 0; i < _fc1OutSize; i++)
                hidden[i] = SiLU(hidden[i] + At(_fc1Bias, i));
            MatMul(_fc2Weight, hidden, output);
            for (int i = 0; i < _hiddenSize; i++)
                output[i] += At(_fc2Bias, i);
        }

        public void TalkerCodecEmbedding(int tokenId, Span<float> output)
        {
            if (output.Length != _hiddenSize)
                throw new ArgumentException($"Output must be length {_hiddenSize}");
            _talkerCodecEmbedding.CopyRow(tokenId, output);
        }

        public void CpCodecEmbedding(int groupIndex, int tokenId, Span<float> output)
        {
            if (groupIndex < 0 || groupIndex >= CpGroupCount)
                throw new ArgumentException($"groupIndex must be 0-14, got {groupIndex}");
            if (output.Length != _cpHiddenSize)
                throw new ArgumentException($"Output must be length {_cpHiddenSize}");
            _cpCodecEmbeddings[groupIndex].CopyRow(tokenId, output);
        }

        public void CpProjection(ReadOnlySpan<float> input, Span<float> output)
        {
            if (!HasCpProjection)
                throw new InvalidOperationException("CP projection weights not loaded");
            if (input.Length < _cpProjectionWeight.Cols)
                throw new ArgumentException(
                    $"CP projection input too short: got {input.Length}, need {_cpProjectionWeight.Cols}");
            if (output.Length < _cpProjectionWeight.Rows)
                throw new ArgumentException(
                    $"CP projection output too short: got {output.Length}, need {_cpProjectionWeight.Rows}");

            MatMul(_cpProjectionWeight, input, output);
            for (int i = 0; i < _cpProjectionWeight.Rows; i++)
                output[i] += At(_cpProjectionBias, i);
        }

        public void ProjectedCpCodecEmbedding(int groupIndex, int tokenId, Span<float> output)
        {
            if (_projectedCpCodecEmbeddings == null)
                throw new InvalidOperationException("Projected CP codec embeddings not available");
            if (groupIndex < 0 || groupIndex >= CpGroupCount)
                throw new ArgumentException($"groupIndex must be 0-14, got {groupIndex}");
            _projectedCpCodecEmbeddings[groupIndex].CopyRow(tokenId, output);
        }

        public void ProjectedTalkerCodecEmbedding(int tokenId, Span<float> output)
        {
            if (_projectedTalkerCodecEmbedding == null)
                throw new InvalidOperationException("Projected talker codec embedding not available");
            _projectedTalkerCodecEmbedding.CopyRow(tokenId, output);
        }

        public int GetSpeakerId(string speaker)
        {
            if (!_speakerIds.TryGetValue(speaker, out var id))
                throw new ArgumentException($"Unknown speaker: {speaker}");
            return id;
        }

        public IReadOnlyCollection<string> GetAvailableSpeakers() => _speakerIds.Keys;

        public float[] GetSpeakerEmbedding(int speakerId)
        {
            var embedding = new float[_hiddenSize];
            _talkerCodecEmbedding.CopyRow(speakerId, embedding);
            return embedding;
        }

        public IEnumerable<(string name, float[] embedding)> GetAllSpeakerEmbeddings()
        {
            foreach (var (name, id) in _speakerIds)
                yield return (name, GetSpeakerEmbedding(id));
        }

        public void Dispose()
        {
            if (!_ownsNative)
                return;
            _ownsNative = false;
            _textEmbedding?.Free();
            _fc1Weight?.Free();
            _fc1Bias?.Free();
            _fc2Weight?.Free();
            _fc2Bias?.Free();
            _talkerCodecEmbedding?.Free();
            for (int i = 0; i < CpGroupCount; i++)
                _cpCodecEmbeddings[i]?.Free();
            _cpProjectionWeight?.Free();
            _cpProjectionBias?.Free();
            if (_projectedCpCodecEmbeddings != null)
            {
                for (int i = 0; i < _projectedCpCodecEmbeddings.Length; i++)
                    _projectedCpCodecEmbeddings[i]?.Free();
            }
            _projectedTalkerCodecEmbedding?.Free();
        }

        static float SiLU(float x) => x / (1.0f + MathF.Exp(-x));

        static unsafe float At(NativeFloatBuffer buf, int i)
        {
            return ((float*)buf.Ptr)[i];
        }

        static unsafe void MatMul(NativeFloatBuffer weight, ReadOnlySpan<float> input, Span<float> output)
        {
            MatMul((float*)weight.Ptr, weight.Rows, weight.Cols, input, output);
        }

        static unsafe void MatMul(float* weight, int M, int N, ReadOnlySpan<float> input, Span<float> output)
        {
            for (int i = 0; i < M; i++)
            {
                float sum = 0;
                float* row = weight + (long)i * N;
                for (int j = 0; j < N; j++)
                    sum += row[j] * input[j];
                output[i] = sum;
            }
        }
    }

    internal sealed class ModelConfig
    {
        public TalkerConfig talker { get; set; } = new();
        public CodePredictorConfig code_predictor { get; set; } = new();
        public TtsConfig tts { get; set; } = new();
        public Dictionary<string, int> language_ids { get; set; } = new();
        public Dictionary<string, object> speaker_dialect { get; set; } = new();
    }

    internal sealed class TalkerConfig
    {
        public int codec_eos_token_id { get; set; }
        public int codec_pad_id { get; set; }
        public int codec_bos_id { get; set; }
        public int codec_think_id { get; set; }
        public int codec_nothink_id { get; set; }
        public int codec_think_bos_id { get; set; }
        public int codec_think_eos_id { get; set; }
        public int num_code_groups { get; set; }
        public int hidden_size { get; set; }
        public int text_hidden_size { get; set; }
        public int num_hidden_layers { get; set; }
        public int num_key_value_heads { get; set; }
        public int head_dim { get; set; }
        public int vocab_size { get; set; }
    }

    internal sealed class CodePredictorConfig
    {
        public int num_hidden_layers { get; set; }
        public int num_key_value_heads { get; set; }
        public int head_dim { get; set; }
        public int vocab_size { get; set; }
        public int hidden_size { get; set; }
    }

    internal sealed class TtsConfig
    {
        public int tts_bos_token_id { get; set; }
        public int tts_eos_token_id { get; set; }
        public int tts_pad_token_id { get; set; }
    }
}
