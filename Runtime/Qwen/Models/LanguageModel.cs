// Ported from ElBruno.QwenTTS (MIT) — https://github.com/elbruno/ElBruno.QwenTTS
// Qwen3-TTS ONNX inference. Public SparkTTS CharacterVoice APIs stay unchanged.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using SparkTTS.Core;
using SparkTTS.Models;
using SparkTTS.Qwen;

namespace SparkTTS.Qwen.Models
{

/// <summary>
/// ONNX language model for Qwen3-TTS VoiceDesign (same graph split as CustomVoice).
/// Sessions are Spark ORTModel instances (QwenOnnxModel).
/// </summary>
internal sealed class LanguageModel : IDisposable
{
    private readonly EmbeddingStore _embeddings;
    private readonly QwenOnnxModel _prefill;
    private readonly QwenOnnxModel _decode;
    private readonly QwenOnnxModel _codePredictor;
    private readonly QwenTokenSampler _sampler = new();
    private int[] _suppressTokens;

    // Dimensions from config — set once after EmbeddingStore loads config.json
    private readonly int _hiddenSize;       // talker hidden_size (1024 for 0.6B, 2048 for 1.7B)
    private readonly int _textHiddenSize;   // text embedding dim (2048 for both)
    private readonly int _cpHiddenSize;     // CP embedding dim (1024 for both)
    private readonly int _numLayers;        // talker num_hidden_layers
    private readonly int _numKvHeads;       // talker num_key_value_heads
    private readonly int _headDim;          // talker head_dim
    private readonly int _cpNumLayers;      // code_predictor num_hidden_layers
    private readonly int _cpNumKvHeads;     // code_predictor num_key_value_heads
    private readonly int _cpHeadDim;        // code_predictor head_dim

    public LanguageModel(EmbeddingStore embeddings, ExecutionProvider executionProvider = ExecutionProvider.CPU)
    {
        _embeddings = embeddings;
        string folder = SparkTTSModelPaths.QwenCustomVoiceFolder;
        _prefill = new QwenOnnxModel(SparkTTSModelPaths.QwenTalkerPrefill, folder, executionProvider);
        _decode = new QwenOnnxModel(SparkTTSModelPaths.QwenTalkerDecode, folder, executionProvider);
        _codePredictor = new QwenOnnxModel(SparkTTSModelPaths.QwenCodePredictor, folder, executionProvider);

        // Read dimensions from config.json (loaded by EmbeddingStore)
        var cfg = embeddings.Config;
        _hiddenSize = cfg.talker.hidden_size;
        _textHiddenSize = cfg.talker.text_hidden_size;
        _cpHiddenSize = embeddings.CpHiddenSize;
        _numLayers = cfg.talker.num_hidden_layers;
        _numKvHeads = cfg.talker.num_key_value_heads;
        _headDim = cfg.talker.head_dim;
        _cpNumLayers = cfg.code_predictor.num_hidden_layers;
        _cpNumKvHeads = cfg.code_predictor.num_key_value_heads;
        _cpHeadDim = cfg.code_predictor.head_dim;
        _suppressTokens = QwenTokenSampler.SuppressUpperCodec(cfg.talker.vocab_size, cfg.talker.codec_eos_token_id);
    }

    private InferenceSession GetPrefillSession() => _prefill.GetSession();

    private InferenceSession GetDecodeSession() => _decode.GetSession();

    private InferenceSession GetCpSession() => _codePredictor.GetSession();

    public long[,,] Generate(int[] tokenIds, string speaker, string language,
                             int maxNewTokens = 2048, float temperature = 0.9f,
                             int topK = 50, float topP = 1.0f,
                             float repetitionPenalty = 1.05f,
                             CancellationToken cancellationToken = default)
    {
        var speakerId = _embeddings.GetSpeakerId(speaker.ToLowerInvariant());
        return GenerateWithSpeakerId(tokenIds, speakerId, language, maxNewTokens, temperature, topK, topP, repetitionPenalty, cancellationToken);
    }

    /// <summary>
    /// Generate audio codes using an explicit speaker ID. Pass -1 to omit the speaker token
    /// from the codec prefix (used by the Base model where voice identity comes from a speaker embedding).
    /// </summary>
    public long[,,] GenerateWithSpeakerId(int[] tokenIds, int speakerId, string language,
                             int maxNewTokens = 2048, float temperature = 0.9f,
                             int topK = 50, float topP = 1.0f,
                             float repetitionPenalty = 1.05f,
                             CancellationToken cancellationToken = default)
    {
        return GenerateInternal(tokenIds, speakerId, language, speakerEmbedding: null,
            maxNewTokens, temperature, topK, topP, repetitionPenalty,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Generate audio codes using a speaker embedding vector (e.g., from ECAPA-TDNN).
    /// The 1024-dim embedding is injected at the speaker position in the codec prefix,
    /// replacing the preset speaker token lookup. Used for voice cloning with the Base model.
    /// </summary>
    public long[,,] GenerateWithSpeakerEmbedding(int[] tokenIds, float[] speakerEmbedding, string language,
                             int maxNewTokens = 2048, float temperature = 0.9f,
                             int topK = 50, float topP = 1.0f,
                             float repetitionPenalty = 1.05f,
                             CancellationToken cancellationToken = default)
    {
        return GenerateInternal(tokenIds, speakerId: -1, language, speakerEmbedding,
            maxNewTokens, temperature, topK, topP, repetitionPenalty,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Generate audio codes using a speaker embedding and optional ICL reference data.
    /// When refTokenIds and refAudioCodes are provided, enables In-Context Learning mode
    /// where the model uses both reference text and reference audio codes for higher-quality
    /// voice cloning.
    /// </summary>
    /// <param name="tokenIds">Target text token IDs from the prompt builder.</param>
    /// <param name="speakerEmbedding">1024-dim ECAPA-TDNN speaker embedding.</param>
    /// <param name="language">Language code (e.g., "english", "russian", "chinese", "auto").</param>
    /// <param name="refTokenIds">Optional reference text token IDs for ICL mode.</param>
    /// <param name="refAudioCodes">Optional reference audio codes [1, T, 16] for ICL mode.</param>
    /// <param name="maxNewTokens">Maximum number of audio frames to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <param name="topK">Top-k sampling parameter.</param>
    /// <param name="topP">Top-p (nucleus) sampling parameter.</param>
    /// <param name="repetitionPenalty">Repetition penalty factor.</param>
    /// <param name="cancellationToken">Cancellation token checked at safe inference loop boundaries.</param>
    public long[,,] GenerateWithSpeakerEmbeddingAndRefText(
        int[] tokenIds, float[] speakerEmbedding, string language,
        int[] refTokenIds = null, long[,,] refAudioCodes = null,
        int maxNewTokens = 2048, float temperature = 0.9f,
        int topK = 50, float topP = 1.0f,
        float repetitionPenalty = 1.05f,
        CancellationToken cancellationToken = default)
    {
        return GenerateInternal(tokenIds, speakerId: -1, language, speakerEmbedding,
            maxNewTokens, temperature, topK, topP, repetitionPenalty,
            refTokenIds, refAudioCodes, cancellationToken);
    }

    /// <summary>
    /// VoiceDesign: no speaker token, instruct embeddings prepended, all text in prefill.
    /// </summary>
    public long[,,] GenerateVoiceDesign(int[] assistantTokenIds, int[] instructTokenIds, string language,
                             int maxNewTokens = 2048, float temperature = 0.9f,
                             int topK = 50, float topP = 1.0f,
                             float repetitionPenalty = 1.05f,
                             CancellationToken cancellationToken = default)
    {
        return GenerateInternal(assistantTokenIds, speakerId: -1, language, speakerEmbedding: null,
            maxNewTokens, temperature, topK, topP, repetitionPenalty,
            cancellationToken: cancellationToken,
            instructTokenIds: instructTokenIds, voiceDesign: true);
    }

    private long[,,] GenerateInternal(int[] tokenIds, int speakerId, string language,
                             float[] speakerEmbedding,
                             int maxNewTokens, float temperature,
                             int topK, float topP,
                             float repetitionPenalty,
                             int[] refTokenIds = null,
                             long[,,] refAudioCodes = null,
                             CancellationToken cancellationToken = default,
                             int[] instructTokenIds = null,
                             bool voiceDesign = false)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var cfg = _embeddings.Config;
        
        var (inputsEmbeds, trailingTextHidden) = voiceDesign
            ? BuildVoiceDesignPrefill(tokenIds, instructTokenIds, language, cfg)
            : BuildPrefillEmbedding(
                tokenIds, speakerId, language, cfg, speakerEmbedding,
                refTokenIds, refAudioCodes);
        int prefillLen = inputsEmbeds.GetLength(1);

        // Attention mask: all 1s
        var attentionMask = new long[1, prefillLen];
        for (int i = 0; i < prefillLen; i++)
            attentionMask[0, i] = 1;

        // Position IDs: cumsum(mask) - 1, broadcast to (3, 1, T)
        var positionIds = new long[3, 1, prefillLen];
        for (int ax = 0; ax < 3; ax++)
            for (int i = 0; i < prefillLen; i++)
                positionIds[ax, 0, i] = i;

        // Run prefill
        var flatEmbeds = ArrayPool<float>.Shared.Rent(1 * prefillLen * _hiddenSize);
        var flatMask = ArrayPool<long>.Shared.Rent(1 * prefillLen);
        var flatPosIds = ArrayPool<long>.Shared.Rent(3 * 1 * prefillLen);
        
        float[] logits, hiddenStates;
        float[] pastKeys, pastValues;

        try
        {
            Buffer.BlockCopy(inputsEmbeds, 0, flatEmbeds, 0, 1 * prefillLen * _hiddenSize * sizeof(float));
            Buffer.BlockCopy(attentionMask, 0, flatMask, 0, 1 * prefillLen * sizeof(long));
            Buffer.BlockCopy(positionIds, 0, flatPosIds, 0, 3 * 1 * prefillLen * sizeof(long));

            var prefillSession = GetPrefillSession();
            using var embedsOrt = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<float>(flatEmbeds, 0, 1 * prefillLen * _hiddenSize), new long[] { 1, prefillLen, _hiddenSize });
            using var maskOrt = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<long>(flatMask, 0, 1 * prefillLen), new long[] { 1, prefillLen });
            using var posOrt = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, new Memory<long>(flatPosIds, 0, 3 * 1 * prefillLen), new long[] { 3, 1, prefillLen });

            var inputNames = new[] { "inputs_embeds", "attention_mask", "position_ids" };
            cancellationToken.ThrowIfCancellationRequested();
            using (var prefillOutputs = prefillSession.Run(new RunOptions(), inputNames,
                new[] { embedsOrt, maskOrt, posOrt }, prefillSession.OutputNames))
            {
                logits = prefillOutputs[0].GetTensorDataAsSpan<float>().ToArray();
                hiddenStates = prefillOutputs[1].GetTensorDataAsSpan<float>().ToArray();
                (pastKeys, pastValues) = StackPrefillKVFromOrtValues(prefillOutputs, prefillLen);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(flatEmbeds);
            ArrayPool<long>.Shared.Return(flatMask);
            ArrayPool<long>.Shared.Return(flatPosIds);
        }

        // Generation loop
        var generatedCodes = new List<long[]>();
        var generatedTokens = new List<int>();
        
        Span<float> tempEmbed2048 = stackalloc float[2048];
        var ttsEosEmbed = new float[_hiddenSize];
        var ttsPadEmbed = new float[_hiddenSize];
        _embeddings.TextEmbedding(cfg.tts.tts_eos_token_id, tempEmbed2048);
        _embeddings.TextProjection(tempEmbed2048, ttsEosEmbed);
        _embeddings.TextEmbedding(cfg.tts.tts_pad_token_id, tempEmbed2048);
        _embeddings.TextProjection(tempEmbed2048, ttsPadEmbed);

        // Reusable buffers for the generation loop
        var group0EmbedBuf = new float[_hiddenSize];
        var cpEmbedBuf = new float[_cpHiddenSize];
        var nextInputBuf = new float[_hiddenSize];

        // CP input dimension: use config-driven value when projection exists, else full hiddenSize
        int cpInputDim = _embeddings.HasCpProjection ? _embeddings.CpModelHiddenSize : _hiddenSize;

        // Rent per-step buffers before loop — size mask to max possible sequence length
        var pooledMask = ArrayPool<long>.Shared.Rent(prefillLen + maxNewTokens + 1);
        var pooledCpInputs = ArrayPool<float>.Shared.Rent(2 * cpInputDim);
        Debug.Assert(pooledCpInputs.Length >= 2 * cpInputDim, "CP prefill buffer too small");

        try
        {
            for (int step = 0; step < maxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Suppress EOS for min_new_tokens=2 (matching Python reference)
                if (step < 2)
                {
                    int eosIdx = logits.Length - cfg.talker.vocab_size + cfg.talker.codec_eos_token_id;
                    logits[eosIdx] = float.NegativeInfinity;
                }

                // Sample group 0
                var group0Token = _sampler.Sample(
                    logits, cfg.talker.vocab_size, temperature, topK, topP,
                    generatedTokens, repetitionPenalty, _suppressTokens);
                if (group0Token == cfg.talker.codec_eos_token_id)
                    break;

                generatedTokens.Add(group0Token);

                // Generate groups 1-15 via Code Predictor
                var codes = new long[16];
                codes[0] = group0Token;
                
                _embeddings.TalkerCodecEmbedding(group0Token, group0EmbedBuf);

                // CP prefill: concat(hidden_states[:,-1,:], group0_embed)
                if (_embeddings.HasCpProjection)
                {
                    Debug.Assert(cpInputDim <= pooledCpInputs.Length / 2, "cpInputDim exceeds half buffer");
                    
                    // Project hidden state: _hiddenSize → cpInputDim
                    int hOffset = hiddenStates.Length - _hiddenSize;
                    var lastHidden = new ReadOnlySpan<float>(hiddenStates, hOffset, _hiddenSize);
                    var projectedHidden = new Span<float>(pooledCpInputs, 0, cpInputDim);
                    _embeddings.CpProjection(lastHidden, projectedHidden);

                    // Use pre-projected talker codec embedding (computed at init)
                    var projectedGroup0 = new Span<float>(pooledCpInputs, cpInputDim, cpInputDim);
                    _embeddings.ProjectedTalkerCodecEmbedding(group0Token, projectedGroup0);
                }
                else
                {
                    // No projection: copy full cpInputDim elements directly
                    int hOffset = hiddenStates.Length - _hiddenSize;
                    BuildCpPrefillDirect(pooledCpInputs, hiddenStates.AsSpan(), hOffset, group0EmbedBuf, cpInputDim);
                }

                var (cpPastKeys, cpPastValues) = InitCPKV();
                int cpPastLen = 0;

                for (int groupIdx = 1; groupIdx < 16; groupIdx++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int cpInputSeqLen = groupIdx == 1 ? 2 : 1;
                    int flatCpSize = cpInputSeqLen * cpInputDim;
                    
                    var flatCpEmbeds = ArrayPool<float>.Shared.Rent(flatCpSize);
                    try
                    {
                        Debug.Assert(flatCpSize <= flatCpEmbeds.Length, "flatCpSize exceeds rented buffer");
                        Buffer.BlockCopy(pooledCpInputs, 0, flatCpEmbeds, 0, flatCpSize * sizeof(float));
                        
                        var cpInputsList = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("inputs_embeds", new DenseTensor<float>(flatCpEmbeds.AsMemory(0, flatCpSize), new[] { 1, cpInputSeqLen, cpInputDim })),
                            NamedOnnxValue.CreateFromTensor("generation_steps", new DenseTensor<long>(new long[] { groupIdx - 1 }, new[] { 1 })),
                            NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(cpPastKeys, new[] { _cpNumLayers, 1, _cpNumKvHeads, cpPastLen, _cpHeadDim })),
                            NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(cpPastValues, new[] { _cpNumLayers, 1, _cpNumKvHeads, cpPastLen, _cpHeadDim }))
                        };

                        cancellationToken.ThrowIfCancellationRequested();
                        using var cpOutputs = GetCpSession().Run(cpInputsList);
                        var cpLogits = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(cpOutputs, "logits"));
                        var cpToken = _sampler.Sample(
                            cpLogits, cfg.code_predictor.vocab_size, temperature, 50, 1f,
                            null, 1f, null);
                        codes[groupIdx] = cpToken;

                        // Update CP KV
                        cpPastLen += cpInputSeqLen;
                        cpPastKeys = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(cpOutputs, "present_keys"));
                        cpPastValues = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(cpOutputs, "present_values"));

                        // Next input
                        if (groupIdx < 15)
                        {
                            Debug.Assert(cpInputDim <= pooledCpInputs.Length, "cpInputDim exceeds pooled buffer for next input");
                            if (_embeddings.HasCpProjection)
                            {
                                // Use pre-projected embedding (computed at init, avoids per-step matrix multiply)
                                _embeddings.ProjectedCpCodecEmbedding(groupIdx - 1, cpToken, new Span<float>(pooledCpInputs, 0, cpInputDim));
                            }
                            else
                            {
                                // No projection needed (0.6B: embedding dim matches CP input dim)
                                _embeddings.CpCodecEmbedding(groupIdx - 1, cpToken, cpEmbedBuf);
                                for (int i = 0; i < cpInputDim; i++)
                                    pooledCpInputs[i] = i < _cpHiddenSize ? cpEmbedBuf[i] : 0f;
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(flatCpEmbeds);
                    }
                }

                generatedCodes.Add(codes);

                // Build next Talker input: sum all group embeddings + trailing text
                _embeddings.TalkerCodecEmbedding((int)codes[0], nextInputBuf);
                for (int g = 1; g < 16; g++)
                {
                    _embeddings.CpCodecEmbedding(g - 1, (int)codes[g], cpEmbedBuf);
                    AccumulateCpEmbedding(nextInputBuf, cpEmbedBuf, _cpHiddenSize);
                }

                // Add trailing text hidden or tts_pad
                if (step < trailingTextHidden.GetLength(0))
                {
                    for (int i = 0; i < _hiddenSize; i++)
                        nextInputBuf[i] += trailingTextHidden[step, i];
                }
                else
                {
                    for (int i = 0; i < _hiddenSize; i++)
                        nextInputBuf[i] += ttsPadEmbed[i];
                }

                // Update attention mask and position IDs
                int newLen = prefillLen + step + 1;
                for (int i = 0; i < newLen; i++)
                    pooledMask[i] = 1;

                var newPositionIds = new long[3, 1, 1];
                for (int ax = 0; ax < 3; ax++)
                    newPositionIds[ax, 0, 0] = prefillLen + step;

                // Run decode
                var flatDecodePos = new long[3];
                Buffer.BlockCopy(newPositionIds, 0, flatDecodePos, 0, 3 * sizeof(long));

                var decodeInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("inputs_embeds", new DenseTensor<float>(nextInputBuf, new[] { 1, 1, _hiddenSize })),
                    NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(pooledMask.AsMemory(0, newLen), new[] { 1, newLen })),
                    NamedOnnxValue.CreateFromTensor("position_ids", new DenseTensor<long>(flatDecodePos, new[] { 3, 1, 1 })),
                    NamedOnnxValue.CreateFromTensor("past_keys", new DenseTensor<float>(pastKeys, new[] { _numLayers, 1, _numKvHeads, prefillLen + step, _headDim })),
                    NamedOnnxValue.CreateFromTensor("past_values", new DenseTensor<float>(pastValues, new[] { _numLayers, 1, _numKvHeads, prefillLen + step, _headDim }))
                };

                cancellationToken.ThrowIfCancellationRequested();
                using var decodeOutputs = GetDecodeSession().Run(decodeInputs);
                logits = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(decodeOutputs, "logits"));
                hiddenStates = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(decodeOutputs, "hidden_states"));
                pastKeys = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(decodeOutputs, "present_keys"));
                pastValues = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(decodeOutputs, "present_values"));
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(pooledMask);
            ArrayPool<float>.Shared.Return(pooledCpInputs);
        }

        // Convert to (1, 16, T)
        int T = generatedCodes.Count;
        var result = new long[1, 16, T];
        for (int t = 0; t < T; t++)
            for (int g = 0; g < 16; g++)
                result[0, g, t] = generatedCodes[t][g];

        return result;
    }

    /// <summary>
    /// VoiceDesign non-streaming prefill: instruct embeds, assistant role, codec prefix
    /// without a speaker token, all text + eos in prefill, trailing is tts_pad only.
    /// Matches tools/qwen3_tts_onnx/generate_onnx.py.
    /// </summary>
    private (float[,,], float[,]) BuildVoiceDesignPrefill(
        int[] assistantTokenIds, int[] instructTokenIds, string language, ModelConfig cfg)
    {
        if (assistantTokenIds == null || assistantTokenIds.Length < 8)
            throw new ArgumentException("Assistant prompt produced too few tokens.", nameof(assistantTokenIds));

        var allEmbeds = new List<float[]>();
        var textEmbBuf = new float[2048];
        var projBuf = new float[_hiddenSize];

        if (instructTokenIds != null)
        {
            for (int i = 0; i < instructTokenIds.Length; i++)
            {
                _embeddings.TextEmbedding(instructTokenIds[i], textEmbBuf);
                _embeddings.TextProjection(textEmbBuf, projBuf);
                allEmbeds.Add((float[])projBuf.Clone());
            }
        }

        for (int i = 0; i < 3; i++)
        {
            _embeddings.TextEmbedding(assistantTokenIds[i], textEmbBuf);
            _embeddings.TextProjection(textEmbBuf, projBuf);
            allEmbeds.Add((float[])projBuf.Clone());
        }

        Span<float> ttsPadTextEmbed = stackalloc float[2048];
        Span<float> ttsBosTextEmbed = stackalloc float[2048];
        Span<float> ttsEosTextEmbed = stackalloc float[2048];
        var ttsPadProj = new float[_hiddenSize];
        var ttsBosProj = new float[_hiddenSize];
        var ttsEosProj = new float[_hiddenSize];
        _embeddings.TextEmbedding(cfg.tts.tts_pad_token_id, ttsPadTextEmbed);
        _embeddings.TextProjection(ttsPadTextEmbed, ttsPadProj);
        _embeddings.TextEmbedding(cfg.tts.tts_bos_token_id, ttsBosTextEmbed);
        _embeddings.TextProjection(ttsBosTextEmbed, ttsBosProj);
        _embeddings.TextEmbedding(cfg.tts.tts_eos_token_id, ttsEosTextEmbed);
        _embeddings.TextProjection(ttsEosTextEmbed, ttsEosProj);

        var codecPrefix = new List<int>();
        if (!string.Equals(language, "auto", StringComparison.OrdinalIgnoreCase))
        {
            codecPrefix.Add(cfg.talker.codec_think_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            var normalizedLanguage = language.ToLowerInvariant();
            if (!cfg.language_ids.TryGetValue(normalizedLanguage, out var languageId))
            {
                var supportedLanguages = string.Join(", ", cfg.language_ids.Keys.OrderBy(x => x));
                throw new ArgumentException(
                    $"Unsupported language '{language}'. Supported languages: {supportedLanguages}",
                    nameof(language));
            }
            codecPrefix.Add(languageId);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }
        else
        {
            codecPrefix.Add(cfg.talker.codec_nothink_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }

        var codecEmbBuf = new float[_hiddenSize];
        for (int i = 0; i < codecPrefix.Count; i++)
        {
            _embeddings.TalkerCodecEmbedding(codecPrefix[i], codecEmbBuf);
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = ttsPadProj[j] + codecEmbBuf[j];
            allEmbeds.Add(combined);
        }

        {
            _embeddings.TalkerCodecEmbedding(cfg.talker.codec_pad_id, codecEmbBuf);
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = ttsBosProj[j] + codecEmbBuf[j];
            allEmbeds.Add(combined);
        }

        var codecPadEmbed = new float[_hiddenSize];
        _embeddings.TalkerCodecEmbedding(cfg.talker.codec_pad_id, codecPadEmbed);
        for (int i = 3; i < assistantTokenIds.Length - 5; i++)
        {
            _embeddings.TextEmbedding(assistantTokenIds[i], textEmbBuf);
            _embeddings.TextProjection(textEmbBuf, projBuf);
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = projBuf[j] + codecPadEmbed[j];
            allEmbeds.Add(combined);
        }
        {
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = ttsEosProj[j] + codecPadEmbed[j];
            allEmbeds.Add(combined);
        }

        {
            var codecBosEmbed = new float[_hiddenSize];
            _embeddings.TalkerCodecEmbedding(cfg.talker.codec_bos_id, codecBosEmbed);
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = ttsPadProj[j] + codecBosEmbed[j];
            allEmbeds.Add(combined);
        }

        var trailingList = new List<float[]> { ttsPadProj };

        var inputsEmbeds = new float[1, allEmbeds.Count, _hiddenSize];
        for (int i = 0; i < allEmbeds.Count; i++)
            for (int j = 0; j < _hiddenSize; j++)
                inputsEmbeds[0, i, j] = allEmbeds[i][j];

        var trailingTextHidden = new float[trailingList.Count, _hiddenSize];
        for (int i = 0; i < trailingList.Count; i++)
            for (int j = 0; j < _hiddenSize; j++)
                trailingTextHidden[i, j] = trailingList[i][j];

        return (inputsEmbeds, trailingTextHidden);
    }

    private (float[,,], float[,]) BuildPrefillEmbedding(int[] tokenIds, int speakerId, string language, ModelConfig cfg,
        float[] speakerEmbeddingOverride = null,
        int[] refTokenIds = null, long[,,] refAudioCodes = null)
    {
        // Role embed: tokens [0:3]
        var roleEmbeds = new List<float[]>();
        var roleTextEmbBuf = new float[2048];
        var roleProjBuf = new float[_hiddenSize];
        for (int i = 0; i < 3; i++)
        {
            _embeddings.TextEmbedding(tokenIds[i], roleTextEmbBuf);
            _embeddings.TextProjection(roleTextEmbBuf, roleProjBuf);
            roleEmbeds.Add((float[])roleProjBuf.Clone());
        }

        // Codec prefix — track speaker position for embedding override
        var codecPrefix = new List<int>();
        int speakerPositionInPrefix = -1; // index of speaker token in codecPrefix
        if (!string.Equals(language, "auto", StringComparison.OrdinalIgnoreCase))
        {
            codecPrefix.Add(cfg.talker.codec_think_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            var normalizedLanguage = language.ToLowerInvariant();
            if (!cfg.language_ids.TryGetValue(normalizedLanguage, out var languageId))
            {
                var supportedLanguages = string.Join(", ", cfg.language_ids.Keys.OrderBy(x => x));
                throw new ArgumentException(
                    $"Unsupported language '{language}'. Supported languages: {supportedLanguages}",
                    nameof(language));
            }
            codecPrefix.Add(languageId);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }
        else
        {
            codecPrefix.Add(cfg.talker.codec_nothink_id);
            codecPrefix.Add(cfg.talker.codec_think_bos_id);
            codecPrefix.Add(cfg.talker.codec_think_eos_id);
        }
        
        // Speaker token: add when speakerId >= 0, or when we have a speaker embedding override
        if (speakerId >= 0)
        {
            codecPrefix.Add(speakerId);
        }
        else if (speakerEmbeddingOverride != null)
        {
            // Use pad_id as placeholder — the embedding will be overridden below
            speakerPositionInPrefix = codecPrefix.Count;
            codecPrefix.Add(cfg.talker.codec_pad_id);
        }
        codecPrefix.Add(cfg.talker.codec_pad_id);
        codecPrefix.Add(cfg.talker.codec_bos_id);

        // TTS token embeds
        Span<float> ttsPadTextEmbed = stackalloc float[2048];
        Span<float> ttsBosTextEmbed = stackalloc float[2048];
        Span<float> ttsEosTextEmbed = stackalloc float[2048];
        var ttsPadProj = new float[_hiddenSize];
        var ttsBosProj = new float[_hiddenSize];
        var ttsEosProj = new float[_hiddenSize];
        
        _embeddings.TextEmbedding(cfg.tts.tts_pad_token_id, ttsPadTextEmbed);
        _embeddings.TextProjection(ttsPadTextEmbed, ttsPadProj);
        _embeddings.TextEmbedding(cfg.tts.tts_bos_token_id, ttsBosTextEmbed);
        _embeddings.TextProjection(ttsBosTextEmbed, ttsBosProj);
        _embeddings.TextEmbedding(cfg.tts.tts_eos_token_id, ttsEosTextEmbed);
        _embeddings.TextProjection(ttsEosTextEmbed, ttsEosProj);

        // _talker_input_embed = concat(tts_pad.expand(codec_prefix_len - 2), tts_bos) + codec_prefix[:-1]
        var talkerInputEmbeds = new List<float[]>();
        int codecPrefixLen = codecPrefix.Count;
        var codecEmbBuf = new float[_hiddenSize];
        
        for (int i = 0; i < codecPrefixLen - 2; i++)
        {
            var combined = new float[_hiddenSize];
            if (i == speakerPositionInPrefix && speakerEmbeddingOverride != null)
            {
                // Inject the speaker embedding directly instead of codec table lookup.
                // Embedding may be shorter than _hiddenSize (e.g., 1024-dim ECAPA-TDNN
                // with 2048-dim talker on 1.7B) — zero-pad the upper dimensions.
                int embLen = speakerEmbeddingOverride.Length;
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = ttsPadProj[j] + (j < embLen ? speakerEmbeddingOverride[j] : 0f);
            }
            else
            {
                _embeddings.TalkerCodecEmbedding(codecPrefix[i], codecEmbBuf);
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = ttsPadProj[j] + codecEmbBuf[j];
            }
            talkerInputEmbeds.Add(combined);
        }

        // Last position: tts_bos + codec_prefix[-2] (always included, both ICL and non-ICL)
        {
            _embeddings.TalkerCodecEmbedding(codecPrefix[codecPrefixLen - 2], codecEmbBuf);
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = ttsBosProj[j] + codecEmbBuf[j];
            talkerInputEmbeds.Add(combined);
        }

        // ICL section: ref_text + target_text + eos (+ codec_pad), then codec_bos + ref_audio (+ tts_pad)
        // Matches official Qwen3-TTS generate_icl_prompt (non-streaming mode).
        bool hasIclData = refTokenIds != null && refAudioCodes != null;
        if (hasIclData)
        {
            var codecPadEmbed = new float[_hiddenSize];
            _embeddings.TalkerCodecEmbedding(cfg.talker.codec_pad_id, codecPadEmbed);

            // Ref text tokens: textProj(textEmbed(refToken)) + codec_pad
            var iclTextEmbBuf = new float[2048];
            var iclProjBuf = new float[_hiddenSize];
            for (int i = 0; i < refTokenIds!.Length; i++)
            {
                _embeddings.TextEmbedding(refTokenIds[i], iclTextEmbBuf);
                _embeddings.TextProjection(iclTextEmbBuf, iclProjBuf);
                var combined = new float[_hiddenSize];
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = iclProjBuf[j] + codecPadEmbed[j];
                talkerInputEmbeds.Add(combined);
            }

            // Target text tokens (tokenIds[3:-5]): textProj(textEmbed(token)) + codec_pad
            for (int i = 3; i < tokenIds.Length - 5; i++)
            {
                _embeddings.TextEmbedding(tokenIds[i], iclTextEmbBuf);
                _embeddings.TextProjection(iclTextEmbBuf, iclProjBuf);
                var combined = new float[_hiddenSize];
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = iclProjBuf[j] + codecPadEmbed[j];
                talkerInputEmbeds.Add(combined);
            }

            // tts_eos + codec_pad
            {
                var combined = new float[_hiddenSize];
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = ttsEosProj[j] + codecPadEmbed[j];
                talkerInputEmbeds.Add(combined);
            }

            // codec_bos + tts_pad
            {
                var codecBosEmbed = new float[_hiddenSize];
                _embeddings.TalkerCodecEmbedding(cfg.talker.codec_bos_id, codecBosEmbed);
                var combined = new float[_hiddenSize];
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = ttsPadProj[j] + codecBosEmbed[j];
                talkerInputEmbeds.Add(combined);
            }

            // Ref audio codes: tts_pad + sum_g(embedding_g(codes[t][g]))
            // Group 0: TalkerCodecEmbedding, Groups 1-15: CpCodecEmbedding
            var cpEmbBuf = new float[_cpHiddenSize];
            int tFrames = refAudioCodes!.GetLength(1);
            for (int t = 0; t < tFrames; t++)
            {
                var combined = new float[_hiddenSize];
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] = ttsPadProj[j];

                // Group 0: talker codec embedding
                _embeddings.TalkerCodecEmbedding((int)refAudioCodes[0, t, 0], codecEmbBuf);
                for (int j = 0; j < _hiddenSize; j++)
                    combined[j] += codecEmbBuf[j];

                // Groups 1-15: CP codec embeddings
                for (int g = 1; g < 16; g++)
                {
                    _embeddings.CpCodecEmbedding(g - 1, (int)refAudioCodes[0, t, g], cpEmbBuf);
                    for (int j = 0; j < _cpHiddenSize; j++)
                        combined[j] += cpEmbBuf[j];
                }
                talkerInputEmbeds.Add(combined);
            }
        }

        // Combine: role_embed + _talker_input_embed
        var allEmbeds = new List<float[]>();
        allEmbeds.AddRange(roleEmbeds);
        allEmbeds.AddRange(talkerInputEmbeds);

        // Non-ICL: append first_text_token + codec_bos
        if (!hasIclData)
        {
            Span<float> textEmbed = stackalloc float[2048];
            var projected = new float[_hiddenSize];
            _embeddings.TextEmbedding(tokenIds[3], textEmbed);
            _embeddings.TextProjection(textEmbed, projected);
            
            var codecBosEmbed = new float[_hiddenSize];
            _embeddings.TalkerCodecEmbedding(cfg.talker.codec_bos_id, codecBosEmbed);
            
            var combined = new float[_hiddenSize];
            for (int j = 0; j < _hiddenSize; j++)
                combined[j] = projected[j] + codecBosEmbed[j];
            allEmbeds.Add(combined);
        }

        // Trailing text hidden
        var trailingList = new List<float[]>();
        if (hasIclData)
        {
            // ICL non-streaming: all text is in prefill, trailing is just tts_pad
            trailingList.Add(ttsPadProj.ToArray());
        }
        else
        {
            // Standard: tokens[4:-5] + tts_eos
            var trailTextEmbBuf = new float[2048];
            var trailProjBuf = new float[_hiddenSize];
            for (int i = 4; i < tokenIds.Length - 5; i++)
            {
                _embeddings.TextEmbedding(tokenIds[i], trailTextEmbBuf);
                _embeddings.TextProjection(trailTextEmbBuf, trailProjBuf);
                trailingList.Add((float[])trailProjBuf.Clone());
            }
            trailingList.Add(ttsEosProj.ToArray());
        }

        // Convert to arrays
        var inputsEmbeds = new float[1, allEmbeds.Count, _hiddenSize];
        for (int i = 0; i < allEmbeds.Count; i++)
            for (int j = 0; j < _hiddenSize; j++)
                inputsEmbeds[0, i, j] = allEmbeds[i][j];

        var trailingTextHidden = new float[trailingList.Count, _hiddenSize];
        for (int i = 0; i < trailingList.Count; i++)
            for (int j = 0; j < _hiddenSize; j++)
                trailingTextHidden[i, j] = trailingList[i][j];

        return (inputsEmbeds, trailingTextHidden);
    }

    private (float[], float[]) StackPrefillKV(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, int seqLen)
    {
        // KV tensors: present_key_0..N, present_value_0..N each (1, numKvHeads, T, headDim)
        var keys = new float[_numLayers * 1 * _numKvHeads * seqLen * _headDim];
        var values = new float[_numLayers * 1 * _numKvHeads * seqLen * _headDim];

        for (int layer = 0; layer < _numLayers; layer++)
        {
            var keyData = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(outputs, $"present_key_{layer}"));
            var valueData = QwenOnnxModel.CopyFloat(QwenOnnxModel.FindNamed(outputs, $"present_value_{layer}"));
            
            Array.Copy(keyData, 0, keys, layer * 1 * _numKvHeads * seqLen * _headDim, keyData.Length);
            Array.Copy(valueData, 0, values, layer * 1 * _numKvHeads * seqLen * _headDim, valueData.Length);
        }

        return (keys, values);
    }

    private (float[], float[]) StackPrefillKVFromOrtValues(IReadOnlyList<OrtValue> outputs, int seqLen)
    {
        // Output order: [0]=logits, [1]=hidden_states, then 56 KV tensors alternating key/value per layer
        // present_key_0 at [2], present_value_0 at [3], present_key_1 at [4], ...
        var keys = new float[_numLayers * 1 * _numKvHeads * seqLen * _headDim];
        var values = new float[_numLayers * 1 * _numKvHeads * seqLen * _headDim];
        int layerSize = 1 * _numKvHeads * seqLen * _headDim;

        for (int layer = 0; layer < _numLayers; layer++)
        {
            var keySpan = outputs[2 + layer * 2].GetTensorDataAsSpan<float>();
            var valSpan = outputs[2 + layer * 2 + 1].GetTensorDataAsSpan<float>();
            keySpan.CopyTo(keys.AsSpan(layer * layerSize));
            valSpan.CopyTo(values.AsSpan(layer * layerSize));
        }

        return (keys, values);
    }

    private (float[], float[]) InitCPKV()
    {
        // Empty KV for CP prefill: (_cpNumLayers, 1, _cpNumKvHeads, 0, _cpHeadDim)
        return (Array.Empty<float>(), Array.Empty<float>());
    }

    /// <summary>
    /// Builds CP prefill buffer by copying hidden state and group0 embedding directly (no projection).
    /// Buffer layout: [hidden[0..cpInputDim], group0[0..cpInputDim]]
    /// </summary>
    internal static void BuildCpPrefillDirect(
        float[] buffer, ReadOnlySpan<float> hiddenStates, int hOffset,
        ReadOnlySpan<float> group0Embed, int cpInputDim)
    {
        Debug.Assert(buffer.Length >= 2 * cpInputDim);
        Debug.Assert(hiddenStates.Length >= hOffset + cpInputDim);
        Debug.Assert(group0Embed.Length >= cpInputDim);
        
        for (int i = 0; i < cpInputDim; i++)
        {
            buffer[i] = hiddenStates[hOffset + i];
            buffer[cpInputDim + i] = group0Embed[i];
        }
    }

    /// <summary>
    /// Accumulates a CP codec embedding into the next talker input buffer.
    /// Only the first cpHiddenSize elements are accumulated (CP space may be smaller than talker space).
    /// </summary>
    internal static void AccumulateCpEmbedding(
        float[] nextInputBuf, ReadOnlySpan<float> cpEmbed, int cpHiddenSize)
    {
        Debug.Assert(nextInputBuf.Length >= cpHiddenSize);
        Debug.Assert(cpEmbed.Length >= cpHiddenSize);
        
        for (int i = 0; i < cpHiddenSize; i++)
            nextInputBuf[i] += cpEmbed[i];
    }

    internal void CollectOnnxModels(List<ORTModel> list)
    {
        list.Add(_prefill);
        list.Add(_decode);
        list.Add(_codePredictor);
    }

    internal void PreloadSessions()
    {
        GetPrefillSession();
        GetDecodeSession();
        GetCpSession();
    }

    public void Dispose()
    {
        _prefill.Dispose();
        _decode.Dispose();
        _codePredictor.Dispose();
    }
}
}
