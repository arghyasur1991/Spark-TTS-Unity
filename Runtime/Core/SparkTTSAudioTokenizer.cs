using System;
using System.Collections.Generic;
using System.Linq;

namespace SparkTTS.Core
{
    using Models;
    using Utils;
    /// <summary>
    /// C# equivalent of Python's BiCodecTokenizer class, responsible for tokenizing audio into
    /// global speaker tokens and semantic tokens.
    /// </summary>
    internal class SparkTTSAudioTokenizer : IDisposable
    {
        private readonly MelSpectrogramModel _melModel;
        private readonly SpeakerEncoderModel _speakerEncoderModel;
        private readonly Wav2Vec2Model _wav2vec2Model;
        private readonly BiCodecEncoderQuantizerModel _encoderQuantizerModel;

        public static DebugLogger Logger = new();

        /// <summary>
        /// Initializes a new instance of SparkTTSAudioTokenizer with required model components
        /// </summary>
        internal SparkTTSAudioTokenizer(
            AudioLoaderService audioLoader,
            MelSpectrogramModel melModel,
            SpeakerEncoderModel speakerEncoderModel,
            Wav2Vec2Model wav2vec2Model,
            BiCodecEncoderQuantizerModel encoderQuantizerModel)
        {
            _melModel = melModel ?? throw new ArgumentNullException(nameof(melModel));
            _speakerEncoderModel = speakerEncoderModel ?? throw new ArgumentNullException(nameof(speakerEncoderModel));
            _wav2vec2Model = wav2vec2Model ?? throw new ArgumentNullException(nameof(wav2vec2Model));
            _encoderQuantizerModel = encoderQuantizerModel ?? throw new ArgumentNullException(nameof(encoderQuantizerModel));

            Logger.Log("[SparkTTSAudioTokenizer] Initialized with all required models.");
        }

        public void Dispose()
        {
            // ORTModel base class handles session disposal
            Logger.Log("[SparkTTSAudioTokenizer] Disposed (models handled by ORTModel base).");
        }

        /// <summary>
        /// Tokenizes audio samples into global tokens and semantic tokens
        /// </summary>
        /// <param name="audioSamples">Raw audio samples</param>
        /// <returns>Tuple of (global tokens, semantic tokens)</returns>
        public (int[] globalTokens, List<long> semanticTokens) Tokenize(float[] audioSamples)
        {
            return TokenizeReferenceAudio(audioSamples);
        }

        /// <summary>
        /// Processes reference audio to extract global speaker tokens and semantic tokens.
        /// This matches Python's BiCodecTokenizer.tokenize method.
        /// </summary>
        /// <param name="refAudioSamples">Raw samples of the reference audio clip.</param>
        /// <returns>A tuple: (int[] globalTokens, List<long> semanticTokens). Returns (null, null) on failure.</returns>
        public (int[] globalTokens, List<long> semanticTokens) TokenizeReferenceAudio(float[] refAudioSamples)
        {
            if (refAudioSamples == null || refAudioSamples.Length == 0)
            {
                Logger.LogError("[SparkTTSAudioTokenizer] Reference audio samples are null or empty.");
                return (null, null);
            }

            int[] globalTokens = null;
            (long[] semanticTokensData, int[] semanticTokensShape)? semanticTokensResult;
            List<long> semanticTokensList;

            try
            { 
                // --- 1. Get Global Speaker Tokens --- 
                Logger.Log("[SparkTTSAudioTokenizer] Generating Mel Spectrogram for Speaker Encoder...");
                (float[] melData, int[] melShape)? rawMelTuple = _melModel.GenerateMelSpectrogram(refAudioSamples);
                if (!rawMelTuple.HasValue) { Logger.LogError("[SparkTTSAudioTokenizer] Failed to generate raw mel spectrogram."); return (null, null); }

                (float[] processedMelData, int[] processedMelShape)? processedMelTuple = _melModel.ProcessMelForSpeakerEncoder(rawMelTuple.Value);
                if (!processedMelTuple.HasValue) { Logger.LogError("[SparkTTSAudioTokenizer] Failed to process mel spectrogram for speaker encoder."); return (null, null); }
                
                Logger.Log("[SparkTTSAudioTokenizer] Generating Global Speaker Tokens...");
                globalTokens = _speakerEncoderModel.GenerateTokens(processedMelTuple.Value);
                if (globalTokens == null || globalTokens.Length == 0)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] Speaker encoding failed or produced null/empty global tokens array.");
                    return (null, null);
                }
                Logger.Log($"[SparkTTSAudioTokenizer] Generated {globalTokens.Length} global speaker tokens.");


                // --- 2. Get Semantic Tokens --- 
                Logger.Log("[SparkTTSAudioTokenizer] Generating Wav2Vec2 Features...");
                (float[] featuresData, int[] featuresShape)? featuresResult = _wav2vec2Model.GenerateFeatures(refAudioSamples);
                if (!featuresResult.HasValue)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] Wav2Vec2 feature generation failed.");
                    return (globalTokens, null); // Return global tokens, but semantic failed
                }
                Logger.Log($"[SparkTTSAudioTokenizer] Generated Wav2Vec2 features with shape: [{string.Join(",", featuresResult.Value.featuresShape)}]");

                Logger.Log("[SparkTTSAudioTokenizer] Generating Semantic Tokens from features...");
                semanticTokensResult = _encoderQuantizerModel.GenerateSemanticTokens(featuresResult.Value.featuresData, featuresResult.Value.featuresShape);
                if (!semanticTokensResult.HasValue)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] BiCodec Encoder/Quantizer failed to generate semantic tokens.");
                    return (globalTokens, null); // Return global tokens, but semantic failed
                }

                semanticTokensList = new List<long>(semanticTokensResult.Value.semanticTokensData);
                Logger.Log($"[SparkTTSAudioTokenizer] BiCodecEncoderQuantizer generated {semanticTokensList.Count} semantic tokens (Shape: [{string.Join(",", semanticTokensResult.Value.semanticTokensShape)}]). First 10: [{string.Join(", ", semanticTokensList.Take(10))}]");
                if (semanticTokensList.Count == 0)
                {
                    Logger.LogWarning("[SparkTTSAudioTokenizer] BiCodecEncoderQuantizer generated an EMPTY list of semantic tokens from reference audio.");
                }

                return (globalTokens, semanticTokensList);
            }
            catch (Exception e)
            {
                Logger.LogError($"[SparkTTSAudioTokenizer] Error during tokenization: {e.Message}\n{e.StackTrace}");
                return (globalTokens, null); // Return any global tokens obtained before error
            }
        }
    }
} 