using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

            Logger.LogVerbose("[SparkTTSAudioTokenizer] Initialized with all required models.");
        }

        public void Dispose()
        {
            // ORTModel base class handles session disposal
            Logger.LogVerbose("[SparkTTSAudioTokenizer] Disposed (models handled by ORTModel base).");
        }

        /// <summary>
        /// Asynchronously tokenizes audio samples into global tokens and semantic tokens
        /// </summary>
        /// <param name="audioSamples">Raw audio samples</param>
        /// <returns>A task containing tuple of (global tokens, semantic tokens)</returns>
        public async Task<(int[] globalTokens, List<long> semanticTokens)> TokenizeAsync(float[] refAudioSamples, bool optimalMemoryUsage = false)
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
                Logger.LogVerbose("[SparkTTSAudioTokenizer] Generating Mel Spectrogram for Speaker Encoder...");
                (float[] processedMelData, int[] processedMelShape)? processedMelTuple = await _melModel.RunAsync(
                    async () => await _melModel.GenerateProcessedMelSpectrogramAsync(refAudioSamples),
                    standaloneLoading: optimalMemoryUsage);
                if (!processedMelTuple.HasValue)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] Failed to generate processed mel spectrogram.");
                    return (null, null);
                }
                Logger.LogVerbose($"[SparkTTSAudioTokenizer] Generated processed mel spectrogram with shape: [{string.Join(",", processedMelTuple.Value.processedMelShape)}]");
                
                
                Logger.LogVerbose("[SparkTTSAudioTokenizer] Generating Global Speaker Tokens...");
                globalTokens = await _speakerEncoderModel.RunAsync(
                    async () => await _speakerEncoderModel.GenerateTokensAsync(processedMelTuple.Value),
                    standaloneLoading: optimalMemoryUsage);
                if (globalTokens == null || globalTokens.Length == 0)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] Speaker encoding failed or produced null/empty global tokens array.");
                    return (null, null);
                }
                Logger.LogVerbose($"[SparkTTSAudioTokenizer] Generated {globalTokens.Length} global speaker tokens.");

                // --- 2. Get Semantic Tokens --- 
                Logger.LogVerbose("[SparkTTSAudioTokenizer] Generating Wav2Vec2 Features...");
                (float[] featuresData, int[] featuresShape)? featuresResult = await _wav2vec2Model.RunAsync(
                    async () => await _wav2vec2Model.GenerateFeaturesAsync(refAudioSamples),
                    standaloneLoading: optimalMemoryUsage);
                if (!featuresResult.HasValue)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] Wav2Vec2 feature generation failed.");
                    return (globalTokens, null); // Return global tokens, but semantic failed
                }
                Logger.LogVerbose($"[SparkTTSAudioTokenizer] Generated Wav2Vec2 features with shape: [{string.Join(",", featuresResult.Value.featuresShape)}]");

                Logger.LogVerbose("[SparkTTSAudioTokenizer] Generating Semantic Tokens from features...");
                semanticTokensResult = await _encoderQuantizerModel.RunAsync(
                    async () => await _encoderQuantizerModel.GenerateSemanticTokensAsync(featuresResult.Value.featuresData, featuresResult.Value.featuresShape),
                    standaloneLoading: optimalMemoryUsage);
                if (!semanticTokensResult.HasValue)
                {
                    Logger.LogError("[SparkTTSAudioTokenizer] BiCodec Encoder/Quantizer failed to generate semantic tokens.");
                    return (globalTokens, null); // Return global tokens, but semantic failed
                }
                semanticTokensList = new List<long>(semanticTokensResult.Value.semanticTokensData);
                Logger.LogVerbose($"[SparkTTSAudioTokenizer] BiCodecEncoderQuantizer generated {semanticTokensList.Count} semantic tokens (Shape: [{string.Join(",", semanticTokensResult.Value.semanticTokensShape)}]). First 10: [{string.Join(", ", semanticTokensList.Take(10))}]");
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