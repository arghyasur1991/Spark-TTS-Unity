using System;
using System.Collections.Generic;

namespace SparkTTS.Core
{
    using Models;
    using Utils;
    /// <summary>
    /// C# implementation mirroring Python's BiCodec class for handling tokenization and detokenization 
    /// of audio content using a combination of encoder/quantizer and vocoder models.
    /// </summary>
    internal class SparkTTSBiCodec : IDisposable
    {
        public static DebugLogger Logger = new();

        private readonly VocoderModel _vocoderModel;
        private readonly BiCodecEncoderQuantizerModel _encoderQuantizerModel;

        public bool IsInitialized { get; private set; } = false;
        private bool _disposed = false;

        /// <summary>
        /// Initializes a new instance of the SparkTTSBiCodec class
        /// </summary>
        /// <param name="vocoderModel">ONNX model for generating waveforms from tokens</param>
        /// <param name="encoderQuantizerModel">ONNX model for encoding features to tokens</param>
        internal SparkTTSBiCodec(VocoderModel vocoderModel, BiCodecEncoderQuantizerModel encoderQuantizerModel)
        {
            _vocoderModel = vocoderModel ?? throw new ArgumentNullException(nameof(vocoderModel));
            _encoderQuantizerModel = encoderQuantizerModel ?? throw new ArgumentNullException(nameof(encoderQuantizerModel));

            if (!_vocoderModel.IsInitialized || !_encoderQuantizerModel.IsInitialized)
            {
                Logger.LogError("[SparkTTSBiCodec] One or more dependent models are not initialized.");
                IsInitialized = false;
            }
            else
            {
                IsInitialized = true;
            }
        }

        /// <summary>
        /// Tokenizes audio features into semantic tokens (Python: BiCodec.tokenize equivalent function)
        /// </summary>
        /// <param name="features">Feature tensor data</param>
        /// <param name="shape">Shape of the feature tensor</param>
        /// <returns>List of semantic token IDs</returns>
        public List<long> Tokenize(float[] features, int[] shape)
        {
            if (!IsInitialized)
            {
                Logger.LogError("[SparkTTSBiCodec.Tokenize] Not initialized.");
                return null;
            }
            
            var result = _encoderQuantizerModel.GenerateSemanticTokens(features, shape);
            if (result.HasValue)
            {
                return new List<long>(result.Value.semanticTokensData);
            }
            
            return null;
        }

        /// <summary>
        /// Detokenizes semantic and global tokens into a waveform (Python: BiCodec.detokenize equivalent)
        /// </summary>
        /// <param name="semanticTokens">Array of semantic tokens</param>
        /// <param name="globalTokens">Array of global tokens</param>
        /// <returns>Synthesized audio waveform as float array</returns>
        public float[] Detokenize(long[] semanticTokens, int[] globalTokens)
        {
            if (!IsInitialized)
            {
                Logger.LogError("[SparkTTSBiCodec.Detokenize] Not initialized.");
                return null;
            }
            
            if (semanticTokens == null || globalTokens == null)
            {
                Logger.LogError("[SparkTTSBiCodec.Detokenize] Input tokens are null.");
                return null;
            }

            // Prepare shapes for VocoderModel
            int[] semanticShape = { 1, semanticTokens.Length };
            int[] globalShape = { 1, 1, globalTokens.Length };

            return _vocoderModel.Synthesize(
                semanticTokens,
                semanticShape, 
                globalTokens, 
                globalShape
            );
        }

        /// <summary>
        /// Legacy method - converts LLM-generated semantic tokens and global speaker tokens into a waveform.
        /// Uses Detokenize internally for consistency.
        /// </summary>
        /// <param name="llmGeneratedSemanticTokens">List of semantic token IDs from the LLM.</param>
        /// <param name="globalSpeakerTokens">List of global speaker token IDs.</param>
        /// <returns>Synthesized waveform as a float array.</returns>
        public float[] DetokenizeToWaveform(List<long> llmGeneratedSemanticTokens, List<int> globalSpeakerTokens)
        {
            if (llmGeneratedSemanticTokens == null || globalSpeakerTokens == null)
            {
                Logger.LogError("[SparkTTSBiCodec.DetokenizeToWaveform] Input tokens are null.");
                return null;
            }

            return Detokenize(
                llmGeneratedSemanticTokens.ToArray(),
                globalSpeakerTokens.ToArray()
            );
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _vocoderModel?.Dispose();
                    _encoderQuantizerModel?.Dispose();
                }

                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
} 