using System;
using System.Collections.Generic;
using System.Threading.Tasks;

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
        private readonly VocoderModel _vocoderModel;
        private bool _disposed = false;

        /// <summary>
        /// Initializes a new instance of the SparkTTSBiCodec class
        /// </summary>
        /// <param name="vocoderModel">ONNX model for generating waveforms from tokens</param>
        /// <param name="encoderQuantizerModel">ONNX model for encoding features to tokens</param>
        internal SparkTTSBiCodec(VocoderModel vocoderModel)
        {
            _vocoderModel = vocoderModel ?? throw new ArgumentNullException(nameof(vocoderModel));
        }

        /// <summary>
        /// Asynchronously detokenizes semantic and global tokens into a waveform (Python: BiCodec.detokenize equivalent)
        /// </summary>
        /// <param name="semanticTokens">Array of semantic tokens</param>
        /// <param name="globalTokens">Array of global tokens</param>
        /// <returns>A task containing synthesized audio waveform as float array</returns>
        public async Task<float[]> DetokenizeAsync(long[] semanticTokens, int[] globalTokens)
        {
            if (semanticTokens == null || globalTokens == null)
            {
                Logger.LogError("[SparkTTSBiCodec.Detokenize] Input tokens are null.");
                return null;
            }

            // Prepare shapes for VocoderModel
            int[] semanticShape = { 1, semanticTokens.Length };
            int[] globalShape = { 1, 1, globalTokens.Length };

            _vocoderModel.StartLoadingAsync();
            var waveform = await _vocoderModel.SynthesizeAsync(
                semanticTokens,
                semanticShape, 
                globalTokens, 
                globalShape
            );
            _vocoderModel.Dispose();
            return waveform;
        }

        /// <summary>
        /// Asynchronously converts LLM-generated semantic tokens and global speaker tokens into a waveform.
        /// Uses DetokenizeAsync internally for consistency.
        /// </summary>
        /// <param name="llmGeneratedSemanticTokens">List of semantic token IDs from the LLM.</param>
        /// <param name="globalSpeakerTokens">List of global speaker token IDs.</param>
        /// <returns>A task containing synthesized waveform as a float array.</returns>
        public async Task<float[]> DetokenizeToWaveformAsync(List<long> llmGeneratedSemanticTokens, List<int> globalSpeakerTokens)
        {
            if (llmGeneratedSemanticTokens == null || globalSpeakerTokens == null)
            {
                Logger.LogError("[SparkTTSBiCodec.DetokenizeToWaveform] Input tokens are null.");
                return null;
            }

            return await DetokenizeAsync(
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