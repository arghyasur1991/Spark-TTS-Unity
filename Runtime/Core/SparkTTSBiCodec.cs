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
        public async Task<float[]> DetokenizeAsync(long[] semanticTokens, int[] globalTokens, bool standaloneLoading = true)
        {
            if (semanticTokens == null || globalTokens == null)
            {
                Logger.LogError("[SparkTTSBiCodec.Detokenize] Input tokens are null.");
                return null;
            }

            // Prepare shapes for VocoderModel
            int[] semanticShape = { 1, semanticTokens.Length };
            int[] globalShape = { 1, 1, globalTokens.Length };
            var waveform = await _vocoderModel.RunAsync(
                async () => await _vocoderModel.SynthesizeAsync(
                semanticTokens,
                semanticShape, 
                globalTokens, 
                globalShape
            ),
            standaloneLoading: standaloneLoading);
            return waveform;
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