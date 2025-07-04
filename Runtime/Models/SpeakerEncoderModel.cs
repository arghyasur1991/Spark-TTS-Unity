using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    
    /// <summary>
    /// Speaker encoder model for generating global tokens from mel spectrograms.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern.
    /// </summary>
    internal class SpeakerEncoderModel : ORTModel
    {
        /// <summary>
        /// Initializes a new instance of the SpeakerEncoderModel class.
        /// </summary>
        /// <param name="logLevel">The logging level for this model instance</param>
        public SpeakerEncoderModel(DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning) 
            : base(SparkTTSModelPaths.SpeakerEncoderModelName, 
                   SparkTTSModelPaths.SpeakerEncoderFolder, 
                   logLevel)
        {
            Logger.Log("[SpeakerEncoderModel] Initialized successfully");
        }

        /// <summary>
        /// Asynchronously generates global tokens from mel spectrogram input.
        /// Uses the consistent LoadInput/Run pattern for professional model execution.
        /// </summary>
        /// <param name="melSpectrogramTuple">The mel spectrogram data and shape tuple</param>
        /// <returns>A task containing the generated global tokens array</returns>
        /// <exception cref="ArgumentNullException">Thrown when input tuple is null or incomplete</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<int[]> GenerateTokensAsync((float[] melData, int[] melShape) melSpectrogramTuple)
        {
            if (melSpectrogramTuple.melData == null || melSpectrogramTuple.melShape == null)
                throw new ArgumentNullException(nameof(melSpectrogramTuple), "Input melSpectrogramTuple is null or incomplete.");

            var melData = melSpectrogramTuple.melData;
            var melShape = melSpectrogramTuple.melShape;
            
            // Create input tensor
            var inputTensor = new DenseTensor<float>(melData, melShape);
            var inputs = new List<Tensor<float>> { inputTensor };
            
            try
            {
                // Use the new LoadInput/Run pattern
                var outputs = await Run(inputs);
                
                // Get the first output (global tokens)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from speaker encoder model");
                }
                
                // Handle different output types
                if (outputValue.Value is DenseTensor<int> outputTensorInt32)
                {
                    return outputTensorInt32.Buffer.ToArray();
                }
                else if (outputValue.Value is DenseTensor<long> outputTensorInt64)
                {
                    Logger.LogWarning("[SpeakerEncoderModel] Received int64 tokens, converting to int32");
                    return outputTensorInt64.Buffer.ToArray().Select(l => (int)l).ToArray();
                }
                else
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}");
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Speaker encoder inference failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Synchronous wrapper for generating global tokens from mel spectrogram input.
        /// Uses the asynchronous implementation internally.
        /// </summary>
        /// <param name="melSpectrogramTuple">The mel spectrogram data and shape tuple</param>
        /// <returns>The generated global tokens array</returns>
        [Obsolete("Use GenerateTokensAsync for better performance. This synchronous method will be removed in future versions.")]
        public int[] GenerateTokens((float[] melData, int[] melShape) melSpectrogramTuple)
        {
            return GenerateTokensAsync(melSpectrogramTuple).GetAwaiter().GetResult();
        }
    }
} 