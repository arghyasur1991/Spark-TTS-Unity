using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SparkTTS.Models
{
    using System.Diagnostics;
    using Core;
    using Utils;
    
    /// <summary>
    /// Wav2Vec2 model for generating audio features from raw audio samples.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern.
    /// </summary>
    internal class Wav2Vec2Model : ORTModel
    {
        /// <summary>
        /// Initializes a new instance of the Wav2Vec2Model class.
        /// </summary>
        public Wav2Vec2Model()
            : base(SparkTTSModelPaths.Wav2Vec2ModelName, 
                   SparkTTSModelPaths.Wav2Vec2Folder,
                  precision: Precision.FP16)
        {
            Logger.LogVerbose("[Wav2Vec2Model] Initialized successfully");
        }

        /// <summary>
        /// Asynchronously generates audio features from raw mono audio samples.
        /// Uses the consistent LoadInput/Run pattern for professional model execution.
        /// </summary>
        /// <param name="monoAudioSamples">The mono audio samples to process</param>
        /// <returns>A task containing a tuple with (features, shape) or null on error</returns>
        /// <exception cref="ArgumentNullException">Thrown when input audio samples are null</exception>
        /// <exception cref="ArgumentException">Thrown when input audio samples are empty</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<(float[] features, int[] shape)?> GenerateFeaturesAsync(float[] monoAudioSamples)
        {
            if (monoAudioSamples == null)
                throw new ArgumentNullException(nameof(monoAudioSamples));
            if (monoAudioSamples.Length == 0)
                throw new ArgumentException("Input audio samples cannot be empty", nameof(monoAudioSamples));

            // Wav2Vec2 often expects shape (batch_size, num_samples)
            var inputShape = new int[] { 1, monoAudioSamples.Length };
            var inputTensor = new DenseTensor<float>(monoAudioSamples, inputShape);
            var inputs = new List<Tensor<float>> { inputTensor };

            Logger.LogVerbose($"[Wav2Vec2Model] Running inference with input shape: [{string.Join(",", inputShape)}]");

            try
            {
                // Use the new LoadInput/Run pattern
                using var outputs = await RunDisposable(inputs);
                
                // Get the first output (features)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from Wav2Vec2 model");
                }
                
                if (!(outputValue.Value is DenseTensor<float> outputTensor))
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}. Expected DenseTensor<float>");
                }
                
                var features = outputTensor.Buffer.ToArray();
                var shape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();
                
                Logger.LogVerbose($"[Wav2Vec2Model] Successfully generated features. Shape: [{string.Join(",", shape)}]");
                
                return (features, shape);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Wav2Vec2 feature generation failed: {ex.Message}", ex);
            }
        }


    }
} 