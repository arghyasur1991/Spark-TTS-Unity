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
    /// Vocoder model for synthesizing waveforms from semantic and global tokens.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern.
    /// </summary>
    internal class VocoderModel : ORTModel
    {
        /// <summary>
        /// Initializes a new instance of the VocoderModel class.
        /// </summary>
        public VocoderModel()
            : base(SparkTTSModelPaths.VocoderModelName, 
                   SparkTTSModelPaths.VocoderFolder)
        {
            Logger.Log("[VocoderModel] Initialized successfully");
        }

        /// <summary>
        /// Asynchronously synthesizes a waveform from semantic and global tokens.
        /// Uses the consistent LoadInput/Run pattern for professional model execution.
        /// </summary>
        /// <param name="semanticTokens">The semantic tokens array (int64)</param>
        /// <param name="semanticTokensShape">Shape of the semantic tokens</param>
        /// <param name="globalTokens">The global tokens array (int32)</param>
        /// <param name="globalTokensShape">Shape of the global tokens</param>
        /// <returns>A task containing the synthesized waveform or null on error</returns>
        /// <exception cref="ArgumentNullException">Thrown when input parameters are null</exception>
        /// <exception cref="ArgumentException">Thrown when global tokens shape is invalid</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<float[]> SynthesizeAsync(
            long[] semanticTokens, int[] semanticTokensShape,
            int[] globalTokens, int[] globalTokensShape)
        {
            if (semanticTokens == null)
                throw new ArgumentNullException(nameof(semanticTokens));
            if (semanticTokensShape == null)
                throw new ArgumentNullException(nameof(semanticTokensShape));
            if (globalTokens == null)
                throw new ArgumentNullException(nameof(globalTokens));
            if (globalTokensShape == null)
                throw new ArgumentNullException(nameof(globalTokensShape));
            if (globalTokensShape.Length != 3)
                throw new ArgumentException("Global tokens shape must be 3D", nameof(globalTokensShape));

            Logger.Log($"[VocoderModel] Synthesizing with:" +
                      $"\n  semanticTokens: {semanticTokens.Length} elements, shape: [{string.Join(",", semanticTokensShape)}]" +
                      $"\n  globalTokens: {globalTokens.Length} elements, shape: [{string.Join(",", globalTokensShape)}]");

            try
            {
                // Create input tensors
                var semanticTensor = new DenseTensor<long>(semanticTokens, semanticTokensShape);
                var globalTensor = new DenseTensor<int>(globalTokens, globalTokensShape);

                // Load inputs using the consistent pattern
                await LoadInput(0, semanticTensor);
                await LoadInput(1, globalTensor);

                // Run inference
                var outputs = await RunDisposable();
                
                // Get the first output (waveform)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from vocoder model");
                }
                
                if (!(outputValue.Value is DenseTensor<float> outputTensor))
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}. Expected DenseTensor<float>");
                }
                
                var waveform = outputTensor.Buffer.ToArray();
                Logger.Log($"[VocoderModel] Successfully synthesized waveform with {waveform.Length} samples");
                
                return waveform;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Vocoder synthesis failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Asynchronously synthesizes a waveform using tensors directly.
        /// Alternative method for when you already have properly formatted tensors.
        /// </summary>
        /// <param name="semanticTokensTensor">The semantic tokens tensor (int64)</param>
        /// <param name="globalTokensTensor">The global tokens tensor (int32)</param>
        /// <returns>A task containing the synthesized waveform</returns>
        /// <exception cref="ArgumentNullException">Thrown when input tensors are null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<float[]> SynthesizeAsync(
            Tensor<long> semanticTokensTensor,
            Tensor<int> globalTokensTensor)
        {
            if (semanticTokensTensor == null)
                throw new ArgumentNullException(nameof(semanticTokensTensor));
            if (globalTokensTensor == null)
                throw new ArgumentNullException(nameof(globalTokensTensor));

            try
            {
                // Load inputs using the consistent pattern
                await LoadInput(0, semanticTokensTensor);
                
                // Convert int tensor to float tensor for compatibility
                if (globalTokensTensor is DenseTensor<int> intTensor)
                {
                    var globalTensorFloat = new DenseTensor<float>(
                        intTensor.Buffer.ToArray().Select(x => (float)x).ToArray(), 
                        intTensor.Dimensions.ToArray());
                    await LoadInput(1, globalTensorFloat);
                }
                else
                {
                    throw new ArgumentException("Global tokens tensor must be DenseTensor<int>", nameof(globalTokensTensor));
                }

                // Run inference
                var outputs = await Run();
                
                // Get the first output (waveform)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from vocoder model");
                }
                
                if (!(outputValue.Value is DenseTensor<float> outputTensor))
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}. Expected DenseTensor<float>");
                }
                
                return outputTensor.Buffer.ToArray();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Vocoder synthesis failed: {ex.Message}", ex);
            }
        }


    }
} 