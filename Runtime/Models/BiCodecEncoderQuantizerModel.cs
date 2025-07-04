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
    /// BiCodec encoder quantizer model for generating semantic tokens from audio features.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern.
    /// </summary>
    internal class BiCodecEncoderQuantizerModel : ORTModel
    {
        // Input/Output names based on export_bicodec_encoder_quantizer_onnx.py
        private const string FeaturesInputName = "features";         // Expects (B, T_feat, D_feat), e.g., (1, 98, 1024) float32
        private const string SemanticTokensOutputName = "semantic_tokens"; // Outputs (B, num_quantizers, T_quantized), e.g., (1, 8, 49) int64

        public static new DebugLogger Logger = new();

        /// <summary>
        /// Initializes a new instance of the BiCodecEncoderQuantizerModel class.
        /// </summary>
        /// <param name="logLevel">The logging level for this model instance</param>
        public BiCodecEncoderQuantizerModel(DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning)
            : base(SparkTTSModelPaths.BiCodecEncoderQuantizerModelName, 
                   SparkTTSModelPaths.BiCodecFolder, 
                   logLevel)
        {
            Logger.Log("[BiCodecEncoderQuantizerModel] Initialized successfully");
        }

        /// <summary>
        /// Asynchronously generates semantic tokens from input features (e.g., Wav2Vec2 output).
        /// Uses the consistent LoadInput/Run pattern for professional model execution.
        /// </summary>
        /// <param name="featuresData">Float array of feature data</param>
        /// <param name="featuresShape">Shape of the features (Batch, SeqLen, FeatureDim)</param>
        /// <returns>A task containing a tuple with (semanticTokensData, semanticTokensShape) or null on error</returns>
        /// <exception cref="ArgumentNullException">Thrown when input parameters are null</exception>
        /// <exception cref="ArgumentException">Thrown when features shape is invalid</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<(long[] semanticTokensData, int[] semanticTokensShape)?> GenerateSemanticTokensAsync(
            float[] featuresData, 
            int[] featuresShape)
        {
            if (featuresData == null)
                throw new ArgumentNullException(nameof(featuresData));
            if (featuresShape == null)
                throw new ArgumentNullException(nameof(featuresShape));
            if (featuresShape.Length != 3)
                throw new ArgumentException("Features shape must have 3 dimensions (Batch, SeqLen, FeatureDim)", nameof(featuresShape));
            
            Logger.Log($"[BiCodecEncoderQuantizerModel] Input features shape: [{string.Join(",", featuresShape)}]");
            
            // Create input tensor
            var featuresTensor = new DenseTensor<float>(featuresData, featuresShape);
            var inputs = new List<Tensor<float>> { featuresTensor };
            
            try
            {
                // Use the new LoadInput/Run pattern
                var outputs = await Run(inputs);
                
                // Get the first output (semantic tokens)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from BiCodec encoder quantizer model");
                }
                
                // Expected output type is int64 based on export script (torch.long)
                if (!(outputValue.Value is DenseTensor<long> outputTensor))
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}. Expected DenseTensor<long>");
                }
                
                var tokensData = outputTensor.Buffer.ToArray(); // Creates a copy
                var tokensShape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray(); // Ensure int[] shape
                
                Logger.Log($"[BiCodecEncoderQuantizerModel] Output semantic tokens shape: [{string.Join(",", tokensShape)}]");
                
                return (tokensData, tokensShape);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"BiCodec encoder quantizer inference failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Synchronous wrapper for generating semantic tokens from input features.
        /// Uses the asynchronous implementation internally.
        /// </summary>
        /// <param name="featuresData">Float array of feature data</param>
        /// <param name="featuresShape">Shape of the features (Batch, SeqLen, FeatureDim)</param>
        /// <returns>A tuple containing (semanticTokensData, semanticTokensShape) or null on error</returns>
        [Obsolete("Use GenerateSemanticTokensAsync for better performance. This synchronous method will be removed in future versions.")]
        public (long[] semanticTokensData, int[] semanticTokensShape)? GenerateSemanticTokens(
            float[] featuresData, 
            int[] featuresShape)
        {
            return GenerateSemanticTokensAsync(featuresData, featuresShape).GetAwaiter().GetResult();
        }
    }
} 