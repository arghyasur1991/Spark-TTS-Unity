using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    internal class BiCodecEncoderQuantizerModel : ORTModel
    {
        // Input/Output names based on export_bicodec_encoder_quantizer_onnx.py
        private const string FeaturesInputName = "features";         // Expects (B, T_feat, D_feat), e.g., (1, 98, 1024) float32
        private const string SemanticTokensOutputName = "semantic_tokens"; // Outputs (B, num_quantizers, T_quantized), e.g., (1, 8, 49) int64

        public static new DebugLogger Logger = new();

        public BiCodecEncoderQuantizerModel(string modelFolder = null, string modelFile = null)
            : base(SparkTTSModelPaths.GetModelPath(modelFolder ?? SparkTTSModelPaths.BiCodecEncoderQuantizerFolder, // Assumes this folder exists in SparkTTSModelPaths
                                                  modelFile ?? SparkTTSModelPaths.BiCodecEncoderQuantizerFile)) // Assumes this file exists in SparkTTSModelPaths
        {
            if (IsInitialized)
            {
                InspectModel("BiCodecEncoderQuantizerModel");
            }
        }

        /// <summary>
        /// Generates semantic tokens from input features (e.g., Wav2Vec2 output).
        /// </summary>
        /// <param name="featuresData">Float array of feature data.</param>
        /// <param name="featuresShape">Shape of the features (Batch, SeqLen, FeatureDim).</param>
        /// <returns>A tuple containing (long[] semanticTokensData, int[] semanticTokensShape) or null on error.</returns>
        public (long[] semanticTokensData, int[] semanticTokensShape)? GenerateSemanticTokens(float[] featuresData, int[] featuresShape)
        {
            if (!IsInitialized || _session == null)
            {
                Logger.LogError("[BiCodecEncoderQuantizerModel] Session not initialized.");
                return null;
            }
            if (featuresData == null || featuresShape == null || featuresShape.Length != 3)
            {
                Logger.LogError($"[BiCodecEncoderQuantizerModel] Invalid features input. Data null: {featuresData == null}, Shape null: {featuresShape == null}, Shape Rank: {featuresShape?.Length ?? 0} (expected 3).");
                return null;
            }
            
            Logger.Log($"[BiCodecEncoderQuantizerModel] Input features shape: [{string.Join(",", featuresShape)}]");
            

            var inputs = new List<NamedOnnxValue>();
            try
            {
                var featuresTensor = new DenseTensor<float>(new Memory<float>(featuresData), new ReadOnlySpan<int>(featuresShape));
                inputs.Add(NamedOnnxValue.CreateFromTensor<float>(FeaturesInputName, featuresTensor));

                using (var outputs = _session.Run(inputs))
                {
                    DisposableNamedOnnxValue outputDisposableValue = outputs.FirstOrDefault(v => v.Name == SemanticTokensOutputName);
                    if (outputDisposableValue == null)
                    {
                        Logger.LogError($"[BiCodecEncoderQuantizerModel] Failed to get output tensor named '{SemanticTokensOutputName}'. Available: {string.Join(", ", outputs.Select(o => o.Name))}");
                        return null;
                    }

                    // Expected output type is int64 based on export script (torch.long)
                    if (!(outputDisposableValue.Value is DenseTensor<long> outputTensor))
                    {
                        Logger.LogError($"[BiCodecEncoderQuantizerModel] Output '{SemanticTokensOutputName}' is not DenseTensor<long>. Actual: {outputDisposableValue.Value?.GetType().FullName}");
                        return null;
                    }

                    long[] tokensData = outputTensor.Buffer.ToArray(); // Creates a copy
                    int[] tokensShape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray(); // Ensure int[] shape

                    Logger.Log($"[BiCodecEncoderQuantizerModel] Output semantic tokens shape: [{string.Join(",", tokensShape)}]");

                    return (tokensData, tokensShape);
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[BiCodecEncoderQuantizerModel] Error during inference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
    }
} 