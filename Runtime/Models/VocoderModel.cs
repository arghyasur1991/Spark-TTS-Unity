using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    internal class VocoderModel : ORTModel
    {
        private const string SemanticTokensInputName = "semantic_tokens"; // Confirm with your ONNX model export
        private const string GlobalTokensInputName = "global_tokens";     // Confirm with your ONNX model export
        private const string WaveformOutputName = "output_waveform"; // Confirm with your ONNX model export

        public static new DebugLogger Logger = new();

        internal VocoderModel(string modelFolder = null, string modelFile = null)
            : base(SparkTTSModelPaths.GetModelPath(modelFolder ?? SparkTTSModelPaths.VocoderFolder,
                                                  modelFile ?? SparkTTSModelPaths.VocoderFile))
        {
            if (IsInitialized && Logger.IsEnabled)
            {
                InspectModel("VocoderModel");
            }
        }

        /// <summary>
        /// Synthesizes a waveform from semantic and global tokens.
        /// Assumes semanticTokens are int64 and globalTokens are int32, matching common setups.
        /// </summary>
        public float[] Synthesize(long[] semanticTokens, int[] semanticTokensShape, 
                                  int[] globalTokens, int[] globalTokensShape)
        {
            if (!IsInitialized || _session == null)
            {
                Logger.LogError("[VocoderModel] Session not initialized.");
                return null;
            }
            if (semanticTokens == null || semanticTokensShape == null)
            {
                Logger.LogError("[VocoderModel] Semantic tokens array or shape is null.");
                return null;
            }
            if (globalTokens == null || globalTokensShape == null)
            {
                Logger.LogError("[VocoderModel] Global tokens array or shape is null.");
                return null;
            }

            Logger.Log($"[VocoderModel.Synthesize M1] Received:" +
                      $"\n  semanticTokens (len): {semanticTokens.Length}" +
                      $"\n  semanticTokensShape: [{string.Join(",", semanticTokensShape)}] ({semanticTokensShape.Length}D)" +
                      $"\n  globalTokens (len): {globalTokens.Length}" +
                      $"\n  globalTokensShape: [{string.Join(",", globalTokensShape)}] ({globalTokensShape.Length}D)");
                
            if (globalTokensShape.Length != 3)
            {
                Logger.LogError($"[VocoderModel.Synthesize M1] CRITICAL: globalTokensShape is not 3D! Rank is {globalTokensShape.Length}");
            }

            var inputs = new List<NamedOnnxValue>();

            try
            {
                // Semantic Tokens (expected int64)
                var semanticDenseTensor = new DenseTensor<long>(new Memory<long>(semanticTokens), new ReadOnlySpan<int>(semanticTokensShape));
                inputs.Add(NamedOnnxValue.CreateFromTensor<long>(SemanticTokensInputName, semanticDenseTensor));

                // Global Tokens (expected int32)
                var globalDenseTensor = new DenseTensor<int>(new Memory<int>(globalTokens), new ReadOnlySpan<int>(globalTokensShape));
                inputs.Add(NamedOnnxValue.CreateFromTensor<int>(GlobalTokensInputName, globalDenseTensor));
                
                using (var outputs = _session.Run(inputs))
                {
                    DisposableNamedOnnxValue outputDisposableValue = outputs.FirstOrDefault(v => v.Name == WaveformOutputName);
                    if (outputDisposableValue == null)
                    {
                        Logger.LogError($"[VocoderModel] Failed to get output tensor named '{WaveformOutputName}'.");
                        return null;
                    }

                    if (!(outputDisposableValue.Value is DenseTensor<float> outputTensor))
                    {
                        Logger.LogError($"[VocoderModel] Output '{WaveformOutputName}' is not DenseTensor<float>. Actual: {outputDisposableValue.Value?.GetType().FullName}");
                        return null;
                    }
                    return outputTensor.Buffer.ToArray();
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[VocoderModel] Error during inference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        // Overload to accept OrtValues if needed for direct passthrough from other ONNX stages,
        // though ideally, this class would abstract OrtValue creation.
        public float[] Synthesize(OrtValue semanticTokensOrtValue, OrtValue globalTokensOrtValue)
        {
            if (!IsInitialized || _session == null) { Logger.LogError("[VocoderModel] Session not initialized."); return null; }
            if (semanticTokensOrtValue == null || globalTokensOrtValue == null) { Logger.LogError("[VocoderModel] Input OrtValues are null."); return null; }

            Logger.Log($"[VocoderModel.Synthesize M2-OrtValue] Received:" +
                      $"\n  semanticTokensOrtValue Shape: [{string.Join(",", semanticTokensOrtValue.GetTensorTypeAndShape().Shape)}] ({semanticTokensOrtValue.GetTensorTypeAndShape().Shape.Length}D)" +
                      $"\n  globalTokensOrtValue Shape: [{string.Join(",", globalTokensOrtValue.GetTensorTypeAndShape().Shape)}] ({globalTokensOrtValue.GetTensorTypeAndShape().Shape.Length}D)");
                
            if (globalTokensOrtValue.GetTensorTypeAndShape().Shape.Length != 3)
            {
                Logger.LogError($"[VocoderModel.Synthesize M2-OrtValue] CRITICAL: globalTokensOrtValue shape is not 3D! Rank is {globalTokensOrtValue.GetTensorTypeAndShape().Shape.Length}");
            }

            var inputs = new List<NamedOnnxValue>();
            
            try
            {
                // Prepare Semantic Tokens Input from OrtValue (expected Int64)
                if (semanticTokensOrtValue.IsTensor && semanticTokensOrtValue.GetTensorTypeAndShape().ElementDataType == TensorElementType.Int64)
                {
                    ReadOnlySpan<long> semanticDataSpan = semanticTokensOrtValue.GetTensorDataAsSpan<long>();
                    Memory<long> semanticDataMemory = new Memory<long>(semanticDataSpan.ToArray()); // Create Memory<T> from a copy
                    long[] semanticShapeLong = semanticTokensOrtValue.GetTensorTypeAndShape().Shape;
                    int[] semanticShapeInt = semanticShapeLong.Select(d => (int)d).ToArray(); // Convert long[] to int[]
                    DenseTensor<long> semanticDenseTensor = new DenseTensor<long>(semanticDataMemory, new ReadOnlySpan<int>(semanticShapeInt));
                    inputs.Add(NamedOnnxValue.CreateFromTensor<long>(SemanticTokensInputName, semanticDenseTensor));
                }
                else 
                {
                    Logger.LogError($"[VocoderModel] Semantic tokens OrtValue is not a Tensor or not Int64. Actual type: {(semanticTokensOrtValue.IsTensor ? semanticTokensOrtValue.GetTensorTypeAndShape().ElementDataType.ToString() : "Not a Tensor")}"); 
                    return null; 
                }

                // Prepare Global Tokens Input from OrtValue (expected Int32)
                if (globalTokensOrtValue.IsTensor && globalTokensOrtValue.GetTensorTypeAndShape().ElementDataType == TensorElementType.Int32)
                {
                    ReadOnlySpan<int> globalDataSpan = globalTokensOrtValue.GetTensorDataAsSpan<int>();
                    Memory<int> globalDataMemory = new Memory<int>(globalDataSpan.ToArray()); // Create Memory<T> from a copy
                    long[] globalShapeLong = globalTokensOrtValue.GetTensorTypeAndShape().Shape;
                    int[] globalShapeInt = globalShapeLong.Select(d => (int)d).ToArray(); 
                    DenseTensor<int> globalDenseTensor = new DenseTensor<int>(globalDataMemory, new ReadOnlySpan<int>(globalShapeInt));
                    inputs.Add(NamedOnnxValue.CreateFromTensor<int>(GlobalTokensInputName, globalDenseTensor));
                }
                else 
                {
                    Logger.LogError($"[VocoderModel] Global tokens OrtValue is not a Tensor or not Int32. Actual type: {(globalTokensOrtValue.IsTensor ? globalTokensOrtValue.GetTensorTypeAndShape().ElementDataType.ToString() : "Not a Tensor")}"); 
                    return null; 
                }

                using (var outputs = _session.Run(inputs))
                {
                    var outputValue = outputs.FirstOrDefault(v => v.Name == WaveformOutputName);
                    if (outputValue == null) { Logger.LogError($"[VocoderModel] Output '{WaveformOutputName}' not found."); return null; }
                    // Ensure the Value within DisposableNamedOnnxValue is a DenseTensor<float>
                    if (!(outputValue.Value is DenseTensor<float> outputTensor)) 
                    { 
                        Logger.LogError($"[VocoderModel] Output '{WaveformOutputName}' Value is not DenseTensor<float>. Actual Type: {outputValue.Value?.GetType().FullName}"); 
                        return null; 
                    }
                    return outputTensor.Buffer.ToArray();
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[VocoderModel] Error during OrtValue-based inference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
    }
} 