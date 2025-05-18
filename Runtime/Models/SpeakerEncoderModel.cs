using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    internal class SpeakerEncoderModel : ORTModel
    {
        private const string MelInputName = "mel_spectrogram"; 
        private const string GlobalTokensOutputName = "global_tokens";

        public static new DebugLogger Logger = new();

        internal SpeakerEncoderModel(string modelFolder = null, string modelFile = null) 
            : base(SparkTTSModelPaths.GetModelPath(modelFolder ?? SparkTTSModelPaths.SpeakerEncoderFolder, 
                                                  modelFile ?? SparkTTSModelPaths.SpeakerEncoderFile))
        {
            if (IsInitialized && Logger.IsEnabled)
            {
                InspectModel("SpeakerEncoderModel");
            }
        }

        public int[] GenerateTokens((float[] melData, int[] melShape) melSpectrogramTuple)
        {
            if (!IsInitialized || _session == null)
            {
                Logger.LogError("[SpeakerEncoderModel] Session not initialized.");
                return null;
            }
            if (melSpectrogramTuple.melData == null || melSpectrogramTuple.melShape == null)
            {
                Logger.LogError("[SpeakerEncoderModel] Input melSpectrogramTuple is null or incomplete.");
                return null;
            }

            float[] melData = melSpectrogramTuple.melData;
            int[] melShapeInt = melSpectrogramTuple.melShape;
            var melShapeReadOnlySpan = new ReadOnlySpan<int>(melShapeInt);

            DenseTensor<float> inputTensor = new DenseTensor<float>(new Memory<float>(melData), melShapeReadOnlySpan);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>(MelInputName, inputTensor) };

            try
            {
                using (var outputs = _session.Run(inputs))
                {
                    DisposableNamedOnnxValue outputDisposableValue = outputs.FirstOrDefault(v => v.Name == GlobalTokensOutputName);
                    if (outputDisposableValue == null)
                    {
                        Logger.LogError($"[SpeakerEncoderModel] Failed to get output tensor named '{GlobalTokensOutputName}'.");
                        return null;
                    }

                    // Model output is int32 for global_tokens
                    if (outputDisposableValue.Value is DenseTensor<int> outputTensorInt32)
                    {
                        return outputTensorInt32.Buffer.ToArray();
                    }
                    else if (outputDisposableValue.Value is DenseTensor<long> outputTensorInt64) // Fallback, though not expected
                    {
                        Logger.LogWarning("[SpeakerEncoderModel] Outputted int64 tokens, converting to int[]. Expected int32.");
                        return outputTensorInt64.Buffer.ToArray().Select(l => (int)l).ToArray();
                    }
                    else
                    {
                        Logger.LogError($"[SpeakerEncoderModel] Output '{GlobalTokensOutputName}' is not DenseTensor<int> or <long>. Actual: {outputDisposableValue.Value?.GetType().FullName}");
                        return null;
                    }
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[SpeakerEncoderModel] Error during inference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
    }
} 