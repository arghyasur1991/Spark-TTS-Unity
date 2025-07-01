using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    internal class MelSpectrogramModel : ORTModel
    {
        private const string InputName = "raw_waveform_with_channel"; 
        private const string OutputName = "mel_spectrogram";

        public int OutputNumMelBands { get; private set; } = 0;
        public const int TargetNumMelBandsForSpeakerEncoder = 128;
        public static new DebugLogger Logger = new();

        internal MelSpectrogramModel(string modelFolder = null, string modelFile = null) 
            : base(SparkTTSModelPaths.GetModelPath(modelFolder ?? SparkTTSModelPaths.MelSpectrogramFolder, 
                                                  modelFile ?? SparkTTSModelPaths.MelSpectrogramFile))
        {
            if (IsInitialized)
            {
                DetermineOutputMelBands();
                if (Logger.IsEnabled)
                {
                    InspectModel("MelSpectrogramModel");
                }
            }
        }

        private void DetermineOutputMelBands()
        {
            if (!IsInitialized || _session == null) return;

            if (_session.OutputMetadata.TryGetValue(OutputName, out var outputMeta))
            {
                if (outputMeta.Dimensions.Length == 3)
                {
                    OutputNumMelBands = outputMeta.Dimensions[1];
                    Logger.Log($"[MelSpectrogramModel] Dynamically determined OutputNumMelBands: {OutputNumMelBands}");
                }
                else
                {
                    Logger.LogWarning($"[MelSpectrogramModel] Could not determine OutputNumMelBands from model output '{OutputName}' shape: ({string.Join(", ", outputMeta.Dimensions)}). Expected 3 dimensions.");
                }
            }
            else
            {
                Logger.LogError($"[MelSpectrogramModel] Output name '{OutputName}' not found in model metadata. Cannot determine OutputNumMelBands.");
            }
        }

        public (float[] melData, int[] melShape)? GenerateMelSpectrogram(float[] monoAudioSamples)
        {
            if (!IsInitialized || _session == null)
            {
                Logger.LogError("[MelSpectrogramModel] Session not initialized.");
                return null;
            }
            if (monoAudioSamples == null || monoAudioSamples.Length == 0)
            {
                Logger.LogError("[MelSpectrogramModel] Input audio samples are null or empty.");
                return null;
            }

            var inputShape = new ReadOnlySpan<int>(new int[] { 1, 1, monoAudioSamples.Length });
            DenseTensor<float> inputTensor = new DenseTensor<float>(new Memory<float>(monoAudioSamples), inputShape);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>(InputName, inputTensor) };

            try
            {
                using (var outputs = _session.Run(inputs))
                {
                    DisposableNamedOnnxValue outputDisposableValue = outputs.FirstOrDefault(v => v.Name == OutputName);
                    if (outputDisposableValue == null)
                    {
                        Logger.LogError($"[MelSpectrogramModel] Failed to get output tensor named '{OutputName}'.");
                        return null;
                    }
                    
                    if (!(outputDisposableValue.Value is DenseTensor<float> outputTensor))
                    {
                        Logger.LogError($"[MelSpectrogramModel] Output '{OutputName}' is not a DenseTensor<float>. Actual type: {outputDisposableValue.Value?.GetType().FullName}");
                        return null;
                    }

                    float[] melData = outputTensor.Buffer.ToArray(); 
                    int[] melShape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();

                    if (melShape.Length == 3 && melShape[1] != OutputNumMelBands && OutputNumMelBands != 0)
                    {
                        Logger.LogWarning($"[MelSpectrogramModel] Model outputted {melShape[1]} bands, but determined OutputNumMelBands was {OutputNumMelBands}. Check consistency.");
                    }
                    else if (melShape.Length != 3)
                    {
                        Logger.LogError($"[MelSpectrogramModel] Output tensor has unexpected rank: {melShape.Length}. Expected 3. Shape: ({string.Join(",", melShape)})");
                        return null;
                    }
                    
                    return (melData, melShape);
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[MelSpectrogramModel] Error during inference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        public (float[] processedMelData, int[] processedMelShape)? ProcessMelForSpeakerEncoder((float[] rawMelData, int[] rawMelShape) rawMelTuple)
        {
            // This logic is identical to MelSpectrogramGeneratorONNX.cs, kept here for now.
            // It could be a static utility if preferred.
            if (rawMelTuple.rawMelData == null || rawMelTuple.rawMelShape == null)
            {
                Logger.LogError("[MelSpectrogramModel] Input rawMelTuple for processing is null or incomplete.");
                return null;
            }

            float[] rawMelData = rawMelTuple.rawMelData;
            int[] rawMelShape = rawMelTuple.rawMelShape;

            if (rawMelShape.Length != 3 || rawMelShape[0] != 1)
            {
                Logger.LogError($"[MelSpectrogramModel] Unexpected shape for rawMelOutput. Expected (1, MelBands, NumFrames), got ({string.Join(",", rawMelShape)})");
                return null;
            }

            int modelOutputNumBandsActual = rawMelShape[1]; // Bands from the actual raw mel output
            int numFrames = rawMelShape[2];

            // Use the dynamically determined OutputNumMelBands if available and matches, otherwise the actual from rawMelShape
            // int effectiveOutputNumBands = (OutputNumMelBands > 0 && OutputNumMelBands == modelOutputNumBandsActual) ? OutputNumMelBands : modelOutputNumBandsActual;

            if (TargetNumMelBandsForSpeakerEncoder > modelOutputNumBandsActual)
            {
                Logger.LogError($"[MelSpectrogramModel] TargetNumMelBandsForSpeakerEncoder ({TargetNumMelBandsForSpeakerEncoder}) is greater than actual modelOutputNumBands ({modelOutputNumBandsActual}). Cannot proceed.");
                return null;
            }
            
            int bandsToProcess = TargetNumMelBandsForSpeakerEncoder; 

            try
            {
                float[] permutedMelData = new float[1 * numFrames * bandsToProcess];
                int writeIdx = 0;

                for (int frame_target = 0; frame_target < numFrames; frame_target++) 
                {
                    for (int band_target = 0; band_target < bandsToProcess; band_target++) 
                    {
                        int source_band_idx = band_target; 
                        int source_frame_idx = frame_target;
                        int sourceFlatIndex = source_frame_idx + numFrames * source_band_idx;
                        permutedMelData[writeIdx++] = rawMelData[sourceFlatIndex];
                    }
                }
                
                int[] processedShape = { 1, numFrames, bandsToProcess };
                return (permutedMelData, processedShape);
            }
            catch (Exception e)
            {
                Logger.LogError($"[MelSpectrogramModel] Error processing mel spectrogram: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
    }
} 