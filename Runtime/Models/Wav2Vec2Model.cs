using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text; // For StringBuilder

namespace SparkTTS.Models
{
    using System.Diagnostics;
    using Core;
    using Utils;
    internal class Wav2Vec2Model : ORTModel
    {
        public static new DebugLogger Logger = new();

        // Input/Output names will be determined dynamically
        private string _inputName = "input_values"; // Default assumption, confirm with inspection
        private string _outputName; // To be populated by InspectAndPopulateNames

        internal Wav2Vec2Model(string modelFolder = null, string modelFile = null)
            : base(SparkTTSModelPaths.GetModelPath(modelFolder ?? SparkTTSModelPaths.Wav2Vec2Folder,
                                                  modelFile ?? SparkTTSModelPaths.Wav2Vec2File))
        {
            if (IsInitialized)
            {
                InspectAndPopulateNames();
            }
            else
            {
                Logger.LogError($"[Wav2Vec2Model] Base model '{_modelPath}' failed to initialize. Wav2Vec2Model will not be functional.");
            }
        }

        private void InspectAndPopulateNames()
        {
            if (_session == null)
            {
                Logger.LogError("[Wav2Vec2Model.Inspect] InferenceSession is null. Cannot inspect.");
                return;
            }

            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"--- Wav2Vec2Model Inspection ({_modelPath}) ---");

            // Input Inspection (Assume one input, verify name)
            if (_session.InputMetadata.Count == 1)
            {
                _inputName = _session.InputMetadata.Keys.First();
                var meta = _session.InputMetadata[_inputName];
                sb.AppendLine($"Input Name (dynamic): {_inputName}, Type: {meta.ElementType}, Shape: ({string.Join(", ", meta.Dimensions)})");
            }
            else
            {
                Logger.LogWarning($"[Wav2Vec2Model.Inspect] Expected 1 input, found {_session.InputMetadata.Count}. Sticking with default '{_inputName}'. Inputs: {string.Join(", ", _session.InputMetadata.Keys)}");
            }


            // Output Inspection (Assume one output, store its name)
            if (_session.OutputMetadata.Count == 1)
            {
                _outputName = _session.OutputMetadata.Keys.First();
                var meta = _session.OutputMetadata[_outputName];
                sb.AppendLine($"Output Name (dynamic): {_outputName}, Type: {meta.ElementType}, Shape: ({string.Join(", ", meta.Dimensions)})");
            }
            else
            {
                // If multiple outputs, try to find one containing "hidden" or "output" as a guess
                _outputName = _session.OutputMetadata.Keys.FirstOrDefault(k => k.ToLower().Contains("hidden") || k.ToLower().Contains("output"));
                if(string.IsNullOrEmpty(_outputName))
                {
                    _outputName = _session.OutputMetadata.Keys.FirstOrDefault(); // Fallback to first if no heuristic match
                }
                Logger.Log($"[Wav2Vec2Model.Inspect] Expected 1 output, found {_session.OutputMetadata.Count}. Dynamically selecting '{_outputName}'. Outputs: {string.Join(", ", _session.OutputMetadata.Keys)}");
                if (!string.IsNullOrEmpty(_outputName))
                {
                   var meta = _session.OutputMetadata[_outputName];
                   sb.AppendLine($"Selected Output Name: {_outputName}, Type: {meta.ElementType}, Shape: ({string.Join(", ", meta.Dimensions)})");
                }
            }
            
            if (string.IsNullOrEmpty(_outputName))
            {
                Logger.LogError("[Wav2Vec2Model.Inspect] Could not determine an output name!");
            }

            Logger.Log(sb.ToString());
        }


        public (float[] features, int[] shape)? GenerateFeatures(float[] monoAudioSamples)
        {
            if (!IsInitialized || _session == null) 
            { 
                Logger.LogError("[Wav2Vec2Model] Not initialized."); 
                return null; 
            }
            if (monoAudioSamples == null || monoAudioSamples.Length == 0) 
            { 
                Logger.LogError("[Wav2Vec2Model] Input audio samples are null or empty."); 
                return null; 
            }
            if (string.IsNullOrEmpty(_inputName) || string.IsNullOrEmpty(_outputName))
            {
                Logger.LogError("[Wav2Vec2Model] Input or Output name not determined. Cannot run inference.");
                return null;
            }

            // Wav2Vec2 often expects shape (batch_size, num_samples)
            var inputShape = new ReadOnlySpan<int>(new int[] { 1, monoAudioSamples.Length });
            var inputTensor = new DenseTensor<float>(new Memory<float>(monoAudioSamples), inputShape);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>(_inputName, inputTensor) };

            Logger.Log($"[Wav2Vec2Model] Running inference with input shape: [{string.Join(",", inputShape.ToArray())}] on input '{_inputName}', expecting output '{_outputName}'");


            try
            {
                using (var outputs = _session.Run(inputs))
                {
                    DisposableNamedOnnxValue outputDisposableValue = outputs.FirstOrDefault(v => v.Name == _outputName); // Use dynamic name
                    if (outputDisposableValue == null)
                    {
                        Logger.LogError($"[Wav2Vec2Model] Failed to get output tensor named '{_outputName}'. Check model output names. Available: {string.Join(", ", outputs.Select(o => o.Name))}");
                        return null;
                    }

                    if (!(outputDisposableValue.Value is DenseTensor<float> outputTensor))
                    {
                        Logger.LogError($"[Wav2Vec2Model] Output '{_outputName}' is not DenseTensor<float>. Actual: {outputDisposableValue.Value?.GetType().FullName}");
                        return null;
                    }

                    float[] features = outputTensor.Buffer.ToArray();
                    int[] shape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();
                    
                    Logger.Log($"[Wav2Vec2Model] Successfully generated features. Shape: [{string.Join(",", shape)}]");

                    return (features, shape);
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[Wav2Vec2Model] Error during feature generation: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }
    }
} 