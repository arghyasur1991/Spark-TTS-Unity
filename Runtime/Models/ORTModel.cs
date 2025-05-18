using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SparkTTS.Models
{
    using Utils;
    internal abstract class ORTModel : IDisposable
    {
        protected InferenceSession _session;
        protected readonly string _modelPath; // Full path to the .onnx model file
        protected bool _disposed = false;
        
        public static DebugLogger Logger = new();

        public bool IsInitialized { get; protected set; } = false;

        protected ORTModel(string modelRelativePath)
        {
            _modelPath = Path.Combine(Application.streamingAssetsPath, modelRelativePath);

            if (!File.Exists(_modelPath))
            {
                Logger.LogError($"[ORTModel] Model file not found at path: {_modelPath}");
                IsInitialized = false;
                return;
            }

            try
            {
                SessionOptions options = new();
                // Configure options as needed, e.g., LogSeverity, Execution Providers
                // options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                _session = new InferenceSession(_modelPath, options);
                IsInitialized = true;
                Logger.Log($"[ORTModel] Successfully loaded ONNX model and created session from: {_modelPath}");
            }
            catch (Exception ex)
            {
                Logger.LogError($"[ORTModel] Failed to create InferenceSession for model at '{_modelPath}'. Exception: {ex.Message}\n{ex.StackTrace}");
                _session = null;
                IsInitialized = false;
            }
        }

        protected void InspectModel(string modelName = "")
        {
            if (!IsInitialized || _session == null)
            {
                Logger.LogWarning($"[ORTModel{(string.IsNullOrEmpty(modelName) ? "" : $" - {modelName}")}] Session not initialized, cannot inspect model.");
                return;
            }

            StringBuilder sb = new();
            sb.AppendLine($"--- {(string.IsNullOrEmpty(modelName) ? "Model" : modelName)} Inspection (ONNX Runtime) ---");
            sb.AppendLine($"Model Path: {_modelPath}");

            sb.AppendLine("\n--- Inputs ---");
            foreach (var inputMeta in _session.InputMetadata)
            {
                string name = inputMeta.Key;
                NodeMetadata meta = inputMeta.Value;
                string typeStr = meta.IsTensor ? meta.ElementDataType.ToString() : meta.OnnxValueType.ToString();
                string shapeStr = meta.IsTensor ? $"({string.Join(", ", meta.Dimensions.Select(d => d.ToString()))})" : "N/A";
                sb.AppendLine($"  Name: {name}, Type: {typeStr}, Shape: {shapeStr}");
            }

            sb.AppendLine("\n--- Outputs ---");
            foreach (var outputMeta in _session.OutputMetadata)
            {
                string name = outputMeta.Key;
                NodeMetadata meta = outputMeta.Value;
                string typeStr = meta.IsTensor ? meta.ElementDataType.ToString() : meta.OnnxValueType.ToString();
                string shapeStr = meta.IsTensor ? $"({string.Join(", ", meta.Dimensions.Select(d => d.ToString()))})" : "N/A";
                sb.AppendLine($"  Name: {name}, Type: {typeStr}, Shape: {shapeStr}");
            }
            Logger.Log(sb.ToString());
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                if (_session != null)
                {
                    _session.Dispose();
                    _session = null;
                    Logger.Log($"[ORTModel] Disposed InferenceSession for model: {_modelPath}");
                }
            }
            IsInitialized = false;
            _disposed = true;
        }

        ~ORTModel()
        {
            Dispose(false);
        }
    }
} 