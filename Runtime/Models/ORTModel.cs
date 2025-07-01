using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

        public bool IsInitialized { get; protected set; } = false;

        protected ORTModel(string modelRelativePath)
        {
            // Initialize ONNX logging for Unity console integration
            InitializeOnnxLogging();
            
            _modelPath = Path.Combine(Application.streamingAssetsPath, modelRelativePath);

            if (!File.Exists(_modelPath))
            {
                Logger.LogError($"[ORTModel] Model file not found at path: {_modelPath}");
                IsInitialized = false;
                return;
            }

            try
            {
                SessionOptions options = new()
                {
                    // Configure options as needed, e.g., LogSeverity, Execution Providers
                    LogSeverityLevel = _ortLogLevel
                };
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

        public static void InitializeEnvironment(DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning)
        {
            Logger.Level = logLevel;
            InitializeOnnxLogging();
        }

        /// <summary>
        /// Initialize ONNX Runtime with Unity logging integration
        /// </summary>
        private static void InitializeOnnxLogging()
        {
            if (OrtEnv.IsCreated || Logger.Level == DebugLogger.LogLevel.None) 
            {
                Logger.Log("[ORTModel] ONNX Runtime logging already initialized");
                return; // Already initialized
            }
            try
            {
                _ortLogLevel = Logger.Level switch
                {
                    DebugLogger.LogLevel.Error => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                    DebugLogger.LogLevel.Warning => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                    DebugLogger.LogLevel.Info => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
                    DebugLogger.LogLevel.Debug => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                    _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                };
                var options = new EnvironmentCreationOptions
                {
                    logLevel = _ortLogLevel,
                    logId = "SparkTTS",
                    loggingFunction = UnityOnnxLoggingCallback,
                    loggingParam = IntPtr.Zero
                };

                OrtEnv.CreateInstanceWithOptions(ref options);
                Logger.Log("[ORTModel] ONNX Runtime logging initialized with Unity integration");
            }
            catch (Exception e)
            {
                Logger.LogError($"[ORTModel] Failed to initialize ONNX Runtime logging: {e.Message}");
            }
        }

        /// <summary>
        /// Unity logging callback for ONNX Runtime in Spark-TTS
        /// </summary>
        private static void UnityOnnxLoggingCallback(IntPtr param, 
                                                   OrtLoggingLevel severity, 
                                                   string category, 
                                                   string logId, 
                                                   string codeLocation, 
                                                   string message)
        {
            string severityStr = severity switch
            {
                OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE => "VERBOSE",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO => "INFO", 
                OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING => "WARN",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR => "ERROR",
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL => "FATAL",
                _ => "UNKNOWN"
            };

            string cleanCategory = !string.IsNullOrEmpty(category) ? $"[{category}]" : "";
            string cleanCodeLocation = !string.IsNullOrEmpty(codeLocation) ? $" ({codeLocation})" : "";
            string formattedMessage = $"[ONNX-{severityStr}]{cleanCategory} {message}{cleanCodeLocation}";

            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    Debug.Log(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    Debug.LogWarning(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL:
                    Debug.LogError(formattedMessage);
                    break;
                    
                default:
                    Debug.Log(formattedMessage);
                    break;
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