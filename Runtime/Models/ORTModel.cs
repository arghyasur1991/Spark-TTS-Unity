using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;

namespace SparkTTS.Models
{
    using Core;
    using Utils;

    /// <summary>
    /// Base ONNX model wrapper for SparkTTS inference operations.
    /// Provides asynchronous loading, input/output management, and execution capabilities
    /// following the LiveTalk Model design pattern.
    /// </summary>
    internal abstract class ORTModel : IDisposable
    {
        #region Private Fields
        
        private readonly ModelConfig _config;
        private InferenceSession _session;
        private List<string> _inputNames = new();
        private List<NamedOnnxValue> _inputs;
        private List<NamedOnnxValue> _preallocatedOutputs;
        private readonly Task<InferenceSession> _loadTask;
        private bool _disposed = false;
        
        // Static logging configuration
        private static bool _loggingInitialized = false;
        private static IntPtr _loggingParam = IntPtr.Zero;
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        
        #endregion

        #region Protected Properties

        /// <summary>
        /// Gets the debug logger instance for this model.
        /// </summary>
        protected static DebugLogger Logger { get; } = new();

        /// <summary>
        /// Gets whether the model has been successfully initialized.
        /// </summary>
        public bool IsInitialized { get; protected set; } = false;

        /// <summary>
        /// Gets the current ONNX Runtime logging level.
        /// </summary>
        protected static OrtLoggingLevel OrtLogLevel => _ortLogLevel;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the ORTModel class with the specified configuration.
        /// </summary>
        /// <param name="modelName">The name of the model file (without extension)</param>
        /// <param name="modelFolder">The folder containing the model (from SparkTTSModelPaths)</param>
        /// <param name="logLevel">The logging level for this model</param>
        protected ORTModel(string modelName, string modelFolder, DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning)
        {
            if (string.IsNullOrEmpty(modelName))
                throw new ArgumentNullException(nameof(modelName));
            if (string.IsNullOrEmpty(modelFolder))
                throw new ArgumentNullException(nameof(modelFolder));

            Logger.Level = logLevel;
            InitializeOnnxLogging();
            
            _config = new ModelConfig
            {
                ModelName = modelName,
                ModelFolder = modelFolder,
                ModelPath = GetModelPath(modelFolder, $"{modelName}.onnx")
            };

            _loadTask = LoadModelAsync();
        }

        #endregion

        #region Public Methods - Input Loading

        /// <summary>
        /// Asynchronously loads a float tensor input at the specified index.
        /// </summary>
        /// <param name="index">The index of the input to load</param>
        /// <param name="inputTensor">The float tensor to load as input</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        public async Task LoadInput(int index, Tensor<float> inputTensor)
        {
            if (inputTensor == null)
                throw new ArgumentNullException(nameof(inputTensor));
                
            await _loadTask;
            
            if (index < 0 || index >= _inputNames.Count)
                throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {_inputNames.Count - 1}");
                
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        /// <summary>
        /// Asynchronously loads a long tensor input at the specified index.
        /// </summary>
        /// <param name="index">The index of the input to load</param>
        /// <param name="inputTensor">The long tensor to load as input</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        public async Task LoadInput(int index, Tensor<long> inputTensor)
        {
            if (inputTensor == null)
                throw new ArgumentNullException(nameof(inputTensor));
                
            await _loadTask;
            
            if (index < 0 || index >= _inputNames.Count)
                throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {_inputNames.Count - 1}");
                
            _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[index], inputTensor));
        }

        /// <summary>
        /// Asynchronously loads multiple float tensor inputs in order.
        /// </summary>
        /// <param name="inputTensors">The list of float tensors to load as inputs</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        public async Task LoadInputs(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            
            if (inputTensors.Count != _inputNames.Count)
                throw new ArgumentException($"Input tensor count mismatch: provided {inputTensors.Count}, expected {_inputNames.Count}");
            
            for (int i = 0; i < inputTensors.Count; i++)
            {
                if (inputTensors[i] == null)
                    throw new ArgumentNullException($"inputTensors[{i}]");
                    
                _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputNames[i], inputTensors[i]));
            }
        }

        #endregion

        #region Public Methods - Model Execution

        /// <summary>
        /// Asynchronously runs the model with the provided input tensors.
        /// </summary>
        /// <param name="inputTensors">The list of input tensors for model execution</param>
        /// <returns>A task containing the list of named output values</returns>
        public async Task<List<NamedOnnxValue>> Run(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            await LoadInputs(inputTensors);
            return await Run();
        }

        /// <summary>
        /// Asynchronously runs the model with previously loaded inputs.
        /// </summary>
        /// <returns>A task containing the list of named output values</returns>
        public async Task<List<NamedOnnxValue>> Run()
        {
            await _loadTask;
            
            if (_inputs == null || _inputs.Count == 0)
                throw new InvalidOperationException("No inputs loaded. Call LoadInputs() or LoadInput() first.");
            
            var start = System.Diagnostics.Stopwatch.StartNew();
            SetLoggingParam(_config.ModelName);
            
            var runOptions = new RunOptions
            {
                LogSeverityLevel = _ortLogLevel
            };
            
            try
            {
                _session.Run(_inputs, _preallocatedOutputs, runOptions);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model execution failed for {_config.ModelName}: {ex.Message}", ex);
            }
            finally
            {
                var elapsed = start.ElapsedMilliseconds;
                Logger.Log($"[{_config.ModelName}] Execution completed in {elapsed}ms");
                
                // Clear inputs for next run
                _inputs.Clear();
            }

            return _preallocatedOutputs;
        }

        /// <summary>
        /// Asynchronously runs the model with disposable outputs.
        /// </summary>
        /// <param name="inputTensors">The list of input tensors for model execution</param>
        /// <returns>A task containing the disposable collection of named output values</returns>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
                
            await _loadTask;
            await LoadInputs(inputTensors);
            return await RunDisposable();
        }

        /// <summary>
        /// Asynchronously runs the model with disposable outputs.
        /// </summary>
        /// <returns>A task containing the disposable collection of named output values</returns>
        public async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable()
        {
            await _loadTask;
            
            if (_inputs == null || _inputs.Count == 0)
                throw new InvalidOperationException("No inputs loaded. Call LoadInputs() or LoadInput() first.");
            
            var start = System.Diagnostics.Stopwatch.StartNew();
            SetLoggingParam(_config.ModelName);
            
            var runOptions = new RunOptions
            {
                LogSeverityLevel = _ortLogLevel
            };
            
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            try
            {
                results = _session.Run(_inputs, _session.OutputNames, runOptions);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Model execution failed for {_config.ModelName}: {ex.Message}", ex);
            }
            finally
            {
                var elapsed = start.ElapsedMilliseconds;
                Logger.Log($"[{_config.ModelName}] Execution completed in {elapsed}ms");
                
                // Clear inputs for next run
                _inputs.Clear();
            }
            
            return results;
        }

        #endregion

        #region Public Methods - Output Management

        /// <summary>
        /// Asynchronously retrieves a preallocated output tensor by name.
        /// </summary>
        /// <typeparam name="T">The tensor element type</typeparam>
        /// <param name="outputName">The name of the output tensor to retrieve</param>
        /// <returns>A task containing the requested preallocated output tensor</returns>
        public async Task<Tensor<T>> GetPreallocatedOutput<T>(string outputName)
        {
            if (string.IsNullOrEmpty(outputName))
                throw new ArgumentNullException(nameof(outputName));
                
            await _loadTask;
            
            var output = _preallocatedOutputs.FirstOrDefault(o => o.Name == outputName);
            if (output != null)
            {
                return output.AsTensor<T>();
            }
            
            throw new ArgumentException($"Output '{outputName}' not found in preallocated outputs", nameof(outputName));
        }

        /// <summary>
        /// Asynchronously retrieves all preallocated output tensors.
        /// </summary>
        /// <returns>A task containing the list of all preallocated output tensors</returns>
        public async Task<List<NamedOnnxValue>> GetPreallocatedOutputs()
        {
            await _loadTask;
            return _preallocatedOutputs;
        }

        #endregion

        #region Static Methods - Environment Management

        /// <summary>
        /// Initializes the ONNX Runtime environment with the specified logging level.
        /// </summary>
        /// <param name="logLevel">The logging level for ONNX Runtime operations</param>
        public static void InitializeEnvironment(DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning)
        {
            Logger.Level = logLevel;
            _ortLogLevel = logLevel switch
            {
                DebugLogger.LogLevel.Error => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                DebugLogger.LogLevel.Warning => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                DebugLogger.LogLevel.Info => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
                DebugLogger.LogLevel.Debug => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };
            InitializeOnnxLogging();
        }

        #endregion

        #region Protected Methods - Utilities

        /// <summary>
        /// Gets the full path to a model file within the SparkTTS model structure.
        /// </summary>
        /// <param name="modelFolder">The subfolder for the model</param>
        /// <param name="modelFileName">The model file name</param>
        /// <returns>The full path to the model file</returns>
        protected static string GetModelPath(string modelFolder, string modelFileName)
        {
            return Path.Combine(
                Application.streamingAssetsPath,
                SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets,
                modelFolder,
                modelFileName);
        }

        /// <summary>
        /// Creates optimized SessionOptions for ONNX Runtime.
        /// </summary>
        /// <returns>A configured SessionOptions object</returns>
        protected static SessionOptions CreateSessionOptions()
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_PARALLEL,
                EnableMemoryPattern = true,
                EnableCpuMemArena = true,
                InterOpNumThreads = Environment.ProcessorCount,
                IntraOpNumThreads = Environment.ProcessorCount,
                LogSeverityLevel = _ortLogLevel
            };

            // Advanced performance optimizations
            options.AddSessionConfigEntry("session.disable_prepacking", "0");
            options.AddSessionConfigEntry("session.use_env_allocators", "1");
            
            return options;
        }

        /// <summary>
        /// Sets the logging parameter context for ONNX Runtime operations.
        /// </summary>
        /// <param name="modelName">The name of the model currently being processed</param>
        protected static void SetLoggingParam(string modelName)
        {
            if (string.IsNullOrEmpty(modelName))
                return;
                
            var loadingInfo = new LoadingInfo { ModelName = modelName };
            if (_loggingParam == IntPtr.Zero)
            {
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
            }
            Marshal.StructureToPtr(loadingInfo, _loggingParam, false);
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Asynchronously loads the ONNX model and initializes input/output metadata.
        /// </summary>
        /// <returns>A task containing the loaded InferenceSession</returns>
        private async Task<InferenceSession> LoadModelAsync()
        {
            return await Task.Run(() =>
            {
                if (!File.Exists(_config.ModelPath))
                {
                    Logger.LogError($"[{_config.ModelName}] Model file not found: {_config.ModelPath}");
                    throw new FileNotFoundException($"Model file not found: {_config.ModelPath}");
                }

                try
                {
                    var options = CreateSessionOptions();
                    _session = new InferenceSession(_config.ModelPath, options);
                    
                    // Initialize input/output metadata
                    _inputNames = _session.InputMetadata.Keys.ToList();
                    _inputs = new List<NamedOnnxValue>(_inputNames.Count);
                    
                    // Pre-allocate output buffers
                    _preallocatedOutputs = new List<NamedOnnxValue>();
                    foreach (var outputMetadata in _session.OutputMetadata)
                    {
                        var outputName = outputMetadata.Key;
                        var nodeMetadata = outputMetadata.Value;
                        
                        if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(float))
                        {
                            CreatePreallocatedTensor<float>(outputName, nodeMetadata.Dimensions);
                        }
                        else if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(long))
                        {
                            CreatePreallocatedTensor<long>(outputName, nodeMetadata.Dimensions);
                        }
                    }
                    
                    IsInitialized = true;
                    Logger.Log($"[{_config.ModelName}] Successfully loaded model: {_config.ModelPath}");
                    return _session;
                }
                catch (Exception ex)
                {
                    Logger.LogError($"[{_config.ModelName}] Failed to load model: {ex.Message}");
                    IsInitialized = false;
                    throw;
                }
            });
        }

        /// <summary>
        /// Creates a preallocated tensor for the specified output.
        /// </summary>
        /// <typeparam name="T">The tensor element type</typeparam>
        /// <param name="outputName">The name of the output tensor</param>
        /// <param name="dimensions">The tensor dimensions</param>
        private void CreatePreallocatedTensor<T>(string outputName, int[] dimensions)
        {
            // Replace dynamic dimensions with 1 for tensor creation
            dimensions = dimensions.Select(d => d == -1 ? 1 : d).ToArray();
            var tensor = new DenseTensor<T>(dimensions);
            _preallocatedOutputs.Add(NamedOnnxValue.CreateFromTensor(outputName, tensor));
        }

        /// <summary>
        /// Initializes ONNX Runtime logging with Unity integration.
        /// </summary>
        private static void InitializeOnnxLogging()
        {
            if (_loggingInitialized || Logger.Level == DebugLogger.LogLevel.None)
                return;

            if (Application.platform == RuntimePlatform.IPhonePlayer)
            {
                _loggingInitialized = true;
                return;
            }

            if (OrtEnv.IsCreated)
            {
                Logger.Log("[ORTModel] ONNX Runtime logging already initialized");
                _loggingInitialized = true;
                return;
            }

            try
            {
                _loggingParam = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LoadingInfo)));
                
                var options = new EnvironmentCreationOptions
                {
                    logLevel = _ortLogLevel,
                    logId = "SparkTTS",
                    loggingFunction = UnityOnnxLoggingCallback,
                    loggingParam = _loggingParam
                };

                OrtEnv.CreateInstanceWithOptions(ref options);
                
                _loggingInitialized = true;
                Logger.Log($"[ORTModel] ONNX Runtime logging initialized (LogLevel: {_ortLogLevel})");
            }
            catch (Exception e)
            {
                Logger.LogError($"[ORTModel] Failed to initialize ONNX Runtime logging: {e.Message}");
                _loggingInitialized = true;
            }
        }

        /// <summary>
        /// Unity logging callback for ONNX Runtime.
        /// </summary>
        private static void UnityOnnxLoggingCallback(IntPtr param, 
                                                   OrtLoggingLevel severity, 
                                                   string category, 
                                                   string logId, 
                                                   string codeLocation, 
                                                   string message)
        {
            if (param == IntPtr.Zero || _loggingParam == IntPtr.Zero)
                return;
                
            var loadingInfo = (LoadingInfo)Marshal.PtrToStructure(param, typeof(LoadingInfo));
            string formattedMessage = FormatOnnxLogMessage(severity, category, logId, codeLocation, message, loadingInfo.ModelName);

            switch (severity)
            {
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO:
                    Logger.Log(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING:
                    Logger.LogWarning(formattedMessage);
                    break;
                    
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR:
                case OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL:
                    Logger.LogError(formattedMessage);
                    break;
                    
                default:
                    Logger.Log(formattedMessage);
                    break;
            }
        }

        /// <summary>
        /// Formats ONNX Runtime log messages for Unity console.
        /// </summary>
        private static string FormatOnnxLogMessage(OrtLoggingLevel severity, 
                                                 string category, 
                                                 string logId, 
                                                 string codeLocation, 
                                                 string message,
                                                 string modelName)
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
            
            return $"[ONNX-{severityStr}][{modelName}]{cleanCategory} {message}{cleanCodeLocation}";
        }

        #endregion

        #region Private Types

        /// <summary>
        /// Configuration for model loading and execution.
        /// </summary>
        private class ModelConfig
        {
            public string ModelName { get; set; }
            public string ModelFolder { get; set; }
            public string ModelPath { get; set; }
        }

        /// <summary>
        /// Loading information structure for ONNX Runtime logging.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        private struct LoadingInfo
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
            public string ModelName;
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Disposes of the ORTModel instance.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Disposes of the ORTModel instance.
        /// </summary>
        /// <param name="disposing">True if called from Dispose(), false if called from finalizer</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                    _inputs?.Clear();
                    _preallocatedOutputs?.Clear();
                    
                    Logger.Log($"[{_config?.ModelName ?? "ORTModel"}] Disposed successfully");
                }
                
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer for the ORTModel class.
        /// </summary>
        ~ORTModel()
        {
            Dispose(false);
        }

        #endregion
    }
} 