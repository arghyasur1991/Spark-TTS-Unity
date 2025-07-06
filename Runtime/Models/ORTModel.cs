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
        private readonly bool _preAllocateOutputs = false;
        private List<NamedOnnxValue> _preallocatedOutputs;
        protected Task<InferenceSession> _loadTask = null;
        private bool _disposed = false;
        
        // Static logging configuration
        private static bool _loggingInitialized = false;
        private static IntPtr _loggingParam = IntPtr.Zero;
        private static OrtLoggingLevel _ortLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
        
        #endregion

        #region Protected Properties

        /// <summary>
        /// Gets whether the model has been successfully initialized.
        /// </summary>
        public bool IsInitialized { get; protected set; } = false;

        /// <summary>
        /// Gets the current ONNX Runtime logging level.
        /// </summary>
        protected static OrtLoggingLevel OrtLogLevel => _ortLogLevel;

        protected enum Precision
        {
            FP32,
            FP16,
            Int8
        }

        protected enum ExecutionProvider
        {
            /// <summary>
            /// CPU execution provider - universal compatibility, moderate performance
            /// </summary>
            CPU,
            
            /// <summary>
            /// CUDA execution provider - GPU acceleration for NVIDIA cards, high performance
            /// </summary>
            CUDA,
            
            /// <summary>
            /// CoreML execution provider - Apple Silicon/macOS acceleration, optimized for Apple hardware
            /// </summary>
            CoreML
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the ORTModel class with the specified configuration.
        /// </summary>
        /// <param name="modelName">The name of the model file (without extension)</param>
        /// <param name="modelFolder">The folder containing the model (from SparkTTSModelPaths)</param>
        /// <param name="precision">The precision of the model</param>
        /// <param name="executionProvider">The execution provider for the model</param>
        protected ORTModel(
            string modelName, 
            string modelFolder, 
            bool preAllocateOutputs = false,
            Precision precision = Precision.FP32, 
            ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            if (string.IsNullOrEmpty(modelName))
                throw new ArgumentNullException(nameof(modelName));
            if (modelFolder == null)
                throw new ArgumentNullException(nameof(modelFolder));

            _config = new ModelConfig
            {
                ModelName = modelName,
                Precision = precision,
                ExecutionProvider = executionProvider,
                ModelPath = Path.Combine(
                    Application.streamingAssetsPath,
                    SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets,
                    modelFolder)
            };
            _preAllocateOutputs = preAllocateOutputs;
        }

        #endregion

        #region Public Methods - Input Loading

        /// <summary>
        /// Starts the asynchronous loading operation.
        /// </summary>
        /// <returns>A task that represents the asynchronous loading operation</returns>
        public void StartLoadingAsync()
        {
            if (IsInitialized || _loadTask != null)
                return;

            _loadTask = LoadModelAsync();
            _loadTask.Start();
        }

        public async Task<T> RunAsync<T>(Func<Task<T>> func, bool standaloneLoading = true)
        {
            if (standaloneLoading)
            {
                StartLoadingAsync();
            }
            var result = await func();
            if (standaloneLoading)
            {
                Dispose();
            }
            return result;
        }

        /// <summary>
        /// Asynchronously loads a tensor input at the specified index.
        /// </summary>
        /// <param name="index">The index of the input to load</param>
        /// <param name="inputTensor">The tensor to load as input</param>
        /// <returns>A task that represents the asynchronous input loading operation</returns>
        public async Task LoadInput<T>(int index, Tensor<T> inputTensor)
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

        #region Protected Methods - Model Execution

        /// <summary>
        /// Asynchronously runs the model with the provided input tensors.
        /// </summary>
        /// <param name="inputTensors">The list of input tensors for model execution</param>
        /// <returns>A task containing the list of named output values</returns>
        protected async Task<List<NamedOnnxValue>> Run(List<Tensor<float>> inputTensors)
        {
            if (inputTensors == null)
                throw new ArgumentNullException(nameof(inputTensors));
            
            if (!_preAllocateOutputs)
            {
                throw new InvalidOperationException("Pre-allocated outputs are not supported for this model. Use RunDisposable() instead.");
            }
            
            await _loadTask;
            await LoadInputs(inputTensors);
            return await Run();
        }

        /// <summary>
        /// Asynchronously runs the model with previously loaded inputs.
        /// </summary>
        /// <returns>A task containing the list of named output values</returns>
        protected async Task<List<NamedOnnxValue>> Run()
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
        protected async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable(List<Tensor<float>> inputTensors)
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
        protected async Task<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> RunDisposable()
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

        #region Protected Methods - Output Management

        /// <summary>
        /// Asynchronously retrieves a preallocated output tensor by name.
        /// </summary>
        /// <typeparam name="T">The tensor element type</typeparam>
        /// <param name="outputName">The name of the output tensor to retrieve</param>
        /// <returns>A task containing the requested preallocated output tensor</returns>
        protected async Task<Tensor<T>> GetPreallocatedOutput<T>(string outputName)
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
        protected async Task<List<NamedOnnxValue>> GetPreallocatedOutputs()
        {
            await _loadTask;
            return _preallocatedOutputs;
        }

        protected async Task<IReadOnlyList<string>> GetOutputNames()
        {
            await _loadTask;
            return _session.OutputNames;
        }

        protected async Task<int[]> GetOutputDimensions(string outputName)
        {
            await _loadTask;
            var outputMetadata = _session.OutputMetadata[outputName];
            if (outputMetadata.IsTensor)
            {
                return outputMetadata.Dimensions;
            }
            return null;
        }

        #endregion

        #region Static Methods - Environment Management

        /// <summary>
        /// Initializes the ONNX Runtime environment with the specified logging level.
        /// </summary>
        /// <param name="logLevel">The logging level for ONNX Runtime operations</param>
        public static void InitializeEnvironment(LogLevel logLevel = LogLevel.WARNING)
        {
            _ortLogLevel = logLevel switch
            {
                LogLevel.ERROR => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                LogLevel.WARNING => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
                LogLevel.INFO => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
                LogLevel.VERBOSE => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };
            InitializeOnnxLogging();
        }

        #endregion

        #region Protected Methods - Utilities

        /// <summary>
        /// Sets the full path to a model file within the SparkTTS model structure.
        /// </summary>
        /// <param name="modelFolder">The subfolder for the model</param>
        /// <param name="modelName">The model name</param>
        protected string GetModelPath(string modelName)
        {
            if (_config.Precision == Precision.FP16)
            {
                modelName = $"{modelName}_fp16";
            }
            else if (_config.Precision == Precision.Int8)
            {
                modelName = $"{modelName}_int8";
            }
            return Path.Combine(
                _config.ModelPath,
                $"{modelName}.onnx");
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
        private Task<InferenceSession> LoadModelAsync()
        {
            return new Task<InferenceSession>(() =>
            {
                string modelPath = GetModelPath(_config.ModelName);
                if (!File.Exists(modelPath))
                {
                    Logger.LogError($"[{_config.ModelName}] Model file not found: {modelPath}");
                    throw new FileNotFoundException($"Model file not found: {modelPath}");
                }
                Logger.Log($"[{_config.ModelName}] Loading model: {_config.ModelName}");

                try
                {
                    var options = CreateSessionOptions();

                    if (_config.ExecutionProvider == ExecutionProvider.CoreML) 
                    {
                        LoadModelWithCoreML(modelPath, options);
                    }
                    else
                    {
                        _session = new InferenceSession(modelPath, options);
                    }
                    
                    // Initialize input/output metadata
                    _inputNames = _session.InputMetadata.Keys.ToList();
                    _inputs = new List<NamedOnnxValue>(_inputNames.Count);
                    
                    // Pre-allocate output buffers
                    if (_preAllocateOutputs)
                    {
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
                            else if (nodeMetadata.IsTensor && nodeMetadata.ElementType == typeof(int))
                            {
                                CreatePreallocatedTensor<int>(outputName, nodeMetadata.Dimensions);
                            }
                        }
                    }
                    IsInitialized = true;
                    Logger.Log($"[{_config.ModelName}] Successfully loaded model: {modelPath}");
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
            if (_loggingInitialized)
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

        #region Private Methods - CoreML Support


        /// <summary>
        /// Loads an ONNX model with CoreML acceleration and comprehensive error handling.
        /// This method configures CoreML provider with caching support, handles cache corruption recovery,
        /// and provides fallback mechanisms for maximum compatibility across different Apple devices.
        /// </summary>
        /// <param name="modelPath">The file path to the ONNX model</param>
        /// <param name="sessionOptions">The base session options to configure with CoreML provider</param>
        private void LoadModelWithCoreML(string modelPath, SessionOptions sessionOptions)
        {
            try
            {
                // Configure CoreML provider with caching support using dictionary API
                string cacheDirectory = GetCoreMLCacheDirectory();
                
                // Ensure cache directory exists and is writable
                EnsureCacheDirectoryExists(cacheDirectory);
                
                var coremlOptions = new Dictionary<string, string>
                {
                    ["ModelFormat"] = "MLProgram",
                    ["MLComputeUnits"] = "CPUAndGPU",
                    ["RequireStaticInputShapes"] = "0",
                    ["EnableOnSubgraphs"] = "1",
                    // Advanced options for optimization
                    // ["SpecializationStrategy"] = "FastPrediction",
                    // ["AllowLowPrecisionAccumulationOnGPU"] = "1",
                    // ["ProfileComputePlan"] = "1"
                };
                
                if (!string.IsNullOrEmpty(cacheDirectory))
                {
                    coremlOptions["ModelCacheDirectory"] = cacheDirectory;
                }
                
                sessionOptions.AppendExecutionProvider("CoreML", coremlOptions);
                Logger.Log($"[ModelUtils] CoreML provider configured with caching (cache: {cacheDirectory})");
                
                // Try creating the session - if it fails due to cache corruption, retry
                try
                {
                    _session = new InferenceSession(modelPath, sessionOptions);
                    Logger.Log($"[ModelUtils] Successfully loaded model with CoreML provider: {modelPath}");
                }
                catch (Exception sessionException)
                {
                    if (sessionException.Message.Contains("Manifest.json") || 
                        sessionException.Message.Contains("coreml_cache") ||
                        sessionException.Message.Contains("manifest does not exist"))
                    {
                        Logger.LogWarning($"[ModelUtils] CoreML cache corruption detected. Retrying: {sessionException.Message}");
                        _session = new InferenceSession(modelPath, sessionOptions);
                        Logger.Log($"[ModelUtils] Successfully loaded model with CoreML provider after retrying: {modelPath}");
                    }
                    else
                    {
                        throw; // Re-throw if it's not a cache-related issue
                    }
                }
            }
            catch (Exception e)
            {
                Logger.LogWarning($"[ModelUtils] CoreML provider configuration failed: {e.Message}");
                
                // Fallback to old CoreML flags approach for compatibility
                try
                {
                    var fallbackOptions = CreateSessionOptions();
                    fallbackOptions.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_USE_CPU_AND_GPU | 
                        CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM |
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    
                    _session = new InferenceSession(modelPath, fallbackOptions);
                    Logger.Log("[ModelUtils] Using fallback CoreML provider (no caching)");
                }
                catch (Exception fallbackException)
                {
                    Logger.LogWarning($"[ModelUtils] CoreML fallback also failed: {fallbackException.Message}. Using CPU provider.");
                }
            }
        }

        /// <summary>
        /// Gets the cache directory for CoreML compiled models with automatic path resolution.
        /// This method determines the best location for CoreML model caching based on configuration
        /// and platform-specific storage locations for optimal performance and persistence.
        /// </summary>
        /// <returns>The full path to the CoreML cache directory</returns>
        private string GetCoreMLCacheDirectory()
        {
            if (!string.IsNullOrEmpty(_config.ModelPath))
            {
                return Path.Combine(_config.ModelPath, "coreml_cache");
            }
            return Path.Combine(Application.persistentDataPath, "coreml_cache");
        }

        /// <summary>
        /// Ensures the CoreML cache directory exists and is writable with proper error handling.
        /// This method creates the cache directory structure if it doesn't exist and handles
        /// permission and filesystem errors gracefully.
        /// </summary>
        /// <param name="cacheDirectory">The cache directory path to create and validate</param>
        private void EnsureCacheDirectoryExists(string cacheDirectory)
        {
            if (string.IsNullOrEmpty(cacheDirectory))
                return;
                
            try
            {
                if (!Directory.Exists(cacheDirectory))
                {
                    Directory.CreateDirectory(cacheDirectory);
                    Logger.Log($"[ModelUtils] Created CoreML cache directory: {cacheDirectory}");
                }
            }
            catch (Exception e)
            {
                Logger.LogWarning($"[ModelUtils] Failed to create cache directory {cacheDirectory}: {e.Message}");
            }
        }
        #endregion

        #region Private Types

        /// <summary>
        /// Configuration for model loading and execution.
        /// </summary>
        private class ModelConfig
        {
            public string ModelName { get; set; }
            public string ModelPath { get; set; }
            public Precision Precision { get; set; } = Precision.FP32;
            public ExecutionProvider ExecutionProvider { get; set; } = ExecutionProvider.CPU;
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
                    IsInitialized = false;
                    _loadTask?.Dispose();
                    _loadTask = null;
                    
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