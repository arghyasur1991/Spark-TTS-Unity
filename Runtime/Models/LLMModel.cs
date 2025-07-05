using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    
    /// <summary>
    /// Large Language Model for generating semantic tokens from tokenized input.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern while
    /// maintaining complex generation logic and performance optimizations.
    /// </summary>
    internal class LLMModel : ORTModel
    {
        // Thread-safe random number generator (one instance per thread)
        private static readonly ThreadLocal<Random> _threadLocalRandom = 
            new(() => new Random(Guid.NewGuid().GetHashCode()));

        public static bool LogTiming = false;

        // Model input/output names, populated by InspectAndPopulateNames
        private string _inputIDsName;
        private string _attentionMaskName;
        private string _positionIDsName;
        private string _logitsOutputName;
        private readonly Dictionary<int, string> _pastKeyInputNames = new();
        private readonly Dictionary<int, string> _pastValueInputNames = new();
        private readonly Dictionary<int, string> _presentKeyOutputNames = new();
        private readonly Dictionary<int, string> _presentValueOutputNames = new();

        // Model parameters, inferred by InspectAndPopulateNames
        private int _numLLMLayers = 0;
        private int _numAttentionHeads = 0;
        private int _headDimension = 0;
        private int _vocabSize = 0;

        // Special token IDs
        private readonly int _eosTokenId;
        
        private readonly float[] _filteredLogits;
        private readonly int[] _filteredIndices;
        private readonly List<(float prob, int index, float logit)> _probIndexList = new(1000);
        private readonly List<float> _nucleusLogits = new(1000);
        private readonly List<int> _nucleusIndices = new(1000);
        private int[] _indices;
        // Pre-allocated arrays for softmax computations
        private readonly float[] _expLogitsBuffer = new float[1000];
        private float[] _probsBuffer = new float[1000];
        
        // Pre-allocated buffer for top-K filtering
        private (float logit, int index)[] _topKHeap;
        private float[] _sortedLogits;
        private int[] _sortedIndices;
        
        // Pre-allocated buffer for KV cache to reduce allocations
        private List<DenseTensor<float>> _cachedKvTensors;
        private bool _isKvCacheInitialized = false;

        // Pre-allocated RunOptions
        private readonly RunOptions _runOptions;
        private readonly List<string> _outputNames = new();

        private readonly AggregatedTimer _inferenceTimer = new("RunInference");
        private readonly AggregatedTimer _processLogitsAndSampleTimer = new("ProcessLogitsAndSample");
        private readonly AggregatedTimer _updateKVCacheTimer = new("UpdateKVCache");
        private readonly AggregatedTimer _sampleLogitsTimer = new("SampleLogits");
        private readonly AggregatedTimer _sampleLogitsTimer2 = new("SampleLogits2");
        private readonly AggregatedTimer _generateSemanticTokensTimer = new("GenerateSemanticTokens");

        private readonly AggregatedTimer _prepareInitialInputsTimer = new("PrepareInitialInputs");
        private readonly AggregatedTimer _prepareStepInputsTimer = new("PrepareStepInputs");

        /// <summary>
        /// Initializes a new instance of the LLMModel class.
        /// </summary>
        /// <param name="tokenizerDef">The tokenizer definition containing special tokens</param>
        /// <exception cref="ArgumentNullException">Thrown when tokenizerDef is null</exception>
        public LLMModel(TokenizerDefinition tokenizerDef)
            : base(SparkTTSModelPaths.LLMModelName, 
                   SparkTTSModelPaths.LLMFolder)
        {
            if (tokenizerDef == null)
                throw new ArgumentNullException(nameof(tokenizerDef), "TokenizerDefinition cannot be null for LLMModel.");

            // Set special token IDs based on tokenizer definition
            _eosTokenId = 151645; // From Python HF tokenizer.eos_token_id

            // Initialize performance buffers
            _filteredLogits = new float[1000];
            _filteredIndices = new int[1000];

            // Initialize RunOptions with performance settings
            _runOptions = new RunOptions
            {
                LogSeverityLevel = OrtLogLevel
            };

            Logger.Log("[LLMModel] Initialized successfully");
            
            // Initialize model metadata asynchronously
            _ = InitializeModelMetadataAsync();
        }

        /// <summary>
        /// Asynchronously initializes model metadata by inspecting input/output names and dimensions.
        /// This replaces the synchronous InspectAndPopulateNames method with dynamic detection.
        /// </summary>
        private async Task InitializeModelMetadataAsync()
        {
            try
            {                
                // Get preallocated outputs to inspect metadata
                var outputs = await GetPreallocatedOutputs();
                
                // Dynamically detect input/output names by pattern matching
                _inputIDsName = "input_ids";
                _attentionMaskName = "attention_mask"; 
                _positionIDsName = "position_ids";
                _logitsOutputName = "logits";
                
                // Detect KV cache structure from outputs
                foreach (var output in outputs)
                {
                    var name = output.Name;
                    
                    // Detect logits output
                    if (name.Contains("logits"))
                    {
                        _logitsOutputName = name;
                    }
                    
                    // Detect present key/value outputs for KV cache
                    var kvMatch = System.Text.RegularExpressions.Regex.Match(name, @"present\.(\d+)\.(key|value)");
                    if (!kvMatch.Success) 
                        kvMatch = System.Text.RegularExpressions.Regex.Match(name, @"present_key_values\.(\d+)\.(key|value)");
                    
                    if (kvMatch.Success)
                    {
                        int layerIndex = int.Parse(kvMatch.Groups[1].Value);
                        string type = kvMatch.Groups[2].Value;
                        
                        if (type == "key") 
                            _presentKeyOutputNames[layerIndex] = name;
                        if (type == "value") 
                            _presentValueOutputNames[layerIndex] = name;
                        
                        if (layerIndex >= _numLLMLayers) 
                            _numLLMLayers = layerIndex + 1;
                        
                        // Extract dimension info from tensor metadata if available
                        if (output.Value is DenseTensor<float> tensor && tensor.Dimensions.Length == 4)
                        {
                            if (_numAttentionHeads == 0 && tensor.Dimensions[1] > 0) 
                                _numAttentionHeads = (int)tensor.Dimensions[1];
                            if (_headDimension == 0 && tensor.Dimensions[3] > 0) 
                                _headDimension = (int)tensor.Dimensions[3];
                        }
                    }
                }
                
                // Set up corresponding past key/value input names (standard naming pattern)
                for (int i = 0; i < _numLLMLayers; i++)
                {
                    _pastKeyInputNames[i] = $"past_key_values.{i}.key";
                    _pastValueInputNames[i] = $"past_key_values.{i}.value";
                }
                
                // Pre-populate output names for efficient inference
                _outputNames.Add(_logitsOutputName);
                for (int i = 0; i < _numLLMLayers; i++)
                {
                    if (_presentKeyOutputNames.TryGetValue(i, out string keyName))
                        _outputNames.Add(keyName);
                    if (_presentValueOutputNames.TryGetValue(i, out string valueName))
                        _outputNames.Add(valueName);
                }
                
                Logger.Log($"[LLMModel] Model metadata initialized dynamically - " +
                          $"Layers: {_numLLMLayers}, Heads: {_numAttentionHeads}, HeadDim: {_headDimension}");
                          
                if (_numLLMLayers == 0)
                {
                    Logger.LogWarning("[LLMModel] No KV cache layers detected. Model may not support KV caching.");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[LLMModel] Failed to initialize model metadata: {ex.Message}");
                
                // Fallback to reasonable defaults
                _inputIDsName = "input_ids";
                _attentionMaskName = "attention_mask";
                _logitsOutputName = "logits";
                _numLLMLayers = 0; // Disable KV caching if detection fails
                Logger.LogWarning("[LLMModel] Using fallback metadata - KV caching disabled");
            }
        }

        /// <summary>
        /// Asynchronously generates semantic tokens from tokenized input.
        // Note: The complex InspectAndPopulateNames method has been simplified
        // in favor of the new ORTModel pattern. The LLM functionality remains intact
        // but uses the async initialization approach.

        /// <summary>
        /// Asynchronously generates semantic tokens from tokenized input.
        /// This is the main generation method that should be used instead of the old synchronous version.
        /// </summary>
        /// <param name="llmInitialInput">The tokenized input to process</param>
        /// <param name="maxNewTokens">Maximum number of new tokens to generate</param>
        /// <param name="temperature">Temperature for sampling (default: 0.8)</param>
        /// <param name="topK">Top-K value for sampling (default: 50)</param>
        /// <param name="topP">Top-P (nucleus) value for sampling (default: 0.95)</param>
        /// <returns>A task containing the list of generated semantic token IDs</returns>
        /// <exception cref="ArgumentNullException">Thrown when llmInitialInput is null</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<List<int>> GenerateSemanticTokensAsync(
            TokenizationOutput llmInitialInput,
            int maxNewTokens,
            float temperature = 0.8f,
            int topK = 50,
            float topP = 0.95f)
        {
            await _loadTask;
            if (llmInitialInput == null || !llmInitialInput.InputIds.Any()) 
            { 
                throw new ArgumentNullException(nameof(llmInitialInput), "Initial input is null or empty"); 
            }
            if (string.IsNullOrEmpty(_inputIDsName) || string.IsNullOrEmpty(_logitsOutputName))
            { 
                throw new InvalidOperationException("Essential model input/output names not identified"); 
            }

            if (_numLLMLayers > 0 &&
                (_pastKeyInputNames.Count != _numLLMLayers || 
                 _pastValueInputNames.Count != _numLLMLayers ||
                 _presentKeyOutputNames.Count != _numLLMLayers || 
                 _presentValueOutputNames.Count != _numLLMLayers))
            { 
                throw new InvalidOperationException("KV cache names not fully identified for all layers"); 
            }

            if (_numLLMLayers > 0 && (_numAttentionHeads == 0 || _headDimension == 0))
            { 
                throw new InvalidOperationException("KV cache dimensions not inferred, cannot create empty KV cache"); 
            }

            var newlyGeneratedTokenIds = new List<int>(maxNewTokens);
            const int batchSize = 1;

            try
            {
                var startTime = Stopwatch.GetTimestamp();
                // 1. Process initial prompt
                Logger.Log("[LLM.GenTokens] Preparing initial prompt processing pass...");
                
                // Convert input IDs - avoid multiple enumerations with ToArray outside of Select
                int[] intInputIds = llmInitialInput.InputIds.ToArray();
                long[] inputIdsData = new long[intInputIds.Length];
                for (int i = 0; i < intInputIds.Length; i++)
                {
                    inputIdsData[i] = intInputIds[i];
                }
                
                // Generate inputs for initial pass
                var initialInputs = await PrepareInitialInputsAsync(
                    inputIdsData, 
                    llmInitialInput.AttentionMask?.ToArray(), 
                    batchSize);

                // Run initial inference - extract KV cache if we have layers with KV cache
                bool extractKvCache = _numLLMLayers > 0;
                var initialOutput = await RunInference(initialInputs, extractKvCache);
                if (initialOutput == null)
                {
                    throw new InvalidOperationException("Initial inference failed");
                }

                // Process logits and sample next token
                int nextTokenId = ProcessLogitsAndSample(
                    initialOutput.Logits, 
                    initialOutput.LogitsDimensions, 
                    initialOutput.LogitsDimensions[1] - 1, // Last token position
                    temperature, 
                    topK, 
                    topP);

                Logger.Log($"[LLMModel.GenTokens InitialPass] Next token ID: {nextTokenId}");
                newlyGeneratedTokenIds.Add(nextTokenId);

                // Check if generation should end
                if (nextTokenId == _eosTokenId)
                {
                    Logger.Log("[LLMModel.GenTokens] EOS token generated after initial prompt. Stopping.");
                    return newlyGeneratedTokenIds;
                }

                // Extract KV cache from initial pass
                var kvCache = UpdateKVCache(initialOutput.RegularKeyValues);
                
                // 2. Autoregressive generation loop
                for (int step = 0; step < maxNewTokens - 1; ++step) // -1 because one token already generated
                {
                    Logger.Log($"--- [LLMModel.GenTokens Loop] Step {step} --- (Generating token {newlyGeneratedTokenIds.Count + 1})");
                    int currentTokenIdForInput = newlyGeneratedTokenIds[newlyGeneratedTokenIds.Count - 1];
                    
                    // Generate inputs for single token step
                    var stepInputs = await PrepareStepInputsAsync(
                        currentTokenIdForInput, 
                        kvCache, 
                        batchSize);

                    // Run inference for this step
                    var stepOutput = await RunInference(stepInputs, kvCache != null);
                    if (stepOutput == null)
                    {
                        break;
                    }

                    // Process logits and sample next token
                    nextTokenId = ProcessLogitsAndSample(
                        stepOutput.Logits, 
                        stepOutput.LogitsDimensions, 
                        0, // Single token position
                        temperature, 
                        topK, 
                        topP);

                    Logger.Log($"[LLMModel.GenTokens Loop Step {step}] Next token ID: {nextTokenId}");
                    newlyGeneratedTokenIds.Add(nextTokenId);

                    // Check if generation should end
                    if (nextTokenId == _eosTokenId)
                    {
                        Logger.Log($"[LLMModel.GenTokens] EOS token generated at step {step}. Stopping.");
                        break;
                    }

                    // Update KV cache for next step if needed
                    if (_numLLMLayers > 0)
                    {
                        kvCache = UpdateKVCache(stepOutput.RegularKeyValues, kvCache);
                    }
                }

                var endTime = Stopwatch.GetTimestamp();
                _generateSemanticTokensTimer.AddTiming(startTime, endTime);
            }
            catch (Exception ex)
            {
                Logger.LogError($"[LLMModel.GenTokens] Exception: {ex.Message}\n{ex.StackTrace}");
                throw new InvalidOperationException($"Token generation failed: {ex.Message}", ex);
            }
            finally
            {
                if (LogTiming)
                {
                    _inferenceTimer.LogTiming();
                    _updateKVCacheTimer.LogTiming();
                    _sampleLogitsTimer.LogTiming();
                    _prepareInitialInputsTimer.LogTiming();
                    _prepareStepInputsTimer.LogTiming();
                    _generateSemanticTokensTimer.LogTiming();
                }
            }

            return newlyGeneratedTokenIds;
        }

        // Class to hold ONNX inference outputs
        private class InferenceOutput
        {
            public float[] Logits { get; set; }
            public int[] LogitsDimensions { get; set; }
            public List<DisposableNamedOnnxValue> PresentKeyValues { get; set; }
            public List<NamedOnnxValue> RegularKeyValues { get; set; }
        }

        // Prepare input tensors for initial prompt
        private List<NamedOnnxValue> PrepareInitialInputs(long[] inputIdsData, int[] attentionMaskData, int batchSize)
        {
            var prepareInitialInputsStartTime = Stopwatch.GetTimestamp();
            List<NamedOnnxValue> inputsForCurrentStep = new(2 + _numLLMLayers * 2);
            
            // Input IDs tensor
            var inputIdsShape = new ReadOnlySpan<int>(new int[] { batchSize, inputIdsData.Length });
            DenseTensor<long> inputIdsTensor = new(new Memory<long>(inputIdsData), inputIdsShape);
            inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_inputIDsName, inputIdsTensor));

            Logger.Log($"[LLM.GenTokens] Input IDs for initial pass ({inputIdsData.Length}): [{string.Join(", ", inputIdsData.Take(50))}{(inputIdsData.Length > 50 ? "..." : "")}]");

            // Attention mask tensor
            if (!string.IsNullOrEmpty(_attentionMaskName) && attentionMaskData != null)
            {
                // Convert to longs
                long[] attentionMaskDataLong = new long[attentionMaskData.Length];
                for (int i = 0; i < attentionMaskData.Length; i++)
                {
                    attentionMaskDataLong[i] = attentionMaskData[i];
                }
                
                var attentionMaskShape = new ReadOnlySpan<int>(new int[] { batchSize, attentionMaskData.Length });
                DenseTensor<long> attentionMaskTensor = new(new Memory<long>(attentionMaskDataLong), attentionMaskShape);
                inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_attentionMaskName, attentionMaskTensor));
            }

            // Position IDs tensor
            if (!string.IsNullOrEmpty(_positionIDsName))
            {
                long[] positionIdsData = new long[inputIdsData.Length];
                for (int i = 0; i < inputIdsData.Length; i++) 
                {
                    positionIdsData[i] = i;
                }
                var positionIdsShape = new ReadOnlySpan<int>(new int[] { batchSize, positionIdsData.Length });
                DenseTensor<long> positionIdsTensor = new(new Memory<long>(positionIdsData), positionIdsShape);
                inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_positionIDsName, positionIdsTensor));
            }

            // Empty KV cache tensors
            if (_numLLMLayers > 0)
            {
                var emptyKvShape = new ReadOnlySpan<int>(new int[] { batchSize, _numAttentionHeads, 0, _headDimension });
                float[] emptyKvData = Array.Empty<float>();
                
                for (int i = 0; i < _numLLMLayers; ++i)
                {
                    if (_pastKeyInputNames.TryGetValue(i, out string pkName) && _pastValueInputNames.TryGetValue(i, out string pvName))
                    {
                        DenseTensor<float> pastKeyTensor = new(new Memory<float>(emptyKvData), emptyKvShape);
                        DenseTensor<float> pastValueTensor = new(new Memory<float>(emptyKvData), emptyKvShape);
                        
                        inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<float>(pkName, pastKeyTensor));
                        inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<float>(pvName, pastValueTensor));
                    }
                    else 
                    { 
                        UnityEngine.Debug.LogError($"[LLMModel.PrepareInitialInputs] Past KV input names not found for layer {i}."); 
                    }
                }
            }

            var prepareInitialInputsEndTime = Stopwatch.GetTimestamp();
            _prepareInitialInputsTimer.AddTiming(prepareInitialInputsStartTime, prepareInitialInputsEndTime);

            return inputsForCurrentStep;
        }

        // Prepare input tensors for a single token step
        private List<NamedOnnxValue> PrepareStepInputs(int tokenId, List<DenseTensor<float>> kvCache, int batchSize)
        {
            var prepareStepInputsStartTime = Stopwatch.GetTimestamp();
            List<NamedOnnxValue> inputsForCurrentStep = new(2 + _numLLMLayers * 2);
            
            // Input ID tensor
            long[] singleTokenIdData = new long[] { tokenId };
            var singleTokenShape = new ReadOnlySpan<int>(new int[] { batchSize, 1 });
            DenseTensor<long> singleTokenIdTensor = new(new Memory<long>(singleTokenIdData), singleTokenShape);
            inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_inputIDsName, singleTokenIdTensor));

            // Determine past KV sequence length for attention mask and position IDs
            int pastKvSequenceLength = 0;
            if (_numLLMLayers > 0 && kvCache != null && kvCache.Count > 0 && kvCache[0] != null)
            {
                pastKvSequenceLength = kvCache[0].Dimensions[2]; // Shape: (batch, num_heads, seq_len, head_dim)
            }

            // Attention mask tensor
            if (!string.IsNullOrEmpty(_attentionMaskName))
            {
                int maskLength = pastKvSequenceLength + 1;
                long[] attentionMaskData = new long[maskLength];
                FillArray(attentionMaskData, maskLength, 1L);
                
                var attentionMaskShape = new ReadOnlySpan<int>(new int[] { batchSize, maskLength });
                DenseTensor<long> attentionMaskTensor = new(new Memory<long>(attentionMaskData), attentionMaskShape);
                inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_attentionMaskName, attentionMaskTensor));
            }

            // Position ID tensor
            if (!string.IsNullOrEmpty(_positionIDsName))
            {
                long[] positionIdsData = new long[] { pastKvSequenceLength };
                DenseTensor<long> positionIdsTensor = new(new Memory<long>(positionIdsData), singleTokenShape);
                inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<long>(_positionIDsName, positionIdsTensor));
            }

            // KV cache tensors
            if (_numLLMLayers > 0 && kvCache != null)
            {
                for (int i = 0; i < _numLLMLayers; ++i)
                {
                    inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<float>(_pastKeyInputNames[i], kvCache[i * 2]));       // Key
                    inputsForCurrentStep.Add(NamedOnnxValue.CreateFromTensor<float>(_pastValueInputNames[i], kvCache[i * 2 + 1])); // Value
                }
            }

            var prepareStepInputsEndTime = Stopwatch.GetTimestamp();
            _prepareStepInputsTimer.AddTiming(prepareStepInputsStartTime, prepareStepInputsEndTime);

            return inputsForCurrentStep;
        }

        // Async wrapper for PrepareInitialInputs
        private async Task<List<NamedOnnxValue>> PrepareInitialInputsAsync(long[] inputIdsData, int[] attentionMaskData, int batchSize)
        {
            return await Task.Run(() => PrepareInitialInputs(inputIdsData, attentionMaskData, batchSize));
        }

        // Async wrapper for PrepareStepInputs  
        private async Task<List<NamedOnnxValue>> PrepareStepInputsAsync(int tokenId, List<DenseTensor<float>> kvCache, int batchSize)
        {
            return await Task.Run(() => PrepareStepInputs(tokenId, kvCache, batchSize));
        }

        // Run ONNX inference with prepared inputs using the new ORTModel pattern
        private async Task<InferenceOutput> RunInference(List<NamedOnnxValue> inputs, bool extractKvCache)
        {
            var startTime = Stopwatch.GetTimestamp();
            if (inputs == null || inputs.Count == 0)
            {
                Logger.LogError("[LLMModel.RunInference] No inputs provided.");
                return null;
            }

            Logger.Log($"[LLMModel.RunInference] Running inference with {inputs.Count} inputs. Extract KV: {extractKvCache}");

            try
            {
                // Convert NamedOnnxValue inputs to tensors and load them using the new pattern
                for (int i = 0; i < inputs.Count; i++)
                {
                    var input = inputs[i];
                    if (input.Value is Tensor<long> longTensor)
                    {
                        await LoadInput(i, longTensor);
                    }
                    else if (input.Value is Tensor<float> floatTensor)
                    {
                        await LoadInput(i, floatTensor);
                    }
                    else
                    {
                        Logger.LogError($"[LLMModel.RunInference] Unsupported input tensor type: {input.Value?.GetType()}");
                        return null;
                    }
                }

                // Run inference using the new pattern
                var outputs = await RunDisposable();
                
                // Create output structure
                var output = new InferenceOutput();
                
                // Extract logits output
                var logitsOutput = outputs.FirstOrDefault(o => o.Name == _logitsOutputName);
                if (logitsOutput?.Value is DenseTensor<float> logitsTensor)
                {
                    output.Logits = logitsTensor.Buffer.ToArray();
                    output.LogitsDimensions = logitsTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();
                }
                else
                {
                    Logger.LogError("[LLMModel.RunInference] Logits output not found or invalid format.");
                    return null;
                }
                
                // Extract KV cache if needed - store as NamedOnnxValue for now
                if (extractKvCache && _numLLMLayers > 0)
                {
                    output.PresentKeyValues = new List<DisposableNamedOnnxValue>();
                    output.RegularKeyValues = new List<NamedOnnxValue>();
                    
                    // Store the outputs for KV cache processing
                    for (int i = 0; i < _numLLMLayers; i++)
                    {
                        // Get key output
                        if (_presentKeyOutputNames.TryGetValue(i, out string keyName))
                        {
                            var keyOutput = outputs.FirstOrDefault(o => o.Name == keyName);
                            if (keyOutput != null)
                            {
                                output.RegularKeyValues.Add(keyOutput);
                            }
                        }
                        
                        // Get value output  
                        if (_presentValueOutputNames.TryGetValue(i, out string valueName))
                        {
                            var valueOutput = outputs.FirstOrDefault(o => o.Name == valueName);
                            if (valueOutput != null)
                            {
                                output.RegularKeyValues.Add(valueOutput);
                            }
                        }
                    }
                }

                var endTime = Stopwatch.GetTimestamp();
                _inferenceTimer.AddTiming(startTime, endTime);
                
                return output;
            }
            catch (Exception ex)
            {
                Logger.LogError($"[LLMModel.RunInference] Error during inference: {ex.Message}");
                return null;
            }
        }

        // Helper method to convert NamedOnnxValue to DenseTensor for KV cache processing
        private DenseTensor<float> ConvertNamedValueToTensor(NamedOnnxValue namedValue)
        {
            if (namedValue.Value is DenseTensor<float> tensor)
            {
                return tensor;
            }
            
            Logger.LogError($"[LLMModel] Expected DenseTensor<float> but got {namedValue.Value?.GetType()}");
            return null;
        }

        // Process logits and sample next token
        private int ProcessLogitsAndSample(float[] logitsData, int[] logitsDimensions, int tokenPosition, float temperature, int topK, float topP)
        {
            var startTime = Stopwatch.GetTimestamp();
            if (logitsData == null || logitsDimensions == null || logitsDimensions.Length < 3)
            {
                UnityEngine.Debug.LogError("[LLMModel.ProcessLogitsAndSample] Invalid logits data or dimensions.");
                return -1;
            }

            _vocabSize = logitsDimensions[2];
            ReadOnlySpan<float> tokenLogits = tokenPosition == 0 ? 
                logitsData : // For single token, entire buffer is for this token
                new ReadOnlySpan<float>(logitsData, tokenPosition * _vocabSize, _vocabSize); // For sequence, get the correct slice
            
            var endTime = Stopwatch.GetTimestamp();
            _processLogitsAndSampleTimer.AddTiming(startTime, endTime);

            return SampleLogits(tokenLogits.ToArray(), temperature, topK, topP);
        }

        // Update KV cache with new values for next step (overload for NamedOnnxValue)
        private List<DenseTensor<float>> UpdateKVCache(List<NamedOnnxValue> presentKeyValues, List<DenseTensor<float>> currentKvCache = null)
        {
            if (presentKeyValues == null || presentKeyValues.Count == 0 || _numLLMLayers == 0)
            {
                return currentKvCache; // Return existing cache if no new values
            }
            
            var startTime = Stopwatch.GetTimestamp();
            
            // Initialize the cached tensors if this is the first call
            if (_cachedKvTensors == null || !_isKvCacheInitialized)
            {
                _cachedKvTensors = new List<DenseTensor<float>>(_numLLMLayers * 2);
                _isKvCacheInitialized = true;
                
                // Pre-allocate all tensors
                for (int i = 0; i < _numLLMLayers * 2; i++)
                {
                    _cachedKvTensors.Add(null);
                }
            }
            
            // Re-use existing buffer list if already initialized
            List<DenseTensor<float>> updatedKvCache = _cachedKvTensors;
            
            // Enhanced memory copy using unsafe context for better performance
            unsafe
            {
                for (int i = 0; i < _numLLMLayers; ++i)
                {
                    var pkNamedValue = presentKeyValues[i * 2]; // Key
                    var pvNamedValue = presentKeyValues[i * 2 + 1]; // Value
                    
                    var pkDt = ConvertNamedValueToTensor(pkNamedValue);
                    var pvDt = ConvertNamedValueToTensor(pvNamedValue);
                    
                    if (pkDt == null || pvDt == null)
                    {
                        Logger.LogError($"[LLMModel.UpdateKVCache] Failed to get present KV DenseTensor for layer {i}.");
                        // Return old cache if we can't update
                        return currentKvCache;
                    }
                    
                    // Process the key tensor
                    int keyIndex = i * 2;
                    updatedKvCache[keyIndex] = ProcessTensor(pkDt, updatedKvCache[keyIndex]);
                    
                    // Process the value tensor
                    int valueIndex = i * 2 + 1;
                    updatedKvCache[valueIndex] = ProcessTensor(pvDt, updatedKvCache[valueIndex]);
                }
            }
            
            var endTime = Stopwatch.GetTimestamp();
            _updateKVCacheTimer.AddTiming(startTime, endTime);
            
            return updatedKvCache;
        }

        // Update KV cache with new values for next step (original overload for DisposableNamedOnnxValue)
        private List<DenseTensor<float>> UpdateKVCache(List<DisposableNamedOnnxValue> presentKeyValues, List<DenseTensor<float>> currentKvCache = null)
        {
            if (presentKeyValues == null || presentKeyValues.Count == 0 || _numLLMLayers == 0)
            {
                return currentKvCache; // Return existing cache if no new values
            }
            
            var startTime = Stopwatch.GetTimestamp();
            
            // Initialize the cached tensors if this is the first call
            if (_cachedKvTensors == null || !_isKvCacheInitialized)
            {
                _cachedKvTensors = new List<DenseTensor<float>>(_numLLMLayers * 2);
                _isKvCacheInitialized = true;
                
                // Pre-allocate all tensors
                for (int i = 0; i < _numLLMLayers * 2; i++)
                {
                    _cachedKvTensors.Add(null);
                }
            }
            
            // Re-use existing buffer list if already initialized
            List<DenseTensor<float>> updatedKvCache = _cachedKvTensors;
            
            // Enhanced memory copy using unsafe context for better performance
            unsafe
            {
                for (int i = 0; i < _numLLMLayers; ++i)
                {
                    DisposableNamedOnnxValue pkDisp = presentKeyValues[i * 2]; // Key
                    DisposableNamedOnnxValue pvDisp = presentKeyValues[i * 2 + 1]; // Value
                    
                    if (pkDisp.Value is not DenseTensor<float> pkDt || pvDisp.Value is not DenseTensor<float> pvDt)
                    {
                        UnityEngine.Debug.LogError($"[LLMModel.UpdateKVCache] Failed to get present KV DenseTensor for layer {i}.");
                        // Return old cache if we can't update
                        return currentKvCache;
                    }
                    
                    // Process the key tensor
                    int keyIndex = i * 2;
                    updatedKvCache[keyIndex] = ProcessTensor(pkDt, updatedKvCache[keyIndex]);
                    
                    // Process the value tensor
                    int valueIndex = i * 2 + 1;
                    updatedKvCache[valueIndex] = ProcessTensor(pvDt, updatedKvCache[valueIndex]);
                }
            }
            
            var endTime = Stopwatch.GetTimestamp();
            _updateKVCacheTimer.AddTiming(startTime, endTime);
            
            return updatedKvCache;
        }
        
        // Helper method to process a tensor efficiently
        private unsafe DenseTensor<float> ProcessTensor(DenseTensor<float> sourceTensor, DenseTensor<float> destTensor)
        {
            // Get dimensions and total size
            ReadOnlySpan<int> dimensions = sourceTensor.Dimensions;
            int totalSize = sourceTensor.Buffer.Length;
            
            // Create or reuse destination tensor
            if (destTensor == null || !TensorDimensionsMatch(destTensor.Dimensions, dimensions))
            {
                // Need to create a new tensor with the right dimensions
                float[] newBuffer = new float[totalSize];
                destTensor = new DenseTensor<float>(newBuffer, dimensions.ToArray());
            }
            else if (destTensor.Buffer.Length < totalSize)
            {
                // Existing tensor is too small, need to resize
                float[] newBuffer = new float[totalSize];
                destTensor = new DenseTensor<float>(newBuffer, dimensions.ToArray());
            }
            
            // Copy data using direct memory manipulation
            fixed (float* sourcePtr = sourceTensor.Buffer.Span)
            fixed (float* destPtr = destTensor.Buffer.Span)
            {
                // Memory copy is fastest for continuous buffers
                Buffer.MemoryCopy(
                    sourcePtr,  // Source pointer
                    destPtr,    // Destination pointer
                    totalSize * sizeof(float),  // Destination buffer size in bytes
                    totalSize * sizeof(float)   // Source bytes to copy
                );
            }
            
            return destTensor;
        }
        
        // Helper to compare tensor dimensions
        private bool TensorDimensionsMatch(ReadOnlySpan<int> dim1, ReadOnlySpan<int> dim2)
        {
            if (dim1.Length != dim2.Length)
                return false;
                
            for (int i = 0; i < dim1.Length; i++)
            {
                if (dim1[i] != dim2[i])
                    return false;
            }
            
            return true;
        }

        private int SampleLogits(float[] logits, float temperature, int topK, float topP)
        {
            var startTime = Stopwatch.GetTimestamp();
            if (temperature <= 1e-6) // Greedy decoding for near-zero temp
            {
                int argMaxResult = ArgMax(logits);
                return argMaxResult;
            }
            
            // Optimize: Only find top K elements instead of sorting the entire array
            int actualTopK = Math.Min(topK > 0 ? topK : _vocabSize, _vocabSize);
            
            // Ensure our buffers are properly sized
            int heapSize = actualTopK + 1;
            if (_topKHeap == null || _topKHeap.Length < heapSize)
            {
                _topKHeap = new (float, int)[heapSize * 2]; // Allocate with room to grow
            }
            
            if (_sortedLogits == null || _sortedLogits.Length < heapSize)
            {
                _sortedLogits = new float[heapSize * 2];
                _sortedIndices = new int[heapSize * 2];
            }
            
            int numFiltered;
            
            // Use unsafe code for better performance
            unsafe
            {
                // Fast top-K using heap for partial sorting
                if (actualTopK >= _vocabSize / 10)
                {
                    // For large K relative to vocab size, use a full sort
                    numFiltered = FullSortTopK(logits, actualTopK);
                }
                else
                {
                    // For small K, use a min-heap for partial sorting
                    numFiltered = HeapSelectTopK(logits, actualTopK);
                }
            }
            
            var sampleLogitsStartTime3 = Stopwatch.GetTimestamp();
            _sampleLogitsTimer2.AddTiming(startTime, sampleLogitsStartTime3);

            // Apply Top-P (Nucleus) filtering on the Top-K results
            if (topP < 1.0f && topP > 0.0f && numFiltered > 1) // Only apply if topP < 1.0 and we have multiple tokens
            {
                // Ensure our buffer is large enough
                EnsureBufferSize(ref _probsBuffer, numFiltered);
                
                // Apply softmax to get probabilities for nucleus sampling
                ComputeSoftmax(_sortedLogits, 0, numFiltered, _probsBuffer);
                
                // Apply nucleus sampling (top-p)
                numFiltered = ApplyNucleusSampling(_probsBuffer, numFiltered, topP);
            }

            // Ensure we have at least one token to sample from after filtering
            if (numFiltered == 0)
            {
                if (actualTopK > 0 && _sortedIndices != null && _sortedIndices.Length > 0) 
                {
                    return _sortedIndices[0]; // Return highest probability token
                }
                else 
                {
                    return ArgMax(logits); // Fallback
                }
            }
            
            // Apply temperature directly to the filtered logits
            for (int i = 0; i < numFiltered; i++)
            {
                _sortedLogits[i] /= temperature;
            }

            // Ensure our buffer is large enough
            EnsureBufferSize(ref _probsBuffer, numFiltered);

            // Calculate probabilities using Softmax on the final filtered & temperature-scaled logits
            ComputeSoftmax(_sortedLogits, 0, numFiltered, _probsBuffer);

            // Sample from the final filtered distribution using the computed probabilities
            int result = SampleFromIndices(_probsBuffer, _sortedIndices, numFiltered);
            
            var sampleLogitsEndTime = Stopwatch.GetTimestamp();
            _sampleLogitsTimer.AddTiming(startTime, sampleLogitsEndTime);
            return result;
        }
        
        // Fast implementation of top-K selection using full sort (for large K)
        private unsafe int FullSortTopK(float[] logits, int k)
        {
            int length = Math.Min(k + 1, _vocabSize);
            
            // Initialize indices array if needed
            if (_indices == null || _indices.Length != _vocabSize)
            {
                _indices = new int[_vocabSize];
                for (int i = 0; i < _vocabSize; i++)
                {
                    _indices[i] = i;
                }
            }
            
            // Use direct array sort with comparison for better performance
            fixed (float* logitsPtr = logits)
            fixed (int* indicesPtr = _indices)
            {
                // Sort indices based on logits values (descending)
                Array.Sort(_indices, 0, _vocabSize, new LogitComparer(logits));
                
                // Copy the top K elements to our pre-allocated buffers
                for (int i = 0; i < length; i++)
                {
                    _sortedLogits[i] = logits[_indices[i]];
                    _sortedIndices[i] = _indices[i];
                }
            }
            
            return length;
        }
        
        // Fast implementation of top-K selection using a min-heap (for small K)
        private unsafe int HeapSelectTopK(float[] logits, int k)
        {
            int length = Math.Min(k + 1, _vocabSize);
            
            // Initialize heap
            fixed (float* logitsPtr = logits)
            {
                // First k elements form initial heap
                for (int i = 0; i < length; i++)
                {
                    _topKHeap[i] = (logitsPtr[i], i);
                    
                    // Bubble up to maintain min-heap property
                    int current = i;
                    while (current > 0)
                    {
                        int parent = (current - 1) / 2;
                        if (_topKHeap[current].logit < _topKHeap[parent].logit)
                        {
                            // Swap with parent
                            (_topKHeap[parent], _topKHeap[current]) = (_topKHeap[current], _topKHeap[parent]);
                            current = parent;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                
                // Process remaining elements
                for (int i = length; i < _vocabSize; i++)
                {
                    // If current element is greater than minimum in heap, replace it
                    if (logitsPtr[i] > _topKHeap[0].logit)
                    {
                        _topKHeap[0] = (logitsPtr[i], i);
                        
                        // Heapify down
                        int current = 0;
                        while (true)
                        {
                            int leftChild = 2 * current + 1;
                            int rightChild = 2 * current + 2;
                            int smallest = current;
                            
                            if (leftChild < length && _topKHeap[leftChild].logit < _topKHeap[smallest].logit)
                                smallest = leftChild;
                            
                            if (rightChild < length && _topKHeap[rightChild].logit < _topKHeap[smallest].logit)
                                smallest = rightChild;
                            
                            if (smallest != current)
                            {
                                // Swap with smaller child
                                (_topKHeap[current], _topKHeap[smallest]) = (_topKHeap[smallest], _topKHeap[current]);
                                current = smallest;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            }
            
            // Sort the selected top-K items in descending order
            Array.Sort(_topKHeap, 0, length, Comparer<(float, int)>.Create((a, b) => b.Item1.CompareTo(a.Item1)));
            
            // Copy to output arrays
            for (int i = 0; i < length; i++)
            {
                _sortedLogits[i] = _topKHeap[i].logit;
                _sortedIndices[i] = _topKHeap[i].index;
            }
            
            return length;
        }
        
        // Comparer class for sorting logits
        private class LogitComparer : IComparer<int>
        {
            private readonly float[] _logits;
            
            public LogitComparer(float[] logits)
            {
                _logits = logits;
            }
            
            public int Compare(int x, int y)
            {
                // Descending order (larger logits first)
                return _logits[y].CompareTo(_logits[x]);
            }
        }
        
        // Apply nucleus sampling (top-p) by filtering to tokens that cumulatively exceed p
        private int ApplyNucleusSampling(float[] probs, int count, float topP)
        {
            // We assume probs are already sorted in descending order
            float cumulativeProb = 0.0f;
            int nucleusSize = 0;
            
            for (int i = 0; i < count; i++)
            {
                cumulativeProb += probs[i];
                nucleusSize++;
                
                if (cumulativeProb >= topP)
                {
                    break;
                }
            }
            
            return nucleusSize; // Return the new filtered size
        }
        
        // Optimized softmax implementation that works on a subset of an array
        private unsafe void ComputeSoftmax(float[] logits, int offset, int count, float[] outProbs)
        {
            if (count == 0) return;
            
            // Find max for numerical stability
            float maxLogit = float.MinValue;
            fixed (float* logitsPtr = logits)
            {
                for (int i = 0; i < count; i++)
                {
                    if (logitsPtr[offset + i] > maxLogit)
                    {
                        maxLogit = logitsPtr[offset + i];
                    }
                }
                
                // Compute exp(logit - maxLogit) and sum
                float sumExp = 0.0f;
                for (int i = 0; i < count; i++)
                {
                    outProbs[i] = MathF.Exp(logitsPtr[offset + i] - maxLogit);
                    sumExp += outProbs[i];
                }
                
                // Normalize
                if (sumExp > 1e-6f)
                {
                    float invSum = 1.0f / sumExp;
                    for (int i = 0; i < count; i++)
                    {
                        outProbs[i] *= invSum;
                    }
                }
                else
                {
                    // Handle potential division by zero with uniform distribution
                    float uniformProb = 1.0f / count;
                    for (int i = 0; i < count; i++)
                    {
                        outProbs[i] = uniformProb;
                    }
                }
            }
        }
        
        // Sample an index based on probabilities
        private int SampleFromIndices(float[] probabilities, int[] indices, int count)
        {
            if (count == 0)
            {
                UnityEngine.Debug.LogError("[SampleFromIndices] Empty probability array");
                return indices.Length > 0 ? indices[0] : -1;
            }
            
            // For a single item, just return it directly
            if (count == 1)
            {
                return indices[0];
            }
            
            // Use thread-safe random instead of Unity's Random.value
            float randomValue = (float)_threadLocalRandom.Value.NextDouble();
            float cumulativeProbability = 0.0f;
            
            for (int i = 0; i < count; i++)
            {
                cumulativeProbability += probabilities[i];
                if (randomValue <= cumulativeProbability)
                {
                    return indices[i];
                }
            }
            
            // Fallback: return the last index (due to floating point inaccuracies)
            return indices[count - 1];
        }

        private int ArgMax(ReadOnlySpan<float> array)
        {
            if (array.IsEmpty) return -1;
            
            int maxIndex = 0;
            float maxValue = array[0];
            int i = 1;
            
            // Use loop unrolling for better performance
            int limit = array.Length - (array.Length % 4);
            for (; i < limit; i += 4)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
                
                if (array[i+1] > maxValue)
                {
                    maxValue = array[i+1];
                    maxIndex = i+1;
                }
                
                if (array[i+2] > maxValue)
                {
                    maxValue = array[i+2];
                    maxIndex = i+2;
                }
                
                if (array[i+3] > maxValue)
                {
                    maxValue = array[i+3];
                    maxIndex = i+3;
                }
            }
            
            // Handle remaining elements
            for (; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }
            
            return maxIndex;
        }
        
        // Ensure a buffer is large enough, resize if needed
        private void EnsureBufferSize(ref float[] buffer, int requiredSize)
        {
            if (buffer.Length < requiredSize)
            {
                buffer = new float[requiredSize * 2]; // Double size for future growth
            }
        }

        private void ComputeSoftmax(float[] logits, float[] outProbs)
        {
            ComputeSoftmax(logits, 0, logits.Length, outProbs);
        }

        // Helper method for filling an array with a specific value
        private void FillArray(long[] array, int count, long value)
        {
            // For small arrays, direct assignment is fastest
            if (count <= 16)
            {
                for (int i = 0; i < count; i++)
                {
                    array[i] = value;
                }
                return;
            }
            
            // For larger arrays, use doubling strategy for better performance
            // Fill first element
            array[0] = value;
            
            // Double the filled portion until we've filled the whole array
            int filled = 1;
            while (filled < count)
            {
                int toCopy = Math.Min(filled, count - filled);
                Buffer.BlockCopy(array, 0, array, filled * sizeof(long), toCopy * sizeof(long));
                filled += toCopy;
            }
        }
    }
}
