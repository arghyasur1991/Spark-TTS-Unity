using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEngine;

namespace SparkTTS.Core
{
    using Models;
    using Utils;
    
    public class TTSInferenceConfig
    {
        public string TextToSynthesize { get; set; }
        public AudioClip ReferenceAudioClip { get; set; } // For voice cloning
        public int TargetSampleRate { get; set; } = 16000;

        // For Style Control (if not using reference audio clip)
        public string StyleControlPrompt { get; set; } // If set, this is used as LLM input text
        public int[] StyleControlGlobalTokens { get; set; } // Pre-defined global tokens for style control

        // LLM Generation Parameters
        public int MaxNewLLMTokens { get; set; } = 768;
        public float LLMTemperature { get; set; } = 0.8f; // Default from LLMModel
        public int LLMTopK { get; set; } = 50;          // Default from LLMModel
        public float LLMTopP { get; set; } = 0.95f;        // Default from LLMModel

        // --- Optional: Control Prompt Construction ---
        public bool UseAcousticTokensInPrompt { get; set; } = true; // Default to true
        
        // Style control parameters
        public string Gender { get; set; } = null;
        public string Pitch { get; set; } = null;
        public string Speed { get; set; } = null; 

        // Paths for model configuration (can be overridden if not using defaults from SparkTTSModelPaths)
        // These are advanced and optional. If null, the defaults in each model class will be used.
        public string LLMModelFolder { get; set; } = null;
        public string LLMModelFile { get; set; } = null;
        public string MelModelFolder { get; set; } = null;
        public string MelModelFile { get; set; } = null;
        public string SpeakerEncoderFolder { get; set; } = null;
        public string SpeakerEncoderFile { get; set; } = null;
        public string VocoderModelFolder { get; set; } = null;
        public string VocoderModelFile { get; set; } = null;
        public string Wav2Vec2ModelFolder { get; set; } = null;
        public string Wav2Vec2ModelFile { get; set; } = null;
        public string EncoderQuantizerModelFolder { get; set; } = null;
        public string EncoderQuantizerModelFile { get; set; } = null;

        // Path to HuggingFace BPE Tokenizer files (vocab.bpe, merges.txt, added_tokens.json)
        // This is needed if TokenizerService is also to be made non-MonoBehaviour and initialized here.
        // For now, assuming an existing TokenizerService instance is provided or accessible.
        // public string TokenizerFilesDirectory { get; set; } = "LLM_data/tokenizer_files"; // Relative to StreamingAssets
    }
    public class TTSInferenceResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }
        public float[] Waveform { get; set; }
        public TokenizationOutput ModelInputs { get; set; }
        public int[] GlobalTokenIds { get; set; }
    }

    /// <summary>
    /// Main SparkTTS model for text-to-speech generation, aligning with Python's SparkTTS class
    /// </summary>
    public class SparkTTS : IDisposable
    {
        private LLMModel _llmModel;
        private SparkTTSAudioTokenizer _audioTokenizer;
        private SparkTTSBiCodec _biCodec; 
        private MelSpectrogramModel _melModel;
        private SpeakerEncoderModel _speakerEncoderModel;
        private Wav2Vec2Model _wav2Vec2Model;
        private BiCodecEncoderQuantizerModel _encoderQuantizerModel;
        private VocoderModel _vocoderModel;

        private TokenizerService _textTokenizerService;
        private AudioLoaderService _audioLoaderService; // For processing AudioClip
        private readonly AggregatedTimer _totalTimer;
        private readonly AggregatedTimer _tokenizationTimer;
        private readonly AggregatedTimer _modelGenerationTimer;
        private readonly AggregatedTimer _semanticTokenExtractionTimer;
        private readonly AggregatedTimer _globalTokenExtractionTimer;
        private readonly AggregatedTimer _vocoderTimer;
        private readonly AggregatedTimer _audioLoaderTimer;
        private readonly AggregatedTimer _updateTextInTokenizedInputsTimer;
        public static bool LogTiming = false;
        public bool OptimalMemoryUsage { get; set; } = false;

        public bool IsInitialized { get; private set; } = false;
        private bool _disposed = false;

        // Config-related constants
        private const string END_GLOBAL_TOKEN = "<|end_global_token|>";
        private const string END_SEMANTIC_TOKEN = "<|end_semantic_token|>";
        private const string START_CONTENT = "<|start_content|>";
        private const string END_CONTENT = "<|end_content|>";
        private const string START_GLOBAL_TOKEN = "<|start_global_token|>";
        private const string START_SEMANTIC_TOKEN = "<|start_semantic_token|>";

        public SparkTTS(TTSInferenceConfig config = null)
        {
            try
            {
                _audioLoaderService = new AudioLoaderService();
                // --- Initialize Text Tokenizer Service ---
                Logger.LogVerbose("[SparkTTS] Initializing TokenizerService...");
                string llmModelFolder = config.LLMModelFolder ?? SparkTTSModelPaths.LLMFolder;
                string tokenizerRelativePath = Path.Combine(SparkTTSModelPaths.BaseSparkTTSPathInStreamingAssets, llmModelFolder, "tokenizer.json");
                string fullTokenizerPath = Path.Combine(Application.streamingAssetsPath, tokenizerRelativePath);

                Logger.LogVerbose($"[SparkTTS] Attempting to load tokenizer.json from: {fullTokenizerPath}");

                if (!File.Exists(fullTokenizerPath))
                {
                    throw new FileNotFoundException($"tokenizer.json not found at expected path: {fullTokenizerPath}");
                }
                string tokenizerJsonText = File.ReadAllText(fullTokenizerPath);

                if (string.IsNullOrEmpty(tokenizerJsonText))
                {
                    throw new InvalidOperationException("Tokenizer JSON file is empty.");
                }
                TokenizerDefinition tokenizerDef = JsonConvert.DeserializeObject<TokenizerDefinition>(tokenizerJsonText);
                if (tokenizerDef == null)
                {
                    throw new InvalidOperationException("Failed to deserialize Tokenizer JSON.");
                }
                // Use hardcoded special token IDs consistent with LLMInference/SparkTTS
                int bosTokenId = -1; 
                int eosTokenId = 151645; 
                int padTokenId = 151643;
                _textTokenizerService = new TokenizerService(tokenizerDef, bosTokenId, eosTokenId, padTokenId);
                Logger.LogVerbose("[SparkTTS] TokenizerService Initialized.");

                // --- Initialize ONNX Models ---
                Logger.LogVerbose("[SparkTTS] Initializing ONNX Models...");
                _melModel = new MelSpectrogramModel();
                _speakerEncoderModel = new SpeakerEncoderModel();
                _llmModel = new LLMModel(tokenizerDef); // Pass tokenizer definition
                _vocoderModel = new VocoderModel();
                _wav2Vec2Model = new Wav2Vec2Model();
                _encoderQuantizerModel = new BiCodecEncoderQuantizerModel();
                Logger.LogVerbose("[SparkTTS] ONNX Models Initialized.");

                _audioTokenizer = new SparkTTSAudioTokenizer(
                    _audioLoaderService,
                    _melModel,
                    _speakerEncoderModel,
                    _wav2Vec2Model,
                    _encoderQuantizerModel
                );
                _biCodec = new SparkTTSBiCodec(_vocoderModel);

                _totalTimer = new AggregatedTimer("SparkTTS");
                _tokenizationTimer = new AggregatedTimer("Tokenization");
                _modelGenerationTimer = new AggregatedTimer("Model Generation");
                _semanticTokenExtractionTimer = new AggregatedTimer("Semantic Token Extraction");
                _globalTokenExtractionTimer = new AggregatedTimer("Global Token Extraction");
                _vocoderTimer = new AggregatedTimer("Vocoder");
                _audioLoaderTimer = new AggregatedTimer("Audio Loader");
                _updateTextInTokenizedInputsTimer = new AggregatedTimer("Update Text in Tokenized Inputs");
                IsInitialized = true;
                Logger.LogVerbose("[SparkTTS] Successfully initialized all components.");
            }
            catch (Exception e)
            {
                Logger.LogError($"[SparkTTS] Initialization failed: {e.Message}\n{e.StackTrace}");
                IsInitialized = false;
                // Dispose any partially initialized models if necessary
                Dispose();
            }
        }
        
        /// <summary>
        /// Sets the execution provider for the ONNX model.
        /// This determines the hardware backend (e.g., CPU, CUDA, CoreML) to be used for inference.
        /// This method must be called before the model loading is initiated.
        /// </summary>
        /// <param name="executionProvider">The execution provider to use for the model.</param>
        public void SetExecutionProvider(ExecutionProvider executionProvider)
        {
            _melModel.SetExecutionProvider(executionProvider);
            _speakerEncoderModel.SetExecutionProvider(executionProvider);
            _llmModel.SetExecutionProvider(executionProvider);
            _vocoderModel.SetExecutionProvider(executionProvider);
            _wav2Vec2Model.SetExecutionProvider(executionProvider);
            _encoderQuantizerModel.SetExecutionProvider(executionProvider);
            Logger.Log($"[SparkTTS] Set Execution Provider for all models to: {executionProvider}");
        }

        /// <summary>
        /// Processes input for voice cloning, aligning with Python's process_prompt method
        /// </summary>
        public string ProcessPrompt(string text, int[] globalTokenIds, List<long> semanticTokenIds = null)
        {
            string globalTokens = string.Join("", globalTokenIds.Select(id => $"<|bicodec_global_{id}|>"));
            
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.Append(SparkTokenBuilder.Task("tts"));
            sb.Append(START_CONTENT);
            sb.Append(text);
            sb.Append(END_CONTENT);
            sb.Append(START_GLOBAL_TOKEN);
            sb.Append(globalTokens);
            sb.Append(END_GLOBAL_TOKEN);

            if (semanticTokenIds != null && semanticTokenIds.Count > 0)
            {
                string semanticTokens = string.Join("", semanticTokenIds.Select(id => $"<|bicodec_semantic_{id}|>"));
                sb.Append(START_SEMANTIC_TOKEN);
                sb.Append(semanticTokens);
            }

            return sb.ToString();
        }

        /// <summary>
        /// Processes input for voice control using attributes, aligning with Python's process_prompt_control
        /// </summary>
        public string ProcessPromptControl(string gender, string pitch, string speed, string text)
        {
            // Mappings are now handled by SparkTokenBuilder
            string genderToken = SparkTokenBuilder.Gender(gender);
            string pitchToken = SparkTokenBuilder.PitchLevel(pitch);
            string speedToken = SparkTokenBuilder.SpeedLevel(speed);
            string attributeTokens = genderToken + pitchToken + speedToken;

            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            sb.Append(SparkTokenBuilder.Task("controllable_tts"));
            sb.Append(START_CONTENT);
            sb.Append(text);
            sb.Append(END_CONTENT);
            sb.Append("<|start_style_label|>");
            sb.Append(attributeTokens);
            sb.Append("<|end_style_label|>");

            return sb.ToString();
        }

        /// <summary>
        /// Asynchronously tokenizes inputs without performing inference, matching Python implementation
        /// </summary>
        public async Task<(TokenizationOutput modelInputs, int[] globalTokenIds)> TokenizeInputsAsync(
            string text,
            float[] promptSpeechSamples = null,
            string promptText = null,
            string gender = null,
            string pitch = null,
            string speed = null)
        {
            int[] globalTokenIds = null;
            string prompt;
            
            if (promptSpeechSamples != null)
            {
                // Voice cloning mode
                var (globalTokens, semanticTokens) = await _audioTokenizer.TokenizeAsync(promptSpeechSamples, OptimalMemoryUsage);
                globalTokenIds = globalTokens;
                List<long> acousticSemanticTokens = semanticTokens;

                if (globalTokenIds == null || globalTokenIds.Length == 0)
                {
                    Logger.LogError("[SparkTTS.TokenizeInputs] Failed to get global speaker tokens");
                    return (null, null);
                }

                prompt = ProcessPrompt(text, globalTokenIds, promptText != null ? acousticSemanticTokens : null);
            }
            else if (gender != null)
            {
                // Control mode
                prompt = ProcessPromptControl(gender, pitch, speed, text);
            }
            else
            {
                Logger.LogError("[SparkTTS.TokenizeInputs] Neither gender (for control) nor promptSpeech (for cloning) provided");
                return (null, null);
            }

            TokenizationOutput modelInputs = _textTokenizerService.Encode(prompt);
            return (modelInputs, globalTokenIds);
        }

        public float[] LoadAudioClip(AudioClip audioClip, int targetSampleRate)
        {
            long startTime = Stopwatch.GetTimestamp();
            if (_audioLoaderService == null)
            { 
                Logger.LogError("[TTSInferenceOrchestrator] AudioLoaderService is null after setup.");
                return null;
            }

            try
            {
                // Process reference audio if provided
                float[] referenceAudioSamples = null;
                if (audioClip != null)
                {
                    AudioLoaderService.AudioSegments? audioSegments = _audioLoaderService.LoadAndProcessAudio(
                        audioClip, targetSampleRate);
                    
                    if (!audioSegments.HasValue)
                    {
                        Logger.LogError("[TTSInferenceOrchestrator] Failed to load and process reference audio.");
                        return null;
                    }
                    
                    referenceAudioSamples = audioSegments.Value.ReferenceClip.Samples;
                }
                var endTime = Stopwatch.GetTimestamp();
                _audioLoaderTimer.AddTiming(startTime, endTime);
                return referenceAudioSamples;
            }
            catch (Exception e)
            {
                Logger.LogError($"[SparkTTS] LoadAudioClip failed: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        /// <summary>
        /// Main inference method matching Python's inference signature and behavior
        /// </summary>
        public async Task<TTSInferenceResult> InferenceAsync(
            string text = null,
            float[] promptSpeechSamples = null,
            string promptText = null,
            string gender = null,
            string pitch = null,
            string speed = null,
            float temperature = 0.8f,
            int topK = 50,
            float topP = 0.95f,
            int maxNewTokens = 3000,
            bool doSample = true,
            TokenizationOutput modelInputs = null,
            int[] globalTokenIds = null)
        {
            bool isControlMode = promptSpeechSamples == null && gender != null;
            long startTime = 0;
            long endTime = 0;

            startTime = Stopwatch.GetTimestamp();

            // Step 1: Tokenize inputs if not provided
            if (modelInputs == null || isControlMode)
            {
                if (!OptimalMemoryUsage)
                {
                    StartLoadingGeneratorModels();
                }
                if (text == null)
                {
                    Logger.LogError("[SparkTTS.Inference] Either text or modelInputs must be provided");
                    return new TTSInferenceResult { Success = false, ErrorMessage = "Either text or modelInputs must be provided" };
                }

                (modelInputs, globalTokenIds) = await TokenizeInputsAsync(
                    text, promptSpeechSamples, promptText, gender, pitch, speed);
                var endTokenizationTime = Stopwatch.GetTimestamp();
                _tokenizationTimer.AddTiming(startTime, endTokenizationTime);
            }

            if (modelInputs == null)
            {
                Logger.LogError("[SparkTTS.Inference] Failed to create model inputs");
                return new TTSInferenceResult { Success = false, ErrorMessage = "Failed to create model inputs" };
            }
            if (!OptimalMemoryUsage)
            {
                Logger.LogVerbose("[SparkTTS.Inference] Starting loading all voice cloning models");
                StartLoadingVoiceCloningModels();
            }
            // Step 2: Generate tokens using LLM
            var startSemanticTokenGeneration = Stopwatch.GetTimestamp();

            List<int> generatedIds = await _llmModel.RunAsync(
                async () => await _llmModel.GenerateSemanticTokensAsync(
                    modelInputs, 
                    maxNewTokens, 
                    temperature, 
                    topK, 
                    topP),
                standaloneLoading: OptimalMemoryUsage);

            var endSemanticTokenGeneration = Stopwatch.GetTimestamp();
            _modelGenerationTimer.AddTiming(startSemanticTokenGeneration, endSemanticTokenGeneration);

            if (generatedIds == null || generatedIds.Count == 0)
            {
                Logger.LogError("[SparkTTS.Inference] LLM generation failed or produced empty output");
                return new TTSInferenceResult { Success = false, ErrorMessage = "LLM generation failed or produced empty output" };
            }

            // Step 3: Extract semantic tokens from generated output
            var startSemanticTokenExtraction = Stopwatch.GetTimestamp();

            // Using skipSpecialTokens: false here is crucial.
            // We need all generated tokens, including the <|bicodec_semantic_...|>
            // and <|bicodec_global_...|> tokens, to be present in the string
            // for the subsequent Regex-based extraction to work correctly.
            // Python's equivalent uses skip_special_tokens=True, but its tokenizer
            // doesn't strip these specific structural tokens, while our C# version
            // might if they were marked as "special" and this flag was true.
            string decodedOutput = _textTokenizerService.Decode(generatedIds, skipSpecialTokens: false);
            List<long> semanticTokenIds = ExtractSemanticTokens(decodedOutput);
            var endSemanticTokenExtraction = Stopwatch.GetTimestamp();
            _semanticTokenExtractionTimer.AddTiming(startSemanticTokenExtraction, endSemanticTokenExtraction);

            List<int> globalTokensForBiCodec = new();
            if (isControlMode)
            {
                Logger.LogVerbose($"[SparkTTS.Inference DEBUG STYLE] Decoded LLM Output (Style Control):\n{decodedOutput}");
                globalTokensForBiCodec = ExtractGlobalTokens(decodedOutput);
                Logger.LogVerbose($"[SparkTTS.Inference DEBUG STYLE] Extracted Global Tokens (Style Control): [{string.Join(", ", globalTokensForBiCodec)}]");

                if (globalTokensForBiCodec.Count == 0)
                {
                    Logger.LogError("[SparkTTS.Inference] Failed to extract global tokens from LLM output in control mode (Count is 0).");
                    if (globalTokenIds != null) 
                    {
                        globalTokensForBiCodec.AddRange(globalTokenIds);
                    }
                }
            }
            else
            {
                 // In voice cloning mode, use the global tokens derived from the prompt audio
                 if (globalTokenIds != null) 
                 {
                    globalTokensForBiCodec.AddRange(globalTokenIds);
                 }
            }

            var endGlobalTokenExtraction = Stopwatch.GetTimestamp();
            _globalTokenExtractionTimer.AddTiming(endSemanticTokenExtraction, endGlobalTokenExtraction);

            if (semanticTokenIds.Count == 0)
            {
                Logger.LogError("[SparkTTS.Inference] Failed to extract semantic tokens from LLM output");
                return new TTSInferenceResult { Success = false, ErrorMessage = "Failed to extract semantic tokens from LLM output" };
            }

            // Step 4: Generate waveform from tokens
            var startAudioConversion = Stopwatch.GetTimestamp();
            Logger.LogVerbose($"[SparkTTS.Inference] Generating waveform from tokens: {semanticTokenIds.Count} semantic tokens, {globalTokensForBiCodec.Count} global tokens");

            // Use the appropriate global tokens (extracted for control mode, from input for cloning)
            float[] waveform = await _biCodec.DetokenizeAsync(semanticTokenIds.ToArray(), globalTokensForBiCodec.ToArray(), OptimalMemoryUsage);
            var endAudioConversion = Stopwatch.GetTimestamp();
            _vocoderTimer.AddTiming(startAudioConversion, endAudioConversion);

            Logger.LogVerbose($"[SparkTTS.Inference] Waveform generated");
            endTime = Stopwatch.GetTimestamp();
            _totalTimer.AddTiming(startTime, endTime);

            if (LogTiming)
            {
                _totalTimer.LogTiming();
                _tokenizationTimer.LogTiming();
                _modelGenerationTimer.LogTiming();
                _semanticTokenExtractionTimer.LogTiming();
                _globalTokenExtractionTimer.LogTiming();
                _vocoderTimer.LogTiming();
                _audioLoaderTimer.LogTiming();
                _updateTextInTokenizedInputsTimer.LogTiming();
            }

            if (waveform == null || waveform.Length == 0)
            {
                Logger.LogError("[SparkTTS.Inference] BiCodec failed to generate waveform");
                return new TTSInferenceResult { Success = false, ErrorMessage = "BiCodec failed to generate waveform" };
            }
            return new TTSInferenceResult { Success = true, Waveform = waveform, ModelInputs = modelInputs, GlobalTokenIds = globalTokensForBiCodec.ToArray() };
        }
        private void DisposeModels()
        {
            _llmModel?.Dispose();
            _melModel?.Dispose();
            _speakerEncoderModel?.Dispose();
            _wav2Vec2Model?.Dispose();
            _encoderQuantizerModel?.Dispose();
            _vocoderModel?.Dispose();
        }

        public void DisposeGeneratorOnlyModels()
        {
            if (OptimalMemoryUsage)
            {
                return;
            }
            _melModel?.Dispose();
            _speakerEncoderModel?.Dispose();
            _wav2Vec2Model?.Dispose();
            _encoderQuantizerModel?.Dispose();
        }

        /// <summary>
        /// Helper method to extract semantic tokens from decoded output
        /// </summary>
        private List<long> ExtractSemanticTokens(string decodedOutput)
        {
            List<long> semanticTokenIds = new List<long>();
            Regex semanticTokenRegex = new Regex(@"<\|bicodec_semantic_(\d+)\|>");
            MatchCollection matches = semanticTokenRegex.Matches(decodedOutput);

            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1 && long.TryParse(match.Groups[1].Value, out long tokenId))
                {
                    semanticTokenIds.Add(tokenId);
                }
                else
                {
                    Logger.LogWarning($"[SparkTTS.ExtractSemanticTokens] Failed to parse semantic token ID from: {match.Value}");
                }
            }

            return semanticTokenIds;
        }

        /// <summary>
        /// Helper method to extract global tokens from decoded output (Added for style control fix)
        /// </summary>
        private List<int> ExtractGlobalTokens(string decodedOutput)
        {
            List<int> globalTokenIds = new List<int>();
            Regex globalTokenRegex = new Regex(@"<\|bicodec_global_(\d+)\|>");
            MatchCollection matches = globalTokenRegex.Matches(decodedOutput);

            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1 && int.TryParse(match.Groups[1].Value, out int tokenId))
                {
                    globalTokenIds.Add(tokenId);
                }
                else
                {
                    Logger.LogWarning($"[SparkTTS.ExtractGlobalTokens] Failed to parse global token ID from: {match.Value}");
                }
            }
            Logger.LogVerbose($"[SparkTTS.ExtractGlobalTokens] Extracted {globalTokenIds.Count} global tokens.");
            return globalTokenIds;
        }

        /// <summary>
        /// Updates text in pre-tokenized inputs without retokenizing everything
        /// </summary>
        public TokenizationOutput UpdateTextInTokenizedInputs(
            TokenizationOutput modelInputs,
            string newText,
            bool isControlMode = false,
            string gender = null,
            string pitch = null,
            string speed = null)
        {
            var startTime = Stopwatch.GetTimestamp();

            if (modelInputs == null)
            {
                Logger.LogError("[SparkTTS.UpdateTextInTokenizedInputs] Model inputs are null");
                return null;
            }

            // Decode the tokenized inputs back to text
            string prompt = _textTokenizerService.Decode(modelInputs.InputIds, skipSpecialTokens: false);
            string newPrompt;

            if (isControlMode)
            {
                // For style-based voices, regenerate the entire prompt with new text
                if (string.IsNullOrEmpty(gender))
                {
                    Logger.LogError("[SparkTTS.UpdateTextInTokenizedInputs] Gender is required for control mode");
                    return null;
                }
                newPrompt = ProcessPromptControl(gender, pitch, speed, newText);
            }
            else
            {
                // For voice cloning, replace only the text between content markers
                int startContentIdx = prompt.IndexOf(START_CONTENT);
                int endContentIdx = prompt.IndexOf(END_CONTENT);
                
                if (startContentIdx != -1 && endContentIdx != -1)
                {
                    string prefix = prompt[..(startContentIdx + START_CONTENT.Length)];
                    string suffix = prompt[endContentIdx..];
                    newPrompt = prefix + newText + suffix;
                }
                else
                {
                    Logger.LogError("[SparkTTS.UpdateTextInTokenizedInputs] Could not find content markers in prompt");
                    return null;
                }
            }

            Logger.LogVerbose($"[SparkTTS.UpdateTextInTokenizedInputs] New prompt: {newPrompt}");

            // Re-tokenize with the new text
            var result = _textTokenizerService.Encode(newPrompt);
            var endTime = Stopwatch.GetTimestamp();
            _updateTextInTokenizedInputsTimer.AddTiming(startTime, endTime);
            return result;
        }

        private void StartLoadingGeneratorModels()
        {
            _llmModel.StartLoadingAsync();
            _melModel.StartLoadingAsync();
            _speakerEncoderModel.StartLoadingAsync();
            _wav2Vec2Model.StartLoadingAsync();
            _encoderQuantizerModel.StartLoadingAsync();
            _vocoderModel.StartLoadingAsync();
        }

        private void StartLoadingVoiceCloningModels()
        {
            _llmModel.StartLoadingAsync();
            _vocoderModel.StartLoadingAsync();
        }

        /// <summary>
        /// Waits for all models to be fully loaded.
        /// Use this in Performance mode to ensure all models are ready before inference.
        /// </summary>
        /// <param name="cancellationToken">Optional cancellation token</param>
        /// <returns>A task that completes when all models are loaded</returns>
        public async Task WaitForAllModelsAsync(System.Threading.CancellationToken cancellationToken = default)
        {
            Logger.Log("[SparkTTS] Waiting for all models to load...");
            
            var loadTasks = new List<Task>();
            
            if (_llmModel?.LoadTask != null) loadTasks.Add(_llmModel.LoadTask);
            if (_melModel?.LoadTask != null) loadTasks.Add(_melModel.LoadTask);
            if (_speakerEncoderModel?.LoadTask != null) loadTasks.Add(_speakerEncoderModel.LoadTask);
            if (_wav2Vec2Model?.LoadTask != null) loadTasks.Add(_wav2Vec2Model.LoadTask);
            if (_encoderQuantizerModel?.LoadTask != null) loadTasks.Add(_encoderQuantizerModel.LoadTask);
            if (_vocoderModel?.LoadTask != null) loadTasks.Add(_vocoderModel.LoadTask);
            
            if (loadTasks.Count > 0)
            {
                // Wait for all tasks with cancellation support
                var combinedTask = Task.WhenAll(loadTasks);
                var cancelTask = Task.Delay(-1, cancellationToken);
                
                await Task.WhenAny(combinedTask, cancelTask);
                
                cancellationToken.ThrowIfCancellationRequested();
            }
            
            Logger.Log("[SparkTTS] All models loaded successfully");
        }

        /// <summary>
        /// Gets whether all models are loaded.
        /// </summary>
        public bool AreAllModelsLoaded =>
            (_llmModel?.IsInitialized ?? false) &&
            (_melModel?.IsInitialized ?? false) &&
            (_speakerEncoderModel?.IsInitialized ?? false) &&
            (_wav2Vec2Model?.IsInitialized ?? false) &&
            (_encoderQuantizerModel?.IsInitialized ?? false) &&
            (_vocoderModel?.IsInitialized ?? false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed) return;
            if (disposing)
            {
                _biCodec?.Dispose();
                _audioTokenizer?.Dispose();
                DisposeModels();

                _llmModel = null;
                _vocoderModel = null;
                _speakerEncoderModel = null;
                _melModel = null;
                _wav2Vec2Model = null;
                _encoderQuantizerModel = null;

                _audioTokenizer = null;
                _biCodec = null;
                _textTokenizerService = null;
                _audioLoaderService = null; // Ditto
                _textTokenizerService = null;
                Logger.LogVerbose("[SparkTTS] Disposed (references cleared).");
            }
            IsInitialized = false;
            _disposed = true;
        }

        ~SparkTTS()
        {
            Dispose(false);
        }
    }
} 