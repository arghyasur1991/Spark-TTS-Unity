using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SparkTTS.Models
{
    using Core;
    using Utils;
    
    /// <summary>
    /// Mel spectrogram model for generating mel spectrograms from raw audio waveforms.
    /// Inherits from ORTModel and follows the consistent LoadInput/Run pattern.
    /// </summary>
    internal class MelSpectrogramModel : ORTModel
    {
        private const string InputName = "raw_waveform_with_channel"; 
        private const string OutputName = "mel_spectrogram";

        public int OutputNumMelBands { get; private set; } = 0;
        public const int TargetNumMelBandsForSpeakerEncoder = 128;
        public static new DebugLogger Logger = new();

        /// <summary>
        /// Initializes a new instance of the MelSpectrogramModel class.
        /// </summary>
        /// <param name="logLevel">The logging level for this model instance</param>
        public MelSpectrogramModel(DebugLogger.LogLevel logLevel = DebugLogger.LogLevel.Warning) 
            : base(SparkTTSModelPaths.MelSpectrogramModelName, 
                   SparkTTSModelPaths.MelSpectrogramFolder, 
                   logLevel)
        {
            Logger.Log("[MelSpectrogramModel] Initialized successfully");
            
            // Initialize the output mel bands determination asynchronously
            _ = InitializeOutputMelBandsAsync();
        }

        /// <summary>
        /// Asynchronously determines the number of mel bands from the model output metadata.
        /// </summary>
        private async Task InitializeOutputMelBandsAsync()
        {
            try
            {
                var outputs = await GetPreallocatedOutputs();
                var firstOutput = outputs.FirstOrDefault();
                if (firstOutput?.Value is DenseTensor<float> tensor)
                {
                    var dimensions = tensor.Dimensions.ToArray();
                    if (dimensions.Length == 3)
                    {
                        OutputNumMelBands = dimensions[1];
                        Logger.Log($"[MelSpectrogramModel] Determined OutputNumMelBands: {OutputNumMelBands}");
                    }
                    else
                    {
                        Logger.LogWarning($"[MelSpectrogramModel] Unexpected output dimensions: {dimensions.Length}. Expected 3D tensor.");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"[MelSpectrogramModel] Failed to determine mel bands: {ex.Message}");
            }
        }

        /// <summary>
        /// Asynchronously generates a mel spectrogram from raw mono audio samples.
        /// Uses the consistent LoadInput/Run pattern for professional model execution.
        /// </summary>
        /// <param name="monoAudioSamples">The mono audio samples to process</param>
        /// <returns>A task containing a tuple with (melData, melShape) or null on error</returns>
        /// <exception cref="ArgumentNullException">Thrown when input audio samples are null</exception>
        /// <exception cref="ArgumentException">Thrown when input audio samples are empty</exception>
        /// <exception cref="InvalidOperationException">Thrown when model execution fails</exception>
        public async Task<(float[] melData, int[] melShape)?> GenerateMelSpectrogramAsync(float[] monoAudioSamples)
        {
            if (monoAudioSamples == null)
                throw new ArgumentNullException(nameof(monoAudioSamples));
            if (monoAudioSamples.Length == 0)
                throw new ArgumentException("Input audio samples cannot be empty", nameof(monoAudioSamples));

            // Expected input shape: (batch, channels, samples) = (1, 1, length)
            var inputShape = new int[] { 1, 1, monoAudioSamples.Length };
            var inputTensor = new DenseTensor<float>(monoAudioSamples, inputShape);
            var inputs = new List<Tensor<float>> { inputTensor };

            try
            {
                // Use the new LoadInput/Run pattern
                var outputs = await Run(inputs);
                
                // Get the first output (mel spectrogram)
                var outputValue = outputs.FirstOrDefault();
                if (outputValue == null)
                {
                    throw new InvalidOperationException("No outputs received from mel spectrogram model");
                }
                
                if (!(outputValue.Value is DenseTensor<float> outputTensor))
                {
                    throw new InvalidOperationException($"Unexpected output type: {outputValue.Value?.GetType().FullName}. Expected DenseTensor<float>");
                }
                
                var melData = outputTensor.Buffer.ToArray();
                var melShape = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();
                
                // Validate output shape
                if (melShape.Length != 3)
                {
                    throw new InvalidOperationException($"Unexpected output tensor rank: {melShape.Length}. Expected 3. Shape: ({string.Join(",", melShape)})");
                }
                
                // Check consistency with determined mel bands
                if (melShape[1] != OutputNumMelBands && OutputNumMelBands != 0)
                {
                    Logger.LogWarning($"[MelSpectrogramModel] Model outputted {melShape[1]} bands, but determined OutputNumMelBands was {OutputNumMelBands}");
                }
                
                Logger.Log($"[MelSpectrogramModel] Successfully generated mel spectrogram with shape: [{string.Join(",", melShape)}]");
                
                return (melData, melShape);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Mel spectrogram generation failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Processes raw mel spectrogram data for speaker encoder compatibility.
        /// Performs permutation and band selection to match speaker encoder requirements.
        /// </summary>
        /// <param name="rawMelTuple">The raw mel spectrogram data and shape tuple</param>
        /// <returns>A tuple containing the processed mel data and shape, or null on error</returns>
        /// <exception cref="ArgumentNullException">Thrown when input tuple is null or incomplete</exception>
        /// <exception cref="ArgumentException">Thrown when mel shape is invalid</exception>
        /// <exception cref="InvalidOperationException">Thrown when processing fails</exception>
        public (float[] processedMelData, int[] processedMelShape)? ProcessMelForSpeakerEncoder(
            (float[] rawMelData, int[] rawMelShape) rawMelTuple)
        {
            if (rawMelTuple.rawMelData == null || rawMelTuple.rawMelShape == null)
                throw new ArgumentNullException(nameof(rawMelTuple), "Input rawMelTuple is null or incomplete");

            var rawMelData = rawMelTuple.rawMelData;
            var rawMelShape = rawMelTuple.rawMelShape;

            if (rawMelShape.Length != 3 || rawMelShape[0] != 1)
                throw new ArgumentException($"Unexpected mel shape. Expected (1, MelBands, NumFrames), got ({string.Join(",", rawMelShape)})", nameof(rawMelTuple));

            var modelOutputNumBandsActual = rawMelShape[1];
            var numFrames = rawMelShape[2];

            if (TargetNumMelBandsForSpeakerEncoder > modelOutputNumBandsActual)
            {
                throw new InvalidOperationException($"TargetNumMelBandsForSpeakerEncoder ({TargetNumMelBandsForSpeakerEncoder}) " +
                    $"is greater than actual model output bands ({modelOutputNumBandsActual})");
            }
            
            var bandsToProcess = TargetNumMelBandsForSpeakerEncoder;

            try
            {
                var permutedMelData = new float[1 * numFrames * bandsToProcess];
                var writeIdx = 0;

                for (var frameTarget = 0; frameTarget < numFrames; frameTarget++) 
                {
                    for (var bandTarget = 0; bandTarget < bandsToProcess; bandTarget++) 
                    {
                        var sourceBandIdx = bandTarget; 
                        var sourceFrameIdx = frameTarget;
                        var sourceFlatIndex = sourceFrameIdx + numFrames * sourceBandIdx;
                        permutedMelData[writeIdx++] = rawMelData[sourceFlatIndex];
                    }
                }
                
                var processedShape = new int[] { 1, numFrames, bandsToProcess };
                
                Logger.Log($"[MelSpectrogramModel] Processed mel spectrogram: {rawMelShape[1]} -> {bandsToProcess} bands, " +
                          $"{numFrames} frames");
                
                return (permutedMelData, processedShape);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Mel spectrogram processing failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Synchronous wrapper for generating mel spectrogram from raw mono audio samples.
        /// Uses the asynchronous implementation internally.
        /// </summary>
        /// <param name="monoAudioSamples">The mono audio samples to process</param>
        /// <returns>A tuple containing (melData, melShape) or null on error</returns>
        [Obsolete("Use GenerateMelSpectrogramAsync for better performance. This synchronous method will be removed in future versions.")]
        public (float[] melData, int[] melShape)? GenerateMelSpectrogram(float[] monoAudioSamples)
        {
            return GenerateMelSpectrogramAsync(monoAudioSamples).GetAwaiter().GetResult();
        }
    }
} 