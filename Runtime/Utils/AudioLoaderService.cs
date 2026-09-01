using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace SparkTTS.Utils
{
    public class AudioLoaderService
    {
        public struct ProcessedAudio
        {
            public float[] Samples; // Mono, resampled, normalized
            public int SampleRate;
        }

        public struct AudioSegments
        {
            public ProcessedAudio FullAudio;
            public ProcessedAudio ReferenceClip; // For speaker encoder
        }

        private static readonly Dictionary<string, AudioClip> s_silenceCache = new();

        // TODO: Make targetSampleRate, refSegmentDuration, latentHopLength configurable
        private const int DEFAULT_TARGET_SAMPLE_RATE = 16000;
        private const float DEFAULT_REF_SEGMENT_DURATION_SECONDS = 3.0f; 
        // private const int DEFAULT_LATENT_HOP_LENGTH = 256; // Example, check config
        
        /// <summary>
        /// Converts a WAV byte array to a Unity AudioClip
        /// </summary>
        /// <param name="wavData">WAV file data</param>
        /// <param name="sampleRate">Sample rate of the audio</param>
        /// <returns>AudioClip generated from the WAV data</returns>
        /// <summary>
        /// Reads a PCM WAV into an AudioClip, honouring the rate, channel count
        /// and bit depth declared in its header.
        /// </summary>
        /// <param name="wavData">WAV file bytes.</param>
        /// <param name="sampleRate">Fallback rate, used only if the header cannot be parsed.</param>
        public static AudioClip WAVToAudioClip(byte[] wavData, int sampleRate = 16000)
        {
            try
            {
                if (!TryParseWav(wavData, out float[] audioData, out int rate, out int channels))
                {
                    // Last resort: assume the canonical 44-byte 16-bit mono header.
                    Debug.LogWarning("[AudioLoaderService] Unparsable WAV header; assuming 16-bit mono at " +
                                     sampleRate + " Hz.");
                    const int headerSize = 44;
                    int fallbackCount = Math.Max(0, (wavData.Length - headerSize) / 2);
                    audioData = new float[fallbackCount];
                    for (int i = 0; i < fallbackCount; i++)
                    {
                        short s = (short)((wavData[headerSize + i * 2 + 1] << 8) | wavData[headerSize + i * 2]);
                        audioData[i] = s / 32768.0f;
                    }
                    rate = sampleRate;
                    channels = 1;
                }

                if (audioData.Length == 0)
                {
                    Debug.LogError("[AudioLoaderService] WAV contained no samples.");
                    return null;
                }

                int frames = audioData.Length / Math.Max(1, channels);
                AudioClip audioClip = AudioClip.Create("Generated Audio", frames, channels, rate, false);
                audioClip.SetData(audioData, 0);
                return audioClip;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error converting WAV to AudioClip: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Minimal RIFF chunk walk. Handles PCM (format 1) at 8/16/24/32-bit and
        /// IEEE float (format 3), any channel count. Returns interleaved samples.
        /// </summary>
        static bool TryParseWav(byte[] d, out float[] samples, out int sampleRate, out int channels)
        {
            samples = Array.Empty<float>();
            sampleRate = 0;
            channels = 0;

            if (d == null || d.Length < 44)
                return false;
            if (d[0] != 'R' || d[1] != 'I' || d[2] != 'F' || d[3] != 'F')
                return false;
            if (d[8] != 'W' || d[9] != 'A' || d[10] != 'V' || d[11] != 'E')
                return false;

            int format = 0;
            int bits = 0;
            int dataOffset = -1;
            int dataLength = 0;

            int pos = 12;
            while (pos + 8 <= d.Length)
            {
                string id = "" + (char)d[pos] + (char)d[pos + 1] + (char)d[pos + 2] + (char)d[pos + 3];
                int size = BitConverter.ToInt32(d, pos + 4);
                int body = pos + 8;
                if (size < 0 || body + size > d.Length)
                    size = d.Length - body;

                if (id == "fmt ")
                {
                    format = BitConverter.ToInt16(d, body);
                    channels = BitConverter.ToInt16(d, body + 2);
                    sampleRate = BitConverter.ToInt32(d, body + 4);
                    bits = BitConverter.ToInt16(d, body + 14);
                }
                else if (id == "data")
                {
                    dataOffset = body;
                    dataLength = size;
                }

                pos = body + size + (size & 1); // chunks are word-aligned
            }

            if (dataOffset < 0 || channels <= 0 || sampleRate <= 0 || bits <= 0)
                return false;

            int bytesPerSample = bits / 8;
            int count = dataLength / bytesPerSample;
            var outBuf = new float[count];

            if (format == 3 && bits == 32)
            {
                for (int i = 0; i < count; i++)
                    outBuf[i] = BitConverter.ToSingle(d, dataOffset + i * 4);
            }
            else if (format == 1 || format == 0xFFFE)
            {
                switch (bits)
                {
                    case 8:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = (d[dataOffset + i] - 128) / 128f;
                        break;
                    case 16:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = BitConverter.ToInt16(d, dataOffset + i * 2) / 32768f;
                        break;
                    case 24:
                        for (int i = 0; i < count; i++)
                        {
                            int o = dataOffset + i * 3;
                            int v = (d[o] | (d[o + 1] << 8) | (sbyte)d[o + 2] << 16);
                            outBuf[i] = v / 8388608f;
                        }
                        break;
                    case 32:
                        for (int i = 0; i < count; i++)
                            outBuf[i] = BitConverter.ToInt32(d, dataOffset + i * 4) / 2147483648f;
                        break;
                    default:
                        return false;
                }
            }
            else
            {
                return false;
            }

            samples = outBuf;
            return true;
        }

        public static AudioClip CreateSilence(int sampleRate = 16000, float duration = 0.25f)
        {
            // Search cache for silence clip
            string cacheKey = $"{sampleRate}_{duration}";
            if (s_silenceCache.ContainsKey(cacheKey))
            {
                return s_silenceCache[cacheKey];
            }

            float[] silence = new float[(int)(duration * sampleRate)];
            AudioClip silenceClip = AudioClip.Create("Silence", silence.Length, 1, sampleRate, false);
            silenceClip.SetData(silence, 0);

            // Cache silence clip based on sample rate and duration
            s_silenceCache[cacheKey] = silenceClip;
            return silenceClip;
        }

        public static AudioClip ConcatenateAudioClips(List<AudioClip> clips, int sampleRate = 16000)
        {
            if (clips == null || clips.Count == 0)
                return null;
            
            // Create a new AudioClip with the combined duration
            float totalDuration = clips.Sum(clip => clip.length);
            Debug.Log($"C# AudioLoaderService: Total duration: {totalDuration}");
            float silenceDuration = 0.25f;
            Debug.Log($"C# AudioLoaderService: Silence duration: {silenceDuration}");
            // add 0.5 second of silences to total duration
            totalDuration += silenceDuration * (clips.Count - 1);
            Debug.Log($"C# AudioLoaderService: Total duration with silences: {totalDuration}");
            AudioClip combinedClip = AudioClip.Create("Combined Audio", (int)(totalDuration * sampleRate), 1, sampleRate, false);

            // Concatenate the clips
            int offset = 0;
            foreach (var clip in clips)
            {
                float[] samples = new float[clip.samples];
                clip.GetData(samples, 0);
                combinedClip.SetData(samples, offset);
                if (clip != clips.Last())
                {
                    // Add some silence of 0.5 second between clips
                    float[] silence = new float[(int)(silenceDuration * sampleRate)];
                    combinedClip.SetData(silence, offset + samples.Length);
                    offset += samples.Length + silence.Length;
                }
            }

            return combinedClip;
        }
        
        /// <summary>
        /// Loads an audio clip from a file path asynchronously
        /// </summary>
        public static async Task<AudioClip> LoadAudioClipAsync(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Debug.LogError($"Audio file not found: {filePath}");
                return null;
            }
            
            try
            {
                // Load the audio file
                byte[] fileData = await Task.Run(() => File.ReadAllBytes(filePath));
                
                // Create an AudioClip
                AudioClip clip = WAVToAudioClip(fileData);
                return clip;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error loading audio clip: {ex.Message}");
                return null;
            }
        }

        public static async Task SaveAudioClipToFile(AudioClip audioClip, string filePath)
        {
            float[] samples = new float[audioClip.samples];
            audioClip.GetData(samples, 0);
            // write audio clip to file
            byte[] wavData = ConvertAudioClipToWAV(samples, audioClip.frequency);
            await File.WriteAllBytesAsync(filePath, wavData);
        }

        /// <summary>
        /// Converts audio clip data to WAV format
        /// </summary>
        /// <param name="samples">Audio samples</param>
        /// <param name="frequency">Sample rate</param>
        /// <returns>WAV data</returns>
        public static byte[] ConvertAudioClipToWAV(float[] samples, int frequency)
        {
            try
            {
                using (MemoryStream stream = new())
                {
                    using (BinaryWriter writer = new(stream))
                    {
                        // WAV header
                        writer.Write(new char[] { 'R', 'I', 'F', 'F' });
                        writer.Write(36 + samples.Length * 2); 
                        writer.Write(new char[] { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
                        writer.Write(16); // Subchunk1Size
                        writer.Write((short)1); // AudioFormat (PCM)
                        writer.Write((short)1); // NumChannels (Mono)
                        writer.Write(frequency); // SampleRate
                        writer.Write(frequency * 2); // ByteRate
                        writer.Write((short)2); // BlockAlign
                        writer.Write((short)16); // BitsPerSample
                        writer.Write(new char[] { 'd', 'a', 't', 'a' });
                        writer.Write(samples.Length * 2); // Subchunk2Size
                        
                        // Audio data. Round rather than truncate, and clamp:
                        // a clone reference makes several of these round trips
                        // before it reaches the 12 Hz tokenizer, and the
                        // quantizer is sensitive enough that truncation bias
                        // shifts reference codes.
                        foreach (float sample in samples)
                        {
                            float scaled = Mathf.Clamp(sample, -1f, 1f) * 32767f;
                            writer.Write((short)Mathf.RoundToInt(scaled));
                        }
                    }
                    
                    return stream.ToArray();
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error converting AudioClip to WAV: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Loads, resamples, and normalizes an audio clip.
        /// Also extracts a reference segment for speaker encoding.
        /// </summary>
        /// <param name="audioClip">The AudioClip to process.</param>
        /// <param name="targetSampleRate">Target sample rate for all output audio.</param>
        /// <param name="refSegmentDurationSeconds">Duration of the reference clip in seconds.</param>
        /// <param name="volumeNormCoeff">Coefficient for audio_volume_normalize.</param>
        /// <returns>Processed audio segments or null on error.</returns>
        public AudioSegments? LoadAndProcessAudio(
            AudioClip audioClip, 
            int targetSampleRate = DEFAULT_TARGET_SAMPLE_RATE, 
            float refSegmentDurationSeconds = DEFAULT_REF_SEGMENT_DURATION_SECONDS,
            float volumeNormCoeff = 0.2f)
        {
            if (audioClip == null)
            {
                Debug.LogError("Input AudioClip is null.");
                return null;
            }

            float[] rawSamples = new float[audioClip.samples * audioClip.channels];
            audioClip.GetData(rawSamples, 0);

            // 1. Convert to Mono
            float[] monoSamples = ToMono(rawSamples, audioClip.channels);
            if (monoSamples == null) return null;

            // 2. Resample full audio
            float[] resampledFullAudio;
            if (audioClip.frequency == targetSampleRate)
            {
                Debug.Log($"C# AudioLoaderService: Full audio sample rate ({audioClip.frequency} Hz) matches target ({targetSampleRate} Hz). Skipping resampling.");
                resampledFullAudio = monoSamples;
            }
            else
            {
                Debug.Log($"C# AudioLoaderService: Resampling full audio from {audioClip.frequency} Hz to {targetSampleRate} Hz.");
                resampledFullAudio = Resample(monoSamples, audioClip.frequency, targetSampleRate);
            }
            if (resampledFullAudio == null) return null;
            
            // 3. Volume Normalize full audio
            float[] normalizedFullAudio = AudioVolumeNormalize(resampledFullAudio, volumeNormCoeff);
            // float[] normalizedFullAudio = resampledFullAudio; // Use resampled audio directly

            ProcessedAudio processedFullAudio = new ProcessedAudio
            {
                Samples = normalizedFullAudio,
                SampleRate = targetSampleRate
            };

            // 4. Extract Reference Clip (from original mono, then resample and normalize)
            // BiCodecTokenizer.get_ref_clip uses original SR audio then implies downstream components handle SR.
            // Here, we get ref clip from original mono, then resample and normalize it separately for consistency.
            
            // For get_ref_clip, we need original sample rate for duration calculation
            int refClipLengthInOriginalSr = (int)(audioClip.frequency * refSegmentDurationSeconds); 
            // The Python code uses latent_hop_length for precise ref_segment_length.
            // This detail is missing here for now without knowing latent_hop_length.
            // Using a simplified length based on duration.
            // TODO: Replicate get_ref_clip logic more precisely using latent_hop_length if available/needed.
            
            float[] refClipOriginalSr = ExtractSegment(monoSamples, 0, refClipLengthInOriginalSr, true); // Pad if shorter
            if(refClipOriginalSr == null) return null;

            float[] resampledRefClip;
            if (audioClip.frequency == targetSampleRate) 
            {
                // If original was targetRate, refClipOriginalSr is already at targetRate (extracted from monoSamples which would also be at targetRate if not resampled above)
                // However, refClipOriginalSr was extracted based on original SR duration. If we skipped full audio resampling because it was already targetRate,
                // then monoSamples are at targetRate. refClipOriginalSr is fine.
                Debug.Log($"C# AudioLoaderService: Reference clip effective sample rate ({audioClip.frequency} Hz) matches target ({targetSampleRate} Hz). Skipping resampling for ref clip.");
                resampledRefClip = refClipOriginalSr;
            }
            else
            {
                Debug.Log($"C# AudioLoaderService: Resampling reference clip from {audioClip.frequency} Hz to {targetSampleRate} Hz.");
                resampledRefClip = Resample(refClipOriginalSr, audioClip.frequency, targetSampleRate);
            }
            if (resampledRefClip == null) return null;

            float[] normalizedRefClip = AudioVolumeNormalize(resampledRefClip, volumeNormCoeff);
            // float[] normalizedRefClip = resampledRefClip; // Use resampled audio directly

            ProcessedAudio processedRefClip = new ProcessedAudio
            {
                Samples = normalizedRefClip,
                SampleRate = targetSampleRate
            };

            return new AudioSegments
            {
                FullAudio = processedFullAudio,
                ReferenceClip = processedRefClip
            };
        }

        public float[] ToMono(float[] interleavedSamples, int channels)
        {
            if (channels == 1)
            {
                return interleavedSamples; // Already mono
            }
            if (channels == 0) {
                Debug.LogError("Audio clip has 0 channels.");
                return null;
            }

            int numSamples = interleavedSamples.Length / channels;
            float[] monoSamples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                float sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    sum += interleavedSamples[i * channels + c];
                }
                monoSamples[i] = sum / channels;
            }
            return monoSamples;
        }

        /// <summary>
        /// Basic linear interpolation resampler.
        /// TODO: Consider a higher quality resampling algorithm for parity with soxr VHQ.
        /// </summary>
        public float[] Resample(float[] samples, int fromRate, int toRate)
        {
            if (fromRate == toRate)
            {
                return samples;
            }

            if (samples == null || samples.Length == 0) return new float[0];
            
            double ratio = (double)toRate / fromRate;
            int newLength = (int)(samples.Length * ratio);
            float[] resampled = new float[newLength];

            for (int i = 0; i < newLength; i++)
            {
                double originalIndex = (double)i / ratio;
                int index1 = (int)Math.Floor(originalIndex);
                int index2 = index1 + 1;

                double fraction = originalIndex - index1;

                float val1 = (index1 >= 0 && index1 < samples.Length) ? samples[index1] : 0;
                float val2 = (index2 >= 0 && index2 < samples.Length) ? samples[index2] : 0;

                resampled[i] = (float)(val1 * (1.0 - fraction) + val2 * fraction);
            }
            return resampled;
        }
        
        /// <summary>
        /// Extracts a segment from the audio. If segment is beyond bounds, it pads or truncates.
        /// </summary>
        /// <param name="samples">Input audio samples.</param>
        /// <param name="startIndex">Start index of the segment.</param>
        /// <param name="length">Desired length of the segment.</param>
        /// <param name="padIfShorter">If true, pads with zeros if audio is shorter than length. Otherwise truncates/returns what's available.</param>
        /// <returns>The extracted (and possibly padded/truncated) segment.</returns>
        public float[] ExtractSegment(float[] samples, int startIndex, int length, bool padIfShorter)
        {
            if (samples == null) return null;
            if (length <= 0) return new float[0];
            if (startIndex < 0) startIndex = 0;

            int availableLength = Math.Max(0, samples.Length - startIndex);
            int segmentLengthToCopy = Math.Min(length, availableLength);
            
            float[] segment = new float[length]; // Initialize to zeros (default padding)

            if (segmentLengthToCopy > 0)
            {
                Array.Copy(samples, startIndex, segment, 0, segmentLengthToCopy);
            }
            
            if (!padIfShorter && availableLength < length) // Truncate if not padding and source is shorter
            {
                float[] truncatedSegment = new float[availableLength];
                Array.Copy(segment, 0, truncatedSegment, 0, availableLength);
                return truncatedSegment;
            }
            
            // If padIfShorter is true, 'segment' is already the correct length and padded with zeros
            // if segmentLengthToCopy < length.
            // If padIfShorter is false and availableLength >= length, we copied 'length' samples.
            return segment;
        }


        /// <summary>
        /// Normalizes the volume of an audio signal.
        /// Ported from sparktts/utils/audio.py:audio_volume_normalize
        /// </summary>
        /// <param name="audio">Input audio signal array.</param>
        /// <param name="coeff">Target coefficient for normalization, default is 0.2f.</param>
        /// <returns>The volume-normalized audio signal.</returns>
        public float[] AudioVolumeNormalize(float[] audio, float coeff = 0.2f)
        {
            if (audio == null || audio.Length == 0)
            {
                return audio;
            }

            float[] temp = audio.Select(x => Math.Abs(x)).ToArray();
            Array.Sort(temp);

            if (temp.Length == 0 || temp[temp.Length - 1] < 0.1f)
            {
                float scalingFactor = temp.Length > 0 ? Math.Max(temp[temp.Length - 1], 1e-3f) : 1e-3f;
                if (scalingFactor == 0) scalingFactor = 1e-3f; // Avoid division by zero
                
                float[] scaledAudio = new float[audio.Length];
                for (int i = 0; i < audio.Length; i++)
                {
                    scaledAudio[i] = audio[i] / scalingFactor * 0.1f;
                }
                audio = scaledAudio; // Update audio with the scaled version
                
                // Re-calculate temp for subsequent logic if audio was scaled
                temp = audio.Select(x => Math.Abs(x)).ToArray();
                Array.Sort(temp);
            }

            // Filter out values less than 0.01f from temp
            List<float> filteredTempList = new List<float>();
            foreach (float val in temp)
            {
                if (val > 0.01f)
                {
                    filteredTempList.Add(val);
                }
            }
            float[] filteredTemp = filteredTempList.ToArray();
            int L = filteredTemp.Length;

            if (L <= 10)
            {
                return audio;
            }

            // Compute the average of the top 10% to 1% of values in temp
            // Python: np.mean(temp[int(0.9 * L) : int(0.99 * L)])
            int startIndex = (int)(0.9 * L);
            int endIndex = (int)(0.99 * L);
            if (startIndex >= endIndex) // Ensure there's a valid range, at least one element
            {
                // If range is too small or invalid, use a fallback like the top few elements or just the max.
                // For simplicity, if the intended slice is empty or invalid,
                // let's take a small portion from the top, e.g. last 1% or at least 1 element.
                startIndex = Math.Max(0, L - Math.Max(1, (int)(0.01 * L)));
                endIndex = L;
                if (startIndex >= endIndex && L > 0) startIndex = L -1; // Ensure at least one element if L > 0
            }
            
            float volume = 0;
            int count = 0;
            if (L > 0) // Check if filteredTemp has elements
            {
                for (int i = startIndex; i < endIndex; i++)
                {
                    volume += filteredTemp[i];
                    count++;
                }
                if (count > 0)
                {
                    volume /= count;
                }
                else if (L > 0) // Fallback if count is 0 but L > 0 (e.g. startIndex == endIndex after adjustment)
                {
                    volume = filteredTemp[L-1]; // Use the loudest significant value
                }
                else // L is 0, filteredTemp is empty
                {
                    volume = coeff; // Fallback to avoid division by zero if no significant values
                }
            } else {
                volume = coeff; // If L is 0, filteredTemp is empty. Avoid division by zero.
            }


            if (volume == 0) volume = 1e-5f; // Avoid division by zero if volume calculation resulted in zero

            float[] normalizedAudio = new float[audio.Length];
            float scale = Mathf.Clamp(coeff / volume, 0.1f, 10.0f);

            for (int i = 0; i < audio.Length; i++)
            {
                normalizedAudio[i] = audio[i] * scale;
            }

            // Ensure the maximum absolute value in the audio does not exceed 1
            float maxValue = 0;
            for (int i = 0; i < normalizedAudio.Length; i++)
            {
                if (Math.Abs(normalizedAudio[i]) > maxValue)
                {
                    maxValue = Math.Abs(normalizedAudio[i]);
                }
            }

            if (maxValue > 1.0f)
            {
                for (int i = 0; i < normalizedAudio.Length; i++)
                {
                    normalizedAudio[i] /= maxValue;
                }
            }
            return normalizedAudio;
        }
    } 
}