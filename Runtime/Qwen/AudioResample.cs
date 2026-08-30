using System;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Linear resampler. Qwen vocoder is 24 kHz; CharacterVoice.GenerateSpeechAsync defaults to 16 kHz.
    /// </summary>
    internal static class AudioResample
    {
        public static float[] Resample(float[] input, int srcRate, int dstRate)
        {
            if (input == null || input.Length == 0)
                return Array.Empty<float>();
            if (srcRate <= 0 || dstRate <= 0)
                throw new ArgumentOutOfRangeException("Sample rates must be positive.");
            if (srcRate == dstRate)
                return input;

            double ratio = (double)srcRate / dstRate;
            int outLen = Math.Max(1, (int)Math.Round(input.Length / ratio));
            var output = new float[outLen];
            int last = input.Length - 1;
            for (int i = 0; i < outLen; i++)
            {
                double srcPos = i * ratio;
                int i0 = (int)srcPos;
                if (i0 >= last)
                {
                    output[i] = input[last];
                    continue;
                }
                float t = (float)(srcPos - i0);
                output[i] = input[i0] * (1f - t) + input[i0 + 1] * t;
            }
            return output;
        }
    }
}
