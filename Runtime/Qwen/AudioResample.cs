using System;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Band-limited resampler (windowed sinc). The Qwen vocoder is 24 kHz and
    /// <see cref="SparkTTS.CharacterVoice.GenerateSpeechAsync"/> hands callers
    /// whatever rate they ask for.
    ///
    /// Linear interpolation is not good enough here: downsampling without a
    /// low-pass folds 8-12 kHz back into the speech band, and upsampling a
    /// clone reference leaves a stair-stepped spectrum. Either one moves the
    /// speaker-encoder mel and the 12 Hz codes away from the take being cloned.
    /// </summary>
    internal static class AudioResample
    {
        /// <summary>Sinc lobes kept either side of the sample being written.</summary>
        const int Lobes = 16;

        public static float[] Resample(float[] input, int srcRate, int dstRate)
        {
            if (input == null || input.Length == 0)
                return Array.Empty<float>();
            if (srcRate <= 0 || dstRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcRate), "Sample rates must be positive.");
            if (srcRate == dstRate)
                return input;

            // Source samples advanced per output sample.
            double step = (double)srcRate / dstRate;
            int outLen = Math.Max(1, (int)Math.Round(input.Length / step));

            // Keep everything below the lower of the two Nyquists, as a
            // fraction of the source Nyquist.
            double cutoff = step > 1.0 ? 1.0 / step : 1.0;
            double halfWidth = Lobes / cutoff;
            int taps = (int)Math.Ceiling(halfWidth);

            var output = new float[outLen];
            int last = input.Length - 1;

            for (int i = 0; i < outLen; i++)
            {
                double center = i * step;
                int first = (int)Math.Floor(center) - taps + 1;
                double sum = 0;
                double norm = 0;

                for (int j = first; j <= first + 2 * taps - 1; j++)
                {
                    double x = center - j;
                    if (x <= -halfWidth || x >= halfWidth)
                        continue;

                    double h = cutoff * Sinc(cutoff * x) * Blackman(x / halfWidth);
                    // Out-of-range taps still count toward norm so the gain
                    // stays flat and the edges fade instead of stepping.
                    norm += h;
                    if (j >= 0 && j <= last)
                        sum += h * input[j];
                }

                output[i] = norm > 1e-9 ? (float)(sum / norm) : 0f;
            }

            return output;
        }

        static double Sinc(double x)
        {
            if (Math.Abs(x) < 1e-9)
                return 1.0;
            double px = Math.PI * x;
            return Math.Sin(px) / px;
        }

        /// <summary>Blackman window over t in [-1, 1].</summary>
        static double Blackman(double t)
        {
            double a = Math.PI * (t + 1.0);
            return 0.42 - 0.5 * Math.Cos(a) + 0.08 * Math.Cos(2.0 * a);
        }
    }
}
