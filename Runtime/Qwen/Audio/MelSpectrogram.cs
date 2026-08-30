// Ported from ElBruno.QwenTTS VoiceCloning (MIT).
// Mel for the Qwen3-TTS Base speaker encoder: 24 kHz, 128 bins, hop 256.

using System;

namespace SparkTTS.Qwen.Audio
{
    internal static class MelSpectrogram
    {
        public const int DefaultSampleRate = 24000;
        public const int DefaultNFft = 1024;
        public const int DefaultHopLength = 256;
        public const int DefaultNMels = 128;
        public const float FMin = 0f;
        public const float FMax = 12000f;

        /// <summary>
        /// Mel-spectrogram as float[T_mel, n_mels] (time-first). Feed speaker_encoder as (1, T, 128).
        /// </summary>
        public static float[,] Extract(float[] samples, int sampleRate = DefaultSampleRate,
            int nFft = DefaultNFft, int hopLength = DefaultHopLength, int nMels = DefaultNMels)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
            if (samples.Length == 0)
                throw new ArgumentException("Audio samples cannot be empty.", nameof(samples));

            int nFreqs = nFft / 2 + 1;
            var window = BuildHannWindow(nFft);
            var melFilters = BuildMelFilterBank(nMels, nFreqs, sampleRate, FMin, FMax);

            int padding = (nFft - hopLength) / 2;
            var padded = ReflectPad(samples, padding, padding);

            int numFrames = 1 + (padded.Length - nFft) / hopLength;
            if (numFrames <= 0)
                numFrames = 1;

            var melSpec = new float[numFrames, nMels];
            var fftReal = new double[nFft];
            var fftImag = new double[nFft];

            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * hopLength;
                for (int i = 0; i < nFft; i++)
                {
                    int idx = start + i;
                    float sample = idx < padded.Length ? padded[idx] : 0f;
                    fftReal[i] = sample * window[i];
                    fftImag[i] = 0;
                }

                Fft(fftReal, fftImag, nFft);

                for (int m = 0; m < nMels; m++)
                {
                    double melEnergy = 0;
                    for (int k = 0; k < nFreqs; k++)
                    {
                        if (melFilters[m, k] > 0)
                        {
                            double real = fftReal[k];
                            double imag = fftImag[k];
                            double magnitude = Math.Sqrt(real * real + imag * imag + 1e-9);
                            melEnergy += melFilters[m, k] * magnitude;
                        }
                    }

                    melSpec[frame, m] = (float)Math.Log(Math.Max(melEnergy, 1e-5));
                }
            }

            return melSpec;
        }

        public static float[] FlattenTimeFirst(float[,] mel)
        {
            int t = mel.GetLength(0);
            int mels = mel.GetLength(1);
            var flat = new float[t * mels];
            int n = 0;
            for (int i = 0; i < t; i++)
            {
                for (int j = 0; j < mels; j++)
                    flat[n++] = mel[i, j];
            }
            return flat;
        }

        private static float[] BuildHannWindow(int size)
        {
            var window = new float[size];
            for (int i = 0; i < size; i++)
                window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / size));
            return window;
        }

        private static float[] ReflectPad(float[] input, int padLeft, int padRight)
        {
            int len = input.Length;
            if (len < 2)
            {
                var zeroPadded = new float[padLeft + len + padRight];
                Array.Copy(input, 0, zeroPadded, padLeft, len);
                return zeroPadded;
            }

            var output = new float[padLeft + len + padRight];
            Array.Copy(input, 0, output, padLeft, len);

            for (int i = 0; i < padLeft; i++)
                output[padLeft - 1 - i] = input[ReflectIndex(i + 1, len)];
            for (int i = 0; i < padRight; i++)
                output[padLeft + len + i] = input[ReflectIndex(len - 2 - i, len)];
            return output;
        }

        private static int ReflectIndex(int idx, int length)
        {
            if (idx < 0)
                idx = -idx;
            int period = 2 * (length - 1);
            if (period == 0)
                return 0;
            idx = idx % period;
            if (idx >= length)
                idx = period - idx;
            return idx;
        }

        private static float[,] BuildMelFilterBank(int nMels, int nFreqs, int sampleRate, float fMin, float fMax)
        {
            var filters = new float[nMels, nFreqs];
            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);

            var melPoints = new double[nMels + 2];
            for (int i = 0; i < nMels + 2; i++)
                melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);

            var hzPoints = new double[nMels + 2];
            var binIndices = new double[nMels + 2];
            double fftFreqStep = (double)sampleRate / (2.0 * (nFreqs - 1));
            for (int i = 0; i < nMels + 2; i++)
            {
                hzPoints[i] = MelToHz(melPoints[i]);
                binIndices[i] = hzPoints[i] / fftFreqStep;
            }

            for (int m = 0; m < nMels; m++)
            {
                double left = binIndices[m];
                double center = binIndices[m + 1];
                double right = binIndices[m + 2];
                double enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]);

                for (int k = 0; k < nFreqs; k++)
                {
                    if (k >= left && k <= center && center > left)
                        filters[m, k] = (float)(enorm * (k - left) / (center - left));
                    else if (k > center && k <= right && right > center)
                        filters[m, k] = (float)(enorm * (right - k) / (right - center));
                }
            }

            return filters;
        }

        private static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);

        private static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        private static void Fft(double[] real, double[] imag, int n)
        {
            int bits = CountBits(n);
            for (int i = 0; i < n; i++)
            {
                int j = BitReverse(i, bits);
                if (j > i)
                {
                    double tr = real[i];
                    real[i] = real[j];
                    real[j] = tr;
                    double ti = imag[i];
                    imag[i] = imag[j];
                    imag[j] = ti;
                }
            }

            for (int size = 2; size <= n; size *= 2)
            {
                int half = size / 2;
                double angle = -2.0 * Math.PI / size;
                for (int i = 0; i < n; i += size)
                {
                    for (int k = 0; k < half; k++)
                    {
                        double cos = Math.Cos(angle * k);
                        double sin = Math.Sin(angle * k);
                        double tReal = cos * real[i + k + half] - sin * imag[i + k + half];
                        double tImag = sin * real[i + k + half] + cos * imag[i + k + half];
                        real[i + k + half] = real[i + k] - tReal;
                        imag[i + k + half] = imag[i + k] - tImag;
                        real[i + k] += tReal;
                        imag[i + k] += tImag;
                    }
                }
            }
        }

        private static int CountBits(int n)
        {
            int bits = 0;
            int v = n;
            while (v > 1)
            {
                v >>= 1;
                bits++;
            }
            return bits;
        }

        private static int BitReverse(int x, int bits)
        {
            int result = 0;
            for (int i = 0; i < bits; i++)
            {
                result = (result << 1) | (x & 1);
                x >>= 1;
            }
            return result;
        }
    }
}
