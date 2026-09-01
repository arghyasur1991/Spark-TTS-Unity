// Ported from ElBruno.QwenTTS VoiceCloning (MIT).
// Mel for the Qwen3-TTS Base speaker encoder: 24 kHz, 128 bins, hop 256.
//
// Filterbank is librosa.filters.mel defaults — Slaney mel scale, Slaney
// normalisation — because qwen_tts `mel_spectrogram` calls `librosa_mel_fn`
// without `htk=True`. An HTK filterbank here produces a plausible-looking
// mel and a wrong x-vector, so the clone drifts off the reference speaker.

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

        /// <summary>
        /// <c>librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)</c> with its defaults
        /// (<c>htk=False</c>, <c>norm="slaney"</c>), which is what qwen_tts uses.
        /// </summary>
        private static float[,] BuildMelFilterBank(int nMels, int nFreqs, int sampleRate, float fMin, float fMax)
        {
            var filters = new float[nMels, nFreqs];

            // librosa.fft_frequencies: linspace(0, sr/2, 1 + n_fft/2)
            var fftFreqs = new double[nFreqs];
            double fftFreqStep = (double)sampleRate / (2.0 * (nFreqs - 1));
            for (int k = 0; k < nFreqs; k++)
                fftFreqs[k] = k * fftFreqStep;

            // librosa.mel_frequencies: evenly spaced in the Slaney mel scale
            double melMin = HzToMel(fMin);
            double melMax = HzToMel(fMax);
            var hzPoints = new double[nMels + 2];
            for (int i = 0; i < nMels + 2; i++)
                hzPoints[i] = MelToHz(melMin + (melMax - melMin) * i / (nMels + 1));

            for (int m = 0; m < nMels; m++)
            {
                double lowerHz = hzPoints[m];
                double centerHz = hzPoints[m + 1];
                double upperHz = hzPoints[m + 2];
                double lowerSpan = centerHz - lowerHz;
                double upperSpan = upperHz - centerHz;
                // Slaney norm: unit area per filter rather than unit peak.
                double enorm = 2.0 / (upperHz - lowerHz);

                for (int k = 0; k < nFreqs; k++)
                {
                    double f = fftFreqs[k];
                    double rise = lowerSpan > 0 ? (f - lowerHz) / lowerSpan : 0.0;
                    double fall = upperSpan > 0 ? (upperHz - f) / upperSpan : 0.0;
                    double w = Math.Min(rise, fall);
                    if (w > 0)
                        filters[m, k] = (float)(enorm * w);
                }
            }

            return filters;
        }

        // Slaney mel: linear at 200/3 Hz per mel below 1 kHz, log above.
        private const double SlaneyFSp = 200.0 / 3.0;
        private const double SlaneyMinLogHz = 1000.0;
        private static readonly double SlaneyMinLogMel = SlaneyMinLogHz / SlaneyFSp;
        private static readonly double SlaneyLogStep = Math.Log(6.4) / 27.0;

        private static double HzToMel(double hz)
        {
            if (hz >= SlaneyMinLogHz)
                return SlaneyMinLogMel + Math.Log(hz / SlaneyMinLogHz) / SlaneyLogStep;
            return hz / SlaneyFSp;
        }

        private static double MelToHz(double mel)
        {
            if (mel >= SlaneyMinLogMel)
                return SlaneyMinLogHz * Math.Exp(SlaneyLogStep * (mel - SlaneyMinLogMel));
            return mel * SlaneyFSp;
        }

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
