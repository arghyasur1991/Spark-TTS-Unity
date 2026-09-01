// Token sampling shared by CustomVoice and Base talkers.
// Heap top-k + reused buffers follow Spark LLMModel; ThreadLocal RNG avoids per-step Random().

using System;
using System.Collections.Generic;
using System.Threading;

namespace SparkTTS.Qwen
{
    internal sealed class QwenTokenSampler
    {
        private static readonly ThreadLocal<Random> Rng =
            new(() => new Random(Guid.NewGuid().GetHashCode()));

        private float[] _work;
        private float[] _sortedLogits;
        private int[] _sortedIndices;
        private (float logit, int index)[] _heap;
        private float[] _probs;

        /// <param name="logitsLength">
        /// Valid elements in <paramref name="logits"/>. Pass this when the array
        /// is a reused buffer that is larger than the tensor it holds — the last
        /// <c>vocabSize</c> entries are the ones sampled from.
        /// </param>
        public int Sample(
            float[] logits,
            int vocabSize,
            float temperature,
            int topK,
            float topP,
            List<int> history,
            float repetitionPenalty,
            int[] suppressTokens,
            int logitsLength = -1)
        {
            EnsureWork(vocabSize);
            int valid = logitsLength >= 0 ? Math.Min(logitsLength, logits.Length) : logits.Length;
            int src = Math.Max(0, valid - vocabSize);
            int n = Math.Min(vocabSize, valid - src);
            Array.Copy(logits, src, _work, 0, n);
            if (n < vocabSize)
                Array.Clear(_work, n, vocabSize - n);

            if (repetitionPenalty != 1f && history != null && history.Count > 0)
            {
                for (int i = 0; i < history.Count; i++)
                {
                    int tok = history[i];
                    if (tok < 0 || tok >= vocabSize)
                        continue;
                    if (_work[tok] >= 0)
                        _work[tok] /= repetitionPenalty;
                    else
                        _work[tok] *= repetitionPenalty;
                }
            }

            if (suppressTokens != null)
            {
                for (int i = 0; i < suppressTokens.Length; i++)
                {
                    int tok = suppressTokens[i];
                    if (tok >= 0 && tok < vocabSize)
                        _work[tok] = -1e9f;
                }
            }

            if (temperature <= 1e-6f)
                return ArgMax(_work, vocabSize);

            int k = topK > 0 ? Math.Min(topK, vocabSize) : vocabSize;
            int filtered = SelectTopK(_work, vocabSize, k);

            if (temperature > 0)
            {
                for (int i = 0; i < filtered; i++)
                    _sortedLogits[i] /= temperature;
            }

            EnsureProbs(filtered);
            Softmax(_sortedLogits, filtered, _probs);

            if (topP > 0f && topP < 1f && filtered > 1)
                filtered = Nucleus(_probs, _sortedIndices, filtered, topP);

            double sum = 0;
            for (int i = 0; i < filtered; i++)
                sum += _probs[i];
            if (sum <= 0 || double.IsNaN(sum))
                return _sortedIndices[0];

            double r = Rng.Value.NextDouble() * sum;
            double acc = 0;
            for (int i = 0; i < filtered; i++)
            {
                acc += _probs[i];
                if (r < acc)
                    return _sortedIndices[i];
            }

            return _sortedIndices[filtered - 1];
        }

        public static int[] SuppressUpperCodec(int vocabSize, int eosId)
        {
            int start = Math.Max(0, vocabSize - 1024);
            var list = new List<int>(1024);
            for (int i = start; i < vocabSize; i++)
            {
                if (i != eosId)
                    list.Add(i);
            }
            return list.ToArray();
        }

        private void EnsureWork(int vocab)
        {
            if (_work == null || _work.Length < vocab)
                _work = new float[vocab];
        }

        private void EnsureProbs(int n)
        {
            if (_probs == null || _probs.Length < n)
                _probs = new float[n];
        }

        private int SelectTopK(float[] logits, int vocab, int k)
        {
            if (_sortedLogits == null || _sortedLogits.Length < k)
            {
                _sortedLogits = new float[k];
                _sortedIndices = new int[k];
            }

            if (k >= vocab)
            {
                if (_sortedLogits.Length < vocab)
                {
                    _sortedLogits = new float[vocab];
                    _sortedIndices = new int[vocab];
                }
                for (int i = 0; i < vocab; i++)
                {
                    _sortedLogits[i] = logits[i];
                    _sortedIndices[i] = i;
                }
                return vocab;
            }

            int heapSize = k;
            if (_heap == null || _heap.Length < heapSize)
                _heap = new (float, int)[heapSize];

            for (int i = 0; i < k; i++)
                _heap[i] = (logits[i], i);
            HeapifyMin(k);

            for (int i = k; i < vocab; i++)
            {
                if (logits[i] > _heap[0].logit)
                {
                    _heap[0] = (logits[i], i);
                    SiftDown(0, k);
                }
            }

            Array.Sort(_heap, 0, k, Comparer<(float logit, int index)>.Create(
                (a, b) => b.logit.CompareTo(a.logit)));

            for (int i = 0; i < k; i++)
            {
                _sortedLogits[i] = _heap[i].logit;
                _sortedIndices[i] = _heap[i].index;
            }

            return k;
        }

        private void HeapifyMin(int length)
        {
            for (int i = length / 2 - 1; i >= 0; i--)
                SiftDown(i, length);
        }

        private void SiftDown(int i, int length)
        {
            while (true)
            {
                int smallest = i;
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                if (left < length && _heap[left].logit < _heap[smallest].logit)
                    smallest = left;
                if (right < length && _heap[right].logit < _heap[smallest].logit)
                    smallest = right;
                if (smallest == i)
                    return;
                var tmp = _heap[i];
                _heap[i] = _heap[smallest];
                _heap[smallest] = tmp;
                i = smallest;
            }
        }

        private static void Softmax(float[] logits, int n, float[] dest)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < n; i++)
            {
                if (logits[i] > max)
                    max = logits[i];
            }

            float sum = 0;
            for (int i = 0; i < n; i++)
            {
                dest[i] = MathF.Exp(logits[i] - max);
                sum += dest[i];
            }

            if (sum <= 0)
                return;
            for (int i = 0; i < n; i++)
                dest[i] /= sum;
        }

        private static int Nucleus(float[] probs, int[] indices, int n, float topP)
        {
            float cum = 0;
            int keep = 0;
            for (int i = 0; i < n; i++)
            {
                cum += probs[i];
                keep++;
                if (cum >= topP && keep > 0)
                    break;
            }

            return Math.Max(1, keep);
        }

        private static int ArgMax(float[] logits, int n)
        {
            int best = 0;
            for (int i = 1; i < n; i++)
            {
                if (logits[i] > logits[best])
                    best = i;
            }
            return best;
        }
    }
}
