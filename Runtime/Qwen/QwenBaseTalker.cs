// zukky/Qwen3-TTS-ONNX-DLL talker graph: embeddings as ONNX (text_project / codec_embed /
// code_predictor_embed), prefill without position_ids, tokenizer12hz_decode vocoder.
// Pipeline matches examples/python_dll_call/run_pipeline.py (x-vector clone). No Windows DLL.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SparkTTS.Qwen
{
    internal sealed class QwenBaseTalker : IDisposable
    {
        private const long MaxOnnxBytes = 8_000_000_000;

        private readonly string _modelDir;
        private readonly Func<SessionOptions> _sessionOptionsFactory;
        private readonly QwenBaseConfig _config;
        private readonly Lazy<InferenceSession> _textProject;
        private readonly Lazy<InferenceSession> _codecEmbed;
        private readonly Lazy<InferenceSession> _cpEmbed;
        private readonly Lazy<InferenceSession> _prefill;
        private readonly Lazy<InferenceSession> _decode;
        private readonly Lazy<InferenceSession> _codePredictor;
        private readonly Lazy<InferenceSession> _vocoder;
        private readonly Random _rng = new();
        private bool _disposed;

        public QwenBaseTalker(string modelDir, QwenBaseConfig config, Func<SessionOptions> sessionOptionsFactory)
        {
            _modelDir = modelDir;
            _config = config;
            _sessionOptionsFactory = sessionOptionsFactory;
            _textProject = CreateLazy("text_project.onnx");
            _codecEmbed = CreateLazy("codec_embed.onnx");
            _cpEmbed = CreateLazy("code_predictor_embed.onnx");
            _prefill = CreateLazy("talker_prefill.onnx");
            _decode = CreateLazy("talker_decode.onnx");
            _codePredictor = CreateLazy("code_predictor.onnx");
            _vocoder = CreateLazy("tokenizer12hz_decode.onnx");
        }

        public int HiddenSize => _config.HiddenSize;

        public float[,,] TextProject(long[] tokenIds)
        {
            return RunEmbed2d(_textProject.Value, tokenIds, "input_ids");
        }

        public float[,,] CodecEmbed(long[] tokenIds)
        {
            return RunEmbed2d(_codecEmbed.Value, tokenIds, "input_ids");
        }

        public float[,,] CodePredictorEmbed(long tokenId, int generationStep)
        {
            var session = _cpEmbed.Value;
            var ids = new long[] { tokenId };
            var idsTensor = new DenseTensor<long>(ids, new[] { 1, 1 });
            var stepName = ResolveInputName(session, "generation_step", "generation_steps");
            var stepTensor = new DenseTensor<long>(new long[] { generationStep }, new[] { 1 });
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsTensor),
                NamedOnnxValue.CreateFromTensor(stepName, stepTensor)
            };
            using var results = session.Run(feeds);
            return ToFloat3(results[0]);
        }

        public long[,] GenerateCodes(
            float[,,] inputsEmbeds,
            long[,] attentionMask,
            float[,,] trailingTextHidden,
            float[] ttsPadEmbed,
            int maxNewTokens,
            CancellationToken cancellationToken)
        {
            int hidden = _config.HiddenSize;
            int numGroups = _config.NumCodeGroups;
            int eosId = _config.CodecEosTokenId;
            int vocab = _config.VocabSize;
            int prefillT = inputsEmbeds.GetLength(1);

            var inputs = Flatten3(inputsEmbeds);
            var mask = FlattenMask(attentionMask);
            int seqLen = prefillT;

            var prefillFeeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds",
                    new DenseTensor<float>(inputs, new[] { 1, seqLen, hidden })),
                NamedOnnxValue.CreateFromTensor("attention_mask",
                    new DenseTensor<long>(mask, new[] { 1, seqLen }))
            };

            float[] logits;
            float[] lastHidden;
            List<OrtCopy> past;
            string[] decodePastNames;

            using (var prefillOut = _prefill.Value.Run(prefillFeeds))
            {
                if (prefillOut.Count < 2)
                    throw new InvalidOperationException("talker_prefill.onnx must output logits and last_hidden.");
                logits = CopyFloat(prefillOut[0]);
                lastHidden = CopyLastHidden(prefillOut[1], hidden);
                past = CopyPast(prefillOut, 2);
            }

            decodePastNames = DecodePastNames(_decode.Value);

            var generated = new List<long[]>();
            var firstCodes = new List<int>();
            var suppress = BuildSuppress(vocab, eosId);

            for (int step = 0; step < maxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var stepLogits = LastLogits(logits, vocab);
                ApplySuppress(stepLogits, suppress);
                ApplyRepetitionPenalty(stepLogits, firstCodes, 1.05f);
                int nextId = SampleToken(stepLogits, 0.9f, 50, 1.0f, true);
                firstCodes.Add(nextId);

                if (nextId == eosId && generated.Count >= 1)
                    break;

                var firstEmbed = Flatten3(CodecEmbed(new long[] { nextId }));
                var embedSeq = new List<float[]> { lastHidden, firstEmbed };
                var codes = new long[numGroups];
                codes[0] = nextId;

                for (int j = 0; j < numGroups - 1; j++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    float[] cpIn = ConcatSeq(embedSeq, hidden);
                    int cpT = embedSeq.Count;
                    var cpFeeds = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("inputs_embeds",
                            new DenseTensor<float>(cpIn, new[] { 1, cpT, hidden })),
                        NamedOnnxValue.CreateFromTensor(
                            ResolveInputName(_codePredictor.Value, "generation_step", "generation_steps"),
                            new DenseTensor<long>(new long[] { j }, new[] { 1 }))
                    };
                    using var cpOut = _codePredictor.Value.Run(cpFeeds);
                    var cpLogits = LastLogits(CopyFloat(cpOut[0]), 2048);
                    int sub = SampleToken(cpLogits, 0.9f, 50, 1.0f, true);
                    codes[j + 1] = sub;
                    var subEmbed = Flatten3(CodePredictorEmbed(sub, j));
                    embedSeq.Add(subEmbed);
                }

                var codecSum = (float[])firstEmbed.Clone();
                for (int e = 2; e < embedSeq.Count; e++)
                    AddInPlace(codecSum, embedSeq[e]);

                int trailT = trailingTextHidden.GetLength(1);
                if (step < trailT)
                {
                    for (int i = 0; i < hidden; i++)
                        codecSum[i] += trailingTextHidden[0, step, i];
                }
                else
                {
                    for (int i = 0; i < hidden; i++)
                        codecSum[i] += ttsPadEmbed[i];
                }

                generated.Add(codes);
                if (nextId == eosId)
                    break;

                seqLen++;
                var newMask = new long[seqLen];
                Array.Copy(mask, newMask, mask.Length);
                newMask[seqLen - 1] = 1;
                mask = newMask;

                if (past == null || decodePastNames.Length == 0)
                {
                    var grown = ConcatHidden(inputs, seqLen - 1, codecSum, hidden);
                    inputs = grown;
                    var again = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("inputs_embeds",
                            new DenseTensor<float>(grown, new[] { 1, seqLen, hidden })),
                        NamedOnnxValue.CreateFromTensor("attention_mask",
                            new DenseTensor<long>(mask, new[] { 1, seqLen }))
                    };
                    using var nextPrefill = _prefill.Value.Run(again);
                    logits = CopyFloat(nextPrefill[0]);
                    lastHidden = CopyLastHidden(nextPrefill[1], hidden);
                    past = CopyPast(nextPrefill, 2);
                }
                else
                {
                    var decodeFeeds = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("inputs_embeds",
                            new DenseTensor<float>(codecSum, new[] { 1, 1, hidden })),
                        NamedOnnxValue.CreateFromTensor("attention_mask",
                            new DenseTensor<long>(mask, new[] { 1, seqLen }))
                    };
                    for (int p = 0; p < decodePastNames.Length && p < past.Count; p++)
                    {
                        decodeFeeds.Add(NamedOnnxValue.CreateFromTensor(
                            decodePastNames[p],
                            new DenseTensor<float>(past[p].Data, past[p].Dims)));
                    }

                    using var decodeOut = _decode.Value.Run(decodeFeeds);
                    logits = CopyFloat(decodeOut[0]);
                    lastHidden = CopyLastHidden(decodeOut[1], hidden);
                    past = CopyPast(decodeOut, 2);
                }
            }

            int keep = generated.Count;
            for (int t = 0; t < generated.Count; t++)
            {
                if (generated[t][0] == eosId)
                {
                    keep = t;
                    break;
                }
            }

            var result = new long[Math.Max(0, keep), numGroups];
            for (int t = 0; t < keep; t++)
            {
                for (int g = 0; g < numGroups; g++)
                    result[t, g] = generated[t][g];
            }

            return result;
        }

        public float[] DecodeCodes(long[,] codes, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int t = codes.GetLength(0);
            int groups = codes.GetLength(1);
            if (t == 0)
                return Array.Empty<float>();

            var flat = new long[1 * t * groups];
            int n = 0;
            for (int i = 0; i < t; i++)
            {
                for (int g = 0; g < groups; g++)
                    flat[n++] = codes[i, g];
            }

            var session = _vocoder.Value;
            var inputName = ResolveInputName(session, "audio_codes", "codes");
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName,
                    new DenseTensor<long>(flat, new[] { 1, t, groups }))
            };

            using var results = session.Run(feeds);
            var wav = CopyFloat(results[0]);
            int wavLen = wav.Length;
            var wavTensor = results[0].AsTensor<float>();
            if (wavTensor.Dimensions.Length >= 2)
                wavLen = wavTensor.Dimensions[1];

            int target = t * 1920;
            if (target > wavLen)
                target = wavLen;
            if (results.Count > 1)
            {
                var lengths = CopyLong(results[1]);
                if (lengths.Length > 0 && lengths[0] > 0 && lengths[0] < target)
                    target = (int)lengths[0];
            }

            if (target >= wav.Length)
                return wav;
            var trimmed = new float[target];
            Array.Copy(wav, trimmed, target);
            return trimmed;
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            DisposeLazy(_textProject);
            DisposeLazy(_codecEmbed);
            DisposeLazy(_cpEmbed);
            DisposeLazy(_prefill);
            DisposeLazy(_decode);
            DisposeLazy(_codePredictor);
            DisposeLazy(_vocoder);
        }

        private Lazy<InferenceSession> CreateLazy(string fileName)
        {
            return new Lazy<InferenceSession>(
                () => CreateSession(fileName),
                LazyThreadSafetyMode.ExecutionAndPublication);
        }

        private InferenceSession CreateSession(string fileName)
        {
            var path = Path.Combine(_modelDir, fileName);
            var info = new FileInfo(path);
            if (!info.Exists)
                throw new FileNotFoundException("Qwen3-TTS Base ONNX file missing.", path);
            if (info.Length > MaxOnnxBytes)
                throw new InvalidOperationException($"ONNX file too large ({info.Length / 1e9:F2} GB).");
            return new InferenceSession(path, _sessionOptionsFactory());
        }

        private static void DisposeLazy(Lazy<InferenceSession> lazy)
        {
            if (lazy.IsValueCreated)
                lazy.Value.Dispose();
        }

        private float[,,] RunEmbed2d(InferenceSession session, long[] tokenIds, string inputName)
        {
            var tensor = new DenseTensor<long>(tokenIds, new[] { 1, tokenIds.Length });
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            };
            using var results = session.Run(feeds);
            return ToFloat3(results[0]);
        }

        private static float[,,] ToFloat3(DisposableNamedOnnxValue value)
        {
            var tensor = value.AsTensor<float>();
            var data = value.AsEnumerable<float>().ToArray();
            if (tensor.Dimensions.Length == 3)
            {
                int d0 = tensor.Dimensions[0];
                int d1 = tensor.Dimensions[1];
                int d2 = tensor.Dimensions[2];
                var arr = new float[d0, d1, d2];
                Buffer.BlockCopy(data, 0, arr, 0, data.Length * sizeof(float));
                return arr;
            }

            int hiddenGuess = data.Length;
            var fallback = new float[1, 1, hiddenGuess];
            for (int i = 0; i < hiddenGuess; i++)
                fallback[0, 0, i] = data[i];
            return fallback;
        }

        private static float[] Flatten3(float[,,] a)
        {
            int b = a.GetLength(0);
            int t = a.GetLength(1);
            int h = a.GetLength(2);
            var flat = new float[b * t * h];
            Buffer.BlockCopy(a, 0, flat, 0, flat.Length * sizeof(float));
            return flat;
        }

        private static long[] FlattenMask(long[,] mask)
        {
            int t = mask.GetLength(1);
            var flat = new long[t];
            for (int i = 0; i < t; i++)
                flat[i] = mask[0, i];
            return flat;
        }

        private static float[] CopyFloat(DisposableNamedOnnxValue value)
        {
            return value.AsEnumerable<float>().ToArray();
        }

        private static long[] CopyLong(DisposableNamedOnnxValue value)
        {
            return value.AsEnumerable<long>().ToArray();
        }

        private static float[] CopyLastHidden(DisposableNamedOnnxValue value, int hidden)
        {
            var all = CopyFloat(value);
            if (all.Length == hidden)
                return all;
            var last = new float[hidden];
            Array.Copy(all, all.Length - hidden, last, 0, hidden);
            return last;
        }

        private static List<OrtCopy> CopyPast(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, int start)
        {
            var list = new List<OrtCopy>();
            for (int i = start; i < outputs.Count; i++)
            {
                var tensor = outputs[i].AsTensor<float>();
                if (tensor == null)
                    continue;
                var dims = new int[tensor.Dimensions.Length];
                for (int d = 0; d < dims.Length; d++)
                    dims[d] = tensor.Dimensions[d];
                list.Add(new OrtCopy { Data = outputs[i].AsEnumerable<float>().ToArray(), Dims = dims });
            }

            return list.Count == 0 ? null : list;
        }

        private static string[] OrderedInputNames(InferenceSession session)
        {
            var keys = new List<string>(session.InputMetadata.Keys);
            return keys.ToArray();
        }

        private static string[] DecodePastNames(InferenceSession session)
        {
            var names = OrderedInputNames(session);
            var past = new List<string>();
            for (int i = 0; i < names.Length; i++)
            {
                string n = names[i];
                if (n == "inputs_embeds" || n == "attention_mask" || n == "position_ids")
                    continue;
                past.Add(n);
            }
            return past.ToArray();
        }

        private static string ResolveInputName(InferenceSession session, params string[] candidates)
        {
            foreach (var c in candidates)
            {
                if (session.InputMetadata.ContainsKey(c))
                    return c;
            }

            foreach (var key in session.InputMetadata.Keys)
                return key;
            return candidates[0];
        }

        private static float[] LastLogits(float[] logits, int vocab)
        {
            if (logits.Length == vocab)
                return (float[])logits.Clone();
            var slice = new float[vocab];
            int src = Math.Max(0, logits.Length - vocab);
            int n = Math.Min(vocab, logits.Length - src);
            Array.Copy(logits, src, slice, 0, n);
            return slice;
        }

        private static int[] BuildSuppress(int vocab, int eosId)
        {
            var list = new List<int>();
            int start = Math.Max(0, vocab - 1024);
            for (int i = start; i < vocab; i++)
            {
                if (i != eosId)
                    list.Add(i);
            }
            return list.ToArray();
        }

        private static void ApplySuppress(float[] logits, int[] tokens)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                int t = tokens[i];
                if (t >= 0 && t < logits.Length)
                    logits[t] = -1e9f;
            }
        }

        private static void ApplyRepetitionPenalty(float[] logits, List<int> hist, float penalty)
        {
            if (penalty == 1f || hist.Count == 0)
                return;
            var seen = new HashSet<int>(hist);
            foreach (var tok in seen)
            {
                if (tok < 0 || tok >= logits.Length)
                    continue;
                if (logits[tok] >= 0)
                    logits[tok] /= penalty;
                else
                    logits[tok] *= penalty;
            }
        }

        private int SampleToken(float[] logits, float temperature, int topK, float topP, bool doSample)
        {
            int vocab = logits.Length;
            var scaled = new float[vocab];
            float temp = temperature <= 0 ? 1f : temperature;
            for (int i = 0; i < vocab; i++)
                scaled[i] = logits[i] / temp;

            if (!doSample)
            {
                int argmax = 0;
                for (int i = 1; i < vocab; i++)
                {
                    if (scaled[i] > scaled[argmax])
                        argmax = i;
                }
                return argmax;
            }

            if (topK > 0 && topK < vocab)
            {
                var copy = (float[])scaled.Clone();
                Array.Sort(copy);
                float thresh = copy[vocab - topK];
                for (int i = 0; i < vocab; i++)
                {
                    if (scaled[i] < thresh)
                        scaled[i] = -1e9f;
                }
            }

            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < vocab; i++)
            {
                if (scaled[i] > maxLogit)
                    maxLogit = scaled[i];
            }

            double sum = 0;
            var probs = new double[vocab];
            for (int i = 0; i < vocab; i++)
            {
                probs[i] = Math.Exp(scaled[i] - maxLogit);
                sum += probs[i];
            }

            if (sum <= 0 || double.IsNaN(sum))
            {
                int argmax = 0;
                for (int i = 1; i < vocab; i++)
                {
                    if (logits[i] > logits[argmax])
                        argmax = i;
                }
                return argmax;
            }

            double r = _rng.NextDouble() * sum;
            double acc = 0;
            for (int i = 0; i < vocab; i++)
            {
                acc += probs[i];
                if (r < acc)
                    return i;
            }

            return vocab - 1;
        }

        private static float[] ConcatSeq(List<float[]> parts, int hidden)
        {
            var flat = new float[parts.Count * hidden];
            for (int i = 0; i < parts.Count; i++)
                Array.Copy(parts[i], 0, flat, i * hidden, hidden);
            return flat;
        }

        private static void AddInPlace(float[] dst, float[] src)
        {
            int n = Math.Min(dst.Length, src.Length);
            for (int i = 0; i < n; i++)
                dst[i] += src[i];
        }

        private static float[] ConcatHidden(float[] prefix, int prefixT, float[] next, int hidden)
        {
            var grown = new float[(prefixT + 1) * hidden];
            Array.Copy(prefix, grown, prefixT * hidden);
            Array.Copy(next, 0, grown, prefixT * hidden, hidden);
            return grown;
        }

        private sealed class OrtCopy
        {
            public float[] Data;
            public int[] Dims;
        }
    }
}
