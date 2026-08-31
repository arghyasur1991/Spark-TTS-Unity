// zukky/Qwen3-TTS-ONNX-DLL talker graph: embeddings as ONNX (text_project / codec_embed /
// code_predictor_embed), prefill without position_ids, tokenizer12hz_decode vocoder.
// Pipeline matches examples/python_dll_call/run_pipeline.py (x-vector clone). No Windows DLL.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SparkTTS.Core;
using SparkTTS.Models;
using SparkTTS.Qwen.Models;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS.Qwen
{
    internal sealed class QwenBaseTalker : IDisposable
    {
        private readonly QwenBaseConfig _config;
        private readonly QwenOnnxModel _textProject;
        private readonly QwenOnnxModel _codecEmbed;
        private readonly QwenOnnxModel _cpEmbed;
        private readonly QwenOnnxModel _prefill;
        private readonly QwenOnnxModel _decode;
        private readonly QwenOnnxModel _codePredictor;
        private readonly QwenVocoderModel _vocoder;
        private readonly QwenTokenSampler _sampler = new();
        private readonly int[] _suppress;
        private bool _disposed;

        public QwenBaseTalker(QwenBaseConfig config, ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            _config = config;
            string folder = SparkTTSModelPaths.QwenBaseFolder;
            _textProject = new QwenOnnxModel(SparkTTSModelPaths.QwenTextProject, folder, executionProvider);
            _codecEmbed = new QwenOnnxModel(SparkTTSModelPaths.QwenCodecEmbed, folder, executionProvider);
            _cpEmbed = new QwenOnnxModel(SparkTTSModelPaths.QwenCodePredictorEmbed, folder, executionProvider);
            _prefill = new QwenOnnxModel(SparkTTSModelPaths.QwenTalkerPrefill, folder, executionProvider);
            _decode = new QwenOnnxModel(SparkTTSModelPaths.QwenTalkerDecode, folder, executionProvider);
            _codePredictor = new QwenOnnxModel(SparkTTSModelPaths.QwenCodePredictor, folder, executionProvider);
            _vocoder = QwenVocoderModel.Base(executionProvider);
            _suppress = QwenTokenSampler.SuppressUpperCodec(config.VocabSize, config.CodecEosTokenId);
        }

        public int HiddenSize => _config.HiddenSize;

        public float[,,] TextProject(long[] tokenIds)
        {
            return RunEmbed2d(_textProject, tokenIds);
        }

        public float[,,] CodecEmbed(long[] tokenIds)
        {
            return RunEmbed2d(_codecEmbed, tokenIds);
        }

        public float[,,] CodePredictorEmbed(long tokenId, int generationStep)
        {
            var idsTensor = new DenseTensor<long>(new long[] { tokenId }, new[] { 1, 1 });
            var stepName = _cpEmbed.ResolveInputName("generation_step", "generation_steps");
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsTensor),
                NamedOnnxValue.CreateFromTensor(stepName, new DenseTensor<long>(new long[] { generationStep }, new[] { 1 }))
            };
            using var results = _cpEmbed.Run(feeds);
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

            using (var prefillOut = _prefill.Run(prefillFeeds))
            {
                if (prefillOut.Count < 2)
                    throw new InvalidOperationException("talker_prefill.onnx must output logits and last_hidden.");
                logits = QwenOnnxModel.CopyFloat(prefillOut[0]);
                lastHidden = QwenOnnxModel.LastHidden(prefillOut[1], hidden);
                past = CopyPast(prefillOut, 2);
            }

            decodePastNames = DecodePastNames(_decode);

            var generated = new List<long[]>();
            var firstCodes = new List<int>();

            for (int step = 0; step < maxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int nextId = _sampler.Sample(logits, vocab, 0.9f, 50, 1f, firstCodes, 1.05f, _suppress);
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
                            _codePredictor.ResolveInputName("generation_step", "generation_steps"),
                            new DenseTensor<long>(new long[] { j }, new[] { 1 }))
                    };
                    using var cpOut = _codePredictor.Run(cpFeeds);
                    int sub = _sampler.Sample(QwenOnnxModel.CopyFloat(cpOut[0]), 2048, 0.9f, 50, 1f, null, 1f, null);
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
                    using var nextPrefill = _prefill.Run(again);
                    logits = QwenOnnxModel.CopyFloat(nextPrefill[0]);
                    lastHidden = QwenOnnxModel.LastHidden(nextPrefill[1], hidden);
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

                    using var decodeOut = _decode.Run(decodeFeeds);
                    logits = QwenOnnxModel.CopyFloat(decodeOut[0]);
                    lastHidden = QwenOnnxModel.LastHidden(decodeOut[1], hidden);
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
            return _vocoder.Decode(codes, cancellationToken);
        }

        internal void CollectOnnxModels(List<ORTModel> list)
        {
            list.Add(_textProject);
            list.Add(_codecEmbed);
            list.Add(_cpEmbed);
            list.Add(_prefill);
            list.Add(_decode);
            list.Add(_codePredictor);
            list.Add(_vocoder);
        }

        internal void PreloadSessions()
        {
            var sw = Stopwatch.StartNew();
            _textProject.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] text_project {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _codecEmbed.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] codec_embed {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _cpEmbed.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] code_predictor_embed {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _prefill.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] talker_prefill {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _decode.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] talker_decode {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _codePredictor.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] code_predictor {sw.ElapsedMilliseconds}ms");
            sw.Restart();
            _vocoder.GetSession();
            TTSLogger.Log($"[QwenBaseTalker] vocoder {sw.ElapsedMilliseconds}ms");
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _textProject.Dispose();
            _codecEmbed.Dispose();
            _cpEmbed.Dispose();
            _prefill.Dispose();
            _decode.Dispose();
            _codePredictor.Dispose();
            _vocoder.Dispose();
        }

        private float[,,] RunEmbed2d(QwenOnnxModel model, long[] tokenIds)
        {
            var tensor = new DenseTensor<long>(tokenIds, new[] { 1, tokenIds.Length });
            var feeds = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", tensor)
            };
            using var results = model.Run(feeds);
            return ToFloat3(results[0]);
        }

        private static float[,,] ToFloat3(DisposableNamedOnnxValue value)
        {
            var tensor = value.AsTensor<float>();
            var data = QwenOnnxModel.CopyFloat(value);
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
                list.Add(new OrtCopy { Data = QwenOnnxModel.CopyFloat(outputs[i]), Dims = dims });
            }

            return list.Count == 0 ? null : list;
        }

        private static string[] DecodePastNames(QwenOnnxModel decode)
        {
            var names = decode.GraphInputNames;
            var past = new List<string>();
            for (int i = 0; i < names.Count; i++)
            {
                string n = names[i];
                if (n == "inputs_embeds" || n == "attention_mask" || n == "position_ids")
                    continue;
                past.Add(n);
            }
            return past.ToArray();
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
