# Qwen3-TTS on this branch

The `qwen3-tts` branch keeps the Spark-TTS-Unity **public API** (`CharacterVoiceFactory`, `CharacterVoice.GenerateSpeechAsync`, style knobs `gender` / `pitch` / `speed`). Internals synthesize with **Qwen3-TTS 12Hz 1.7B CustomVoice ONNX**, ported from [ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS) (MIT).

This package **does not download** weights (no HuggingFace client). Place files locally the same way as the original Spark ONNX tree, then copy into StreamingAssets. A later Google Drive zip is fine; the layout below is the contract.

## Layout

Drop the 1.7B CustomVoice export under:

```
Assets/StreamingAssets/SparkTTS/Qwen3-1.7B/
  talker_prefill.onnx
  talker_prefill.onnx.data
  talker_decode.onnx
  talker_decode.onnx.data
  code_predictor.onnx
  code_predictor.onnx.data
  vocoder.onnx
  vocoder.onnx.data
  embeddings/
    config.json
    speaker_ids.json
    text_embedding.npy
    text_projection_fc1_weight.npy
    text_projection_fc1_bias.npy
    text_projection_fc2_weight.npy
    text_projection_fc2_bias.npy
    talker_codec_embedding.npy
    codec_head_weight.npy
    cp_projection_weight.npy
    cp_projection_bias.npy
    cp_codec_embedding_0.npy … cp_codec_embedding_14.npy
  tokenizer/
    vocab.json
    merges.txt
```

Source repo for the ONNX export: [elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX) (~10 GB).

`SparkTTS/Model Deployment Tool` has an **Include Qwen3-TTS 1.7B** toggle. Point the source folder at a tree that already contains `SparkTTS/Qwen3-1.7B/…` (or copy the folder into StreamingAssets by hand).

`QwenModelPaths.IsPresent()` / `GetMissingFiles()` are the runtime checklist. `CharacterVoiceFactory.IsReady` is false until every expected file exists.

## Public API mapping

| Spark call | Qwen 1.7B CustomVoice |
|---|---|
| `CreateFromStyleAsync(gender, pitch, speed)` | Preset speaker (`male` → `ryan`, `female` → `serena`) plus instruct text for non-moderate pitch/speed |
| `GenerateSpeechAsync(text, sampleRate)` | Tokenize → talker LM → vocoder at **24 kHz**, then resample to `sampleRate` (default **16000**) |
| `CreateFromReference` / clone | **Not supported.** Needs the Base ONNX + speaker encoder. Returns `null` and logs. |
| `CreateFromFolderAsync` | Reloads `voice_config.json` style knobs only (no Spark global-token cache) |

Use **CPU** `ExecutionProvider`. CoreML/CUDA arguments are accepted on `Initialize` and ignored.

`WaitForModelsLoadedAsync` loads embeddings + tokenizer (~1 GB+ npy). ONNX sessions stay lazy until the first `GenerateSpeechAsync`.

## What this branch does not do

- Auto-download from HuggingFace
- Voice cloning / ICL (CustomVoice presets only)
- 0.6B variant
- Packing weights into the git repo
