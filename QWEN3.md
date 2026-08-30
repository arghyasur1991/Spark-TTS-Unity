# Qwen3-TTS on this branch

The `qwen3-tts` branch keeps the Spark-TTS-Unity **public API** (`CharacterVoiceFactory`, `CharacterVoice.GenerateSpeechAsync`, style knobs `gender` / `pitch` / `speed`). Internals:

| Path | Backend | Weights |
|---|---|---|
| `CreateFromStyleAsync` | Qwen3-TTS 12Hz **1.7B CustomVoice** | ElBruno npy + sharded ONNX |
| `CreateFromReference` | Qwen3-TTS 12Hz **1.7B Base** (x-vector clone) | zukky single-file ONNX |

This package **does not download** weights (no HuggingFace client). Place files locally, then copy into StreamingAssets. A later Google Drive zip is fine; the layouts below are the contract.

Do **not** ship `qwen3_tts_rust.dll`. That DLL is Windows-only. Mac editor and Android use the C# tokenizer, resampler, and mel spectrogram in this package.

## Layout — CustomVoice (style)

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
    …
  tokenizer/
    vocab.json
    merges.txt
```

Source: [elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX) (~10 GB).

`QwenModelPaths.IsPresent()` is the checklist. Style TTS is unavailable until every expected file exists.

## Layout — Base (voice cloning)

Drop the zukky **1.7B Base** bundle under:

```
Assets/StreamingAssets/SparkTTS/Qwen3-1.7B-Base/
  talker_prefill.onnx
  talker_decode.onnx
  code_predictor.onnx
  code_predictor_embed.onnx
  codec_embed.onnx
  text_project.onnx
  speaker_encoder.onnx
  tokenizer12hz_decode.onnx
  config.json
  vocab.json
  merges.txt
```

From [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL) (~14 GB):

1. Copy `dist/dll_release/onnx_kv/*.onnx` into that folder (single-file ONNX, no `.onnx.data`).
2. Copy `dist/dll_release/models/Qwen3-TTS-12Hz-1.7B-Base/{config.json,vocab.json,merges.txt}` next to them (or into a `tokenizer/` subfolder for the vocab/merges pair).
3. Skip `qwen3_tts_rust.dll`, `qwen3_tts.h`, and `tokenizer12hz_encode.onnx` (encode is ICL-only; Spark’s clone API has no reference transcript).

`QwenBaseModelPaths.IsPresent()` / `GetMissingFiles()` are the runtime checklist. `CharacterVoiceFactory.IsReady` is true when **either** CustomVoice **or** Base is complete.

`SparkTTS/Model Deployment Tool` has **Include Qwen3-TTS 1.7B CustomVoice** and **Include Qwen3-TTS 1.7B Base (clone)** toggles. Base is off by default so a CustomVoice-only copy still deploys.

## Public API mapping

| Spark call | Backend |
|---|---|
| `CreateFromStyleAsync(gender, pitch, speed)` | CustomVoice preset speaker (`male` → `ryan`, `female` → `serena`) plus instruct text for non-moderate pitch/speed |
| `CreateFromReference(AudioClip)` | Base **x-vector** clone: 24 kHz mel → `speaker_encoder` → talker. Needs Base files. |
| `GenerateSpeechAsync(text, sampleRate)` | Talker + vocoder at **24 kHz**, then resample to `sampleRate` (default **16000**) |
| `CreateFromFolderAsync` | Reloads `voice_config.json` style knobs only (CustomVoice) |

Use **CPU** `ExecutionProvider`. CoreML/CUDA arguments are accepted on `Initialize` and ignored.

`WaitForModelsLoadedAsync` loads CustomVoice embeddings / Base tokenizer. Large ONNX sessions stay lazy until the first generate or clone extract.

## Clone notes

Spark’s `CreateFromReference(AudioClip)` has **no reference transcript**. That matches zukky `--xvec-only` (speaker embedding only). Full ICL (`--ref-audio` + `--ref-text` + `tokenizer12hz_encode`) is not on this public API.

Reference audio is converted to mono 24 kHz on the calling thread (`AudioClip.GetData`). First clone also loads `speaker_encoder.onnx` (~48 MB). First `GenerateSpeechAsync` on a cloned voice loads the ~5.7 GB talker sessions — expect a long hitch.

## What this branch does not do

- Auto-download from HuggingFace
- Windows Rust DLL
- ICL (reference transcript + codec codes)
- 0.6B variant
- Packing weights into the git repo
