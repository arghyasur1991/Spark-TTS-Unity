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

1. Copy `onnx_kv/*.onnx` into that folder (single-file ONNX, no `.onnx.data`). On the Hub those files live at repo-root `onnx_kv/`, not `dist/dll_release/`.
2. Copy `models/Qwen3-TTS-12Hz-1.7B-Base/{config.json,vocab.json,merges.txt}` next to them (or into a `tokenizer/` subfolder for the vocab/merges pair).
3. Skip `qwen3_tts_rust.dll`, `qwen3_tts.h`, and `tokenizer12hz_encode.onnx` (encode is ICL-only; Spark’s clone API has no reference transcript).

This package does not download Hub files. A host app copies the layouts above into `StreamingAssets/SparkTTS/`.

`QwenModelPaths.IsCustomVoicePresent()` / `IsBasePresent()` / `GetMissingBaseFiles()` are the runtime checklist. `CharacterVoiceFactory.IsReady` is true when **either** CustomVoice **or** Base is complete.

`SparkTTS/Model Deployment Tool` has **Include Qwen3-TTS 1.7B CustomVoice** and **Include Qwen3-TTS 1.7B Base (clone)** toggles. Base is off by default so a CustomVoice-only copy still deploys.

## Public API mapping

| Spark call | Backend |
|---|---|
| `CreateFromStyleAsync(gender, pitch, speed)` | CustomVoice preset speaker (`male` → `ryan`, `female` → `serena`) plus instruct text for non-moderate pitch/speed |
| `CreateFromReference(AudioClip)` | Base **x-vector** clone: 24 kHz mel → `speaker_encoder` → talker. Needs Base files. |
| `GenerateSpeechAsync(text, sampleRate)` | Talker + vocoder at **24 kHz**, then resample to `sampleRate` (default **16000**) |
| `CreateFromFolderAsync` | Reloads `voice_config.json` style knobs only (CustomVoice) |

`Initialize` passes `ExecutionProvider` into Spark `ORTModel` (same CPU / CUDA / CoreML path as the original Spark graphs). Default is **CPU**. CoreML still has a known load NRE on some machines — prefer CPU until that is fixed in `ORTModel`.

Every Qwen ONNX file is an `ORTModel` (`QwenOnnxModel` or a named subclass). Sessions defer-load (the 5–14 GB talkers must not all open at factory init). The decode loop is synchronous `Session.Run` after `EnsureLoaded` — do not wrap each token in `Task.Run` / `RunDisposable`. Sampling matches Spark `LLMModel`: reused buffers, min-heap top-K, `ThreadLocal<Random>`, `DenseTensor.Buffer.ToArray()`. `BackgroundWork` suppresses ExecutionContext flow so session construct stays off the Unity main thread. CustomVoice vocoder output layout is `[batch, 1, samples]` — never treat dim 1 as the waveform length.

CustomVoice (npy embeddings + `position_ids`) and Base (ONNX embeddings, no `position_ids`) keep **two generate loops** — the graphs are not drop-in. One engine (`QwenTtsEngine`), one vocoder class, one sampler, one path class.

`WaitForModelsLoadedAsync` constructs the engine (tokenizer / embeddings) off the Unity main thread. Large ONNX sessions stay deferred until the first generate or clone extract; that construct also uses the thread pool. `CharacterVoiceFactory.UnloadModels()` drops the engine so RAM can be freed without a domain reload. In the editor, `KeepNativeSessionsAcrossReload` detaches native ONNX sessions **and** AllocHGlobal embedding matrices before a domain reload and wraps them after so a script compile does not rebuild the graphs or re-read 1.5 GB of npy.

## Clone notes

Spark’s `CreateFromReference(AudioClip)` has **no reference transcript**. That matches zukky `--xvec-only` (speaker embedding only). Full ICL (`--ref-audio` + `--ref-text` + `tokenizer12hz_encode`) is not on this public API.

Reference audio is converted to mono 24 kHz on the calling thread (`AudioClip.GetData`). First clone also loads `speaker_encoder.onnx` (~48 MB). First `GenerateSpeechAsync` on a cloned voice loads the ~5.7 GB talker sessions — expect a long hitch.

## What this branch does not do

- Auto-download from HuggingFace
- Windows Rust DLL
- ICL (reference transcript + codec codes)
- 0.6B variant
- Packing weights into the git repo
