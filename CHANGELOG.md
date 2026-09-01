# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — `qwen3-tts`

### Changed
- Style TTS now runs Qwen3-TTS 1.7B CustomVoice ONNX (ElBruno.QwenTTS port). `CharacterVoiceFactory` / `CharacterVoice` signatures are unchanged.
- `GenerateSpeechAsync` still defaults to 16 kHz; native vocoder output is 24 kHz and is resampled.
- `CreateFromReference` clones via Qwen3-TTS 1.7B **Base** ONNX. With `refText`, this is official ICL (`tokenizer_encoder` + speaker embedding). Without `refText`, x-vector-only. CustomVoice still cannot clone.

### Added
- Local 1.7B CustomVoice layout under `StreamingAssets/SparkTTS/Qwen3-1.7B/` (`QwenModelPaths`). No HuggingFace download.
- Local 1.7B Base layout under `StreamingAssets/SparkTTS/Qwen3-1.7B-Base/` (`QwenModelPaths`) from [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL). C# mel + speaker encoder; no Windows DLL.
- Model Deployment Tool categories for CustomVoice and Base.
- Qwen ONNX graphs load through Spark `ORTModel` (`QwenOnnxModel`). Shared `QwenTokenSampler` / `QwenVocoderModel` / one `QwenTtsEngine`. Decode uses Spark LLM-style buffer reuse; sessions defer-load. Session construct and embedding load use `TaskScheduler.Default` with ExecutionContext flow suppressed so `InferenceSession` cannot marshal onto the Unity main thread. `GenerateSpeechAsync` runs preload+synth in one worker. `UnloadModels()` drops the engine without permanently disposing the factory.
- Editor domain reload: `KeepNativeSessionsAcrossReload` detaches native `OrtSession` / `OrtEnv` handles **and** AllocHGlobal CustomVoice embedding matrices (npy + precomputed CP projections, ~1.5 GB) before unload, then wraps them after. Script compile / Play does not rebuild graphs or re-read npy. Default `OrtEnv` (no custom logger fn ptr — a Unity callback here SIGSEGVs after reload). First load still streams npy into native RAM with unsafe `FileStream.Read` + parallel projection. `UnloadModels()` still releases.
- CustomVoice `vocoder.onnx` output is `[batch, 1, samples]`. Do not read `Dimensions[1]` as waveform length (that trimmed speech to 1 sample).

## [0.1.0] - 2025-05-17

### Added
- Initial release of Spark TTS Unity package
- Basic text-to-speech and voice cloning functionality
- Sample scripts and example implementation
- Support for Unity 6000.0.46f1 and newer 