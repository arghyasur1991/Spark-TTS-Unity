# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — `qwen3-tts`

### Changed
- Style TTS now runs Qwen3-TTS 1.7B CustomVoice ONNX (ElBruno.QwenTTS port). `CharacterVoiceFactory` / `CharacterVoice` signatures are unchanged.
- `GenerateSpeechAsync` still defaults to 16 kHz; native vocoder output is 24 kHz and is resampled.

### Added
- Local 1.7B layout under `StreamingAssets/SparkTTS/Qwen3-1.7B/` (`QwenModelPaths`). No HuggingFace download.
- Model Deployment Tool category for the Qwen3 1.7B file list.

### Removed (this branch)
- Voice cloning via `CreateFromReference` (CustomVoice has no speaker encoder). Returns null.

## [0.1.0] - 2025-05-17

### Added
- Initial release of Spark TTS Unity package
- Basic text-to-speech and voice cloning functionality
- Sample scripts and example implementation
- Support for Unity 6000.0.46f1 and newer 