# Third-party notices

## ElBruno.QwenTTS

Portions of `Runtime/Qwen/` are derived from [ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS)
(C) 2026 Bruno Capuano, MIT License.

The original license text:

MIT License

Copyright (c) 2026 Bruno Capuano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING WITHOUT LIMITATION THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The in-repo GPT-2 byte-level BPE (`TextTokenizer`) replaces
`Microsoft.ML.Tokenizers`, which is not available in this Unity package.
It follows the HuggingFace GPT-2 / Qwen2 tokenizer algorithm used by that
project.

The mel spectrogram used by 1.7B Base cloning (`Runtime/Qwen/Audio/MelSpectrogram.cs`)
is also from that project's VoiceCloning module.

## zukky/Qwen3-TTS-ONNX-DLL

The 1.7B Base ONNX graph layout and inference order follow
[zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL)
(Apache-2.0), itself derived from [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).
This package does not redistribute those weights or the Windows `qwen3_tts_rust.dll`.
C# reimplements the Python sample pipeline for Mac/Android.

