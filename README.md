# Spark-TTS-Unity

Unity package for using Spark-TTS on-device models. This is a C# port of [https://github.com/SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) by SparkAudio team and uses converted ONNX models instead of the PyTorch models in the original repo.

## What is Spark-TTS?

Spark-TTS is an open-source text-to-speech system capable of generating high-quality, natural-sounding speech directly on your device. This Unity package makes it easy to incorporate this technology into your Unity projects.

## Key Features:

* 🎮 **Unity-Native Integration**: Simple API designed specifically for Unity
* 🔊 **Voice Styling**: Customize gender, pitch, and speed parameters
* 🎭 **Voice Cloning**: Clone voices from reference audio clips
* 💻 **Runs Offline**: All processing happens on-device
* ⚡ **Optimized Performance**: Caching system for faster repeated generation

## Perfect For:

* Indie games with lots of dialogue
* Accessibility features
* Prototyping narrative content
* Dynamic content generation
* Interactive tutorials

## Installation

### Using Unity Package Manager (Recommended)

1. Open your Unity project
2. Open the Package Manager (Window > Package Manager)
3. Click the "+" button in the top-left corner
4. Select "Add package from git URL..."
5. Enter the repository URL: `https://github.com/arghyasur1991/Spark-TTS-Unity.git`
6. Click "Add"

### Manual Installation

1. Clone this repository
2. Copy the contents into your Unity project's Assets folder

## Dependencies

This package requires the following Unity packages:
- com.github.asus4.onnxruntime (0.4.0)
- com.github.asus4.onnxruntime-extensions (0.4.0)
- com.unity.nuget.newtonsoft-json (3.2.1)

### Setting up Package Dependencies

Some dependencies require additional scoped registry configuration. Add the following to your project's `Packages/manifest.json` file:

```json
{
  "scopedRegistries": [
    {
      "name": "NPM",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ],
  "dependencies": {
    "com.genesis.sparktts.unity": "file:/path/to/Spark-TTS-Unity",
    // ... other dependencies
  }
}
```

**Note**: Replace `/path/to/Spark-TTS-Unity` with the actual path to your Spark-TTS-Unity package folder.

## Features

- On-device text-to-speech synthesis
- Voice styling with adjustable gender, pitch, and speed
- Voice cloning from reference audio clips
- Optimized for runtime performance
- Simple API for integration into games and applications

## Usage

### Basic Voice Generation Using Styles

```csharp
using UnityEngine;
using SparkTTS;
using System.Threading.Tasks;

public class TTSExample : MonoBehaviour
{
    private CharacterVoice characterVoice;
    private AudioSource audioSource;

    async void Start()
    {
        // Initialize the TTS system (optional - sets log levels)
        CharacterVoiceFactory.Initialize(DebugLogger.LogLevel.Warning);
        
        // Get reference to AudioSource
        audioSource = GetComponent<AudioSource>();
        
        // Get the singleton instance of the factory
        var voiceFactory = CharacterVoiceFactory.Instance;
        
        // Create a styled voice (gender: male/female, pitch: very_low/low/moderate/high/very_high, speed: very_low/low/moderate/high/very_high)
        characterVoice = await voiceFactory.CreateFromStyleAsync(
            gender: "female",
            pitch: "moderate", 
            speed: "moderate",
            referenceText: "Hello, I am a character voice sample."
        );
        
        // Generate and play speech
        if (characterVoice != null)
        {
            await GenerateAndPlaySpeech("Hello, welcome to my game! I'm an on-device TTS voice.");
        }
    }
    
    public async Task GenerateAndPlaySpeech(string text)
    {
        if (characterVoice == null) return;
        
        AudioClip generatedClip = await characterVoice.GenerateSpeechAsync(text);
        
        if (generatedClip != null && audioSource != null)
        {
            audioSource.clip = generatedClip;
            audioSource.Play();
        }
    }
    
    private void OnDestroy()
    {
        // Clean up resources
        characterVoice?.Dispose();
        // Note: Don't dispose the factory instance as it's a singleton
    }
}
```

### Voice Cloning from Reference Audio

```csharp
using UnityEngine;
using SparkTTS;
using System.Threading.Tasks;

public class VoiceCloningExample : MonoBehaviour
{
    public AudioClip referenceClip; // Assign in inspector
    private CharacterVoice clonedVoice;
    private AudioSource audioSource;
    
    async void Start()
    {
        // Initialize the TTS system (optional - sets log levels)
        CharacterVoiceFactory.Initialize(DebugLogger.LogLevel.Warning);
        
        audioSource = GetComponent<AudioSource>();
        
        // Get the singleton instance of the factory
        var voiceFactory = CharacterVoiceFactory.Instance;
        
        if (referenceClip != null)
        {
            // Clone voice from reference audio (note: this is async)
            clonedVoice = await voiceFactory.CreateFromReferenceAsync(referenceClip);
            Debug.Log("Voice cloned from reference audio");
        }
    }
    
    public async void SpeakText(string text)
    {
        if (clonedVoice == null) return;
        
        AudioClip generatedClip = await clonedVoice.GenerateSpeechAsync(text);
        
        if (generatedClip != null && audioSource != null)
        {
            audioSource.clip = generatedClip;
            audioSource.Play();
        }
    }
    
    private void OnDestroy()
    {
        clonedVoice?.Dispose();
        // Note: Don't dispose the factory instance as it's a singleton
    }
}
```

## Model Setup

This package requires Spark-TTS models in the following location:

```
Assets/StreamingAssets/SparkTTS/
  ├── bicodec_encoder_quantizer.onnx
  ├── bicodec_vocoder.onnx
  ├── mel_spectrogram.onnx
  ├── speaker_encoder_tokenizer.onnx
  ├── wav2vec2_model.onnx
  └── LLM/
      ├── model.onnx
      ├── model.onnx_data
      ├── config.json
      ├── generation_config.json
      ├── added_tokens.json
      ├── tokenizer.json
      ├── tokenizer_config.json
      ├── special_tokens_map.json
      ├── vocab.json
      └── merges.txt
```

You can obtain these models by using the `export_all_sparktts_onnx.py` script from the [Spark-TTS repository](https://github.com/arghyasur1991/Spark-TTS). This script converts the original PyTorch models to ONNX format for use in Unity.

### Exporting Models

1. Clone the Spark-TTS repository: `git clone https://github.com/arghyasur1991/Spark-TTS.git`
2. Install the required dependencies
3. Run the export script: `python export_all_sparktts_onnx.py`
4. Copy the `SparkTTS` folder with exported ONNX models inside `onnx_models_exported` to your Unity project's StreamingAssets folder

### Pre-Exported ONNX Models

As an alternative to running the export script, you can download pre-exported ONNX models from this [Google Drive link](https://drive.google.com/file/d/1YXj81ApcEasY17a8Zj9RqTpvn4s1UKk7/view?usp=sharing).

1. Download the ZIP file from the link
2. Extract the contents
3. Copy the extracted `SparkTTS` folder with models to your Unity project's `Assets/StreamingAssets/` directory

## Sample Demo

The package includes a CharacterVoiceDemo that demonstrates:
- Creating male and female voices with adjustable pitch and speed
- Generating speech from text input
- Playing generated audio

To use the demo:
1. Import the package
2. Add the demo prefab to your scene
3. Set up the required models in StreamingAssets
4. Run the scene and interact with the UI

## Requirements

- Unity 6000.0.46f1 or newer
- Supported platforms: Windows, macOS
- Minimum 32GB RAM for runtime operation
- Storage space for TTS models (~3GB)

## License

This Unity package is provided under the [LICENSE](LICENSE) file terms.

### Model License

The original Spark-TTS models and code are licensed under the [Apache License 2.0](https://github.com/SparkAudio/Spark-TTS/blob/main/LICENSE).

```
Copyright 2025 The Spark-TTS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.
- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.
- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. 
