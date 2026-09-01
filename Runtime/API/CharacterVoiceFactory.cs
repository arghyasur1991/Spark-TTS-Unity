using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;
using SparkTTS.Qwen.Models;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS
{
    using Models;
    using Qwen;
    using Utils;
    /// <summary>
    /// Factory class for creating CharacterVoice objects using either voice cloning or style-based generation.
    /// </summary>
    public class CharacterVoiceFactory : IDisposable
    {
        public static CharacterVoiceFactory Instance { get; private set; } = new();

        public bool LogTiming
        {
            get => SparkTTS.Core.SparkTTS.LogTiming;
            set => SparkTTS.Core.SparkTTS.LogTiming = value;
        }

        /// <summary>
        /// True after embeddings/tokenizers are constructed (ONNX sessions may still be deferred),
        /// or editor keep-alive has wrapped native sessions / embedding buffers waiting to be adopted.
        /// </summary>
        public static bool HasEngine
        {
            get
            {
                if (Instance._engine != null && !Instance._disposed)
                    return true;
#if UNITY_EDITOR
                return NativeSessionKeepAlive.HasEngineKeepAlive;
#else
                return false;
#endif
            }
        }

        static bool _keepAcrossReload;

        /// <summary>
        /// When true, editor domain reload detaches native ONNX sessions and
        /// AllocHGlobal embedding matrices instead of releasing them so the next
        /// domain can wrap them. Set from the host.
        /// Persisted in a per-process file so stash still runs if the static was reset.
        /// </summary>
        public static bool KeepNativeSessionsAcrossReload
        {
            get
            {
#if UNITY_EDITOR
                return _keepAcrossReload || NativeSessionKeepAlive.KeepRequested;
#else
                return _keepAcrossReload;
#endif
            }
            set
            {
                _keepAcrossReload = value;
#if UNITY_EDITOR
                NativeSessionKeepAlive.SetKeepRequested(value);
#endif
            }
        }

        /// <summary>
        /// Gets whether the factory is initialized and ready for use.
        /// </summary>
        public static bool IsReady => Instance._initialized && !Instance._disposed;

        private QwenTtsEngine _engine;
        private Task _engineTask;
        private ExecutionProvider _executionProvider = ExecutionProvider.CPU;
        private bool _disposed;
        private bool _initialized;

        internal CharacterVoiceFactory()
        {
            bool style = QwenModelPaths.IsCustomVoicePresent();
            bool clone = QwenModelPaths.IsBasePresent();
            _initialized = style || clone;
            if (!_initialized)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Qwen3-TTS files not found. Style needs " +
                    $"{QwenModelPaths.Root}; clone needs {QwenModelPaths.BaseRoot}. See QWEN3.md.");
            }
            else if (!style)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] VoiceDesign 1.7B missing — CreateFromStyleAsync unavailable. " +
                    "Place exported ONNX at " + QwenModelPaths.Root);
            }
            else if (!clone)
            {
                TTSLogger.LogWarning(
                    "[CharacterVoiceFactory] Base 1.7B missing — CreateFromReference unavailable. " +
                    "Place zukky onnx_kv + tokenizer at " + QwenModelPaths.BaseRoot);
            }
        }

        /// <summary>
        /// Initializes or re-initializes the CharacterVoiceFactory with the specified settings.
        /// </summary>
        public static void Initialize(LogLevel logLevel, MemoryUsage memoryUsage, ExecutionProvider executionProvider = ExecutionProvider.CPU)
        {
            TTSLogger.LogLevel = logLevel;
            ORTModel.InitializeEnvironment(logLevel);
            ORTModel.SetMemoryUsage(memoryUsage);

            Instance._executionProvider = executionProvider;
            Instance._disposed = false;
            bool style = QwenModelPaths.IsCustomVoicePresent();
            bool clone = QwenModelPaths.IsBasePresent();
            Instance._initialized = style || clone;
            if (!Instance._initialized)
            {
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] Qwen3-TTS files missing. Style: " +
                    $"{QwenModelPaths.Root}; clone: {QwenModelPaths.BaseRoot}");
                return;
            }

            TTSLogger.Log(
                $"[CharacterVoiceFactory] Initialized Qwen3-TTS (style={style}, clone={clone}), " +
                $"MemoryUsage: {memoryUsage}, ExecutionProvider: {executionProvider}");
        }

        /// <summary>
        /// Waits for models to be ready. Constructs the engine (tokenizer / embeddings).
        /// Large ONNX sessions stay deferred until the first generate or clone extract.
        /// </summary>
        public static async Task WaitForModelsLoadedAsync()
        {
            if (!Instance._initialized)
            {
                throw new InvalidOperationException("CharacterVoiceFactory is not initialized");
            }

            TTSLogger.Log("[CharacterVoiceFactory] Loading Qwen3-TTS...");
            await Instance.EnsureEngineAsync();
            TTSLogger.Log("[CharacterVoiceFactory] Qwen3-TTS ready");
        }

        /// <summary>
        /// Creates a character voice using style-based generation with specified voice parameters.
        /// </summary>
        public async Task<CharacterVoice> CreateFromStyleAsync(
            string gender, string pitch, string speed, string referenceText = "I am a character voice",
            string instruct = null)
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            if (!QwenModelPaths.IsCustomVoicePresent())
            {
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] CreateFromStyleAsync needs VoiceDesign ONNX at " +
                    QwenModelPaths.Root);
                return null;
            }

            if (string.IsNullOrEmpty(gender))
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Gender parameter is required for style-based voices.");
                return null;
            }

            try
            {
                await EnsureEngineAsync();
                CharacterVoice voice = new(
                    _engine,
                    gender: gender.ToLower(),
                    pitch: pitch?.ToLower() ?? "moderate",
                    speed: speed?.ToLower() ?? "moderate",
                    referenceText: referenceText,
                    instruct: instruct
                );

                await voice.GenerateVoiceAsync(referenceText);
                return voice;
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception creating voice from style: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        public async Task<CharacterVoice> CreateFromFolderAsync(string voiceFolder)
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            try
            {
                await EnsureEngineAsync();
                string configPath = Path.Combine(voiceFolder, "voice_config.json");
                if (File.Exists(configPath))
                {
                    var voiceConfig = JsonConvert.DeserializeObject<VoiceConfig>(
                        File.ReadAllText(configPath));
                    if (voiceConfig != null && voiceConfig.clone)
                    {
                        if (!QwenModelPaths.IsBasePresent())
                        {
                            TTSLogger.LogError(
                                "[CharacterVoiceFactory] Folder is a cloned voice but Base ONNX is missing at " +
                                QwenModelPaths.BaseRoot);
                            return null;
                        }

                        string audioFilePath = Path.Combine(
                            voiceFolder, voiceConfig.audioFile ?? "sample.wav");
                        if (!File.Exists(audioFilePath))
                        {
                            TTSLogger.LogError(
                                "[CharacterVoiceFactory] Clone folder has no sample wav: " + audioFilePath);
                            return null;
                        }

                        var clip = await AudioLoaderService.LoadAudioClipAsync(audioFilePath);
                        TTSLogger.LogVerbose(
                            "[CharacterVoiceFactory] Reloading cloned voice from " + audioFilePath);
                        return CreateFromReference(clip);
                    }
                }

                if (!QwenModelPaths.IsCustomVoicePresent())
                {
                    TTSLogger.LogError(
                        "[CharacterVoiceFactory] CreateFromFolderAsync needs VoiceDesign at " +
                        QwenModelPaths.Root);
                    return null;
                }

                CharacterVoice voice = new(_engine);
                await voice.LoadVoiceAsync(voiceFolder);
                return voice;
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception creating voice from folder: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        public CharacterVoice CreateFromReference(AudioClip referenceClip)
        {
            if (!_initialized || _disposed)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Factory is not initialized or has been disposed.");
                return null;
            }

            if (!QwenModelPaths.IsBasePresent())
            {
                var missing = QwenModelPaths.GetMissingBaseFiles();
                TTSLogger.LogError(
                    "[CharacterVoiceFactory] Voice cloning needs Qwen3-TTS 1.7B Base ONNX at " +
                    $"{QwenModelPaths.BaseRoot} (missing {missing.Count} file(s)). " +
                    "Source: https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL — copy onnx_kv/*.onnx " +
                    "and models/Qwen3-TTS-12Hz-1.7B-Base/{config.json,vocab.json,merges.txt}. " +
                    "Do not use qwen3_tts_rust.dll (Windows-only).");
                return null;
            }

            if (referenceClip == null)
            {
                TTSLogger.LogError("[CharacterVoiceFactory] Reference clip is null.");
                return null;
            }

            try
            {
                EnsureEngineAsync().GetAwaiter().GetResult();
                float[] samples = QwenTtsEngine.ClipToMono24k(referenceClip);
                float[] embedding = _engine.ExtractSpeakerEmbedding(samples);
                return new CharacterVoice(_engine, referenceClip, embedding);
            }
            catch (Exception e)
            {
                TTSLogger.LogError($"[CharacterVoiceFactory] Exception cloning from reference: {e.Message}\n{e.StackTrace}");
                return null;
            }
        }

        private Task EnsureEngineAsync()
        {
            if (_engine != null)
                return Task.CompletedTask;
            if (_engineTask != null)
            {
                if (!_engineTask.IsFaulted && !_engineTask.IsCanceled)
                    return _engineTask;
                _engineTask = null;
            }

            var ep = _executionProvider;
            _engineTask = BackgroundWork.Run(() =>
            {
                if (_engine != null)
                    return;
#if UNITY_EDITOR
                if (TryApplyPendingSessions(ep))
                    return;
#endif
                _engine = new QwenTtsEngine(ep);
                TTSLogger.LogVerbose(
                    $"[CharacterVoiceFactory] Qwen engine ready (EP {ep}, " +
                    $"style={_engine.HasCustomVoice}, clone={_engine.HasClone})");
            });
            return _engineTask;
        }

#if UNITY_EDITOR
        static bool TryApplyPendingSessions(ExecutionProvider ep)
        {
            if (Instance._engine != null)
                return true;
            bool hasSessions = NativeSessionKeepAlive.HasPending;
            bool hasEmb = NativeSessionKeepAlive.HasPendingEmbeddings;
            if (!hasSessions && !hasEmb)
                return false;

            var pending = hasSessions ? NativeSessionKeepAlive.TakePending() : null;
            Instance._disposed = false;
            Instance._engine = new QwenTtsEngine(ep);

            int offered = pending?.Count ?? 0;
            int leftover = 0;
            if (pending != null && pending.Count > 0)
            {
                Instance._engine.AdoptNativeSessions(pending);
                leftover = pending.Count;
                foreach (var leftoverSession in pending.Values)
                {
                    try { leftoverSession.Dispose(); }
                    catch (Exception ex) { TTSLogger.LogWarning("[CharacterVoiceFactory] leftover session: " + ex.Message); }
                }
            }

            if (offered > 0)
            {
                TTSLogger.Log(
                    $"[CharacterVoiceFactory] Adopted {offered - leftover}/{offered} ONNX sessions after domain reload");
            }
            return true;
        }
#endif

        /// <summary>
        /// Drops loaded ONNX sessions and embeddings. Next create/speak reconstructs them.
        /// Does not mark the factory permanently disposed.
        /// </summary>
        public static void UnloadModels()
        {
            Instance._engine?.Dispose();
            Instance._engine = null;
            Instance._engineTask = null;
            Instance._disposed = false;
#if UNITY_EDITOR
            NativeSessionKeepAlive.DisposePending();
#endif
            TTSLogger.Log("[CharacterVoiceFactory] Unloaded Qwen3-TTS engine");
        }

#if UNITY_EDITOR
        /// <summary>
        /// Detach native ONNX handles so domain reload does not OrtReleaseSession.
        /// No-op unless <see cref="KeepNativeSessionsAcrossReload"/> is set.
        /// </summary>
        public static void StashNativeForReload()
        {
            if (!KeepNativeSessionsAcrossReload)
                return;

            NativeEmbSlot[] embeddings = null;
            var sessions = new List<(string key, IntPtr handle)>();
            if (Instance._engine != null)
            {
                embeddings = Instance._engine.DetachEmbeddings();
                var models = new List<ORTModel>();
                Instance._engine.CollectOnnxModels(models);
                foreach (var model in models)
                {
                    if (model.TryStealNativeSession(out var key, out var handle) && handle != IntPtr.Zero)
                        sessions.Add((key, handle));
                }
            }
            else
            {
                embeddings = NativeSessionKeepAlive.TakePendingEmbeddings();
                var pending = NativeSessionKeepAlive.TakePending();
                if (pending != null)
                {
                    foreach (var kv in pending)
                    {
                        var handle = NativeSessionKeepAlive.DetachSessionHandle(kv.Value);
                        if (handle != IntPtr.Zero)
                            sessions.Add((kv.Key, handle));
                    }
                }
            }

            if (sessions.Count > 0)
            {
                var env = NativeSessionKeepAlive.DetachOrtEnv();
                NativeSessionKeepAlive.Stash(env, sessions);
            }

            if (embeddings != null)
                NativeSessionKeepAlive.StashEmbeddings(embeddings);
        }

        /// <summary>
        /// Wrap stashed native ONNX sessions and AllocHGlobal embedding matrices.
        /// Tokenizer + config.json still come from disk (small). Do not re-read npy.
        /// </summary>
        public static void TryRestoreNativeAfterReload()
        {
            NativeSessionKeepAlive.TryRestore();
            NativeSessionKeepAlive.TryRestoreEmbeddings();
            if (!NativeSessionKeepAlive.HasEngineKeepAlive && Instance._engine == null)
                return;
            TTSLogger.Log(
                "[CharacterVoiceFactory] Native ONNX/embeddings restored; wrapping off the main thread");
            Instance.EnsureEngineAsync();
        }
#endif

        public void Dispose()
        {
            if (!_disposed)
            {
                _engine?.Dispose();
                _engine = null;
                _engineTask = null;
                _initialized = false;
                _disposed = true;
            }

            GC.SuppressFinalize(this);
        }

        ~CharacterVoiceFactory()
        {
#if UNITY_EDITOR
            if (KeepNativeSessionsAcrossReload)
            {
                _engine = null;
                return;
            }
#endif
            Dispose();
        }
    }
}
