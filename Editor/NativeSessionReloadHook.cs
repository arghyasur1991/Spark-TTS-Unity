using UnityEditor;
using SparkTTS;

namespace SparkTTS.Editor
{
    /// <summary>
    /// Stash/restore native ONNX sessions across script domain reload.
    /// Do not construct InferenceSession from afterAssemblyReload — Play enter
    /// can start a second reload while that ctor is still running (editor SIGSEGV).
    /// Open CustomVoice graphs once the editor is idle, or from WaitForModelsLoadedAsync.
    /// </summary>
    [InitializeOnLoad]
    static class NativeSessionReloadHook
    {
        static NativeSessionReloadHook()
        {
            AssemblyReloadEvents.beforeAssemblyReload += CharacterVoiceFactory.StashNativeForReload;
            AssemblyReloadEvents.afterAssemblyReload += CharacterVoiceFactory.TryRestoreNativeAfterReload;
            EditorApplication.playModeStateChanged += OnPlayMode;
            CharacterVoiceFactory.TryRestoreNativeAfterReload();
            EditorApplication.delayCall += TryPreloadWhenIdle;
        }

        static void OnPlayMode(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.EnteredPlayMode ||
                change == PlayModeStateChange.EnteredEditMode)
                EditorApplication.delayCall += TryPreloadWhenIdle;
        }

        static void TryPreloadWhenIdle()
        {
            if (EditorApplication.isCompiling || EditorApplication.isUpdating)
            {
                EditorApplication.delayCall += TryPreloadWhenIdle;
                return;
            }

            // Play enter sets isPlayingOrWillChangePlaymode before the domain
            // reload finishes. Starting new InferenceSession here is the crash.
            if (EditorApplication.isPlayingOrWillChangePlaymode && !EditorApplication.isPlaying)
                return;

            if (!CharacterVoiceFactory.IsReady || !CharacterVoiceFactory.HasEngine)
                return;

            _ = CharacterVoiceFactory.WaitForModelsLoadedAsync();
        }
    }
}
