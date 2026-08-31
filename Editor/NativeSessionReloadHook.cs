using UnityEditor;
using SparkTTS;

namespace SparkTTS.Editor
{
    /// <summary>
    /// Stash/restore native ONNX sessions across script domain reload.
    /// </summary>
    [InitializeOnLoad]
    static class NativeSessionReloadHook
    {
        static NativeSessionReloadHook()
        {
            AssemblyReloadEvents.beforeAssemblyReload += CharacterVoiceFactory.StashNativeForReload;
            CharacterVoiceFactory.TryRestoreNativeAfterReload();
        }
    }
}
