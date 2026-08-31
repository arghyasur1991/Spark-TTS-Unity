#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using TTSLogger = SparkTTS.Utils.Logger;

namespace SparkTTS.Utils
{
    /// <summary>
    /// Keeps OrtEnv + InferenceSession native handles alive across an editor
    /// domain reload. Managed wrappers die with the AppDomain; the process-wide
    /// ONNX allocations do not, if we skip OrtReleaseSession / OrtReleaseEnv.
    /// </summary>
    internal static class NativeSessionKeepAlive
    {
        const int Magic = 0x514B4131; // QKA1
        const int MaxSessions = 24;

        static readonly FieldInfo SessionHandleField = typeof(InferenceSession).GetField(
            "_nativeHandle", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly FieldInfo SessionDisposedField = typeof(InferenceSession).GetField(
            "_disposed", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly MethodInfo InitWithHandle = typeof(InferenceSession).GetMethod(
            "InitWithSessionHandle", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly ConstructorInfo OrtEnvCtor = typeof(OrtEnv).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            null,
            new[] { typeof(IntPtr), typeof(OrtLoggingLevel) },
            null);
        static readonly FieldInfo OrtEnvInstanceField = typeof(OrtEnv).GetField(
            "_instance", BindingFlags.Static | BindingFlags.NonPublic);

        static Dictionary<string, InferenceSession> _pending;
        static bool _envInstalled;

        internal static bool HasPending => _pending != null && _pending.Count > 0;

        internal static void SetKeepRequested(bool value)
        {
            try
            {
                if (value)
                    File.WriteAllText(KeepFlagPath(), "1");
                else if (File.Exists(KeepFlagPath()))
                    File.Delete(KeepFlagPath());
            }
            catch (Exception ex)
            {
                TTSLogger.LogWarning("[SparkTTS] Keep-alive flag: " + ex.Message);
            }
        }

        internal static bool KeepRequested
        {
            get
            {
                try { return File.Exists(KeepFlagPath()); }
                catch { return false; }
            }
        }

        internal static Dictionary<string, InferenceSession> TakePending()
        {
            var p = _pending;
            _pending = null;
            return p;
        }

        internal static void DisposePending()
        {
            if (_pending == null)
                return;
            foreach (var s in _pending.Values)
            {
                try { s?.Dispose(); }
                catch (Exception ex) { TTSLogger.LogWarning("[SparkTTS] Keep-alive pending dispose: " + ex.Message); }
            }
            _pending = null;
        }

        internal static IntPtr DetachSessionHandle(InferenceSession session)
        {
            if (session == null || SessionHandleField == null)
                return IntPtr.Zero;
            var handle = (IntPtr)SessionHandleField.GetValue(session);
            SessionHandleField.SetValue(session, IntPtr.Zero);
            GC.SuppressFinalize(session);
            return handle;
        }

        internal static void Stash(IntPtr envHandle, List<(string key, IntPtr handle)> sessions)
        {
            ClearTokenFile();
            if (envHandle == IntPtr.Zero || sessions == null || sessions.Count == 0)
                return;
            if (sessions.Count > MaxSessions)
            {
                TTSLogger.LogError("[SparkTTS] Keep-alive: too many sessions to stash.");
                return;
            }

            int bytes = 4 + 4 + 8 + 8 + 4;
            var keysUtf8 = new byte[sessions.Count][];
            for (int i = 0; i < sessions.Count; i++)
            {
                keysUtf8[i] = Encoding.UTF8.GetBytes(sessions[i].key ?? "");
                bytes += 8 + 4 + keysUtf8[i].Length;
            }

            IntPtr blob = Marshal.AllocHGlobal(bytes);
            IntPtr p = blob;
            WriteI32(ref p, Magic);
            WriteI32(ref p, Process.GetCurrentProcess().Id);
            WriteI64(ref p, Process.GetCurrentProcess().StartTime.ToUniversalTime().Ticks);
            WriteI64(ref p, envHandle.ToInt64());
            WriteI32(ref p, sessions.Count);
            for (int i = 0; i < sessions.Count; i++)
            {
                WriteI64(ref p, sessions[i].handle.ToInt64());
                WriteI32(ref p, keysUtf8[i].Length);
                Marshal.Copy(keysUtf8[i], 0, p, keysUtf8[i].Length);
                p = IntPtr.Add(p, keysUtf8[i].Length);
            }

            File.WriteAllText(TokenPath(), blob.ToInt64().ToString());
            TTSLogger.Log($"[SparkTTS] Stashed {sessions.Count} ONNX session(s) across domain reload.");
            for (int i = 0; i < sessions.Count; i++)
                TTSLogger.LogVerbose("[SparkTTS] stash " + sessions[i].key);
        }

        internal static bool TryRestore()
        {
            if (_pending != null && _pending.Count > 0)
                return true;

            string path = TokenPath();
            if (!File.Exists(path))
                return false;
            string raw = File.ReadAllText(path).Trim();
            ClearTokenFile();
            if (string.IsNullOrEmpty(raw) || !long.TryParse(raw, out long addr) || addr == 0)
                return false;
            var blob = new IntPtr(addr);
            try
            {
                IntPtr p = blob;
                int magic = ReadI32(ref p);
                int pid = ReadI32(ref p);
                long startTicks = ReadI64(ref p);
                long envBits = ReadI64(ref p);
                int count = ReadI32(ref p);
                var proc = Process.GetCurrentProcess();
                if (magic != Magic || pid != proc.Id ||
                    startTicks != proc.StartTime.ToUniversalTime().Ticks ||
                    count < 1 || count > MaxSessions)
                {
                    TTSLogger.LogWarning("[SparkTTS] Keep-alive blob is stale; ignoring.");
                    return false;
                }

                var envHandle = new IntPtr(envBits);
                if (!InstallOrtEnv(envHandle))
                    return false;

                var pending = new Dictionary<string, InferenceSession>(count);
                for (int i = 0; i < count; i++)
                {
                    var sessionHandle = new IntPtr(ReadI64(ref p));
                    int keyLen = ReadI32(ref p);
                    if (keyLen < 0 || keyLen > 2048)
                        throw new InvalidOperationException("Keep-alive key length is invalid.");
                    var keyBytes = new byte[keyLen];
                    Marshal.Copy(p, keyBytes, 0, keyLen);
                    p = IntPtr.Add(p, keyLen);
                    string key = Encoding.UTF8.GetString(keyBytes);
                    var wrapped = WrapSession(sessionHandle);
                    if (wrapped == null)
                        throw new InvalidOperationException("Failed to wrap InferenceSession for " + key);
                    pending[key] = wrapped;
                }

                _pending = pending;
                TTSLogger.Log($"[SparkTTS] Restored {pending.Count} ONNX session(s) after domain reload.");
                return true;
            }
            catch (Exception ex)
            {
                TTSLogger.LogError("[SparkTTS] Keep-alive restore failed: " + ex.Message);
                return false;
            }
            finally
            {
                Marshal.FreeHGlobal(blob);
            }
        }

        static bool InstallOrtEnv(IntPtr envHandle)
        {
            if (_envInstalled && OrtEnv.IsCreated)
                return true;
            if (OrtEnvCtor == null || OrtEnvInstanceField == null)
            {
                TTSLogger.LogError("[SparkTTS] Keep-alive: OrtEnv reflection failed.");
                return false;
            }

            if (OrtEnv.IsCreated)
            {
                TTSLogger.LogError(
                    "[SparkTTS] OrtEnv already created before keep-alive restore; " +
                    "stashed sessions belong to a different env and will not be adopted.");
                return false;
            }

            var env = (OrtEnv)OrtEnvCtor.Invoke(new object[]
            {
                envHandle, OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            });
            OrtEnvInstanceField.SetValue(null, new Lazy<OrtEnv>(() => env));
            _envInstalled = true;
            TTSLogger.Log("[SparkTTS] Reattached OrtEnv after domain reload.");
            return true;
        }

        internal static IntPtr DetachOrtEnv()
        {
            if (!OrtEnv.IsCreated)
                return IntPtr.Zero;
            var env = OrtEnv.Instance();
            IntPtr handle = env.DangerousGetHandle();
            env.SetHandleAsInvalid();
            return handle;
        }

        static InferenceSession WrapSession(IntPtr nativeHandle)
        {
            if (nativeHandle == IntPtr.Zero || InitWithHandle == null)
                return null;
            var session = (InferenceSession)FormatterServices.GetUninitializedObject(typeof(InferenceSession));
            SessionDisposedField?.SetValue(session, false);
            InitWithHandle.Invoke(session, new object[] { nativeHandle });
            return session;
        }

        static string KeepFlagPath()
        {
            return Path.Combine(Path.GetTempPath(),
                "SparkTTS-QwenKeepAlive-" + Process.GetCurrentProcess().Id + ".keep");
        }

        static string TokenPath()
        {
            return Path.Combine(Path.GetTempPath(),
                "SparkTTS-QwenKeepAlive-" + Process.GetCurrentProcess().Id + ".ptr");
        }

        static void ClearTokenFile()
        {
            try
            {
                string path = TokenPath();
                if (File.Exists(path))
                    File.Delete(path);
            }
            catch { /* ignore */ }
        }

        static void WriteI32(ref IntPtr p, int v)
        {
            Marshal.WriteInt32(p, v);
            p = IntPtr.Add(p, 4);
        }

        static void WriteI64(ref IntPtr p, long v)
        {
            Marshal.WriteInt64(p, v);
            p = IntPtr.Add(p, 8);
        }

        static int ReadI32(ref IntPtr p)
        {
            int v = Marshal.ReadInt32(p);
            p = IntPtr.Add(p, 4);
            return v;
        }

        static long ReadI64(ref IntPtr p)
        {
            long v = Marshal.ReadInt64(p);
            p = IntPtr.Add(p, 8);
            return v;
        }
    }
}
#endif
