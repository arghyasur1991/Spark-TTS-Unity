// Ported from ElBruno.QwenTTS (MIT) — https://github.com/elbruno/ElBruno.QwenTTS
// Qwen3-TTS ONNX inference. Public SparkTTS CharacterVoice APIs stay unchanged.

using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using System.Text;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// Loads NumPy .npy files (format v1.0/v2.0). Float payloads can stream
    /// into AllocHGlobal so 1.7B text_embedding.npy is not doubled in managed RAM.
    /// </summary>
    internal static class NpyReader
    {
        public static NativeFloatBuffer ReadNative2D(string path)
        {
            var (dtype, shape, fs) = OpenNpy(path);
            using (fs)
            {
                if (dtype != "<f4")
                    throw new InvalidDataException($"Expected float32 (<f4), got {dtype}");
                if (shape.Length != 2)
                    throw new InvalidDataException($"Expected 2D array, got {shape.Length}D");
                var buf = NativeFloatBuffer.Alloc(shape[0], shape[1]);
                SparkTTS.Qwen.IOUtil.ReadExact(fs, buf.Ptr, buf.ByteCount);
                return buf;
            }
        }

        public static NativeFloatBuffer ReadNative1D(string path)
        {
            var (dtype, shape, fs) = OpenNpy(path);
            using (fs)
            {
                if (dtype != "<f4")
                    throw new InvalidDataException($"Expected float32 (<f4), got {dtype}");
                if (shape.Length != 1)
                    throw new InvalidDataException($"Expected 1D array, got {shape.Length}D");
                var buf = NativeFloatBuffer.Alloc(shape[0], 1);
                SparkTTS.Qwen.IOUtil.ReadExact(fs, buf.Ptr, buf.ByteCount);
                return buf;
            }
        }

        static (string dtype, int[] shape, FileStream fs) OpenNpy(string path)
        {
            var fileInfo = new FileInfo(path);
            const long maxNpySize = 2_000_000_000;
            if (fileInfo.Length > maxNpySize)
                throw new InvalidOperationException($"NPY file too large ({fileInfo.Length / 1e6:F2} MB). Maximum allowed: {maxNpySize / 1e6:F2} MB.");

            var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
            try
            {
                Span<byte> magic = stackalloc byte[6];
                byte[] expected = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };
                SparkTTS.Qwen.IOUtil.ReadExact(fs, magic);
                if (!MemoryExtensions.SequenceEqual<byte>(magic, expected))
                    throw new InvalidDataException("Not a valid NPY file (bad magic)");

                int major = fs.ReadByte();
                int minor = fs.ReadByte();
                if (major != 1 && major != 2)
                    throw new NotSupportedException($"Unsupported NPY version {major}.{minor}");

                int headerLen;
                if (major == 1)
                {
                    Span<byte> lenBytes = stackalloc byte[2];
                    SparkTTS.Qwen.IOUtil.ReadExact(fs, lenBytes);
                    headerLen = BinaryPrimitives.ReadUInt16LittleEndian(lenBytes);
                }
                else
                {
                    Span<byte> lenBytes = stackalloc byte[4];
                    SparkTTS.Qwen.IOUtil.ReadExact(fs, lenBytes);
                    headerLen = (int)BinaryPrimitives.ReadUInt32LittleEndian(lenBytes);
                }

                var headerBytes = new byte[headerLen];
                SparkTTS.Qwen.IOUtil.ReadExact(fs, headerBytes);
                var header = Encoding.ASCII.GetString(headerBytes).Trim();
                var dtype = ExtractValue(header, "'descr':");
                var shapeStr = ExtractValue(header, "'shape':");
                var shape = ParseShape(shapeStr);
                return (dtype, shape, fs);
            }
            catch
            {
                fs.Dispose();
                throw;
            }
        }

        static string ExtractValue(string header, string key)
        {
            var idx = header.IndexOf(key);
            if (idx < 0)
                throw new InvalidDataException($"Missing key {key} in NPY header");

            idx += key.Length;
            while (idx < header.Length && char.IsWhiteSpace(header[idx]))
                idx++;

            if (header[idx] == '\'')
            {
                int start = idx + 1;
                int end = header.IndexOf('\'', start);
                return header[start..end];
            }
            if (header[idx] == '(')
            {
                int start = idx;
                int end = header.IndexOf(')', start);
                return header[start..(end + 1)];
            }

            int begin = idx;
            while (idx < header.Length && !char.IsWhiteSpace(header[idx]) && header[idx] != ',')
                idx++;
            return header[begin..idx];
        }

        static int[] ParseShape(string shapeStr)
        {
            var inner = shapeStr.Trim('(', ')').Trim();
            var parts = inner.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            return parts.Select(p => int.Parse(p.Trim())).ToArray();
        }
    }
}
