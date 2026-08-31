// Ported helpers for ElBruno.QwenTTS (MIT) — Unity does not have Stream.ReadExactly.

using System;
using System.IO;

namespace SparkTTS.Qwen
{
    internal static class IOUtil
    {
        public static void ReadExact(Stream stream, byte[] buffer)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            ReadExact(stream, buffer, 0, buffer.Length);
        }

        public static void ReadExact(Stream stream, byte[] buffer, int offset, int count)
        {
            int remaining = count;
            int pos = offset;
            while (remaining > 0)
            {
                int n = stream.Read(buffer, pos, remaining);
                if (n <= 0)
                    throw new EndOfStreamException();
                pos += n;
                remaining -= n;
            }
        }

        public static void ReadExact(Stream stream, Span<byte> destination)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            while (!destination.IsEmpty)
            {
                int n = stream.Read(destination);
                if (n <= 0)
                    throw new EndOfStreamException();
                destination = destination.Slice(n);
            }
        }

        public static unsafe void ReadExact(Stream stream, IntPtr dest, int byteCount)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (dest == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dest));
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            byte* p = (byte*)dest;
            int remaining = byteCount;
            while (remaining > 0)
            {
                int n = stream.Read(new Span<byte>(p, remaining));
                if (n <= 0)
                    throw new EndOfStreamException();
                p += n;
                remaining -= n;
            }
        }
    }
}
