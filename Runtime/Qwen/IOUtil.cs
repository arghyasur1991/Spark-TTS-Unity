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
            var tmp = new byte[destination.Length];
            ReadExact(stream, tmp);
            tmp.CopyTo(destination);
        }
    }
}
