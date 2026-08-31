using System;
using System.Runtime.InteropServices;

namespace SparkTTS.Qwen.Models
{
    /// <summary>
    /// Row-major float32 matrix in AllocHGlobal. Survives an editor domain
    /// reload if we skip FreeHGlobal (same idea as OrtSession keep-alive).
    /// </summary>
    internal sealed class NativeFloatBuffer
    {
        public IntPtr Ptr;
        public int Rows;
        public int Cols;

        public int Count => Rows * Cols;
        public int ByteCount => Count * 4;
        public bool IsEmpty => Ptr == IntPtr.Zero || Count == 0;

        public static NativeFloatBuffer Alloc(int rows, int cols)
        {
            if (rows < 0 || cols < 0)
                throw new ArgumentOutOfRangeException();
            long bytes = (long)rows * cols * 4;
            if (bytes > int.MaxValue)
                throw new InvalidOperationException($"Native float buffer too large ({bytes} bytes).");
            var ptr = bytes == 0 ? IntPtr.Zero : Marshal.AllocHGlobal((int)bytes);
            return new NativeFloatBuffer { Ptr = ptr, Rows = rows, Cols = cols };
        }

        public static NativeFloatBuffer Wrap(IntPtr ptr, int rows, int cols)
        {
            return new NativeFloatBuffer { Ptr = ptr, Rows = rows, Cols = cols };
        }

        public void Free()
        {
            if (Ptr == IntPtr.Zero)
                return;
            Marshal.FreeHGlobal(Ptr);
            Ptr = IntPtr.Zero;
            Rows = 0;
            Cols = 0;
        }

        public unsafe void CopyRow(int row, Span<float> dst)
        {
            if ((uint)row >= (uint)Rows)
                throw new ArgumentOutOfRangeException(nameof(row));
            if (dst.Length < Cols)
                throw new ArgumentException("Destination is shorter than a row.");
            float* src = (float*)Ptr + (long)row * Cols;
            int bytes = Cols * 4;
            fixed (float* d = dst)
                Buffer.MemoryCopy(src, d, (long)dst.Length * 4, bytes);
        }

        public unsafe void CopyTo(Span<float> dst)
        {
            if (dst.Length < Count)
                throw new ArgumentException("Destination is shorter than the buffer.");
            int bytes = ByteCount;
            fixed (float* d = dst)
                Buffer.MemoryCopy((float*)Ptr, d, (long)dst.Length * 4, bytes);
        }
    }

    internal struct NativeEmbSlot
    {
        public int Rows;
        public int Cols;
        public IntPtr Ptr;
    }
}
