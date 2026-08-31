using System;
using System.Threading;
using System.Threading.Tasks;

namespace SparkTTS.Utils
{
    /// <summary>
    /// Thread-pool work that does not use Unity's synchronization context.
    /// <c>new Task().Start()</c> and <c>Task.Run</c> can still marshal onto the
    /// editor main thread via ExecutionContext; that is what made 5 GB ONNX
    /// session construct look like a main-thread hang.
    /// </summary>
    internal static class BackgroundWork
    {
        public static Task Run(Action action)
        {
            return Task.Factory.StartNew(
                action,
                CancellationToken.None,
                TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
                TaskScheduler.Default);
        }

        public static Task<T> Run<T>(Func<T> func)
        {
            return Task.Factory.StartNew(
                func,
                CancellationToken.None,
                TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
                TaskScheduler.Default);
        }
    }
}
