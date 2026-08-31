using System;
using System.Threading;
using System.Threading.Tasks;

namespace SparkTTS.Utils
{
    /// <summary>
    /// Thread-pool work that does not use Unity's synchronization context.
    /// <c>Task.Run</c> / <c>new Task().Start()</c> flow ExecutionContext, so
    /// <c>new InferenceSession</c> can marshal onto the editor thread and
    /// deadlock with <c>GetResult</c> / coroutine pumps.
    /// </summary>
    internal static class BackgroundWork
    {
        public static Task Run(Action action)
        {
            return Start(() =>
            {
                action();
                return true;
            });
        }

        public static Task<T> Run<T>(Func<T> func)
        {
            return Start(func);
        }

        static Task<T> Start<T>(Func<T> func)
        {
            AsyncFlowControl flow = default;
            bool suppressed = false;
            if (!ExecutionContext.IsFlowSuppressed())
            {
                flow = ExecutionContext.SuppressFlow();
                suppressed = true;
            }

            try
            {
                return Task.Factory.StartNew(
                    func,
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning | TaskCreationOptions.DenyChildAttach,
                    TaskScheduler.Default);
            }
            finally
            {
                if (suppressed)
                    flow.Undo();
            }
        }
    }
}
