using System.Diagnostics;

namespace SparkTTS.Utils
{
    public static class TimerUtils
    {
        public static double GetDuration(long startTimeTicks, long endTimeTicks)
        {
            return ((double)(endTimeTicks - startTimeTicks)) / Stopwatch.Frequency * 1000;
        }

        public static void LogTiming(string message, long startTimeTicks, long endTimeTicks)
        {
            UnityEngine.Debug.Log($"[LLMModel.Timing] {message} time: {GetDuration(startTimeTicks, endTimeTicks)}ms");
        }
    }

    public class AggregatedTimer
    {
        private readonly string _name;
        private long _totalTimeTicks;
        private long _totalCalls;

        public AggregatedTimer(string name)
        {
            _name = name;
            _totalTimeTicks = 0;
            _totalCalls = 0;
        }

        public void AddTiming(long startTimeTicks, long endTimeTicks)
        {
            _totalTimeTicks += endTimeTicks - startTimeTicks;
            _totalCalls++;
        }

        public void Reset()
        {
            _totalTimeTicks = 0;
            _totalCalls = 0;
        }

        public void LogTiming()
        {
            var totalTimeMs = (double)_totalTimeTicks / Stopwatch.Frequency * 1000;
            UnityEngine.Debug.Log($"[SparkTTS.Timing] {_name} Total time: {totalTimeMs}ms, Total calls: {_totalCalls}, Avg time: {totalTimeMs / _totalCalls}ms");
            Reset();
        }
    }
}