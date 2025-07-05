using UnityEngine;

namespace SparkTTS.Utils
{
    public enum LogLevel
    {
        VERBOSE,
        INFO,
        WARNING,
        ERROR,
    }
    public static class Logger
    {

        public static LogLevel LogLevel { get; set; } = LogLevel.INFO;

        public static void LogVerbose(string message)
        {
            if (LogLevel <= LogLevel.VERBOSE)
            {
                Debug.Log(message);
            }
        }

        public static void Log(string message)
        {
            if (LogLevel <= LogLevel.INFO)
            {
                Debug.Log(message);
            }
        }

        public static void LogWarning(string message)
        {
            if (LogLevel <= LogLevel.WARNING)
            {
                Debug.LogWarning(message);
            }
        }

        public static void LogError(string message)
        {
            if (LogLevel <= LogLevel.ERROR)
            {
                Debug.LogError(message);
            }
        }
    }
}