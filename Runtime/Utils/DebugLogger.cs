using UnityEngine;

namespace SparkTTS.Utils
{
    public class DebugLogger
    {
        public enum LogLevel
        {
            None,
            Error,
            Warning,
            Info,
            Debug
        }

        public LogLevel Level { get => _logLevel; set => _logLevel = value; }
        private LogLevel _logLevel = LogLevel.Warning;
        
        public DebugLogger(LogLevel logLevel = LogLevel.Warning)
        {
            _logLevel = logLevel;
        }
        
        /// <summary>
        /// Gets or sets whether debug logging is enabled
        /// </summary>
        public bool IsEnabled
        {
            get { return _logLevel >= LogLevel.Debug; }
        }

        public void Log(string message)
        {
            if (_logLevel >= LogLevel.Debug)
            {
                Debug.Log(message);
            }
        }

        public void LogWarning(string message)
        {
            if (_logLevel >= LogLevel.Warning)
            {
                Debug.LogWarning(message);
            }
        }

        public void LogError(string message)
        {
            if (_logLevel >= LogLevel.Error)
            {
                Debug.LogError(message);
            }
        }
    }
}