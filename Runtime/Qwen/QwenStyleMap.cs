using System.Collections.Generic;
using System.Text;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Maps Spark CharacterVoice style knobs onto Qwen3-TTS CustomVoice (preset speaker + instruct).
    /// 1.7B supports instruct; pitch/speed become instruction text rather than Spark global tokens.
    /// </summary>
    public static class QwenStyleMap
    {
        public const string DefaultLanguage = "auto";

        public static string SpeakerForGender(string gender)
        {
            if (string.IsNullOrEmpty(gender))
                return "ryan";
            switch (gender.Trim().ToLowerInvariant())
            {
                case "female":
                case "woman":
                case "f":
                    return "serena";
                default:
                    return "ryan";
            }
        }

        /// <summary>
        /// Builds an instruct string for 1.7B. Returns null when pitch and speed are moderate/default
        /// so the prompt stays a single assistant turn.
        /// </summary>
        public static string InstructFor(string pitch, string speed)
        {
            var parts = new List<string>();
            AppendPitch(parts, pitch);
            AppendSpeed(parts, speed);
            if (parts.Count == 0)
                return null;
            var sb = new StringBuilder();
            sb.Append("Speak ");
            sb.Append(string.Join(", ", parts));
            sb.Append('.');
            return sb.ToString();
        }

        private static void AppendPitch(List<string> parts, string pitch)
        {
            if (string.IsNullOrEmpty(pitch))
                return;
            switch (pitch.Trim().ToLowerInvariant())
            {
                case "very_low":
                case "low":
                    parts.Add("in a lower pitch");
                    break;
                case "high":
                case "very_high":
                    parts.Add("in a higher pitch");
                    break;
            }
        }

        private static void AppendSpeed(List<string> parts, string speed)
        {
            if (string.IsNullOrEmpty(speed))
                return;
            switch (speed.Trim().ToLowerInvariant())
            {
                case "very_low":
                case "low":
                    parts.Add("slowly");
                    break;
                case "high":
                case "very_high":
                    parts.Add("quickly");
                    break;
            }
        }
    }
}
