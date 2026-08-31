using System.Collections.Generic;
using System.Text;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Maps Spark CharacterVoice style knobs onto Qwen3-TTS VoiceDesign.
    /// The voice is a natural-language instruct string, not a preset speaker id.
    /// </summary>
    public static class QwenStyleMap
    {
        public const string DefaultLanguage = "english";

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
        /// VoiceDesign instruct. An explicit description wins; otherwise gender/pitch/speed
        /// are folded into one sentence so Call Studio knobs still do something.
        /// </summary>
        public static string VoiceDesignInstruct(string gender, string pitch, string speed, string instruct = null)
        {
            if (!string.IsNullOrWhiteSpace(instruct))
                return instruct.Trim();

            var parts = new List<string>();
            if (!string.IsNullOrEmpty(gender))
            {
                switch (gender.Trim().ToLowerInvariant())
                {
                    case "female":
                    case "woman":
                    case "f":
                        parts.Add("a female speaker");
                        break;
                    default:
                        parts.Add("a male speaker");
                        break;
                }
            }
            AppendPitch(parts, pitch);
            AppendSpeed(parts, speed);
            if (parts.Count == 0)
                return "A natural conversational speaking voice.";
            var sb = new StringBuilder();
            sb.Append("Speak as ");
            sb.Append(string.Join(", ", parts));
            sb.Append('.');
            return sb.ToString();
        }

        /// <summary>
        /// Builds an instruct string for 1.7B. Returns null when pitch and speed are moderate/default
        /// so the prompt stays a single assistant turn.
        /// </summary>
        public static string InstructFor(string pitch, string speed)
        {
            return VoiceDesignInstruct(null, pitch, speed, null);
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
