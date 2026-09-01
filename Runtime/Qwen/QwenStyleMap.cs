using System.Text;

namespace SparkTTS.Qwen
{
    /// <summary>
    /// Maps Spark CharacterVoice style knobs onto Qwen3-TTS VoiceDesign.
    /// Gender / pitch / speed always become the identity prefix. Extra
    /// is optional notes (age, timbre, mic, use-case) — not a replacement
    /// for the knobs.
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
        /// VoiceDesign instruct. Dropdowns always contribute; extra is appended.
        /// Host UIs that preview this string must use the same phrases.
        /// </summary>
        public static string VoiceDesignInstruct(string gender, string pitch, string speed, string extra = null)
        {
            var sb = new StringBuilder();
            sb.Append(GenderPhrase(gender));
            sb.Append(", ");
            sb.Append(PitchPhrase(pitch));
            sb.Append(" pitch, ");
            sb.Append(SpeedPhrase(speed));
            sb.Append(" speaking rate.");
            if (!string.IsNullOrWhiteSpace(extra))
            {
                sb.Append(' ');
                sb.Append(extra.Trim());
            }
            return sb.ToString();
        }

        /// <summary>
        /// Builds an instruct string for 1.7B from pitch/speed only.
        /// </summary>
        public static string InstructFor(string pitch, string speed)
        {
            return VoiceDesignInstruct(null, pitch, speed, null);
        }

        public static string GenderPhrase(string gender)
        {
            if (string.IsNullOrEmpty(gender))
                return "Male";
            switch (gender.Trim().ToLowerInvariant())
            {
                case "female":
                case "woman":
                case "f":
                    return "Female";
                default:
                    return "Male";
            }
        }

        public static string PitchPhrase(string pitch)
        {
            if (string.IsNullOrEmpty(pitch))
                return "medium";
            switch (pitch.Trim().ToLowerInvariant())
            {
                case "very_low":
                    return "very low";
                case "low":
                    return "low";
                case "high":
                    return "high";
                case "very_high":
                    return "very high";
                default:
                    return "medium";
            }
        }

        public static string SpeedPhrase(string speed)
        {
            if (string.IsNullOrEmpty(speed))
                return "medium";
            switch (speed.Trim().ToLowerInvariant())
            {
                case "very_low":
                    return "very slow";
                case "low":
                    return "slow";
                case "high":
                    return "fast";
                case "very_high":
                    return "very fast";
                default:
                    return "medium";
            }
        }
    }
}
