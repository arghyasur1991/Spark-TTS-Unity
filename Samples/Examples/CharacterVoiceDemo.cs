using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Threading.Tasks;
using SparkTTS;
using TMPro;

/// <summary>
/// Demo script showing how to use CharacterVoiceFactory and CharacterVoice classes.
/// </summary>
public class CharacterVoiceDemo : MonoBehaviour
{
    [Header("References")]
    public AudioSource audioSource;
    public AudioClip referenceAudioClip;
    
    [Header("UI Elements")]
    public TMP_InputField textInput;
    public Button generateButton;
    public Button createMaleButton;
    public Button createFemaleButton;
    // public Button cloneVoiceButton;
    public TMP_Dropdown pitchDropdown;
    public TMP_Dropdown speedDropdown;
    
    [Header("Generation Settings")]
    public string defaultText = "Hello, I am a character in your game!";
    
    // Character voice components
    private CharacterVoiceFactory _voiceFactory;
    private CharacterVoice _currentVoice;
    private bool _isGenerating = false;
    
    void Start()
    {
        // Set up UI
        if (textInput != null)
        {
            textInput.text = defaultText;
        }
        
        // Set up pitch dropdown
        if (pitchDropdown != null && pitchDropdown.options.Count == 0)
        {
            pitchDropdown.options.Add(new TMP_Dropdown.OptionData("Very Low"));
            pitchDropdown.options.Add(new TMP_Dropdown.OptionData("Low"));
            pitchDropdown.options.Add(new TMP_Dropdown.OptionData("Moderate"));
            pitchDropdown.options.Add(new TMP_Dropdown.OptionData("High"));
            pitchDropdown.options.Add(new TMP_Dropdown.OptionData("Very High"));
            pitchDropdown.value = 2; // Default to Moderate
        }
        
        // Set up speed dropdown
        if (speedDropdown != null && speedDropdown.options.Count == 0)
        {
            speedDropdown.options.Add(new TMP_Dropdown.OptionData("Very Low"));
            speedDropdown.options.Add(new TMP_Dropdown.OptionData("Low"));
            speedDropdown.options.Add(new TMP_Dropdown.OptionData("Moderate"));
            speedDropdown.options.Add(new TMP_Dropdown.OptionData("High"));
            speedDropdown.options.Add(new TMP_Dropdown.OptionData("Very High"));
            speedDropdown.value = 2; // Default to Moderate
        }
        
        // Assign button handlers
        if (generateButton != null)
        {
            generateButton.onClick.AddListener(GenerateSpeech);
        }
        
        if (createMaleButton != null)
        {
            createMaleButton.onClick.AddListener(() => CreateStyleVoice("male"));
        }
        
        if (createFemaleButton != null)
        {
            createFemaleButton.onClick.AddListener(() => CreateStyleVoice("female"));
        }
        
        // Initialize factory
        _voiceFactory = new CharacterVoiceFactory();
        
        Debug.Log("CharacterVoiceDemo initialized. Ready to create voices and generate speech.");
    }
    
    private string GetPitchFromDropdown()
    {
        if (pitchDropdown == null) return "moderate";
        
        switch (pitchDropdown.value)
        {
            case 0: return "very_low";
            case 1: return "low";
            case 2: return "moderate";
            case 3: return "high";
            case 4: return "very_high";
            default: return "moderate";
        }
    }
    
    private string GetSpeedFromDropdown()
    {
        if (speedDropdown == null) return "moderate";
        
        switch (speedDropdown.value)
        {
            case 0: return "very_low";
            case 1: return "low";
            case 2: return "moderate";
            case 3: return "high";
            case 4: return "very_high";
            default: return "moderate";
        }
    }
    
    /// <summary>
    /// Creates a style-based voice with the specified gender and current dropdown settings.
    /// </summary>
    private async void CreateStyleVoice(string gender)
    {
        if (_isGenerating)
        {
            Debug.Log("Already generating or creating a voice. Please wait.");
            return;
        }
        
        _isGenerating = true;
        SetButtonsInteractable(false);
        
        Debug.Log($"Creating {gender} voice with pitch: {GetPitchFromDropdown()} and speed: {GetSpeedFromDropdown()}");
        
        // Dispose previous voice if any
        if (_currentVoice != null)
        {
            _currentVoice.Dispose();
            _currentVoice = null;
        }
        
        // Create new voice with pre-generation of sample text
        _currentVoice = await _voiceFactory.CreateFromStyleAsync(
            gender, 
            GetPitchFromDropdown(), 
            GetSpeedFromDropdown(),
            textInput.text);
        
        if (_currentVoice != null)
        {
            Debug.Log("Voice created successfully. Sample speech pre-generated.");
            PlayLastGeneratedSpeech();
        }
        else
        {
            Debug.LogError("Failed to create voice.");
        }
        
        _isGenerating = false;
        SetButtonsInteractable(true);
    }
    
    /// <summary>
    /// Generates speech using the current character voice.
    /// </summary>
    private async void GenerateSpeech()
    {
        if (_isGenerating)
        {
            Debug.Log("Already generating speech. Please wait.");
            return;
        }
        
        if (_currentVoice == null)
        {
            Debug.LogError("No character voice created. Please create a voice first.");
            return;
        }
        
        _isGenerating = true;
        SetButtonsInteractable(false);
        
        await GenerateSpeechAsync();
        _isGenerating = false;
        SetButtonsInteractable(true);
    }
    
    private async Task GenerateSpeechAsync()
    {
        string text = textInput != null ? textInput.text : defaultText;
        
        if (string.IsNullOrEmpty(text))
        {
            Debug.LogError("Text is empty. Nothing to generate.");
            return;
        }
        
        Debug.Log($"Generating speech for text: {text}");
        
        // Generate speech using the character voice
        AudioClip generatedClip = await _currentVoice.GenerateSpeechAsync(text);
        
        if (generatedClip != null)
        {
            Debug.Log("Speech generated successfully. Playing audio...");
            PlayAudioClip(generatedClip);
        }
        else
        {
            Debug.LogError("Failed to generate speech.");
        }
    }
    
    /// <summary>
    /// Plays the last generated speech.
    /// </summary>
    private void PlayLastGeneratedSpeech()
    {
        if (_currentVoice == null)
        {
            Debug.LogError("No character voice available.");
            return;
        }
        
        AudioClip lastClip = _currentVoice.GetLastGeneratedClip();
        
        if (lastClip != null)
        {
            PlayAudioClip(lastClip);
        }
        else
        {
            Debug.LogError("No speech has been generated yet.");
        }
    }
    
    /// <summary>
    /// Plays an audio clip using the audio source.
    /// </summary>
    private void PlayAudioClip(AudioClip clip)
    {
        if (audioSource == null || clip == null)
        {
            Debug.LogError("AudioSource or AudioClip is null.");
            return;
        }
        
        audioSource.Stop();
        audioSource.clip = clip;
        audioSource.Play();
    }
    
    /// <summary>
    /// Sets all buttons to be interactable or not.
    /// </summary>
    private void SetButtonsInteractable(bool interactable)
    {
        if (generateButton != null) generateButton.interactable = interactable;
        if (createMaleButton != null) createMaleButton.interactable = interactable;
        if (createFemaleButton != null) createFemaleButton.interactable = interactable;
        // if (cloneVoiceButton != null) cloneVoiceButton.interactable = interactable;
    }
    
    void OnDestroy()
    {
        // Clean up resources
        _currentVoice?.Dispose();
        _voiceFactory?.Dispose();
    }
}

/// <summary>
/// Helper class for executing actions on the main Unity thread.
/// This is needed because async operations complete on background threads.
/// </summary>
public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static UnityMainThreadDispatcher _instance;
    private readonly Queue<System.Action> _executionQueue = new Queue<System.Action>();
    private readonly object _lock = new object();

    public static UnityMainThreadDispatcher Instance()
    {
        if (_instance == null)
        {
            GameObject go = new GameObject("UnityMainThreadDispatcher");
            _instance = go.AddComponent<UnityMainThreadDispatcher>();
            DontDestroyOnLoad(go);
        }
        return _instance;
    }

    public void Enqueue(System.Action action)
    {
        lock (_lock)
        {
            _executionQueue.Enqueue(action);
        }
    }

    void Update()
    {
        lock (_lock)
        {
            while (_executionQueue.Count > 0)
            {
                _executionQueue.Dequeue().Invoke();
            }
        }
    }
} 