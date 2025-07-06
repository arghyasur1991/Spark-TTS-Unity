using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace SparkTTS.Editor
{
    /// <summary>
    /// Editor tool for deploying SparkTTS models to StreamingAssets.
    /// Can be used standalone or integrated with other packages.
    /// </summary>
    public class ModelDeploymentTool : EditorWindow
    {
        #region Model Configuration Data

        /// <summary>
        /// Configuration for a SparkTTS model
        /// </summary>
        [Serializable]
        public class ModelConfig
        {
            public string modelName;
            public string relativePath;
            public string precision;
            public bool isRequired;
            public long fileSize;
            public string fullPath;

            public ModelConfig(string name, string path, string prec, bool required = true)
            {
                modelName = name;
                relativePath = path;
                precision = prec;
                isRequired = required;
            }
        }

        /// <summary>
        /// SparkTTS model configurations based on codebase analysis
        /// </summary>
        private static readonly Dictionary<string, List<ModelConfig>> ModelConfigurations = new()
        {
            ["SparkTTS"] = new List<ModelConfig>
            {
                new("wav2vec2_model", "SparkTTS", "fp16"),
                new("bicodec_encoder_quantizer", "SparkTTS", "fp32"),
                new("bicodec_vocoder", "SparkTTS", "fp32"),
                new("mel_spectrogram", "SparkTTS", "fp32"),
                new("speaker_encoder_tokenizer", "SparkTTS", "fp32"),
                new("onnx_config.json", "SparkTTS", "none"),
            },
            ["SparkTTS_LLM"] = new List<ModelConfig>
            {
                new("model", "SparkTTS/LLM", "fp32"),
                new("config.json", "SparkTTS/LLM", "none"),
                new("tokenizer.json", "SparkTTS/LLM", "none"),
                new("vocab.json", "SparkTTS/LLM", "none"),
                new("merges.txt", "SparkTTS/LLM", "none"),
                new("added_tokens.json", "SparkTTS/LLM", "none"),
                new("special_tokens_map.json", "SparkTTS/LLM", "none"),
                new("tokenizer_config.json", "SparkTTS/LLM", "none"),
                new("generation_config.json", "SparkTTS/LLM", "none"),
            }
        };

        #endregion

        #region UI Fields

        [SerializeField] private Vector2 scrollPosition;
        [SerializeField] private bool showAdvancedOptions = false;
        [SerializeField] private bool includeSparkTTS = true;
        [SerializeField] private bool includeLLM = true;
        [SerializeField] private bool overwriteExisting = true;
        [SerializeField] private bool createBackup = true;
        [SerializeField] private bool dryRun = false;

        // Model source and destination paths
        private string sourceModelsPath;
        private string streamingAssetsPath;
        private List<ModelConfig> selectedModels;
        private long totalSelectedSize;

        #endregion

        #region Public API

        /// <summary>
        /// Gets all SparkTTS model configurations
        /// </summary>
        /// <returns>Dictionary of model configurations by category</returns>
        public static Dictionary<string, List<ModelConfig>> GetAllModelConfigurations()
        {
            return new Dictionary<string, List<ModelConfig>>(ModelConfigurations);
        }

        /// <summary>
        /// Gets model configurations for a specific category
        /// </summary>
        /// <param name="category">The category name (e.g., "SparkTTS", "SparkTTS_LLM")</param>
        /// <returns>List of model configurations for the category</returns>
        public static List<ModelConfig> GetModelConfigurations(string category)
        {
            if (ModelConfigurations.ContainsKey(category))
            {
                return new List<ModelConfig>(ModelConfigurations[category]);
            }
            return new List<ModelConfig>();
        }

        /// <summary>
        /// Updates model information including full path and file size
        /// </summary>
        /// <param name="model">The model configuration to update</param>
        /// <param name="sourceModelsPath">The base path where models are located</param>
        /// <returns>The updated model configuration</returns>
        public static ModelConfig UpdateModelInfo(ModelConfig model, string sourceModelsPath)
        {
            // Build the full path based on precision
            string fileName = model.modelName;
            string extension = ".onnx";
            
            // Handle special cases
            if (model.modelName.Contains(".json") || model.modelName.Contains(".txt"))
            {
                fileName = model.modelName;
                extension = "";
            }
            else if (model.precision == "fp16")
            {
                fileName += "_fp16";
            }
            else if (model.precision == "int8")
            {
                fileName += "_int8";
            }
            
            // Handle special case for LLM model with data file
            if (model.modelName == "model" && model.relativePath.Contains("LLM"))
            {
                string dataFileName = fileName + ".onnx_data";
                string dataPath = Path.Combine(sourceModelsPath, model.relativePath, dataFileName);
                if (File.Exists(dataPath))
                {
                    model.fullPath = Path.Combine(sourceModelsPath, model.relativePath, fileName + extension);
                    if (File.Exists(model.fullPath))
                    {
                        model.fileSize = new FileInfo(model.fullPath).Length + new FileInfo(dataPath).Length;
                    }
                    return model;
                }
            }
            
            model.fullPath = Path.Combine(sourceModelsPath, model.relativePath, fileName + extension);
            
            if (File.Exists(model.fullPath))
            {
                model.fileSize = new FileInfo(model.fullPath).Length;
            }
            else
            {
                model.fileSize = 0;
                Debug.LogWarning($"SparkTTS model file not found: {model.fullPath}");
            }
            
            return model;
        }

        #endregion

        #region Unity Editor Window

        [MenuItem("SparkTTS/Model Deployment Tool")]
        public static void ShowWindow()
        {
            var window = GetWindow<ModelDeploymentTool>("SparkTTS Model Deployment");
            window.minSize = new Vector2(600, 400);
            window.Show();
        }

        private void OnEnable()
        {
            sourceModelsPath = Path.Combine(Application.dataPath, "Models");
            streamingAssetsPath = Application.streamingAssetsPath;
            RefreshModelList();
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("SparkTTS Model Deployment Tool", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            DrawPathConfiguration();
            EditorGUILayout.Space();
            
            DrawModelSelection();
            EditorGUILayout.Space();
            
            DrawAdvancedOptions();
            EditorGUILayout.Space();
            
            DrawDeploymentActions();
        }

        #endregion

        #region UI Drawing Methods

        private void DrawPathConfiguration()
        {
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);
            
            EditorGUI.BeginChangeCheck();
            sourceModelsPath = EditorGUILayout.TextField("Source Models Path:", sourceModelsPath);
            streamingAssetsPath = EditorGUILayout.TextField("Destination Path:", streamingAssetsPath);
            
            if (EditorGUI.EndChangeCheck())
            {
                RefreshModelList();
            }

            // Validation
            if (!Directory.Exists(sourceModelsPath))
            {
                EditorGUILayout.HelpBox($"Source path does not exist: {sourceModelsPath}", MessageType.Error);
            }
            else
            {
                EditorGUILayout.HelpBox($"✓ Source path found with {CountAvailableModels()} models", MessageType.Info);
            }
        }

        private void DrawModelSelection()
        {
            EditorGUILayout.LabelField("Model Selection", EditorStyles.boldLabel);
            
            // Component toggles
            includeSparkTTS = EditorGUILayout.Toggle("Include SparkTTS Models", includeSparkTTS);
            includeLLM = EditorGUILayout.Toggle("Include LLM Models", includeLLM);
            
            EditorGUILayout.Space();
            
            // Model details
            if (selectedModels != null && selectedModels.Count > 0)
            {
                EditorGUILayout.LabelField($"Selected Models ({selectedModels.Count}):", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Total Size: {FormatFileSize(totalSelectedSize)}");
                
                scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition, GUILayout.Height(200));
                
                foreach (var model in selectedModels.OrderBy(m => m.relativePath).ThenBy(m => m.modelName))
                {
                    DrawModelItem(model);
                }
                
                EditorGUILayout.EndScrollView();
            }
            else
            {
                EditorGUILayout.HelpBox("No models selected or available.", MessageType.Warning);
            }
        }

        private void DrawModelItem(ModelConfig model)
        {
            EditorGUILayout.BeginHorizontal();
            
            // Model info
            string displayName = $"{model.modelName}";
            if (model.precision != "none")
                displayName += $" ({model.precision.ToUpper()})";
            
            EditorGUILayout.LabelField(displayName, GUILayout.Width(200));
            EditorGUILayout.LabelField(model.relativePath, GUILayout.Width(200));
            EditorGUILayout.LabelField(FormatFileSize(model.fileSize), GUILayout.Width(80));
            
            // Status
            if (File.Exists(model.fullPath))
            {
                EditorGUILayout.LabelField("✓", GUILayout.Width(20));
            }
            else
            {
                EditorGUILayout.LabelField("✗", GUILayout.Width(20));
            }
            
            EditorGUILayout.EndHorizontal();
        }

        private void DrawAdvancedOptions()
        {
            showAdvancedOptions = EditorGUILayout.Foldout(showAdvancedOptions, "Advanced Options");
            
            if (showAdvancedOptions)
            {
                EditorGUI.indentLevel++;
                
                overwriteExisting = EditorGUILayout.Toggle("Overwrite Existing Files", overwriteExisting);
                createBackup = EditorGUILayout.Toggle("Create Backup", createBackup);
                dryRun = EditorGUILayout.Toggle("Dry Run (Preview Only)", dryRun);
                
                if (dryRun)
                {
                    EditorGUILayout.HelpBox("Dry run mode: No files will be copied, only operations will be logged.", MessageType.Info);
                }
                
                EditorGUI.indentLevel--;
            }
        }

        private void DrawDeploymentActions()
        {
            EditorGUILayout.LabelField("Deployment", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            
            if (GUILayout.Button("Refresh Model List", GUILayout.Height(30)))
            {
                RefreshModelList();
            }
            
            if (GUILayout.Button("Deploy SparkTTS Models", GUILayout.Height(30)))
            {
                DeployModels();
            }
            
            EditorGUILayout.EndHorizontal();
            
            if (GUILayout.Button("Clean StreamingAssets", GUILayout.Height(25)))
            {
                CleanStreamingAssets();
            }
        }

        #endregion

        #region Model Management

        private void RefreshModelList()
        {
            selectedModels = new List<ModelConfig>();
            totalSelectedSize = 0;

            if (includeSparkTTS)
                AddModelsFromCategory("SparkTTS");
            
            if (includeLLM)
                AddModelsFromCategory("SparkTTS_LLM");
            
            // Calculate file sizes and validate paths
            foreach (var model in selectedModels)
            {
                UpdateModelInfo(model);
            }
            
            Repaint();
        }

        private void AddModelsFromCategory(string category)
        {
            if (ModelConfigurations.ContainsKey(category))
            {
                selectedModels.AddRange(ModelConfigurations[category]);
            }
        }

        private void UpdateModelInfo(ModelConfig model)
        {
            model = UpdateModelInfo(model, sourceModelsPath);
            totalSelectedSize += model.fileSize;
        }

        private int CountAvailableModels()
        {
            if (!Directory.Exists(sourceModelsPath))
                return 0;
            
            int count = 0;
            string[] searchPatterns = { "*.onnx", "*.json", "*.txt" };
            
            foreach (var pattern in searchPatterns)
            {
                count += Directory.GetFiles(sourceModelsPath, pattern, SearchOption.AllDirectories).Length;
            }
            
            return count;
        }

        #endregion

        #region Public API Methods

        /// <summary>
        /// Deploy SparkTTS models programmatically (can be called from other packages)
        /// </summary>
        /// <param name="sourceModelsPath">Path to source models</param>
        /// <param name="destinationPath">Path to destination (StreamingAssets)</param>
        /// <param name="options">Deployment options</param>
        /// <returns>True if deployment was successful</returns>
        public static bool DeploySparkTTSModels(
            string sourceModelsPath, 
            string destinationPath, 
            DeploymentOptions options = null)
        {
            if (options == null)
            {
                options = new DeploymentOptions
                {
                    OverwriteExisting = true,
                    CreateBackup = true,
                    DryRun = false,
                    IncludeSparkTTS = true,
                    IncludeLLM = true
                };
            }

            try
            {
                var tool = new ModelDeploymentTool();
                tool.sourceModelsPath = sourceModelsPath;
                tool.streamingAssetsPath = destinationPath;
                tool.overwriteExisting = options.OverwriteExisting;
                tool.createBackup = options.CreateBackup;
                tool.dryRun = options.DryRun;
                tool.includeSparkTTS = options.IncludeSparkTTS;
                tool.includeLLM = options.IncludeLLM;
                
                tool.RefreshModelList();
                
                if (tool.selectedModels == null || tool.selectedModels.Count == 0)
                {
                    Debug.LogWarning("[SparkTTS] No models found to deploy");
                    return false;
                }

                var missingModels = tool.selectedModels.Where(m => !File.Exists(m.fullPath)).ToList();
                if (missingModels.Any())
                {
                    Debug.LogError($"[SparkTTS] Missing models: {string.Join(", ", missingModels.Select(m => m.modelName))}");
                    return false;
                }

                return tool.InternalDeployModels();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SparkTTS] Deployment failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Options for SparkTTS model deployment
        /// </summary>
        public class DeploymentOptions
        {
            public bool OverwriteExisting { get; set; } = true;
            public bool CreateBackup { get; set; } = true;
            public bool DryRun { get; set; } = false;
            public bool IncludeSparkTTS { get; set; } = true;
            public bool IncludeLLM { get; set; } = true;
        }

        #endregion

        #region Deployment Operations

        private void DeployModels()
        {
            if (selectedModels == null || selectedModels.Count == 0)
            {
                EditorUtility.DisplayDialog("No Models Selected", "Please select at least one model category to deploy.", "OK");
                return;
            }

            var missingModels = selectedModels.Where(m => !File.Exists(m.fullPath)).ToList();
            if (missingModels.Any())
            {
                string missingList = string.Join("\n", missingModels.Select(m => $"- {m.modelName} at {m.fullPath}"));
                EditorUtility.DisplayDialog("Missing Models", $"The following models are missing:\n{missingList}", "OK");
                return;
            }

            if (!dryRun)
            {
                bool proceed = EditorUtility.DisplayDialog("Deploy Models", 
                    $"Deploy {selectedModels.Count} SparkTTS models ({FormatFileSize(totalSelectedSize)}) to StreamingAssets?", 
                    "Deploy", "Cancel");
                
                if (!proceed)
                    return;
            }

            bool success = InternalDeployModels();
            
            if (success && !dryRun)
            {
                AssetDatabase.Refresh();
                EditorUtility.DisplayDialog("Deployment Complete", 
                    $"Successfully deployed {selectedModels.Count} SparkTTS models to StreamingAssets.", "OK");
            }
        }

        private bool InternalDeployModels()
        {
            try
            {
                int progress = 0;
                int total = selectedModels.Count;
                
                foreach (var model in selectedModels)
                {
                    if (!dryRun)
                    {
                        EditorUtility.DisplayProgressBar("Deploying SparkTTS Models", $"Copying {model.modelName}...", (float)progress / total);
                    }
                    
                    DeployModel(model);
                    progress++;
                }
                
                if (!dryRun)
                {
                    EditorUtility.ClearProgressBar();
                }
                else
                {
                    Debug.Log($"[SparkTTS DRY RUN] Would have deployed {selectedModels.Count} models");
                }
                
                return true;
            }
            catch (Exception ex)
            {
                if (!dryRun)
                {
                    EditorUtility.ClearProgressBar();
                }
                Debug.LogError($"[SparkTTS] Model deployment failed: {ex}");
                return false;
            }
        }

        private void DeployModel(ModelConfig model)
        {
            string destinationDir = Path.Combine(streamingAssetsPath, model.relativePath);
            string destinationPath = Path.Combine(destinationDir, Path.GetFileName(model.fullPath));
            
            if (dryRun)
            {
                Debug.Log($"[SparkTTS DRY RUN] Would copy: {model.fullPath} -> {destinationPath}");
                
                // Check for LLM model data file
                if (model.modelName == "model" && model.relativePath.Contains("LLM"))
                {
                    string dataSourcePath = model.fullPath + "_data";
                    string dataDestPath = destinationPath + "_data";
                    if (File.Exists(dataSourcePath))
                    {
                        Debug.Log($"[SparkTTS DRY RUN] Would copy: {dataSourcePath} -> {dataDestPath}");
                    }
                }
                return;
            }

            // Create destination directory
            Directory.CreateDirectory(destinationDir);
            
            // Create backup if requested
            if (createBackup && File.Exists(destinationPath))
            {
                string backupPath = destinationPath + ".backup";
                File.Copy(destinationPath, backupPath, true);
                Debug.Log($"[SparkTTS] Created backup: {backupPath}");
            }
            
            // Copy the main model file
            File.Copy(model.fullPath, destinationPath, overwriteExisting);
            Debug.Log($"[SparkTTS] Copied: {model.fullPath} -> {destinationPath}");
            
            // Handle LLM model data file
            if (model.modelName == "model" && model.relativePath.Contains("LLM"))
            {
                string dataSourcePath = model.fullPath + "_data";
                string dataDestPath = destinationPath + "_data";
                if (File.Exists(dataSourcePath))
                {
                    // Create backup for data file if requested
                    if (createBackup && File.Exists(dataDestPath))
                    {
                        string backupDataPath = dataDestPath + ".backup";
                        File.Copy(dataDestPath, backupDataPath, true);
                        Debug.Log($"[SparkTTS] Created backup: {backupDataPath}");
                    }
                    
                    File.Copy(dataSourcePath, dataDestPath, overwriteExisting);
                    Debug.Log($"[SparkTTS] Copied: {dataSourcePath} -> {dataDestPath}");
                }
            }
        }

        private void CleanStreamingAssets()
        {
            if (!Directory.Exists(streamingAssetsPath))
            {
                EditorUtility.DisplayDialog("Nothing to Clean", "StreamingAssets SparkTTS directory does not exist.", "OK");
                return;
            }

            bool proceed = EditorUtility.DisplayDialog("Clean StreamingAssets", 
                "This will delete all SparkTTS models from StreamingAssets. Continue?", 
                "Delete", "Cancel");
            
            if (!proceed)
                return;

            try
            {
                Directory.Delete(streamingAssetsPath, true);
                AssetDatabase.Refresh();
                
                EditorUtility.DisplayDialog("Clean Complete", "StreamingAssets SparkTTS directory has been cleaned.", "OK");
                Debug.Log("[SparkTTS] StreamingAssets SparkTTS directory cleaned");
            }
            catch (Exception ex)
            {
                EditorUtility.DisplayDialog("Clean Error", $"Error cleaning StreamingAssets: {ex.Message}", "OK");
                Debug.LogError($"[SparkTTS] Clean operation failed: {ex}");
            }
        }

        #endregion

        #region Utility Methods

        private string FormatFileSize(long bytes)
        {
            if (bytes == 0) return "0 B";
            
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;
            
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }
            
            return $"{size:0.##} {sizes[order]}";
        }

        #endregion
    }
} 