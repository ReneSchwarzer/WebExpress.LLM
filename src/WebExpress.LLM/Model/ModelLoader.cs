using System;
using System.IO;
using System.Text.Json;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Model;

/// <summary>
/// Provides functionality to load a machine learning model definition from a specified directory containing
/// configuration and weights files.
/// </summary>
public sealed class ModelLoader
{
    public const string DefaultConfigurationFileName = "config.json";
    public const string DefaultWeightsFileName = "model.weights";

    private static readonly string[] SupportedWeightFileNames =
    [
        "model.safetensors",
        "model.weights",
        "pytorch_model.bin"
    ];

    /// <summary>
    /// Loads a model definition from the specified directory, including its configuration and weights files.
    /// </summary>
    /// <param name="modelDirectory">
    /// The path to the directory containing the model configuration and weights files. Cannot be null, empty, or
    /// consist only of white-space characters.</param>
    /// <returns>
    /// A ModelDefinition instance containing the deserialized configuration and weights loaded from the specified
    /// directory.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown if modelDirectory is null, empty, or consists only of white-space characters.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    /// Thrown if the required configuration or weights file does not exist in the specified directory.
    /// </exception>
    /// <exception cref="InvalidDataException">
    /// Thrown if the model configuration file cannot be deserialized.
    /// </exception>
    public ModelDefinition Load(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory must be provided.", nameof(modelDirectory));
        }

        var configurationPath = Path.Combine(modelDirectory, DefaultConfigurationFileName);

        if (!File.Exists(configurationPath))
        {
            throw new FileNotFoundException("Model configuration file was not found.", configurationPath);
        }

        var configurationJson = File.ReadAllText(configurationPath);
        var configuration = JsonSerializer.Deserialize<ModelConfiguration>(configurationJson)
            ?? throw new InvalidDataException("Model configuration could not be deserialized.");

        // Validate critical configuration parameters
        if (configuration.VocabularySize <= 0)
        {
            throw new InvalidDataException(
                $"Model configuration has invalid vocabulary size: {configuration.VocabularySize}. " +
                "Vocabulary size must be greater than zero.");
        }

        if (configuration.ContextLength <= 0)
        {
            throw new InvalidDataException(
                $"Model configuration has invalid context length: {configuration.ContextLength}. " +
                "Context length must be greater than zero.");
        }

        // Check for sharded SafeTensors index file
        var indexPath = Path.Combine(modelDirectory, SafeTensorIndex.DefaultFileName);

        if (File.Exists(indexPath))
        {
            return LoadSharded(modelDirectory, configuration, indexPath);
        }

        // Try to find a single weights file using supported file names
        string weightsPath = null;
        foreach (var weightsFileName in SupportedWeightFileNames)
        {
            var candidatePath = Path.Combine(modelDirectory, weightsFileName);
            if (File.Exists(candidatePath))
            {
                weightsPath = candidatePath;
                break;
            }
        }

        if (weightsPath == null)
        {
            var supportedFormats = string.Join(", ", SupportedWeightFileNames);
            throw new FileNotFoundException(
                $"Model weights file was not found. Supported formats: {supportedFormats}",
                Path.Combine(modelDirectory, "<weights-file>"));
        }

        // Load weights using ModelWeights class which supports files larger than 2GB
        var weights = ModelWeights.FromFile(weightsPath);

        return new ModelDefinition
        {
            Configuration = configuration,
            Weights = weights
        };
    }

    /// <summary>
    /// Loads a model definition using sharded SafeTensor files and the specified configuration.
    /// </summary>
    /// <param name="modelDirectory">
    /// The directory containing the model shard files.  
    /// Must be a valid path to the model resources.
    /// </param>
    /// <param name="configuration">
    /// The configuration settings to be used for the model.
    /// </param>
    /// <param name="indexPath">
    /// The path to the index file that describes the model’s shard structure.  
    /// Must not be null or empty.
    /// </param>
    /// <returns>
    /// A ModelDefinition instance containing the loaded model configuration  
    /// and the associated ShardedLoader.
    /// </returns>
    private static ModelDefinition LoadSharded(
        string modelDirectory,
        ModelConfiguration configuration,
        string indexPath)
    {
        var index = SafeTensorIndex.FromFile(indexPath);
        var shardedLoader = new ShardedSafeTensorLoader(index, modelDirectory);

        return new ModelDefinition
        {
            Configuration = configuration,
            ShardedLoader = shardedLoader
        };
    }
}
