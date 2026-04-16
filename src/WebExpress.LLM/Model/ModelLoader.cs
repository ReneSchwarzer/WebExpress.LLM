using System;
using System.IO;
using System.Text.Json;

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

        // Try to find weights file using supported file names
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

        var configurationJson = File.ReadAllText(configurationPath);
        var configuration = JsonSerializer.Deserialize<ModelConfiguration>(configurationJson);

        return configuration is null
            ? throw new InvalidDataException("Model configuration could not be deserialized.")
            : new ModelDefinition
            {
                Configuration = configuration,
                Weights = File.ReadAllBytes(weightsPath)
            };
    }
}
