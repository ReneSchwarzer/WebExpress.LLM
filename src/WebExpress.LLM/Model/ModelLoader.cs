using System.Text.Json;

namespace WebExpress.LLM.Model;

public sealed class ModelLoader
{
    public const string DefaultConfigurationFileName = "config.json";
    public const string DefaultWeightsFileName = "model.weights";

    public ModelDefinition Load(string modelDirectory)
    {
        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory must be provided.", nameof(modelDirectory));
        }

        var configurationPath = Path.Combine(modelDirectory, DefaultConfigurationFileName);
        var weightsPath = Path.Combine(modelDirectory, DefaultWeightsFileName);

        if (!File.Exists(configurationPath))
        {
            throw new FileNotFoundException("Model configuration file was not found.", configurationPath);
        }

        if (!File.Exists(weightsPath))
        {
            throw new FileNotFoundException("Model weights file was not found.", weightsPath);
        }

        var configurationJson = File.ReadAllText(configurationPath);
        var configuration = JsonSerializer.Deserialize<ModelConfiguration>(configurationJson);

        if (configuration is null)
        {
            throw new InvalidDataException("Model configuration could not be deserialized.");
        }

        return new ModelDefinition
        {
            Configuration = configuration,
            Weights = File.ReadAllBytes(weightsPath)
        };
    }
}
