using System;
using System.IO;
using System.Xml.Linq;

namespace WebExpress.LLM.Console;

/// <summary>
/// Provides functionality to load application configuration from an XML file.
/// </summary>
public sealed class ConfigurationLoader
{
    /// <summary>
    /// The default path to the configuration file relative to the application directory.
    /// </summary>
    public const string DefaultConfigurationPath = "config/webexpress.llm.config.xml";

    /// <summary>
    /// Loads the application configuration from the specified XML file.
    /// </summary>
    /// <param name="configPath">
    /// The path to the configuration file. If not provided, uses the default path.
    /// </param>
    /// <returns>
    /// An ApplicationConfiguration instance containing the settings loaded from the XML file.
    /// </returns>
    /// <exception cref="FileNotFoundException">
    /// Thrown if the configuration file does not exist.
    /// </exception>
    /// <exception cref="InvalidDataException">
    /// Thrown if the configuration file cannot be parsed or is missing required elements.
    /// </exception>
    public ApplicationConfiguration Load(string configPath = null)
    {
        configPath ??= DefaultConfigurationPath;

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException("Configuration file was not found.", configPath);
        }

        try
        {
            var document = XDocument.Load(configPath);
            var root = document.Root ?? throw new InvalidDataException("Configuration file has no root element.");

            if (root.Name != "config")
            {
                throw new InvalidDataException("Configuration file root element must be 'config'.");
            }

            var modelElement = root.Element("model") ?? throw new InvalidDataException("Configuration file is missing 'model' element.");
            var modelName = modelElement.Attribute("name")?.Value ?? throw new InvalidDataException("Model element is missing 'name' attribute.");
            var modelPath = modelElement.Element("path")?.Value ?? throw new InvalidDataException("Model element is missing 'path' element.");

            // Load inference configuration (optional)
            var inferenceElement = root.Element("inference");
            var maxNewTokens = ParseInt(inferenceElement?.Element("maxNewTokens")?.Value, 100);
            var temperature = ParseFloat(inferenceElement?.Element("temperature")?.Value, 1.0f);
            var topK = ParseNullableInt(inferenceElement?.Element("topK")?.Value);
            var topP = ParseNullableFloat(inferenceElement?.Element("topP")?.Value);
            var seed = ParseNullableInt(inferenceElement?.Element("seed")?.Value);

            // Load tokenizer configuration (optional)
            var tokenizerElement = root.Element("tokenizer");
            var tokenizerType = tokenizerElement?.Attribute("type")?.Value ?? "byte";

            // Load runtime configuration (optional)
            var runtimeElement = root.Element("runtime");
            var useDeterministicEngine = ParseBool(runtimeElement?.Element("useDeterministicEngine")?.Value, false);

            return new ApplicationConfiguration
            {
                ModelName = modelName,
                ModelPath = modelPath,
                MaxNewTokens = maxNewTokens,
                Temperature = temperature,
                TopK = topK,
                TopP = topP,
                Seed = seed,
                TokenizerType = tokenizerType,
                UseDeterministicEngine = useDeterministicEngine
            };
        }
        catch (Exception ex) when (ex is not FileNotFoundException and not InvalidDataException)
        {
            throw new InvalidDataException($"Failed to parse configuration file: {ex.Message}", ex);
        }
    }

    private static int ParseInt(string value, int defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : int.Parse(value);
    }

    private static float ParseFloat(string value, float defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : float.Parse(value);
    }

    private static bool ParseBool(string value, bool defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : bool.Parse(value);
    }

    private static int? ParseNullableInt(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : int.Parse(value);
    }

    private static float? ParseNullableFloat(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : float.Parse(value);
    }
}
