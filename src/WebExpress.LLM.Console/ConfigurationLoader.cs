using System;
using System.Globalization;
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
            var modelName = modelElement.Value?.Trim();
            if (string.IsNullOrWhiteSpace(modelName))
            {
                throw new InvalidDataException("Model element must contain the model name.");
            }
            var modelPath = root.Element("path")?.Value?.Trim()
                ?? throw new InvalidDataException("Configuration file is missing global 'path' element.");

            modelPath = Path.GetFullPath(modelPath ?? Path.Combine(Environment.CurrentDirectory, "models"));

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
            var tokenizerModelPath = tokenizerElement?.Attribute("modelPath")?.Value ?? "tokenizer.json";

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
                TokenizerModelPath = tokenizerModelPath,
                UseDeterministicEngine = useDeterministicEngine
            };
        }
        catch (Exception ex) when (ex is not FileNotFoundException and not InvalidDataException)
        {
            throw new InvalidDataException($"Failed to parse configuration file: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Parses a string into an integer and returns a default value if the input is null,
    /// empty, or consists only of whitespace.
    /// </summary>
    /// <param name="value">
    /// The string to parse, expected to represent an integer. May be null, empty,
    /// or contain only whitespace.
    /// </param>
    /// <param name="defaultValue">
    /// The value to return when the input is invalid or cannot be parsed.
    /// </param>
    /// <returns>
    /// The parsed integer from the input string, or the specified default value if the
    /// input is null, empty, or consists only of whitespace.
    /// </returns>
    private static int ParseInt(string value, int defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : int.Parse(value, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Parses a string into a floating‑point value and returns a default value if the input is null,
    /// empty, or consists only of whitespace.
    /// </summary>
    /// <param name="value">
    /// The string to parse, expected to represent a floating‑point value. May be null, empty,
    /// or contain only whitespace.
    /// </param>
    /// <param name="defaultValue">
    /// The value to return when <paramref name="value"/> is null, empty, or consists only of whitespace.
    /// </param>
    /// <returns>
    /// The floating‑point value parsed from <paramref name="value"/>, or <paramref name="defaultValue"/>
    /// if the input is invalid.
    /// </returns>
    private static float ParseFloat(string value, float defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : float.Parse(value, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Parses a string as a boolean value and returns a default value if the input is null,
    /// empty, or consists only of whitespace.
    /// </summary>
    /// <remarks>
    /// Throws an exception if <paramref name="value"/> is not null, empty, or whitespace
    /// and does not represent a valid boolean value.
    /// </remarks>
    /// <param name="value">
    /// The string to parse, expected to represent a boolean value. May be null, empty,
    /// or contain only whitespace.
    /// </param>
    /// <param name="defaultValue">
    /// The value to return when <paramref name="value"/> is null, empty, or consists only
    /// of whitespace.
    /// </param>
    /// <returns>
    /// The boolean value parsed from <paramref name="value"/>, or <paramref name="defaultValue"/>
    /// if the input is null, empty, or consists only of whitespace.
    /// </returns>
    private static bool ParseBool(string value, bool defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : bool.Parse(value);
    }

    /// <summary>
    /// Converts the specified string into a nullable integer value.
    /// </summary>
    /// <param name="value">
    /// The string to convert. May be null or empty.
    /// </param>
    /// <returns>
    /// An integer value representing the converted string, or <c>null</c> if the
    /// input cannot be converted.
    /// </returns>
    private static int? ParseNullableInt(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : int.Parse(value, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Parses a string and returns the corresponding floating‑point value, or <c>null</c>
    /// if the input is empty or consists only of whitespace.
    /// </summary>
    /// <param name="value">
    /// The string to parse. May be null, empty, or contain only whitespace.
    /// </param>
    /// <returns>
    /// A nullable float containing the parsed value, or <c>null</c> if the input is
    /// empty or consists only of whitespace.
    /// </returns>
    private static float? ParseNullableFloat(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : float.Parse(value, CultureInfo.InvariantCulture);
    }
}
