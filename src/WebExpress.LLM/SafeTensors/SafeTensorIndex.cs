using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace WebExpress.LLM.SafeTensors;

/// <summary>
/// Represents the parsed contents of a <c>model.safetensors.index.json</c> file,
/// which maps tensor names to their respective shard files for sharded SafeTensors models.
/// </summary>
/// <remarks>
/// The index JSON file has the following structure:
/// <code>
/// {
///   "metadata": {
///     "total_parameters": 26544131376,
///     "total_size": 51611872412
///   },
///   "weight_map": {
///     "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
///     "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
///     ...
///   }
/// }
/// </code>
/// </remarks>
public sealed class SafeTensorIndex
{
    /// <summary>
    /// The default filename for the SafeTensors index file.
    /// </summary>
    public const string DefaultFileName = "model.safetensors.index.json";

    private SafeTensorIndex(
        IReadOnlyDictionary<string, string> weightMap,
        long totalParameters,
        long totalSize,
        IReadOnlyCollection<string> shardFiles)
    {
        WeightMap = weightMap;
        TotalParameters = totalParameters;
        TotalSize = totalSize;
        ShardFiles = shardFiles;
    }

    /// <summary>
    /// Gets the mapping of tensor names to their respective shard filenames.
    /// </summary>
    public IReadOnlyDictionary<string, string> WeightMap { get; }

    /// <summary>
    /// Gets the total number of parameters across all shards, as reported by the index metadata.
    /// </summary>
    public long TotalParameters { get; }

    /// <summary>
    /// Gets the total size in bytes across all shards, as reported by the index metadata.
    /// </summary>
    public long TotalSize { get; }

    /// <summary>
    /// Gets the distinct set of shard filenames referenced by the weight map.
    /// </summary>
    public IReadOnlyCollection<string> ShardFiles { get; }

    /// <summary>
    /// Parses a SafeTensors index from a JSON string.
    /// </summary>
    /// <param name="json">The JSON content of the index file.</param>
    /// <returns>A new <see cref="SafeTensorIndex"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when json is null or whitespace.</exception>
    /// <exception cref="InvalidDataException">Thrown when the JSON is malformed or missing required fields.</exception>
    public static SafeTensorIndex Parse(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            throw new ArgumentException("Index JSON must not be null or empty.", nameof(json));
        }

        JsonDocument doc;

        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException("Failed to parse SafeTensors index JSON.", ex);
        }

        using (doc)
        {
            var root = doc.RootElement;

            // Parse metadata (optional but expected)
            long totalParameters = 0;
            long totalSize = 0;

            if (root.TryGetProperty("metadata", out var metadataElement))
            {
                if (metadataElement.TryGetProperty("total_parameters", out var paramElement))
                {
                    totalParameters = paramElement.GetInt64();
                }

                if (metadataElement.TryGetProperty("total_size", out var sizeElement))
                {
                    totalSize = sizeElement.GetInt64();
                }
            }

            // Parse weight_map (required)
            if (!root.TryGetProperty("weight_map", out var weightMapElement))
            {
                throw new InvalidDataException(
                    "SafeTensors index JSON is missing the required 'weight_map' property.");
            }

            var weightMap = new Dictionary<string, string>();
            var shardFileSet = new HashSet<string>(StringComparer.Ordinal);

            foreach (var property in weightMapElement.EnumerateObject())
            {
                var shardFile = property.Value.GetString();

                if (string.IsNullOrEmpty(shardFile))
                {
                    throw new InvalidDataException(
                        $"Tensor '{property.Name}' has a null or empty shard filename in the weight map.");
                }

                weightMap[property.Name] = shardFile;
                shardFileSet.Add(shardFile);
            }

            if (weightMap.Count == 0)
            {
                throw new InvalidDataException("SafeTensors index weight_map is empty.");
            }

            return new SafeTensorIndex(
                weightMap,
                totalParameters,
                totalSize,
                shardFileSet);
        }
    }

    /// <summary>
    /// Loads and parses a SafeTensors index from a file path.
    /// </summary>
    /// <param name="filePath">The path to the index JSON file.</param>
    /// <returns>A new <see cref="SafeTensorIndex"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when filePath is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    public static SafeTensorIndex FromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must be provided.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("SafeTensors index file not found.", filePath);
        }

        var json = File.ReadAllText(filePath);

        return Parse(json);
    }
}
