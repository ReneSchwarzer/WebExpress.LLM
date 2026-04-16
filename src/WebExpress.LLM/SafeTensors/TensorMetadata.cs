using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.SafeTensors;

/// <summary>
/// Metadata for a single tensor stored in the SafeTensors format, describing its
/// data type, shape, and byte offset range within the data section.
/// </summary>
public sealed class TensorMetadata
{
    /// <summary>
    /// Gets the name/key of this tensor in the SafeTensors file.
    /// </summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Gets the data type identifier (e.g. "F32", "F16", "BF16").
    /// </summary>
    [JsonPropertyName("dtype")]
    public string Dtype { get; init; } = string.Empty;

    /// <summary>
    /// Gets the shape of this tensor as a list of dimension sizes.
    /// </summary>
    [JsonPropertyName("shape")]
    public IReadOnlyList<long> Shape { get; init; } = [];

    /// <summary>
    /// Gets the byte offset range [begin, end) within the data section.
    /// </summary>
    [JsonPropertyName("data_offsets")]
    public IReadOnlyList<long> DataOffsets { get; init; } = [];

    /// <summary>
    /// Gets the number of bytes per element for this tensor's data type.
    /// </summary>
    public int BytesPerElement => Dtype switch
    {
        "F32" => 4,
        "F16" => 2,
        "BF16" => 2,
        "I32" => 4,
        "I64" => 8,
        "U8" => 1,
        "I8" => 1,
        "BOOL" => 1,
        "F64" => 8,
        "I16" => 2,
        _ => throw new NotSupportedException($"Unsupported tensor data type: {Dtype}")
    };

    /// <summary>
    /// Gets the total number of elements in this tensor.
    /// </summary>
    public long ElementCount
    {
        get
        {
            if (Shape.Count == 0) return 1;
            long count = 1;
            for (var i = 0; i < Shape.Count; i++)
            {
                count *= Shape[i];
            }
            return count;
        }
    }
}
