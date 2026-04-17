using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.SafeTensors;

/// <summary>
/// Parses the SafeTensors binary format and provides access to tensor metadata and data.
/// </summary>
/// <remarks>
/// The SafeTensors format consists of:
/// 1. An 8-byte little-endian header length
/// 2. A JSON header describing tensor metadata (name, dtype, shape, data offsets)
/// 3. Raw tensor data referenced by the metadata offsets
///
/// This parser works with <see cref="ModelWeights"/> to read data from either
/// in-memory byte arrays or memory-mapped files.
/// </remarks>
public sealed class SafeTensorLoader : ISafeTensorLoader
{
    private readonly ModelWeights _weights;
    private readonly Dictionary<string, TensorMetadata> _metadata;
    private readonly long _dataOffset;
    private long _baseOffset;

    /// <summary>
    /// Initializes a new SafeTensorLoader by parsing the header from the provided model weights.
    /// </summary>
    /// <param name="weights">The model weights containing the SafeTensors data.</param>
    /// <exception cref="ArgumentNullException">Thrown when weights is null.</exception>
    /// <exception cref="InvalidDataException">Thrown when the SafeTensors header is malformed.</exception>
    public SafeTensorLoader(ModelWeights weights)
    {
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _metadata = [];

        if (weights.Length < 8)
        {
            throw new InvalidDataException("SafeTensors file is too small to contain a valid header.");
        }

        // Read header length (8 bytes, little-endian)
        var headerLengthBytes = weights.ReadBytes(0, 8);
        var headerLength = BinaryPrimitives.ReadInt64LittleEndian(headerLengthBytes);

        if (headerLength <= 0 || 8 + headerLength > weights.Length)
        {
            throw new InvalidDataException(
                $"Invalid SafeTensors header length: {headerLength}.");
        }

        // Read and parse JSON header
        var headerBytes = weights.ReadBytes(8, (int)headerLength);
        var headerJson = Encoding.UTF8.GetString(headerBytes);

        ParseHeader(headerJson);

        _dataOffset = 8 + headerLength;
        _baseOffset = ComputeBaseOffset();
    }

    /// <summary>
    /// Gets the names of all tensors in this SafeTensors file.
    /// </summary>
    public IReadOnlyCollection<string> TensorNames => _metadata.Keys;

    /// <summary>
    /// Gets the size of the data section in bytes (file length minus header).
    /// </summary>
    internal long DataSectionSize => _weights.Length - _dataOffset;

    /// <summary>
    /// Overrides the automatically computed base offset.
    /// Used by <see cref="ShardedSafeTensorLoader"/> to set a shard-specific base offset
    /// computed from only the tensors that the index maps to this shard, rather than
    /// from all tensors in the header (which may include tensors from other shards).
    /// </summary>
    /// <param name="baseOffset">The base offset to subtract from tensor data offsets.</param>
    internal void SetBaseOffset(long baseOffset)
    {
        _baseOffset = baseOffset;
    }

    /// <summary>
    /// Gets the metadata for the specified tensor.
    /// </summary>
    /// <param name="name">The name of the tensor.</param>
    /// <returns>The tensor metadata.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    public TensorMetadata GetMetadata(string name)
    {
        if (!_metadata.TryGetValue(name, out var metadata))
        {
            throw new KeyNotFoundException($"Tensor '{name}' not found in SafeTensors file.");
        }

        return metadata;
    }

    /// <summary>
    /// Checks whether a tensor with the given name exists.
    /// </summary>
    public bool ContainsTensor(string name)
    {
        return _metadata.ContainsKey(name);
    }

    /// <summary>
    /// Loads a tensor as a float array, converting from the stored data type if necessary.
    /// </summary>
    /// <param name="name">The name of the tensor to load.</param>
    /// <returns>A <see cref="Tensor.Tensor"/> containing the tensor data as float32.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    public Tensor.Tensor LoadTensor(string name)
    {
        var meta = GetMetadata(name);
        var begin = meta.DataOffsets[0];
        var end = meta.DataOffsets[1];
        var byteCount = (int)(end - begin);
        var rawBytes = _weights.ReadBytes(_dataOffset + begin - _baseOffset, byteCount);

        var floats = ConvertToFloat32(rawBytes, meta.Dtype);

        var shape = new int[meta.Shape.Count];

        for (var i = 0; i < meta.Shape.Count; i++)
        {
            shape[i] = (int)meta.Shape[i];
        }

        return new Tensor.Tensor(shape, floats);
    }

    /// <summary>
    /// Converts raw bytes to float32 array based on the specified data type.
    /// </summary>
    private static float[] ConvertToFloat32(byte[] rawBytes, string dtype)
    {
        return dtype switch
        {
            "F32" => ConvertF32(rawBytes),
            "F16" => ConvertF16(rawBytes),
            "BF16" => ConvertBF16(rawBytes),
            _ => throw new NotSupportedException($"Conversion from {dtype} to float32 is not supported.")
        };
    }

    /// <summary>
    /// Converts a byte array containing 32-bit IEEE 754 floating-point values in little-endian format to an array of
    /// single-precision floating-point numbers.
    /// </summary>
    /// <remarks>
    /// The method interprets each consecutive group of 4 bytes as a single-precision floating-point
    /// value using little-endian byte order. The caller is responsible for ensuring that the input array length is a
    /// multiple of 4; otherwise, the last incomplete group of bytes will be ignored.
    /// </remarks>
    /// <param name="bytes">
    /// The byte array to convert. The length must be a multiple of 4, with each group of 4 bytes representing a
    /// single-precision floating-point value in little-endian format.
    /// </param>
    /// <returns>
    /// An array of single-precision floating-point numbers converted from the input byte array.
    /// </returns>
    private static float[] ConvertF32(byte[] bytes)
    {
        var count = bytes.Length / 4;
        var result = new float[count];

        for (var i = 0; i < count; i++)
        {
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        }

        return result;
    }

    /// <summary>
    /// Converts an array of bytes containing IEEE 754 half-precision (16-bit) floating-point values in little-endian
    /// order to an array of single-precision (32-bit) floating-point values.
    /// </summary>
    /// <remarks>
    /// The input array must have a length that is a multiple of 2, as each half-precision value
    /// consists of 2 bytes. The conversion assumes the bytes are in little-endian order.
    /// </remarks>
    /// <param name="bytes">
    /// The byte array containing half-precision floating-point values in little-endian format. The length must be an
    /// even number.
    /// </param>
    /// <returns>
    /// An array of single-precision floating-point values converted from the input half-precision values.
    /// </returns>
    private static float[] ConvertF16(byte[] bytes)
    {
        var count = bytes.Length / 2;
        var result = new float[count];

        for (var i = 0; i < count; i++)
        {
            var halfBits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2));
            result[i] = HalfToFloat(halfBits);
        }

        return result;
    }

    /// <summary>
    /// Converts a byte array containing BFloat16‑encoded values into an array of 32‑bit
    /// floating‑point numbers (float).
    /// </summary>
    /// <remarks>
    /// BFloat16 is a 16‑bit floating‑point format commonly used in machine‑learning applications.
    /// Each BFloat16 value corresponds to the upper 16 bits of an IEEE 754 float.  
    /// This method processes two bytes at a time and converts them into a single float value.
    /// </remarks>
    /// <param name="bytes">
    /// The input byte array containing the BFloat16 values in little‑endian format.  
    /// The length must be even, since each value consists of two bytes.
    /// </param>
    /// <returns>
    /// An array of float values representing the converted BFloat16 numbers.  
    /// The length of the returned array is half the length of the input.
    /// </returns>
    private static float[] ConvertBF16(byte[] bytes)
    {
        var count = bytes.Length / 2;
        var result = new float[count];

        Parallel.For(0, count, i =>
        //for (var i = 0; i < count; i++)
        {
            var bfBits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2));
            // BFloat16 is simply the upper 16 bits of a float32
            var floatBits = (uint)bfBits << 16;
            result[i] = BitConverter.Int32BitsToSingle((int)floatBits);
        });

        return result;
    }

    /// <summary>
    /// Converts IEEE 754 half-precision (16-bit) to single-precision (32-bit) float.
    /// </summary>
    private static float HalfToFloat(ushort halfBits)
    {
        var sign = (halfBits >> 15) & 0x1;
        var exponent = (halfBits >> 10) & 0x1F;
        var mantissa = halfBits & 0x3FF;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                // Zero
                var zeroBits = (uint)(sign << 31);
                return BitConverter.Int32BitsToSingle((int)zeroBits);
            }

            // Subnormal: normalize by shifting mantissa until the implicit leading bit is set.
            // The effective exponent for subnormals is 1 (not 0), so initialize to 1.
            exponent = 1;

            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }

            // Remove the now-explicit leading bit from the mantissa
            mantissa &= 0x3FF;
        }
        else if (exponent == 31)
        {
            // Inf or NaN
            var specialBits = (uint)((sign << 31) | (0xFF << 23) | (mantissa << 13));
            return BitConverter.Int32BitsToSingle((int)specialBits);
        }

        // Normal number (or normalized subnormal)
        exponent = exponent + (127 - 15); // Rebias from half to single
        var floatBits2 = (uint)((sign << 31) | (exponent << 23) | (mantissa << 13));

        return BitConverter.Int32BitsToSingle((int)floatBits2);
    }

    /// <summary>
    /// Parses the metadata information from a JSON‑encoded header string and updates the internal
    /// metadata structure for tensors.
    /// </summary>
    /// <remarks>
    /// Properties named "__metadata__" are ignored.  
    /// The method expects each tensor entry to contain the properties "dtype", "shape", and "data_offsets".  
    /// Missing "dtype" properties are treated as "F32".
    /// </remarks>
    /// <param name="headerJson">
    /// The JSON‑encoded string containing the header information for the tensors.  
    /// It must be a valid JSON object with the expected properties.
    /// </param>
    private void ParseHeader(string headerJson)
    {
        using var doc = JsonDocument.Parse(headerJson);
        var root = doc.RootElement;

        foreach (var property in root.EnumerateObject())
        {
            // Skip the __metadata__ key
            if (property.Name == "__metadata__")
            {
                continue;
            }

            var name = property.Name;
            var tensor = property.Value;

            var dtype = tensor.GetProperty("dtype").GetString() ?? "F32";

            var shape = new List<long>();

            foreach (var dim in tensor.GetProperty("shape").EnumerateArray())
            {
                shape.Add(dim.GetInt64());
            }

            var offsets = new List<long>();

            foreach (var offset in tensor.GetProperty("data_offsets").EnumerateArray())
            {
                offsets.Add(offset.GetInt64());
            }

            _metadata[name] = new TensorMetadata
            {
                Name = name,
                Dtype = dtype,
                Shape = shape,
                DataOffsets = offsets
            };
        }
    }

    /// <summary>
    /// Computes the base data offset for this file.
    /// Some sharded SafeTensors files store global offsets (relative to the concatenated
    /// weight data across all shards) rather than per-shard-local offsets.
    /// When global offsets are detected, this method returns the minimum begin offset of
    /// non-empty tensors so that reads can be normalized to the shard's local data section.
    /// </summary>
    /// <returns>
    /// The base offset to subtract from tensor data offsets, or 0 when offsets are already local.
    /// </returns>
    private long ComputeBaseOffset()
    {
        var dataSize = _weights.Length - _dataOffset;
        long maxEnd = 0;
        var minOffset = long.MaxValue;

        foreach (var meta in _metadata.Values)
        {
            if (meta.DataOffsets.Count < 2)
            {
                continue;
            }

            if (meta.DataOffsets[1] > maxEnd)
            {
                maxEnd = meta.DataOffsets[1];
            }

            // Only consider tensors with actual data for the minimum offset calculation.
            // Zero-size tensors (e.g. [0, 0] placeholders) are excluded because their
            // offsets do not reflect the shard's true position in the global address space.
            if (meta.DataOffsets[1] > meta.DataOffsets[0] && meta.DataOffsets[0] < minOffset)
            {
                minOffset = meta.DataOffsets[0];
            }
        }

        // If all offsets fit within the local data section, no adjustment is needed.
        if (maxEnd <= dataSize)
        {
            return 0;
        }

        // Offsets are global; return the minimum non-empty begin offset as the base.
        return minOffset == long.MaxValue ? 0 : minOffset;
    }
}
