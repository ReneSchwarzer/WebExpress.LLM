using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// A JSON converter that deserializes a property that may be either a single integer
/// or a JSON array of integers. When the value is an array, the first element is used.
/// </summary>
internal sealed class IntOrArrayConverter : JsonConverter<int>
{
    /// <summary>
    /// Reads an integer value from the JSON input, accepting either a single integer or an array containing at least
    /// one integer.
    /// </summary>
    /// <remarks>If the JSON input is an array, only the first integer element is returned; any additional
    /// elements are ignored.</remarks>
    /// <param name="reader">
    /// The reader positioned at the JSON token to read. Must be at a number or the start of an array containing at
    /// least one integer.
    /// </param>
    /// <param name="typeToConvert">
    /// The type of the value to convert. This parameter is not used.
    /// </param>
    /// <param name="options">
    /// Options to control the behavior of the deserialization. This parameter is not used.
    /// </param>
    /// <returns>
    /// The integer value read from the JSON input. If the input is an array, returns the first integer element.
    /// </returns>
    /// <exception cref="JsonException">
    /// Thrown if the JSON token is not a number or an array, if the array is empty, or if the first element of the
    /// array is not an integer.
    /// </exception>
    public override int Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number)
        {
            return reader.GetInt32();
        }

        if (reader.TokenType == JsonTokenType.StartArray)
        {
            if (!reader.Read() || reader.TokenType == JsonTokenType.EndArray)
            {
                throw new JsonException("The 'eos_token_id' array must contain at least one integer element.");
            }

            if (reader.TokenType != JsonTokenType.Number)
            {
                throw new JsonException($"The first element of the 'eos_token_id' array must be an integer, but got {reader.TokenType}.");
            }

            var result = reader.GetInt32();

            // Consume any remaining elements.
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
            }

            return result;
        }

        throw new JsonException($"Unexpected token type {reader.TokenType} for an integer or integer-array value.");
    }

    /// <summary>
    /// Writes the specified integer value as a JSON number using the provided Utf8JsonWriter.
    /// </summary>
    /// <param name="writer">The Utf8JsonWriter to which the integer value will be written. Must not be null.</param>
    /// <param name="value">The integer value to write as a JSON number.</param>
    /// <param name="options">
    /// The serialization options to use when writing the value. This parameter can influence formatting and behavior.
    /// </param>
    public override void Write(Utf8JsonWriter writer, int value, JsonSerializerOptions options)
    {
        writer.WriteNumberValue(value);
    }
}
