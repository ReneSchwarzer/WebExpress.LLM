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
    /// <inheritdoc />
    public override int Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number)
        {
            return reader.GetInt32();
        }

        if (reader.TokenType == JsonTokenType.StartArray)
        {
            var result = 0;
            var first = true;

            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
                if (first && reader.TokenType == JsonTokenType.Number)
                {
                    result = reader.GetInt32();
                    first = false;
                }
            }

            return result;
        }

        throw new JsonException($"Unexpected token type {reader.TokenType} for an integer or integer-array value.");
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, int value, JsonSerializerOptions options)
    {
        writer.WriteNumberValue(value);
    }
}
