using System.Collections.Generic;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Declares a tool/function that is made available to the model. Definitions are emitted inside the
/// leading system turn as <c>&lt;|tool&gt;declaration:name{…}&lt;tool|&gt;</c> blocks.
/// </summary>
/// <param name="Name">The function name. Cannot be null or white-space.</param>
/// <remarks>
/// <see cref="Parameters"/> and <see cref="Response"/> follow the JSON-Schema shape used by the
/// Gemma tool declaration format: a parameters object with <c>type</c>, <c>properties</c>, and
/// <c>required</c> keys, where each property is itself a schema map (with <c>type</c>,
/// <c>description</c>, <c>enum</c>, <c>items</c>, nested <c>properties</c>, <c>required</c>, and
/// <c>nullable</c>). Types are upper-cased on output (e.g. <c>STRING</c>, <c>OBJECT</c>).
/// </remarks>
public sealed record ToolDefinition(string Name)
{
    /// <summary>
    /// Gets the human-readable description of the function.
    /// </summary>
    public string Description { get; init; }

    /// <summary>
    /// Gets the parameter schema for the function as a JSON-Schema-shaped map, or
    /// <see langword="null"/> when the function takes no parameters.
    /// </summary>
    public IReadOnlyDictionary<string, object> Parameters { get; init; }

    /// <summary>
    /// Gets an optional response schema (a map that may carry <c>description</c> and <c>type</c>),
    /// or <see langword="null"/> when no response shape is declared.
    /// </summary>
    public IReadOnlyDictionary<string, object> Response { get; init; }
}
