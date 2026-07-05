using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents a function/tool invocation requested by the model within a <c>model</c> turn.
/// Rendered by <see cref="ChatTemplate"/> as a <c>&lt;|tool_call&gt;call:name{…}&lt;tool_call|&gt;</c>
/// block.
/// </summary>
/// <param name="Name">The name of the function to call. Cannot be null or white-space.</param>
public sealed record ToolCall(string Name)
{
    /// <summary>
    /// Gets the call arguments as an ordered map of argument name to value. Values may be strings,
    /// booleans, numbers, nested <see cref="IReadOnlyDictionary{TKey,TValue}"/> maps, or sequences.
    /// When set, arguments are serialized with keys emitted verbatim (unescaped) and string values
    /// wrapped in <c>&lt;|"|&gt;</c> delimiters. Ignored when <see cref="RawArguments"/> is set.
    /// </summary>
    public IReadOnlyDictionary<string, object> Arguments { get; init; }

    /// <summary>
    /// Gets a pre-serialized argument body inserted verbatim between the call braces. Use this when
    /// the arguments are already formatted in the model's argument syntax. Takes precedence over
    /// <see cref="Arguments"/>.
    /// </summary>
    public string RawArguments { get; init; }

    /// <summary>
    /// Gets an optional identifier used to correlate this call with a subsequent
    /// <c>role: tool</c> response message (OpenAI-style tool calling).
    /// </summary>
    public string Id { get; init; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ToolCall"/> record with mapped arguments.
    /// </summary>
    /// <param name="name">The name of the function to call.</param>
    /// <param name="arguments">The call arguments as an ordered map.</param>
    public ToolCall(string name, IReadOnlyDictionary<string, object> arguments)
        : this(name)
    {
        Arguments = arguments;
    }
}
