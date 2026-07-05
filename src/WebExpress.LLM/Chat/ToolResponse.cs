namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents the result of a tool/function execution that is fed back to the model. Rendered by
/// <see cref="ChatTemplate"/> as a <c>&lt;|tool_response&gt;response:name{…}&lt;tool_response|&gt;</c>
/// block.
/// </summary>
/// <param name="Name">
/// The name of the function whose result this represents. When null, <c>unknown</c> is emitted.
/// </param>
/// <param name="Response">
/// The response payload. When it is an <see cref="System.Collections.Generic.IReadOnlyDictionary{TKey,TValue}"/>
/// of string to object, its entries are emitted as <c>key:value</c> pairs; otherwise the value is
/// emitted as a single <c>value:…</c> entry.
/// </param>
public sealed record ToolResponse(string Name, object Response);
