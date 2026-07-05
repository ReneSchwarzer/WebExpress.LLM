using System;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Identifies the kind of a <see cref="ContentPart"/> within a multi-part chat message.
/// </summary>
public enum ContentPartType
{
    /// <summary>
    /// A textual content part. Its <see cref="ContentPart.Text"/> carries the text.
    /// </summary>
    Text,

    /// <summary>
    /// An image placeholder, rendered as the <c>&lt;|image|&gt;</c> token.
    /// </summary>
    Image,

    /// <summary>
    /// An audio placeholder, rendered as the <c>&lt;|audio|&gt;</c> token.
    /// </summary>
    Audio,

    /// <summary>
    /// A video placeholder, rendered as the <c>&lt;|video|&gt;</c> token.
    /// </summary>
    Video
}

/// <summary>
/// Represents a single part of a multi-modal chat message. A message's content may be a plain
/// string or, for multi-modal turns, an ordered sequence of <see cref="ContentPart"/> values that
/// interleave text with image, audio, and video placeholders.
/// </summary>
/// <remarks>
/// Non-text parts are emitted by <see cref="ChatTemplate"/> as the corresponding placeholder token
/// (<c>&lt;|image|&gt;</c>, <c>&lt;|audio|&gt;</c>, <c>&lt;|video|&gt;</c>); the actual media
/// embeddings are supplied to the model separately.
/// </remarks>
public sealed record ContentPart
{
    /// <summary>
    /// Gets the kind of this content part.
    /// </summary>
    public ContentPartType Type { get; }

    /// <summary>
    /// Gets the text for a <see cref="ContentPartType.Text"/> part; <see langword="null"/> for
    /// placeholder parts.
    /// </summary>
    public string Text { get; }

    private ContentPart(ContentPartType type, string text)
    {
        Type = type;
        Text = text;
    }

    /// <summary>
    /// Creates a textual content part.
    /// </summary>
    /// <param name="text">The text carried by the part.</param>
    /// <returns>A new <see cref="ContentPart"/> of type <see cref="ContentPartType.Text"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is null.</exception>
    public static ContentPart FromText(string text) => new(ContentPartType.Text, text ?? throw new ArgumentNullException(nameof(text)));

    /// <summary>
    /// Gets the shared image placeholder part.
    /// </summary>
    public static ContentPart Image { get; } = new(ContentPartType.Image, null);

    /// <summary>
    /// Gets the shared audio placeholder part.
    /// </summary>
    public static ContentPart Audio { get; } = new(ContentPartType.Audio, null);

    /// <summary>
    /// Gets the shared video placeholder part.
    /// </summary>
    public static ContentPart Video { get; } = new(ContentPartType.Video, null);
}
