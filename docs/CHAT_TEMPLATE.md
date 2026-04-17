# Chat Template Integration

## Overview

WebExpress.LLM supports model-specific chat templates stored as `chat_template.jinja` files within the model directory. When present, this template governs how conversation messages are formatted into prompt strings before tokenization and inference.

The Jinja2 template defines the turn-based structure that the model expects, including special tokens for turn boundaries, system instructions, tool declarations, thinking channels, and multi-modal content markers. The C# runtime translates this structure into formatted prompts compatible with the model's training format.

## Directory Layout

A model directory that includes a chat template has the following structure:

```
models/
└── gemma-4-1b-it/
    ├── config.json
    ├── chat_template.jinja      ← chat template (optional)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model.safetensors        (or sharded variant)
```

When the `chat_template.jinja` file is absent, the system falls back to a simple `role: content` format separated by newlines.

## Configuration Requirements

No additional configuration is required. The `ModelLoader` automatically detects the `chat_template.jinja` file within the model directory during loading. The loaded template is stored on the `ModelDefinition.ChatTemplate` property and can be passed to `ChatSession` for prompt formatting.

### Console Application

The console application (`WebExpress.LLM.Console`) loads the model via `ModelLoader`. If a chat template is present, it is available via `model.ChatTemplate` and can be passed to the `ChatSession` constructor:

```csharp
var loader = new ModelLoader();
var model = loader.Load(modelDirectory);

var chatSession = new ChatSession(tokenizer, inferenceEngine, model.ChatTemplate);
```

### Programmatic Usage

```csharp
// Load the chat template directly
var template = ChatTemplate.FromFile("path/to/chat_template.jinja");

// Or construct from a string
var template = new ChatTemplate(templateContent);

// Format messages
var messages = new List<ChatMessage>
{
    new("system", "You are a helpful assistant."),
    new("user", "What is 2+2?")
};

string prompt = template.ApplyTemplate(messages, addGenerationPrompt: true);
```

## Template Format

The chat template uses a turn-based structure with the following special tokens:

| Token | Purpose |
|---|---|
| `<bos>` | Beginning-of-sequence marker |
| `<\|turn>role\n` | Start of a conversation turn for the given role |
| `<turn\|>\n` | End of a conversation turn |
| `<\|tool>…<tool\|>` | Tool/function declaration block |
| `<\|tool_call>…<tool_call\|>` | Tool call invocation block |
| `<\|tool_response>…<tool_response\|>` | Tool response block |
| `<\|think\|>` | Thinking mode activation |
| `<\|channel>thought\n…<channel\|>` | Thinking/reasoning channel |
| `<\|"\|>` | Escaped double-quote within structured content |
| `<\|image\|>`, `<\|audio\|>`, `<\|video\|>` | Multi-modal content placeholders |

### Example Output

For a conversation with a system message and a user question:

```
<bos><|turn>system
You are a helpful assistant.<turn|>
<|turn>user
What is 2+2?<turn|>
<|turn>model
```

For a multi-turn conversation with a prior assistant response:

```
<bos><|turn>system
You are a helpful assistant.<turn|>
<|turn>user
What is 2+2?<turn|>
<|turn>model
4<turn|>
<|turn>user
Thanks!<turn|>
<|turn>model
```

## Runtime Behavior

### Template Loading

1. During `ModelLoader.Load()`, the loader checks for a `chat_template.jinja` file in the model directory.
2. If the file exists, it is read and stored as a `ChatTemplate` instance on `ModelDefinition.ChatTemplate`.
3. If the file does not exist, `ModelDefinition.ChatTemplate` is `null`.

### Prompt Formatting

1. When `ChatSession` is constructed with a `ChatTemplate`, the `Send()` and `SendAsync()` methods use the template's `ApplyTemplate()` method to format the conversation history.
2. When no template is provided (i.e., `chatTemplate` is `null`), the session falls back to the simple `role: content` format.
3. The `ApplyTemplate()` method:
   - Prepends the `BosToken` (default: `<bos>`).
   - Emits a system turn for the first message if its role is `"system"` or `"developer"`.
   - Maps the `"assistant"` role to `"model"` for model-internal consistency.
   - Trims whitespace from user and system message content.
   - Appends a generation prompt (`<|turn>model\n`) when `addGenerationPrompt` is `true`.

### Role Mapping

| Input Role | Output Role |
|---|---|
| `user` | `user` |
| `assistant` | `model` |
| `system` | `system` |
| `developer` | `system` (only at position 0) |

### BOS Token Customization

The `BosToken` property defaults to `<bos>` but can be overridden to match the tokenizer's expected special token:

```csharp
var template = new ChatTemplate(content) { BosToken = "<s>" };
```

## API Reference

### `ChatTemplate` Class

**Namespace:** `WebExpress.LLM.Chat`

| Member | Description |
|---|---|
| `DefaultFileName` | Constant: `"chat_template.jinja"` |
| `TemplateContent` | The raw Jinja2 template content loaded from the file |
| `BosToken` | The beginning-of-sequence token (default: `"<bos>"`) |
| `ChatTemplate(string)` | Constructor accepting the raw template content |
| `FromFile(string)` | Static factory that loads a template from a file path |
| `ApplyTemplate(IReadOnlyList<ChatMessage>, bool)` | Formats messages into a prompt string |

### `ModelDefinition.ChatTemplate` Property

The `ChatTemplate` property on `ModelDefinition` is `null` when no template file exists in the model directory. Always check for `null` before using:

```csharp
if (model.ChatTemplate != null)
{
    var prompt = model.ChatTemplate.ApplyTemplate(messages);
}
```

### `ChatSession` Constructor

The `ChatSession` constructor accepts an optional `ChatTemplate` parameter:

```csharp
public ChatSession(
    ITokenizer tokenizer,
    IInferenceEngine inferenceEngine,
    ChatTemplate chatTemplate = null)
```

## Testing

The following test classes cover the chat template integration:

- **`UnitTestChatTemplate`** — Tests for template construction, file loading, prompt formatting, edge cases (null/empty messages, custom BOS tokens, role mapping, whitespace trimming).
- **`UnitTestChatSession`** — Tests for template-based vs. fallback prompt formatting within chat sessions.
- **`ModelLoaderTests`** — Tests that `ModelLoader` correctly loads or skips the chat template based on file presence.
