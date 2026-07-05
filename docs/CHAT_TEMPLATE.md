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

| Token | Purpose | Status |
|---|---|---|
| `<bos>` | Beginning-of-sequence marker | ✅ Implemented |
| `<|turn>role\n` | Start of a conversation turn for the given role | ✅ Implemented |
| `<turn|>\n` | End of a conversation turn | ✅ Implemented |
| `<|tool>…<tool|>` | Tool/function declaration block | ✅ Implemented |
| `<|tool_call>…<tool_call|>` | Tool call invocation block | ✅ Implemented |
| `<|tool_response>…<tool_response|>` | Tool response block | ✅ Implemented |
| `<|think|>` | Thinking mode activation | ✅ Implemented |
| `<|channel>thought\n…<channel|>` | Thinking/reasoning channel | ✅ Implemented |
| `<|"|>` | String delimiter within structured content | ✅ Implemented |
| `<|image|>`, `<|audio|>`, `<|video|>` | Multi-modal content placeholders | ✅ Implemented |

> **Note:** The C# `ApplyTemplate()` method implements the complete Gemma-4 chat protocol as a faithful port of the reference `chat_template.jinja`. Its output is validated byte-for-byte against the real template (rendered with Jinja2) for every protocol feature — see `UnitTestChatTemplateProtocol` and its golden data file `Gemma4PromptGolden.json`.

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

### Advanced Protocol Features

The following data classes (namespace `WebExpress.LLM.Chat`) drive the advanced protocol:

| Type | Purpose |
|---|---|
| `ContentPart` | A part of a multi-modal message: text, or an image/audio/video placeholder |
| `ToolCall` | A function call issued by the model (`Name`, `Arguments` map or `RawArguments`, optional `Id`) |
| `ToolResponse` | A tool execution result (`Name`, `Response` map or scalar) |
| `ToolDefinition` | A tool/function declaration (`Name`, `Description`, `Parameters`, `Response`) advertised via `tools` |

Additional `ChatMessage` fields: `Parts`, `Reasoning`, `ToolCalls`, `ToolResponses`, `Name`, `ToolCallId`.

**Thinking mode** — passing `enableThinking: true` injects the `<|think|>` marker into the leading
system turn:

```
<bos><|turn>system
<|think|>
You are helpful.<turn|>
<|turn>user
What is water?<turn|>
<|turn>model
```

**Tool declarations** — tools passed via `tools` are emitted inside the system turn:

```
<bos><|turn>system
<|tool>declaration:get_current_temperature{description:<|"|>Gets the current temperature for a location.<|"|>,parameters:{properties:{location:{description:<|"|>The city name, e.g. San Francisco<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Weather in London?<turn|>
<|turn>model
```

**Tool calls and responses** — a model turn carrying `ToolCalls` (with `ToolResponses`, or followed by
`role: tool` messages) renders the call/response blocks:

```
<|turn>model
<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>response:get_current_temperature{value:<|"|>15 degrees, sunny<|"|>}<tool_response|>It is 15 degrees and sunny.<turn|>
```

**Multi-modal content** — a message whose `Parts` interleave text and media placeholders:

```
<|turn>user
Describe this:<|image|><|audio|><|video|><turn|>
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
   - Emits a leading system turn when `enableThinking` is set, when `tools` are supplied, or when the first message is a `"system"`/`"developer"` message — including the `<|think|>` marker, system content, and tool declarations in that order.
   - Maps the `"assistant"` role to `"model"`, and continues the same model turn for consecutive assistant messages.
   - Renders the reasoning channel, tool calls, tool responses (embedded or from `role: tool` messages), and text or multi-modal content per turn.
   - Trims whitespace from user and system message content, and strips generated thinking channels from model content.
   - Appends a generation prompt (`<|turn>model\n`) when `addGenerationPrompt` is `true` and the last turn does not leave an open tool call/response.

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
| `ApplyTemplate(IReadOnlyList<ChatMessage>, bool, IReadOnlyList<ToolDefinition>, bool)` | Formats messages into a prompt string. Optional parameters: `addGenerationPrompt`, `tools`, and `enableThinking` |

### `ModelDefinition.ChatTemplate` Property

The `ChatTemplate` property on `ModelDefinition` is `null` when no template file exists in the model directory. Always check for `null` before using:

```csharp
if (model.ChatTemplate != null)
{
    var prompt = model.ChatTemplate.ApplyTemplate(messages);
}
```

### `ChatSession` Constructor

The `ChatSession` constructor accepts an optional `ChatTemplate` plus protocol options that are
applied when a template is present:

```csharp
public ChatSession(
    ITokenizer tokenizer,
    IInferenceEngine inferenceEngine,
    ChatTemplate chatTemplate = null,
    IReadOnlyList<ToolDefinition> tools = null,
    bool enableThinking = false,
    string systemInstruction = null)
```

## Testing

The following test classes cover the chat template integration:

- **`UnitTestChatTemplate`** — Tests for template construction, file loading, prompt formatting, edge cases (null/empty messages, custom BOS tokens, role mapping, whitespace trimming).
- **`UnitTestChatSession`** — Tests for template-based vs. fallback prompt formatting within chat sessions.
- **`ModelLoaderTests`** — Tests that `ModelLoader` correctly loads or skips the chat template based on file presence.
