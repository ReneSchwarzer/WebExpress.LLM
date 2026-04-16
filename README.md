# WebExpress.LLM

WebExpress.LLM is a local inference framework for the Gemma 4 large language model, implemented in C# on .NET 10.
It provides modular building blocks to run basic chat-style inference entirely on the local machine without external services,
REST APIs, or HTTP endpoints.

## Repository structure

- `/src/WebExpress.LLM` – core framework library
- `/src/WebExpress.LLM.Test` – xUnit test project
- `/WebExpress.LLM.slnx` – solution file

## Included framework components

The framework currently includes:

- model loading support for configuration and weight files
- deterministic tokenizer (encode/decode)
- minimal deterministic inference engine
- chat session abstraction for conversation state and orchestration

The architecture is intentionally modular to enable extensions such as additional sampling strategies, caching layers, or alternative inference backends.

## Build and test

```bash
dotnet build /home/runner/work/WebExpress.LLM/WebExpress.LLM/WebExpress.LLM.slnx
dotnet test /home/runner/work/WebExpress.LLM/WebExpress.LLM/WebExpress.LLM.slnx
```

## Example usage

```csharp
using WebExpress.LLM.Chat;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

var tokenizer = new ByteTokenizer();
var engine = new DeterministicInferenceEngine();
var chat = new ChatSession(tokenizer, engine);

var assistant = chat.Send("Hello local Gemma", maxNewTokens: 16);
Console.WriteLine(assistant.Content);
```
