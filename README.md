![WebExpress-Framework](https://raw.githubusercontent.com/webexpress-framework/.github/main/docs/assets/img/banner.png)

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

### Core Components
- **Model Loading**: Support for loading Gemma-4 model configuration and weight files
- **Tokenization**:
  - `ByteTokenizer`: Simple UTF-8 byte-level tokenizer
  - `VocabularyTokenizer`: Vocabulary-based tokenization foundation
- **Inference Engines**:
  - `DeterministicInferenceEngine`: Deterministic mock for testing
  - `TransformerInferenceEngine`: Basic transformer architecture implementation
- **Sampling Strategies**:
  - `GreedySampling`: Deterministic token selection
  - `TopKSampling`: Sample from top-k tokens
  - `TopPSampling`: Nucleus sampling
- **Chat Session**: Conversation state management and orchestration

The architecture is intentionally modular to enable extensions such as additional sampling strategies, caching layers, or alternative inference backends.

## Build and test

```bash
dotnet build src/WebExpress.LLM.slnx
dotnet test src/WebExpress.LLM.slnx
```

## Example usage

### Basic Chat Session
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

### Using Transformer Engine with Sampling Strategies
```csharp
using WebExpress.LLM.Inference;
using WebExpress.LLM.Model;
using WebExpress.LLM.Tokenization;

// Load model
var loader = new ModelLoader();
var model = loader.Load("/path/to/model");

// Create inference engine with greedy sampling
var greedySampling = new GreedySampling();
var engine = new TransformerInferenceEngine(model, greedySampling);

// Or use top-k sampling
var topKSampling = new TopKSampling(k: 50, seed: 42);
var engineWithTopK = new TransformerInferenceEngine(model, topKSampling);

// Or use nucleus (top-p) sampling
var topPSampling = new TopPSampling(p: 0.9f, seed: 42);
var engineWithTopP = new TransformerInferenceEngine(model, topPSampling);
```

### Using Generation Configuration
```csharp
using WebExpress.LLM.Inference;

// Create generation config with top-k sampling
var config = new GenerationConfig
{
    MaxNewTokens = 100,
    Temperature = 0.8f,
    TopK = 50,
    Seed = 42
};

var sampler = config.CreateSamplingStrategy();
var engine = new TransformerInferenceEngine(model, sampler);
```
