![WebExpress-Framework](https://raw.githubusercontent.com/webexpress-framework/.github/main/docs/assets/img/banner.png)

# WebExpress.LLM

WebExpress.LLM is a high-performance local inference framework for the Gemma-4 large language model, implemented in pure C# on .NET 10.
It provides modular building blocks to run full transformer-based inference entirely on the local machine without external services,
heavy dependencies (like TorchSharp/PyTorch), or cloud APIs.

## Repository structure

- `/src/WebExpress.LLM` – core framework library (transformer, tensors, tokenizers)
- `/src/WebExpress.LLM.Console` – interactive streaming console application
- `/src/WebExpress.LLM.Test` – comprehensive xUnit test suite
- `/docs` – architecture and integration documentation
- `/WebExpress.LLM.slnx` – solution file

## Included framework components

### Core Components
- **Full Gemma-4 Architecture**: Complete implementation of multi-head attention (GQA/Sliding Window), RMS normalization, and gated FFN.
- **Native Tensor Library**: Optimized C# tensor implementation with efficient memory management via `Span<float>`.
- **Model Loading**: Support for single-file and sharded SafeTensors models with memory-mapping for large weights (>2GB).
- **Tokenization**:
  - `GemmaTokenizer`: BPE-based tokenizer specifically for Gemma-4.
  - `SentencePieceTokenizer`: Support for SentencePiece-based model vocabularies.
  - `ByteTokenizer`: UTF-8 byte-level fallback tokenizer.
- **Inference Engines**:
  - `TransformerInferenceEngine`: High-performance engine utilizing the full Gemma-4 transformer architecture.
  - `DeterministicInferenceEngine`: Reproducible mock for testing and development.
- **Sampling Strategies**: Greedy, Top-K, and Nucleus (Top-P) sampling.
- **KV Cache**: Efficient state management for high-speed autoregressive generation.
- **Chat Session**: Conversation history tracking and async streaming orchestration.

## Build and test

```bash
dotnet build src/WebExpress.LLM.slnx
dotnet test src/WebExpress.LLM.slnx
```

## Running the console application

The `WebExpress.LLM.Console` project provides an interactive chat interface with real-time token streaming. It uses an XML configuration file located at `config/webexpress.llm.config.xml`.

### Configuration file

The configuration supports:
- **Model settings**: Name and absolute/relative path to the model directory.
- **Inference settings**: MaxNewTokens, Temperature, TopK, TopP, and Seed.
- **Tokenizer settings**: Type of tokenizer (e.g., "gemma", "byte").
- **Runtime settings**: Toggle between real transformer and deterministic engine.

### Basic usage

```bash
cd src/WebExpress.LLM.Console
dotnet run
```

The application will attempt to load the configured model. If weights are missing or incompatible, it falls back to a placeholder mode that generates readable English text.

## Example usage

### Async Streaming Chat
```csharp
using WebExpress.LLM.Chat;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

var tokenizer = new GemmaTokenizer(vocab, merges);
var engine = new TransformerInferenceEngine(model, new TopPSampling(0.9f));
var chat = new ChatSession(tokenizer, engine);

await foreach (var chunk in chat.SendAsync("What is quantum computing?"))
{
    Console.Write(chunk);
}
```

### Manual Tensor Operations
```csharp
using WebExpress.LLM.Tensor;

var a = new Tensor([2, 2], [1f, 2f, 3f, 4f]);
var b = new Tensor([2, 2], [5f, 6f, 7f, 8f]);
var result = TensorOperations.MatMul(a, b);
```

## Documentation

For more detailed information, see:
- [Gemma-4 Integration Guide](docs/GEMMA4_INTEGRATION.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
