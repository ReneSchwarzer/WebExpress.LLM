# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

WebExpress.LLM is a local inference framework for the Gemma-4 language model, implemented in pure C# on .NET 10 — no TorchSharp, PyTorch, or cloud APIs.

## Build and Test

```bash
dotnet build src/WebExpress.LLM.slnx
dotnet test src/WebExpress.LLM.slnx

# Run a single test class
dotnet test src/WebExpress.LLM.slnx --filter "FullyQualifiedName~ChatSessionTest"

# Run the console app
cd src/WebExpress.LLM.Console && dotnet run
```

The solution file is `src/WebExpress.LLM.slnx` (`.slnx` is the modern format). Target framework is .NET 10.

## Architecture

Three projects:

- **`WebExpress.LLM`** — Core library: transformer, tensors, tokenizers, inference, model loading.
- **`WebExpress.LLM.Console`** — Interactive CLI with real-time streaming. Reads `config/webexpress.llm.config.xml`; falls back to `DeterministicInferenceEngine` if model weights are missing.
- **`WebExpress.LLM.Test`** — xUnit tests (26 test files) organized to mirror the library namespace.

### Key namespaces in `WebExpress.LLM`

| Namespace | Purpose |
|-----------|---------|
| `Gemma/` | Transformer layers: `Gemma4Model`, `MultiHeadAttention` (GQA + sliding window), `FeedForward`, `RotaryEmbedding`, `KvCache` |
| `Tensor/` | Native tensor library — `Tensor` (1-3D float arrays), `TensorOperations` (MatMul, Softmax, RmsNorm, EmbeddingLookup, broadcasting) |
| `Inference/` | `IInferenceEngine` / `TransformerInferenceEngine` / `DeterministicInferenceEngine`; `ISamplingStrategy` / `GreedySampling` / `TopKSampling` / `TopPSampling`; `GenerationConfig` |
| `Model/` | `ModelLoader` (config.json + weights), `ModelConfiguration`, `ModelWeights`, sharded-model support via SafeTensorIndex |
| `SafeTensors/` | `ISafeTensorLoader`, `SafeTensorLoader` (single file), `ShardedSafeTensorLoader` (multi-file); memory-mapped to handle >2 GB weights |
| `Tokenization/` | `ITokenizer`; `GemmaTokenizer` (BPE), `SentencePieceTokenizer`, `ByteTokenizer` (UTF-8 fallback) |
| `Chat/` | `ChatSession` (conversation history + async streaming), `ChatMessage`, `ChatTemplate` (Jinja from `chat_template.jinja`) |

### Generation pipeline

```
User input
  → Tokenizer.Encode()
  → ChatTemplate formatting (optional Jinja)
  → IInferenceEngine.GenerateTokensAsync()   // IAsyncEnumerable<int>
  → ISamplingStrategy selects next token
  → KvCache avoids recomputation per layer
  → Tokenizer.Decode()
  → streamed back via ChatSession.SendAsync()  // IAsyncEnumerable<string>
```

### Design patterns

- **Interface-based, pluggable**: `ITokenizer`, `IInferenceEngine`, `ISamplingStrategy`, `ISafeTensorLoader` can all be swapped.
- **Async streaming first**: generation and chat are `IAsyncEnumerable<T>` throughout.
- **Zero-copy tensors**: `Span<float>` used in hot paths; 32×32 tile-based transpose for cache efficiency.
- **Memory-mapped weight loading**: avoids heap pressure for large SafeTensors files.
- **KV-cache**: keys/values stored per layer; cleared between sessions.

## Documentation

Architecture detail lives in `docs/`:
- `docs/GEMMA4_INTEGRATION.md` — 11-step process flow, component breakdown, known limitations.
- `docs/IMPLEMENTATION_SUMMARY.md` — problem/solution history and completed work.
- `docs/CHAT_TEMPLATE.md` — Jinja chat template format.

Treat `docs/` as authoritative for architectural decisions, naming rules, and component guidelines.
