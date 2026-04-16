![WebExpress-Framework](https://raw.githubusercontent.com/webexpress-framework/.github/main/docs/assets/img/banner.png)

# WebExpress.LLM

WebExpress.LLM is a local inference framework for the Gemma 4 large language model, implemented in C# on .NET 10.
It provides modular building blocks to run basic chat-style inference entirely on the local machine without external services,
REST APIs, or HTTP endpoints.

## Repository structure

- `/src/WebExpress.LLM` – core framework library
- `/src/WebExpress.LLM.Console` – interactive console application
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

## Running the console application

The `WebExpress.LLM.Console` project provides an interactive chat interface for conversing with the language model. The application uses an XML configuration file located at `config/webexpress.llm.config.xml` that defines all runtime parameters including model paths, inference settings, and tokenizer options.

### Configuration file

The configuration file (`config/webexpress.llm.config.xml`) contains:
- **Model settings**: Model name and path to the model directory
- **Inference settings**: MaxNewTokens, Temperature, TopK/TopP sampling, and optional Seed
- **Tokenizer settings**: Type of tokenizer to use (currently "byte")
- **Runtime settings**: Option to use deterministic inference engine for testing

### Basic usage

```bash
cd src/WebExpress.LLM.Console
dotnet run
```

The application will automatically load settings from the default configuration file. By default, it will attempt to load the model specified in the configuration file. If the model path does not exist, it will fall back to the deterministic inference engine.

### Using a custom configuration file

```bash
cd src/WebExpress.LLM.Console
dotnet run -- /path/to/custom/config.xml
```

### Configuration example

The configuration file supports the following settings:

```xml
<?xml version="1.0" encoding="utf-8" ?>
<config version="1">
  <model name="google/gemma-4-E2B-it">
    <path>../../../../../model/google/gemma-4-E2B-it</path>
  </model>

  <inference>
    <maxNewTokens>100</maxNewTokens>
    <temperature>1.0</temperature>
    <!-- Optional: <topK>50</topK> -->
    <!-- Optional: <topP>0.9</topP> -->
    <!-- Optional: <seed>42</seed> -->
  </inference>

  <tokenizer type="byte" />

  <runtime>
    <useDeterministicEngine>false</useDeterministicEngine>
  </runtime>
</config>
```

The model directory must contain:
- `config.json` – model configuration file
- `model.weights` – model weights file

### Interactive commands

Once the console application is running:
- Type your messages and press Enter to receive responses from the assistant
- Type `exit` or `quit` to end the session

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
