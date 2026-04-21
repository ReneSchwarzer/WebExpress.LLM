using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements a transformer-based inference engine for the Gemma-4 model.
/// </summary>
/// <remarks>
/// When the model weights are in SafeTensors format, this engine performs a full
/// Gemma-4 forward pass through:
/// <list type="number">
///   <item>Token embedding lookup with scaling</item>
///   <item>35 transformer layers (multi-head attention, RMS normalization, gated FFN, residual connections)</item>
///   <item>Final RMS normalization</item>
///   <item>Linear projection to vocabulary logits</item>
/// </list>
///
/// When valid SafeTensors weights are not available, the engine falls back to a
/// placeholder implementation that generates readable text-like output for testing
/// purposes.
///
/// All tensor operations, memory handling, and mathematical primitives are implemented
/// using native .NET functionality without external libraries.
/// </remarks>
public sealed class TransformerInferenceEngine : IInferenceEngine
{
    private readonly ModelDefinition _model;
    private readonly ISamplingStrategy _samplingStrategy;
    private readonly Gemma4Model _gemmaModel;

    /// <summary>
    /// Initializes a new instance of the TransformerInferenceEngine class with the specified model  
    /// and sampling strategy.
    /// </summary>
    /// <param name="model">
    /// The model description used for inference operations. Must not be null.
    /// </param>
    /// <param name="samplingStrategy">
    /// The sampling strategy used to control text generation. Must not be null.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="model"/> or <paramref name="samplingStrategy"/> is null.
    /// </exception>
    public TransformerInferenceEngine(ModelDefinition model, ISamplingStrategy samplingStrategy)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _samplingStrategy = samplingStrategy ?? throw new ArgumentNullException(nameof(samplingStrategy));

        // Attempt to initialize the Gemma-4 model from SafeTensors weights.
        // If the weights are not in SafeTensors format or are too small, fall back
        // to the placeholder implementation.
        _gemmaModel = TryCreateGemmaModel(model);
    }

    /// <summary>
    /// Generates a sequence of new tokens based on the provided prompt tokens, up to the specified maximum number of
    /// tokens.
    /// </summary>
    /// <remarks>
    /// The generated tokens are produced sequentially, with each new token appended to the context
    /// for subsequent generation. If the context exceeds the model's maximum context length, the oldest tokens are
    /// removed to maintain the limit.
    /// </remarks>
    /// <param name="promptTokens">The sequence of input tokens that serves as the initial context for generation. Cannot be null.</param>
    /// <param name="maxNewTokens">The maximum number of new tokens to generate. Must be greater than or equal to zero.</param>
    /// <returns>A read-only list containing the generated tokens. The list will be empty if maxNewTokens is zero.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if maxNewTokens is less than zero.</exception>
    public IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);

        if (maxNewTokens < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "Token count must be greater than or equal to zero.");
        }

        if (maxNewTokens == 0)
        {
            return Array.Empty<int>();
        }

        _gemmaModel?.ResetCache();

        var generatedTokens = new List<int>();
        var contextTokens = new List<int>(promptTokens);

        for (var i = 0; i < maxNewTokens; i++)
        {
            var logits = ForwardPass(contextTokens);
            var nextToken = _samplingStrategy.Sample(logits);

            generatedTokens.Add(nextToken);
            contextTokens.Add(nextToken);

            if (contextTokens.Count > _model.Configuration.ContextLength)
            {
                contextTokens.RemoveAt(0);
            }
        }

        return generatedTokens;
    }

    /// <summary>
    /// Asynchronously generates tokens one at a time, yielding each token as it is produced.
    /// This enables streaming token generation for real-time display in chat interfaces.
    /// </summary>
    /// <remarks>
    /// Generates new tokens autoregressively from a given prompt sequence using the model.
    /// For each token step:
    /// - Runs a forward pass using the full current context
    /// - Selects the next token index based on the configured sampling strategy (e.g., argmax or probabilistic sampling)
    /// - Streams each generated token asynchronously via yield
    /// - Extends the context with the new token and manages context length according to model limits
    /// Designed for efficient asynchronous streaming in interactive or server applications.
    /// </remarks>
    /// <param name="promptTokens">The sequence of input tokens that serves as the initial context for generation. Cannot be null.</param>
    /// <param name="maxNewTokens">The maximum number of new tokens to generate. Must be greater than or equal to zero.</param>
    /// <returns>An async enumerable that yields generated tokens one at a time.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if maxNewTokens is less than zero.</exception>
    public async IAsyncEnumerable<int> GenerateTokensAsync(IReadOnlyList<int> promptTokens, int maxNewTokens)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);

        if (maxNewTokens < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "Token count must be greater than or equal to zero.");
        }

        if (maxNewTokens == 0)
        {
            yield break;
        }

        _gemmaModel?.ResetCache();

        var contextTokens = new List<int>(promptTokens);

        for (var i = 0; i < maxNewTokens; i++)
        {
            // Yield to allow the scheduler to process other work
            await Task.Yield();

            var logits = ForwardPass(contextTokens);
            var nextToken = _samplingStrategy.Sample(logits);

            yield return nextToken;

            contextTokens.Add(nextToken);

            if (contextTokens.Count > _model.Configuration.ContextLength)
            {
                contextTokens.RemoveAt(0);
            }

            //System.Console.WriteLine($"Generating {maxNewTokens} tokens with context [{string.Join(",", contextTokens)}]");
        }
    }

    /// <summary>
    /// Computes the logit values for the specified token sequence using either the full
    /// Gemma-4 transformer model or a placeholder implementation.
    /// </summary>
    /// <param name="tokens">
    /// A read‑only list of token IDs used as input for computing the logits.
    /// </param>
    /// <returns>
    /// An array of floating‑point numbers containing the computed logit values
    /// for each token in the vocabulary.
    /// </returns>
    private float[] ForwardPass(IReadOnlyList<int> tokens)
    {
        // Use the real Gemma-4 model when available
        if (_gemmaModel != null && tokens.Count > 0)
        {
            var tokenArray = new int[tokens.Count];

            for (var i = 0; i < tokens.Count; i++)
            {
                tokenArray[i] = tokens[i];
            }

            return _gemmaModel.Forward(tokenArray);
        }

        // Fallback: placeholder implementation for testing and development
        return PlaceholderForwardPass(tokens);
    }

    /// <summary>
    /// Placeholder forward pass that generates logits biased toward readable English
    /// characters. Used when real model weights are not available.
    /// </summary>
    private float[] PlaceholderForwardPass(IReadOnlyList<int> tokens)
    {
        var vocabSize = _model.Configuration.VocabularySize;
        var logits = new float[vocabSize];

        var seed = tokens.Count > 0 ? tokens[^1] : 0;

        for (var i = 0; i < vocabSize; i++)
        {
            var baseValue = (float)((seed * 7 + i * 13) % 1000) / 1000.0f;

            if (i == 32)
            {
                logits[i] = baseValue + 2.0f;
            }
            else if (i >= 97 && i <= 122)
            {
                logits[i] = baseValue + 1.5f;
            }
            else if (i >= 65 && i <= 90)
            {
                logits[i] = baseValue + 1.0f;
            }
            else if (i == 46 || i == 44 || i == 33 || i == 63)
            {
                logits[i] = baseValue + 0.8f;
            }
            else if (i >= 48 && i <= 57)
            {
                logits[i] = baseValue + 0.5f;
            }
            else if (i == 10 || i == 13)
            {
                logits[i] = baseValue + 0.3f;
            }
            else
            {
                logits[i] = baseValue;
            }
        }

        return logits;
    }

    /// <summary>
    /// Attempts to create a Gemma4Model from the model weights. Returns null if the
    /// weights are not in SafeTensors format or cannot be parsed.
    /// </summary>
    private static Gemma4Model TryCreateGemmaModel(ModelDefinition model)
    {
        try
        {
            // If the model uses sharded weights, use the sharded loader directly
            if (model.ShardedLoader != null)
            {
                if (!model.ShardedLoader.ContainsTensor("model.language_model.embed_tokens.weight"))
                {
                    return null;
                }

                return new Gemma4Model(model.Configuration, model.ShardedLoader);
            }

            // SafeTensors requires at least 8 bytes for the header length
            if (model.Weights == null || model.Weights.Length < 8)
            {
                return null;
            }

            var loader = new SafeTensorLoader(model.Weights);

            // Verify that the weights contain the expected embedding tensor
            if (!loader.ContainsTensor("model.language_model.embed_tokens.weight"))
            {
                return null;
            }

            return new Gemma4Model(model.Configuration, loader);
        }
        catch (Exception ex) when (
            ex is InvalidDataException ||
            ex is KeyNotFoundException ||
            ex is ArgumentException ||
            ex is FormatException ||
            ex is System.Text.Json.JsonException)
        {
            // If weight parsing fails for known reasons, fall back to placeholder
            return null;
        }
    }
}
