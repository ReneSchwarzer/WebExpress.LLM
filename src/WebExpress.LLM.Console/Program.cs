using System;
using System.IO;
using System.Threading.Tasks;
using WebExpress.LLM.Chat;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Model;
using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Console;

/// <summary>
/// Entry point for the WebExpress.LLM console application.
/// Provides an interactive chat interface for conversing with the Gemma-4 language model.
/// </summary>
internal class Program
{
    /// <summary>
    /// The main entry point for the application. Initializes the language model and enters
    /// an interactive loop where the user can input messages and receive AI-generated responses.
    /// </summary>
    /// <param name="args">
    /// Command-line arguments. The first argument, if provided, should be the path to the
    /// configuration file. If not provided, the default configuration file will be used.
    /// </param>
    /// /// <returns>
    /// Returns 0 on success; any value greater than 0 indicates an error.
    /// </returns>
    private static async Task<int> Main(string[] args)
    {
        // display welcome message to the user
        System.Console.WriteLine("WebExpress.LLM - Interactive Chat");
        System.Console.WriteLine("==================================");
        System.Console.WriteLine();

        // load application configuration from XML file
        ApplicationConfiguration config;
        var configPath = args.Length > 0 ? args[0] : null;

        try
        {
            var configLoader = new ConfigurationLoader();
            config = configLoader.Load(configPath);
            System.Console.WriteLine($"Configuration loaded: {config.ModelName}");
            System.Console.WriteLine();
        }
        catch (Exception ex)
        {
            System.Console.WriteLine($"Error loading configuration: {ex.Message}");
            System.Console.WriteLine("Application cannot start without a valid configuration file.");
            return 1;
        }

        // initialize the tokenizer based on configuration
        ITokenizer tokenizer = config.TokenizerType.ToLowerInvariant() switch
        {
            "byte" => new ByteTokenizer(),
            _ => throw new InvalidOperationException($"Unsupported tokenizer type: {config.TokenizerType}")
        };

        // initialize the inference engine based on configuration
        IInferenceEngine inferenceEngine;
        ModelDefinition model = null;

        if (config.UseDeterministicEngine)
        {
            // use deterministic inference engine for testing
            System.Console.WriteLine("Using deterministic inference engine (testing mode).");
            System.Console.WriteLine();
            inferenceEngine = new DeterministicInferenceEngine();
        }
        else if (Directory.Exists(config.ModelPath))
        {
            // load the actual model from the configured path
            System.Console.WriteLine($"Loading model from: {config.ModelPath}");
            System.Console.WriteLine();

            try
            {
                // Load the model configuration and weights from the specified directory
                var loader = new ModelLoader();
                model = loader.Load(config.ModelPath);

                // create generation configuration from application settings
                var generationConfig = new GenerationConfig
                {
                    MaxNewTokens = config.MaxNewTokens,
                    Temperature = config.Temperature,
                    TopK = config.TopK,
                    TopP = config.TopP,
                    Seed = config.Seed
                };

                // create sampling strategy based on generation configuration
                var samplingStrategy = generationConfig.CreateSamplingStrategy();
                inferenceEngine = new TransformerInferenceEngine(model, samplingStrategy);

                System.Console.WriteLine("Model loaded successfully.");
                System.Console.WriteLine($"Inference settings: MaxTokens={config.MaxNewTokens}, Temperature={config.Temperature}");
                if (config.TopK.HasValue)
                {
                    System.Console.WriteLine($"  Sampling: Top-K (k={config.TopK.Value})");
                }
                else if (config.TopP.HasValue)
                {
                    System.Console.WriteLine($"  Sampling: Top-P (p={config.TopP.Value})");
                }
                else
                {
                    System.Console.WriteLine("  Sampling: Greedy");
                }
                System.Console.WriteLine();
            }
            catch (Exception ex)
            {
                // handle model loading errors gracefully
                System.Console.WriteLine($"Error loading model: {ex.Message}");

                return 2;
            }
        }
        else
        {
            // model path does not exist
            System.Console.WriteLine($"Model path does not exist: {config.ModelPath}");

            return 3;
        }

        // store max tokens from configuration for use during chat
        var maxNewTokens = config.MaxNewTokens;

        // create a new chat session with the configured tokenizer and inference engine
        var chatSession = new ChatSession(tokenizer, inferenceEngine);

        System.Console.WriteLine("Chat session started. Type 'exit' or 'quit' to end the session.");
        System.Console.WriteLine();

        // enter the interactive chat loop
        while (true)
        {
            // prompt the user for input
            System.Console.Write(">");
            var userInput = System.Console.ReadLine();

            // check if the user wants to exit the application
            if (string.IsNullOrWhiteSpace(userInput))
            {
                continue;
            }

            if (userInput.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                userInput.Trim().Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                System.Console.WriteLine("Goodbye!");
                break;
            }

            try
            {
                // send the user's message to the chat session and stream the response
                System.Console.Write("Assistant: ");

                await foreach (var textChunk in chatSession.SendAsync(userInput, maxNewTokens: maxNewTokens))
                {
                    System.Console.Write(textChunk);
                }

                System.Console.WriteLine();
                System.Console.WriteLine();
            }
            catch (Exception ex)
            {
                // handle any errors that occur during message processing
                System.Console.WriteLine($"Error: {ex.Message}");
                System.Console.WriteLine();
            }
        }

        // dispose model when application exits
        model?.Dispose();

        return 0;
    }
}
