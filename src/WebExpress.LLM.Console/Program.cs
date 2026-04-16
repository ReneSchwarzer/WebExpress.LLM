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
    /// Command-line arguments. The first argument, if provided, should be the path to the model directory
    /// containing the configuration and weights files. If not provided, the application will use a
    /// deterministic inference engine for demonstration purposes.
    /// </param>
    private static void Main(string[] args)
    {
        // Display welcome message to the user
        System.Console.WriteLine("WebExpress.LLM - Interactive Chat");
        System.Console.WriteLine("==================================");
        System.Console.WriteLine();

        // Initialize the tokenizer for converting text to/from token sequences
        ITokenizer tokenizer = new ByteTokenizer();

        // Initialize the inference engine based on whether a model path was provided
        IInferenceEngine inferenceEngine;

        if (args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
        {
            // User provided a model directory path - load the actual model
            var modelPath = args[0];
            System.Console.WriteLine($"Loading model from: {modelPath}");
            System.Console.WriteLine();

            try
            {
                // Load the model configuration and weights from the specified directory
                var loader = new ModelLoader();
                var model = loader.Load(modelPath);

                // Create a transformer inference engine with greedy sampling strategy
                var samplingStrategy = new GreedySampling();
                inferenceEngine = new TransformerInferenceEngine(model, samplingStrategy);

                System.Console.WriteLine("Model loaded successfully.");
            }
            catch (Exception ex)
            {
                // Handle model loading errors gracefully
                System.Console.WriteLine($"Error loading model: {ex.Message}");
                System.Console.WriteLine("Falling back to deterministic inference engine.");
                inferenceEngine = new DeterministicInferenceEngine();
            }
        }
        else
        {
            // No model path provided - use deterministic inference engine for demonstration
            System.Console.WriteLine("No model path provided. Using deterministic inference engine.");
            System.Console.WriteLine("To use a real model, provide the model directory path as a command-line argument.");
            System.Console.WriteLine();
            inferenceEngine = new DeterministicInferenceEngine();
        }

        // Create a new chat session with the configured tokenizer and inference engine
        var chatSession = new ChatSession(tokenizer, inferenceEngine);

        System.Console.WriteLine("Chat session started. Type 'exit' or 'quit' to end the session.");
        System.Console.WriteLine();

        // Enter the interactive chat loop
        while (true)
        {
            // Prompt the user for input
            System.Console.Write("You: ");
            var userInput = System.Console.ReadLine();

            // Check if the user wants to exit the application
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
                // Send the user's message to the chat session and generate a response
                var response = chatSession.Send(userInput, maxNewTokens: 100);

                // Display the assistant's response to the user
                System.Console.WriteLine($"Assistant: {response.Content}");
                System.Console.WriteLine();
            }
            catch (Exception ex)
            {
                // Handle any errors that occur during message processing
                System.Console.WriteLine($"Error: {ex.Message}");
                System.Console.WriteLine();
            }
        }
    }
}
