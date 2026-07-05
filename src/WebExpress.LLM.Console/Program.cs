using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
            System.Console.WriteLine($"Model path loaded: {config.ModelPath}");
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
            "sentencepiece" => CreateSentencePieceTokenizer(config),
            "gemma" => CreateGemmaTokenizer(config),
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
            System.Console.WriteLine($"Loading model: {config.ModelName}");

            try
            {
                // load the model configuration and weights from the specified directory
                var loader = new ModelLoader();
                model = loader.Load(Path.Combine(config.ModelPath, config.ModelName));

                // create generation configuration from application settings
                var generationConfig = new GenerationConfig
                {
                    MaxNewTokens = config.MaxNewTokens,
                    Temperature = config.Temperature,
                    TopK = config.TopK,
                    TopP = config.TopP,
                    Seed = config.Seed,
                    RepetitionPenalty = config.RepetitionPenalty
                };

                // create sampling strategy based on generation configuration
                var samplingStrategy = generationConfig.CreateSamplingStrategy();
                inferenceEngine = new TransformerInferenceEngine(model, samplingStrategy);

                System.Console.Write($"Inference settings: MaxTokens={config.MaxNewTokens}, Temperature={config.Temperature}, RepPenalty={config.RepetitionPenalty}");
                if (config.TopK.HasValue && config.TopP.HasValue)
                {
                    System.Console.WriteLine($", Sampling: Top-K + Top-P (k={config.TopK.Value}, p={config.TopP.Value})");
                }
                else if (config.TopK.HasValue)
                {
                    System.Console.WriteLine($", Sampling: Top-K (k={config.TopK.Value})");
                }
                else if (config.TopP.HasValue)
                {
                    System.Console.WriteLine($", Sampling: Top-P (p={config.TopP.Value})");
                }
                else if (config.Temperature != 1.0f)
                {
                    System.Console.WriteLine(", Sampling: Temperature");
                }
                else
                {
                    System.Console.WriteLine(", Sampling: Greedy");
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
        var chatSession = new ChatSession(tokenizer, inferenceEngine, model.ChatTemplate);

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

                var tokenCount = 0;
                var stopwatch = Stopwatch.StartNew();
                var previousElapsedSeconds = 0.0;
                var process = Process.GetCurrentProcess();
                var previousCpuTime = process.TotalProcessorTime;
                var previousCpuSample = stopwatch.Elapsed;
                var previousIoOps = GetProcessIoOperationCount(process) ?? 0UL;

                await foreach (var textChunk in chatSession.SendAsync(userInput, maxNewTokens: maxNewTokens))
                {
                    tokenCount++;
                    var elapsedSeconds = Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9);
                    var tokensPerSecond = tokenCount / elapsedSeconds;
                    var lastTokenSeconds = Math.Max(elapsedSeconds - previousElapsedSeconds, 0.0);
                    previousElapsedSeconds = elapsedSeconds;

                    process.Refresh();
                    var nowCpuTime = process.TotalProcessorTime;
                    var nowSample = stopwatch.Elapsed;
                    var wallSeconds = Math.Max((nowSample - previousCpuSample).TotalSeconds, 1e-9);
                    var cpuPercent = Math.Clamp(
                        (nowCpuTime - previousCpuTime).TotalSeconds / (wallSeconds * Environment.ProcessorCount) * 100.0,
                        0.0,
                        100.0);
                    previousCpuTime = nowCpuTime;
                    previousCpuSample = nowSample;

                    var ramGb = process.WorkingSet64 / (1024.0 * 1024.0 * 1024.0);
                    var ioOps = GetProcessIoOperationCount(process) ?? previousIoOps;
                    var hddOpsPerSecond = Math.Max((ioOps - previousIoOps) / wallSeconds, 0.0);
                    previousIoOps = ioOps;

                    WriteTopRightStatus(
                        $"Token/s: {tokensPerSecond:0.000000}",
                        $"Last: {lastTokenSeconds:0.000} s",
                        $"CPU:  {cpuPercent:0.0}%",
                        $"RAM:  {ramGb:0.00} GB",
                        $"HDD:  {hddOpsPerSecond:0.0} ops/s");

                    System.Console.Write($"{textChunk} ");
                }

                System.Console.WriteLine();
                System.Console.WriteLine();
            }
            catch (Exception ex)
            {
                // handle any errors that occur during message processing
                System.Console.WriteLine($"Error: {ex.Message}");
                System.Console.WriteLine($"{ex.StackTrace}");
                System.Console.WriteLine();
            }
        }

        // dispose model when application exits
        model?.Dispose();

        return 0;
    }

    /// <summary>
    /// Writes the specified status text to the top-right corner of the console window without altering 
    /// the current cursor position.
    /// </summary>
    private static void WriteTopRightStatus(params string[] lines)
    {
        if (System.Console.IsOutputRedirected)
        {
            return;
        }

        if (lines == null || lines.Length == 0)
        {
            return;
        }

        try
        {
            var cursorLeft = System.Console.CursorLeft;
            var cursorTop = System.Console.CursorTop;

            var width = System.Console.WindowWidth;
            var statusLines = lines
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Select(x => $"     {x}")
                .ToArray();

            if (statusLines.Length == 0)
            {
                return;
            }

            var statusWidth = statusLines.Max(x => x.Length);
            var column = Math.Max(0, width - statusWidth);

            for (var lineIndex = 0; lineIndex < statusLines.Length && lineIndex < System.Console.WindowHeight; lineIndex++)
            {
                System.Console.SetCursorPosition(column, lineIndex);
                System.Console.Write(statusLines[lineIndex].PadRight(statusWidth));
            }

            System.Console.SetCursorPosition(cursorLeft, cursorTop);
        }
        catch (ArgumentOutOfRangeException)
        {
        }
        catch (IOException)
        {
        }
    }

    private static ulong? GetProcessIoOperationCount(Process process)
    {
        ArgumentNullException.ThrowIfNull(process);

        try
        {
            if (!GetProcessIoCounters(process.Handle, out var counters))
            {
                return null;
            }

            return counters.ReadOperationCount + counters.WriteOperationCount;
        }
        catch
        {
            return null;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessIoCounters(IntPtr hProcess, out IoCounters lpIoCounters);

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    /// <summary>
    /// Creates a <see cref="SentencePieceTokenizer"/> from the application configuration.
    /// Loads the binary SentencePiece <c>.model</c> file and optionally reads
    /// <c>tokenizer_config.json</c> for special-token flags (<c>add_bos_token</c>/<c>add_eos_token</c>).
    /// </summary>
    private static SentencePieceTokenizer CreateSentencePieceTokenizer(ApplicationConfiguration config)
    {
        var modelDir = Path.Combine(config.ModelPath, config.ModelName);

        // Load tokenizer_config.json when present (optional)
        TokenizerConfiguration tokenizerConfig = null;
        var tokenizerConfigPath = Path.Combine(modelDir, "tokenizer_config.json");

        if (File.Exists(tokenizerConfigPath))
        {
            tokenizerConfig = TokenizerConfiguration.FromFile(tokenizerConfigPath);
        }

        // Load the SentencePiece binary .model file.
        var spModelPath = Path.IsPathRooted(config.TokenizerModelPath)
            ? config.TokenizerModelPath
            : Path.Combine(modelDir, config.TokenizerModelPath);

        return SentencePieceTokenizer.FromModel(spModelPath, tokenizerConfig);
    }

    /// <summary>
    /// Creates a <see cref="GemmaTokenizer"/> from the application configuration.
    /// Reads <c>tokenizer.json</c> (HuggingFace tokenizers format) and optionally reads
    /// <c>tokenizer_config.json</c> for special-token names and <c>add_bos_token</c>/<c>add_eos_token</c> flags.
    /// </summary>
    private static GemmaTokenizer CreateGemmaTokenizer(ApplicationConfiguration config)
    {
        var modelDir = Path.Combine(config.ModelPath, config.ModelName);

        // Load tokenizer_config.json when present (optional)
        TokenizerConfiguration tokenizerConfig = null;
        var tokenizerConfigPath = Path.Combine(modelDir, "tokenizer_config.json");

        if (File.Exists(tokenizerConfigPath))
        {
            tokenizerConfig = TokenizerConfiguration.FromFile(tokenizerConfigPath);
        }

        // Load tokenizer.json (required for Gemma)
        var tokenizerJsonPath = Path.IsPathRooted(config.TokenizerModelPath)
            ? config.TokenizerModelPath
            : Path.Combine(modelDir, config.TokenizerModelPath);

        return GemmaTokenizer.FromTokenizerJson(tokenizerJsonPath, tokenizerConfig);
    }
}
