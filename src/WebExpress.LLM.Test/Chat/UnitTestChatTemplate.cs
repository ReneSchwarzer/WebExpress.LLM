using WebExpress.LLM.Chat;

namespace WebExpress.LLM.Test.Chat;

/// <summary>
/// Provides unit tests for the <see cref="ChatTemplate"/> class, covering template loading,
/// construction, and prompt formatting behavior.
/// </summary>
public sealed class UnitTestChatTemplate
{
    private const string SampleTemplate = "{%- for message in messages -%}<|turn>{{ message.role }}\n{{ message.content }}<turn|>\n{%- endfor -%}";

    #region Constructor Tests

    /// <summary>
    /// Tests that constructing a ChatTemplate with valid content stores the template content.
    /// </summary>
    [Fact]
    public void Constructor_WithValidContent_ShouldStoreTemplateContent()
    {
        var template = new ChatTemplate(SampleTemplate);

        Assert.Equal(SampleTemplate, template.TemplateContent);
    }

    /// <summary>
    /// Tests that the default BOS token is set to the expected value.
    /// </summary>
    [Fact]
    public void Constructor_DefaultBosToken_ShouldBeBos()
    {
        var template = new ChatTemplate(SampleTemplate);

        Assert.Equal("<bos>", template.BosToken);
    }

    /// <summary>
    /// Tests that the BOS token can be overridden via the init property.
    /// </summary>
    [Fact]
    public void Constructor_CustomBosToken_ShouldBeUsed()
    {
        var template = new ChatTemplate(SampleTemplate) { BosToken = "<s>" };

        Assert.Equal("<s>", template.BosToken);
    }

    /// <summary>
    /// Tests that constructing a ChatTemplate with null content throws an ArgumentException.
    /// </summary>
    [Fact]
    public void Constructor_WithNullContent_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new ChatTemplate(null!));
    }

    /// <summary>
    /// Tests that constructing a ChatTemplate with empty content throws an ArgumentException.
    /// </summary>
    [Fact]
    public void Constructor_WithEmptyContent_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new ChatTemplate(string.Empty));
    }

    /// <summary>
    /// Tests that constructing a ChatTemplate with whitespace content throws an ArgumentException.
    /// </summary>
    [Fact]
    public void Constructor_WithWhitespaceContent_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new ChatTemplate("   "));
    }

    #endregion

    #region FromFile Tests

    /// <summary>
    /// Tests that FromFile loads the template content from a valid file.
    /// </summary>
    [Fact]
    public void FromFile_WithValidFile_ShouldLoadContent()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var templatePath = Path.Combine(tempPath, ChatTemplate.DefaultFileName);
            File.WriteAllText(templatePath, SampleTemplate);

            var template = ChatTemplate.FromFile(templatePath);

            Assert.Equal(SampleTemplate, template.TemplateContent);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    /// <summary>
    /// Tests that FromFile throws FileNotFoundException when the file does not exist.
    /// </summary>
    [Fact]
    public void FromFile_WithNonExistentFile_ShouldThrowFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() => ChatTemplate.FromFile("/nonexistent/path/template.jinja"));
    }

    /// <summary>
    /// Tests that FromFile throws ArgumentException when the file path is null.
    /// </summary>
    [Fact]
    public void FromFile_WithNullPath_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => ChatTemplate.FromFile(null!));
    }

    /// <summary>
    /// Tests that FromFile throws ArgumentException when the file path is empty.
    /// </summary>
    [Fact]
    public void FromFile_WithEmptyPath_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => ChatTemplate.FromFile(string.Empty));
    }

    /// <summary>
    /// Tests that the default file name constant is set correctly.
    /// </summary>
    [Fact]
    public void DefaultFileName_ShouldBeChatTemplateJinja()
    {
        Assert.Equal("chat_template.jinja", ChatTemplate.DefaultFileName);
    }

    #endregion

    #region ApplyTemplate - Basic Formatting Tests

    /// <summary>
    /// Tests that ApplyTemplate with a single user message produces the correct turn-based format.
    /// </summary>
    [Fact]
    public void ApplyTemplate_SingleUserMessage_ShouldFormatCorrectly()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Equal("<bos><|turn>user\nHello<turn|>\n<|turn>model\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with a user-assistant exchange produces the correct format.
    /// </summary>
    [Fact]
    public void ApplyTemplate_UserAndAssistantMessages_ShouldFormatCorrectly()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello"),
            new("assistant", "Hi there!")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Equal(
            "<bos><|turn>user\nHello<turn|>\n<|turn>model\nHi there!<turn|>\n<|turn>model\n",
            result);
    }

    /// <summary>
    /// Tests that ApplyTemplate maps the "assistant" role to "model" in the output.
    /// </summary>
    [Fact]
    public void ApplyTemplate_AssistantRole_ShouldBeMappedToModel()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("assistant", "response text")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Contains("<|turn>model\n", result);
        Assert.DoesNotContain("<|turn>assistant\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with a system message at position 0 produces the system turn.
    /// </summary>
    [Fact]
    public void ApplyTemplate_SystemMessage_ShouldProduceSystemTurn()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("system", "You are helpful."),
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages);

        Assert.StartsWith("<bos><|turn>system\n", result);
        Assert.Contains("You are helpful.<turn|>\n", result);
        Assert.Contains("<|turn>user\nHello<turn|>\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with a developer role at position 0 produces a system turn.
    /// </summary>
    [Fact]
    public void ApplyTemplate_DeveloperMessage_ShouldProduceSystemTurn()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("developer", "System instructions."),
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages);

        Assert.StartsWith("<bos><|turn>system\n", result);
        Assert.Contains("System instructions.<turn|>\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate trims whitespace from user message content.
    /// </summary>
    [Fact]
    public void ApplyTemplate_UserMessage_ShouldTrimContent()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "  Hello  ")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Contains("<|turn>user\nHello<turn|>", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate trims whitespace from system message content.
    /// </summary>
    [Fact]
    public void ApplyTemplate_SystemMessage_ShouldTrimContent()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("system", "  Instructions  ")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Contains("<|turn>system\nInstructions<turn|>", result);
    }

    #endregion

    #region ApplyTemplate - Generation Prompt Tests

    /// <summary>
    /// Tests that ApplyTemplate appends a generation prompt when addGenerationPrompt is true.
    /// </summary>
    [Fact]
    public void ApplyTemplate_WithGenerationPrompt_ShouldAppendModelTurn()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages, addGenerationPrompt: true);

        Assert.EndsWith("<|turn>model\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate does not append a generation prompt when addGenerationPrompt is false.
    /// </summary>
    [Fact]
    public void ApplyTemplate_WithoutGenerationPrompt_ShouldNotAppendModelTurn()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages, addGenerationPrompt: false);

        Assert.Equal("<bos><|turn>user\nHello<turn|>\n", result);
    }

    #endregion

    #region ApplyTemplate - Edge Case Tests

    /// <summary>
    /// Tests that ApplyTemplate with null messages returns BOS token and generation prompt.
    /// </summary>
    [Fact]
    public void ApplyTemplate_NullMessages_ShouldReturnBosAndGenerationPrompt()
    {
        var template = new ChatTemplate(SampleTemplate);

        var result = template.ApplyTemplate(null, addGenerationPrompt: true);

        Assert.Equal("<bos><|turn>model\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with empty messages returns BOS token and generation prompt.
    /// </summary>
    [Fact]
    public void ApplyTemplate_EmptyMessages_ShouldReturnBosAndGenerationPrompt()
    {
        var template = new ChatTemplate(SampleTemplate);

        var result = template.ApplyTemplate([], addGenerationPrompt: true);

        Assert.Equal("<bos><|turn>model\n", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with null messages and no generation prompt returns only BOS token.
    /// </summary>
    [Fact]
    public void ApplyTemplate_NullMessagesNoGenerationPrompt_ShouldReturnOnlyBos()
    {
        var template = new ChatTemplate(SampleTemplate);

        var result = template.ApplyTemplate(null, addGenerationPrompt: false);

        Assert.Equal("<bos>", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with a custom BOS token uses the custom value.
    /// </summary>
    [Fact]
    public void ApplyTemplate_CustomBosToken_ShouldUseCustomValue()
    {
        var template = new ChatTemplate(SampleTemplate) { BosToken = "<s>" };
        var messages = new List<ChatMessage>
        {
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages);

        Assert.StartsWith("<s>", result);
        Assert.DoesNotContain("<bos>", result);
    }

    /// <summary>
    /// Tests that ApplyTemplate with multiple conversation turns formats all messages correctly.
    /// </summary>
    [Fact]
    public void ApplyTemplate_MultipleConversationTurns_ShouldFormatAll()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("system", "Be helpful."),
            new("user", "What is 2+2?"),
            new("assistant", "4"),
            new("user", "Thanks!")
        };

        var result = template.ApplyTemplate(messages);

        Assert.Equal(
            "<bos><|turn>system\nBe helpful.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n4<turn|>\n<|turn>user\nThanks!<turn|>\n<|turn>model\n",
            result);
    }

    /// <summary>
    /// Tests that ApplyTemplate always starts with the BOS token.
    /// </summary>
    [Fact]
    public void ApplyTemplate_ShouldAlwaysStartWithBosToken()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello")
        };

        var result = template.ApplyTemplate(messages);

        Assert.StartsWith("<bos>", result);
    }

    /// <summary>
    /// Tests that a message with null content does not throw and produces an empty content area.
    /// </summary>
    [Fact]
    public void ApplyTemplate_MessageWithNullContent_ShouldNotThrow()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", null!)
        };

        var result = template.ApplyTemplate(messages);

        Assert.Contains("<|turn>user\n<turn|>", result);
    }

    /// <summary>
    /// Tests that only the first message position is treated as a system message.
    /// A system role at a later position is formatted as a regular turn.
    /// </summary>
    [Fact]
    public void ApplyTemplate_SystemMessageNotAtPositionZero_ShouldBeRegularTurn()
    {
        var template = new ChatTemplate(SampleTemplate);
        var messages = new List<ChatMessage>
        {
            new("user", "Hello"),
            new("system", "Late system instruction")
        };

        var result = template.ApplyTemplate(messages);

        // The second system message should appear as a system turn, but not the leading one
        Assert.StartsWith("<bos><|turn>user\n", result);
        Assert.Contains("<|turn>system\nLate system instruction<turn|>", result);
    }

    #endregion
}
