namespace WebExpress.LLM.Inference;

public interface IInferenceEngine
{
    IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens);
}
