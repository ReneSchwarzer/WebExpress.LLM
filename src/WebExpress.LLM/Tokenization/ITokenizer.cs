using System.Collections.Generic;

namespace WebExpress.LLM.Tokenization;

public interface ITokenizer
{
    IReadOnlyList<int> Encode(string text);

    string Decode(IEnumerable<int> tokens);
}
