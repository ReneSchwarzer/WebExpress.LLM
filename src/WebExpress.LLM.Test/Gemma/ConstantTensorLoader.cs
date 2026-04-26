using System.Collections.Generic;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Wraps another <see cref="ISafeTensorLoader"/> and overrides specific tensor
/// values with caller-supplied constants. Useful for tests that need to fix the
/// value of a particular weight (e.g. zero-out a branch) without having to
/// re-implement an entire stub loader.
/// </summary>
internal sealed class ConstantTensorLoader : ISafeTensorLoader
{
    private readonly ISafeTensorLoader _inner;
    private readonly IReadOnlyDictionary<string, float[]> _overrides;

    public ConstantTensorLoader(
        ISafeTensorLoader inner,
        IReadOnlyDictionary<string, float[]> overrides)
    {
        _inner = inner;
        _overrides = overrides;
    }

    public IReadOnlyCollection<string> TensorNames => _inner.TensorNames;

    public TensorMetadata GetMetadata(string name)
    {
        return _inner.GetMetadata(name);
    }

    public bool ContainsTensor(string name)
    {
        return _overrides.ContainsKey(name) || _inner.ContainsTensor(name);
    }

    public WebExpress.LLM.Tensor.Tensor LoadTensor(string name)
    {
        // Always ask the inner loader first so the underlying stub records
        // the request and can resolve the canonical shape.
        var fromInner = _inner.LoadTensor(name);

        if (!_overrides.TryGetValue(name, out var data))
        {
            return fromInner;
        }

        var shape = new int[fromInner.Rank];

        for (var i = 0; i < shape.Length; i++)
        {
            shape[i] = fromInner.Shape[i];
        }

        if (data.Length != fromInner.Length)
        {
            throw new System.ArgumentException(
                $"Override data length {data.Length} does not match inner shape product {fromInner.Length} for '{name}'.");
        }

        return new WebExpress.LLM.Tensor.Tensor(shape, data);
    }
}
