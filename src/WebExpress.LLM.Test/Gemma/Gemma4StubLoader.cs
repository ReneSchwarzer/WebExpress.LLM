using System.Collections.Generic;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Test-only deterministic <see cref="ISafeTensorLoader"/> that infers tensor
/// shapes from their names so the Gemma-4 forward pass can run end-to-end
/// without real safetensors files. Records every name that was requested so
/// tests can pin the weight-name contract.
/// </summary>
internal sealed class Gemma4StubLoader : ISafeTensorLoader
{
    public Gemma4StubLoader(
        int numLayers,
        int hiddenSize,
        int numQueryHeads,
        int numKvHeads,
        int headDim,
        int numExperts,
        int moeIntermediate,
        int intermediateSize,
        int vocabSize,
        int hiddenSizePerLayerInput = 0,
        int vocabSizePerLayerInput = 0,
        int globalHeadDim = 0,
        int numGlobalKvHeads = 0,
        bool attentionKeyEqualsValue = false,
        IReadOnlyList<string>? layerTypes = null,
        IReadOnlyDictionary<string, int[]>? overrideShapes = null,
        ISet<string>? excludeTensors = null)
    {
        NumLayers = numLayers;
        HiddenSize = hiddenSize;
        NumQueryHeads = numQueryHeads;
        NumKvHeads = numKvHeads;
        HeadDim = headDim;
        NumExperts = numExperts;
        MoeIntermediate = moeIntermediate;
        IntermediateSize = intermediateSize;
        VocabSize = vocabSize;
        HiddenSizePerLayerInput = hiddenSizePerLayerInput;
        VocabSizePerLayerInput = vocabSizePerLayerInput > 0 ? vocabSizePerLayerInput : vocabSize;
        GlobalHeadDim = globalHeadDim > 0 ? globalHeadDim : headDim;
        NumGlobalKvHeads = numGlobalKvHeads > 0 ? numGlobalKvHeads : numKvHeads;
        AttentionKeyEqualsValue = attentionKeyEqualsValue;
        LayerTypes = layerTypes ?? new List<string>();
        OverrideShapes = overrideShapes ?? new Dictionary<string, int[]>();
        ExcludeTensors = excludeTensors ?? new HashSet<string>();
    }

    public int NumLayers { get; }

    public int HiddenSize { get; }

    public int NumQueryHeads { get; }

    public int NumKvHeads { get; }

    public int HeadDim { get; }

    public int NumExperts { get; }

    public int MoeIntermediate { get; }

    public int IntermediateSize { get; }

    public int VocabSize { get; }

    public int HiddenSizePerLayerInput { get; }

    public int VocabSizePerLayerInput { get; }

    public int GlobalHeadDim { get; }

    public int NumGlobalKvHeads { get; }

    public bool AttentionKeyEqualsValue { get; }

    public IReadOnlyList<string> LayerTypes { get; }

    public IReadOnlyDictionary<string, int[]> OverrideShapes { get; }

    public ISet<string> ExcludeTensors { get; }

    public HashSet<string> Requested { get; } = [];

    public IReadOnlyCollection<string> TensorNames => [];

    public TensorMetadata GetMetadata(string name)
    {
        throw new KeyNotFoundException(name);
    }

    public bool ContainsTensor(string name)
    {
        if (ExcludeTensors.Contains(name))
        {
            return false;
        }

        return TryShape(name) is not null;
    }

    public WebExpress.LLM.Tensor.Tensor LoadTensor(string name)
    {
        Requested.Add(name);

        if (ExcludeTensors.Contains(name))
        {
            throw new KeyNotFoundException(name);
        }

        var shape = TryShape(name) ?? throw new KeyNotFoundException(name);
        var size = 1;

        foreach (var d in shape)
        {
            size *= d;
        }

        var data = new float[size];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 7 + 3) % 11 - 5) * 0.05f;
        }

        return new WebExpress.LLM.Tensor.Tensor(shape, data);
    }

    private int[]? TryShape(string name)
    {
        if (OverrideShapes.TryGetValue(name, out var overrideShape))
        {
            return overrideShape;
        }

        if (name == "model.language_model.embed_tokens.weight")
        {
            return [VocabSize, HiddenSize];
        }

        if (name == "model.language_model.norm.weight")
        {
            return [HiddenSize];
        }

        if (name == "model.language_model.embed_tokens_per_layer.weight")
        {
            if (HiddenSizePerLayerInput <= 0)
            {
                return null;
            }

            return [VocabSizePerLayerInput, NumLayers * HiddenSizePerLayerInput];
        }

        if (name == "model.language_model.per_layer_model_projection.weight")
        {
            if (HiddenSizePerLayerInput <= 0)
            {
                return null;
            }

            return [NumLayers * HiddenSizePerLayerInput, HiddenSize];
        }

        if (name == "model.language_model.per_layer_projection_norm.weight")
        {
            if (HiddenSizePerLayerInput <= 0)
            {
                return null;
            }

            return [HiddenSizePerLayerInput];
        }

        if (!name.StartsWith("model.language_model.layers."))
        {
            return null;
        }

        var remainder = name["model.language_model.layers.".Length..];
        var dotIndex = remainder.IndexOf('.');

        if (dotIndex <= 0)
        {
            return null;
        }

        var layerStr = remainder[..dotIndex];

        if (!int.TryParse(layerStr, out var layerIndex) ||
            layerIndex < 0 || layerIndex >= NumLayers)
        {
            return null;
        }

        var suffix = remainder[(dotIndex + 1)..];

        // Determine effective head/kv dims for this layer based on attention type.
        var isFullAttention = layerIndex < LayerTypes.Count
            && LayerTypes[layerIndex] == "full_attention";
        var effectiveHeadDim = isFullAttention ? GlobalHeadDim : HeadDim;
        var effectiveKvHeads = isFullAttention && AttentionKeyEqualsValue
            ? NumGlobalKvHeads
            : NumKvHeads;

        // For k_eq_v full-attention layers, v_proj does not exist in the
        // checkpoint — match vLLM behaviour.
        if (suffix == "self_attn.v_proj.weight" && AttentionKeyEqualsValue && isFullAttention)
        {
            return null;
        }

        return suffix switch
        {
            "input_layernorm.weight" => [HiddenSize],
            "post_attention_layernorm.weight" => [HiddenSize],
            "pre_feedforward_layernorm.weight" => [HiddenSize],
            "pre_feedforward_layernorm_2.weight" => [HiddenSize],
            "post_feedforward_layernorm.weight" => [HiddenSize],
            "post_feedforward_layernorm_1.weight" => [HiddenSize],
            "post_feedforward_layernorm_2.weight" => [HiddenSize],
            "self_attn.q_norm.weight" => [effectiveHeadDim],
            "self_attn.k_norm.weight" => [effectiveHeadDim],
            "self_attn.q_proj.weight" => [NumQueryHeads * effectiveHeadDim, HiddenSize],
            "self_attn.k_proj.weight" => [effectiveKvHeads * effectiveHeadDim, HiddenSize],
            "self_attn.v_proj.weight" => [effectiveKvHeads * effectiveHeadDim, HiddenSize],
            "self_attn.o_proj.weight" => [HiddenSize, NumQueryHeads * effectiveHeadDim],
            "mlp.gate_proj.weight" => [IntermediateSize, HiddenSize],
            "mlp.up_proj.weight" => [IntermediateSize, HiddenSize],
            "mlp.down_proj.weight" => [HiddenSize, IntermediateSize],
            "router.proj.weight" => [NumExperts, HiddenSize],
            "experts.gate_up_proj" => [NumExperts, 2 * MoeIntermediate, HiddenSize],
            "experts.down_proj" => [NumExperts, HiddenSize, MoeIntermediate],
            "layer_scalar" => [1],
            "per_layer_input_gate.weight" => HiddenSizePerLayerInput > 0
                ? [HiddenSizePerLayerInput, HiddenSize]
                : null,
            "per_layer_projection.weight" => HiddenSizePerLayerInput > 0
                ? [HiddenSize, HiddenSizePerLayerInput]
                : null,
            "post_per_layer_input_norm.weight" => HiddenSizePerLayerInput > 0
                ? [HiddenSize]
                : null,
            _ => null
        };
    }
}
