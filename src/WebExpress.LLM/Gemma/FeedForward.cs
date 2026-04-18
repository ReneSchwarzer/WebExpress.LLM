using System;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the gated feed-forward network used in Gemma-4 transformer layers.
/// </summary>
/// <remarks>
/// The FFN uses a gated activation pattern:
///   output = down_proj(activation(gate_proj(x)) * up_proj(x))
///
/// Gemma-4 uses GELU as the activation function for the gate.
/// </remarks>
public sealed class FeedForward
{
    /// <summary>
    /// Computes the feed-forward network forward pass.
    /// </summary>
    /// <remarks>
    /// Implements the gated feed-forward block found in modern transformer architectures (e.g., Gemma, Llama).
    /// Steps:
    /// - Projects the input into two intermediate representations (gate and up) using independent weight matrices.
    /// - Applies the GELU activation to the gate projection.
    /// - Multiplies the activated gate element-wise with the up projection (gating mechanism).
    /// - Projects the result back to the hidden size using a final linear transformation ("down" projection).
    /// Produces the output tensor for residual addition after the feed-forward block.
    /// </remarks>
    /// <param name="input">Input tensor of shape [seqLen, hiddenSize].</param>
    /// <param name="gateWeight">Gate projection weight [intermediateSize, hiddenSize].</param>
    /// <param name="upWeight">Up projection weight [intermediateSize, hiddenSize].</param>
    /// <param name="downWeight">Down projection weight [hiddenSize, intermediateSize].</param>
    /// <returns>Output tensor of shape [seqLen, hiddenSize].</returns>
    public static Tensor.Tensor Forward(
        Tensor.Tensor input,
        Tensor.Tensor gateWeight,
        Tensor.Tensor upWeight,
        Tensor.Tensor downWeight)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gateWeight);
        ArgumentNullException.ThrowIfNull(upWeight);
        ArgumentNullException.ThrowIfNull(downWeight);

        // gate = input @ gate_weight^T  -> [seqLen, intermediateSize]
        var gate = TensorOperations.MatMul(input, gateWeight.Transpose());

        // up = input @ up_weight^T  -> [seqLen, intermediateSize]
        var up = TensorOperations.MatMul(input, upWeight.Transpose());

        // Apply GELU activation to gate
        var activated = TensorOperations.Gelu(gate);

        // Element-wise multiply: activated * up
        var combined = activated * up;

        // down = combined @ down_weight^T  -> [seqLen, hiddenSize]
        var output = TensorOperations.MatMul(combined, downWeight.Transpose());

        return output;
    }
}
