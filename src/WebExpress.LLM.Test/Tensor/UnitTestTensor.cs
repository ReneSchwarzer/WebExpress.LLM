using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Tensor;

public sealed class UnitTestTensor
{
    [Fact]
    public void Constructor_WithShape_ShouldCreateZeroFilledTensor()
    {
        var tensor = new WebExpress.LLM.Tensor.Tensor(3, 4);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);
        Assert.Equal(12, tensor.Length);
        Assert.All(tensor.ToArray(), v => Assert.Equal(0.0f, v));
    }

    [Fact]
    public void Constructor_WithShapeAndData_ShouldStoreData()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6 };
        var tensor = new WebExpress.LLM.Tensor.Tensor([2, 3], data);

        Assert.Equal(1.0f, tensor[0, 0]);
        Assert.Equal(2.0f, tensor[0, 1]);
        Assert.Equal(3.0f, tensor[0, 2]);
        Assert.Equal(4.0f, tensor[1, 0]);
        Assert.Equal(5.0f, tensor[1, 1]);
        Assert.Equal(6.0f, tensor[1, 2]);
    }

    [Fact]
    public void Constructor_WithMismatchedDataLength_ShouldThrow()
    {
        var data = new float[] { 1, 2, 3 };
        Assert.Throws<ArgumentException>(() => new WebExpress.LLM.Tensor.Tensor([2, 3], data));
    }

    [Fact]
    public void Constructor_WithEmptyShape_ShouldThrow()
    {
        Assert.Throws<ArgumentException>(() => new WebExpress.LLM.Tensor.Tensor(Array.Empty<int>()));
    }

    [Fact]
    public void Constructor_WithNegativeDimension_ShouldThrow()
    {
        Assert.Throws<ArgumentException>(() => new WebExpress.LLM.Tensor.Tensor(3, -1));
    }

    [Fact]
    public void Zeros_ShouldCreateZeroTensor()
    {
        var tensor = WebExpress.LLM.Tensor.Tensor.Zeros(2, 3);
        Assert.Equal(6, tensor.Length);
        Assert.All(tensor.ToArray(), v => Assert.Equal(0.0f, v));
    }

    [Fact]
    public void Ones_ShouldCreateOnesTensor()
    {
        var tensor = WebExpress.LLM.Tensor.Tensor.Ones(2, 3);
        Assert.Equal(6, tensor.Length);
        Assert.All(tensor.ToArray(), v => Assert.Equal(1.0f, v));
    }

    [Fact]
    public void FromArray_ShouldCreate1DTensor()
    {
        var tensor = WebExpress.LLM.Tensor.Tensor.FromArray([1.0f, 2.0f, 3.0f]);
        Assert.Equal(1, tensor.Rank);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(1.0f, tensor[0]);
        Assert.Equal(3.0f, tensor[2]);
    }

    [Fact]
    public void From2DArray_ShouldCreate2DTensor()
    {
        var tensor = WebExpress.LLM.Tensor.Tensor.From2DArray([[1.0f, 2.0f], [3.0f, 4.0f]]);
        Assert.Equal(2, tensor.Rank);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);
        Assert.Equal(3.0f, tensor[1, 0]);
    }

    [Fact]
    public void Addition_SameShape_ShouldAddElementWise()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 2], [1f, 2, 3, 4]);
        var b = new WebExpress.LLM.Tensor.Tensor([2, 2], [5f, 6, 7, 8]);

        var result = a + b;

        Assert.Equal(6.0f, result[0, 0]);
        Assert.Equal(8.0f, result[0, 1]);
        Assert.Equal(10.0f, result[1, 0]);
        Assert.Equal(12.0f, result[1, 1]);
    }

    [Fact]
    public void Addition_WithBroadcasting_ShouldWork()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var b = new WebExpress.LLM.Tensor.Tensor([1, 3], [10f, 20, 30]);

        var result = a + b;

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(3, result.Shape[1]);
        Assert.Equal(11.0f, result[0, 0]);
        Assert.Equal(22.0f, result[0, 1]);
        Assert.Equal(33.0f, result[0, 2]);
        Assert.Equal(14.0f, result[1, 0]);
        Assert.Equal(25.0f, result[1, 1]);
        Assert.Equal(36.0f, result[1, 2]);
    }

    [Fact]
    public void Subtraction_ShouldSubtractElementWise()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([10f, 20, 30]);
        var b = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3]);

        var result = a - b;

        Assert.Equal(9.0f, result[0]);
        Assert.Equal(18.0f, result[1]);
        Assert.Equal(27.0f, result[2]);
    }

    [Fact]
    public void Multiplication_ShouldMultiplyElementWise()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([3f, 4]);
        var b = WebExpress.LLM.Tensor.Tensor.FromArray([5f, 6]);

        var result = a * b;

        Assert.Equal(15.0f, result[0]);
        Assert.Equal(24.0f, result[1]);
    }

    [Fact]
    public void ScalarMultiplication_ShouldScaleAll()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3]);
        var result = a * 2.0f;

        Assert.Equal(2.0f, result[0]);
        Assert.Equal(4.0f, result[1]);
        Assert.Equal(6.0f, result[2]);
    }

    [Fact]
    public void ScalarMultiplication_Commutative_ShouldWork()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3]);
        var result = 2.0f * a;

        Assert.Equal(2.0f, result[0]);
        Assert.Equal(4.0f, result[1]);
        Assert.Equal(6.0f, result[2]);
    }

    [Fact]
    public void ScalarDivision_ShouldDivideAll()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([10f, 20]);
        var result = a / 5.0f;

        Assert.Equal(2.0f, result[0]);
        Assert.Equal(4.0f, result[1]);
    }

    [Fact]
    public void ScalarAddition_ShouldAddToAll()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2]);
        var result = a + 10.0f;

        Assert.Equal(11.0f, result[0]);
        Assert.Equal(12.0f, result[1]);
    }

    [Fact]
    public void Negation_ShouldNegateAll()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, -2, 3]);
        var result = -a;

        Assert.Equal(-1.0f, result[0]);
        Assert.Equal(2.0f, result[1]);
        Assert.Equal(-3.0f, result[2]);
    }

    [Fact]
    public void Transpose_2DTensor_ShouldSwapRowsAndColumns()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var result = a.Transpose();

        Assert.Equal(3, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
        Assert.Equal(1.0f, result[0, 0]);
        Assert.Equal(4.0f, result[0, 1]);
        Assert.Equal(2.0f, result[1, 0]);
        Assert.Equal(5.0f, result[1, 1]);
        Assert.Equal(3.0f, result[2, 0]);
        Assert.Equal(6.0f, result[2, 1]);
    }

    [Fact]
    public void Transpose_Non2DTensor_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor(3);
        Assert.Throws<InvalidOperationException>(() => a.Transpose());
    }

    [Fact]
    public void GetRow_ShouldExtractRow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([3, 2], [1f, 2, 3, 4, 5, 6]);
        var row = a.GetRow(1);

        Assert.Equal(1, row.Rank);
        Assert.Equal(2, row.Shape[0]);
        Assert.Equal(3.0f, row[0]);
        Assert.Equal(4.0f, row[1]);
    }

    [Fact]
    public void GetLastRow_ShouldExtractLastRow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([3, 2], [1f, 2, 3, 4, 5, 6]);
        var last = a.GetLastRow();

        Assert.Equal(5.0f, last[0]);
        Assert.Equal(6.0f, last[1]);
    }

    [Fact]
    public void Reshape_ShouldChangeShape()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var reshaped = a.Reshape(3, 2);

        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(2, reshaped.Shape[1]);
        Assert.Equal(1.0f, reshaped[0, 0]);
        Assert.Equal(2.0f, reshaped[0, 1]);
        Assert.Equal(3.0f, reshaped[1, 0]);
    }

    [Fact]
    public void Reshape_WithMismatchedLength_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        Assert.Throws<ArgumentException>(() => a.Reshape(2, 2));
    }

    [Fact]
    public void Clone_ShouldCreateIndependentCopy()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2]);
        var clone = a.Clone();

        clone[0] = 99.0f;

        Assert.Equal(1.0f, a[0]);
        Assert.Equal(99.0f, clone[0]);
    }

    [Fact]
    public void ThreeDimensionalAccess_ShouldWork()
    {
        var tensor = new WebExpress.LLM.Tensor.Tensor(2, 3, 4);
        tensor[1, 2, 3] = 42.0f;

        Assert.Equal(42.0f, tensor[1, 2, 3]);
    }

    [Fact]
    public void Division_ElementWise_ShouldWork()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([10f, 20]);
        var b = WebExpress.LLM.Tensor.Tensor.FromArray([2f, 5]);
        var result = a / b;

        Assert.Equal(5.0f, result[0]);
        Assert.Equal(4.0f, result[1]);
    }

    [Fact]
    public void GetRow_On1DTensor_ShouldThrow()
    {
        var a = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3]);
        Assert.Throws<InvalidOperationException>(() => a.GetRow(0));
    }

    [Fact]
    public void GetRow_NegativeIndex_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([3, 2], [1f, 2, 3, 4, 5, 6]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a.GetRow(-1));
    }

    [Fact]
    public void GetRow_IndexEqualToFirstDimension_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([3, 2], [1f, 2, 3, 4, 5, 6]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a.GetRow(3));
    }

    [Fact]
    public void GetRow_IndexBeyondFirstDimension_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => a.GetRow(5));
        Assert.Contains("5", ex.Message);
    }
}
