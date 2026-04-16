using System;
using System.IO;
using System.IO.MemoryMappedFiles;

namespace WebExpress.LLM.Model;

/// <summary>
/// Provides access to model weight data, supporting both small files (≤2GB using byte arrays)
/// and large files (>2GB using memory-mapped files).
/// </summary>
public sealed class ModelWeights : IDisposable
{
    private const long TwoGigabytes = 2L * 1024 * 1024 * 1024;

    private readonly byte[] _smallData;
    private readonly MemoryMappedFile _memoryMappedFile;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly long _length;
    private bool _disposed;

    private ModelWeights(byte[] data)
    {
        _smallData = data ?? throw new ArgumentNullException(nameof(data));
        _memoryMappedFile = null;
        _accessor = null;
        _length = data.Length;
    }

    private ModelWeights(MemoryMappedFile memoryMappedFile, MemoryMappedViewAccessor accessor, long length)
    {
        _memoryMappedFile = memoryMappedFile ?? throw new ArgumentNullException(nameof(memoryMappedFile));
        _accessor = accessor ?? throw new ArgumentNullException(nameof(accessor));
        _smallData = null;
        _length = length;
    }

    /// <summary>
    /// Gets the total length of the weight data in bytes.
    /// </summary>
    public long Length => _length;

    /// <summary>
    /// Gets a value indicating whether this instance uses a memory-mapped file for large data (>2GB).
    /// </summary>
    public bool IsMemoryMapped => _memoryMappedFile != null;

    /// <summary>
    /// Creates a ModelWeights instance from a file on disk.
    /// Uses byte array for files ≤2GB, memory-mapped file for larger files.
    /// </summary>
    /// <param name="filePath">The path to the weights file.</param>
    /// <returns>A new ModelWeights instance.</returns>
    /// <exception cref="ArgumentException">Thrown if filePath is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the file does not exist.</exception>
    public static ModelWeights FromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must be provided.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Weights file not found.", filePath);
        }

        var fileInfo = new FileInfo(filePath);
        var fileLength = fileInfo.Length;

        // For files ≤2GB, use byte array for better performance
        if (fileLength <= TwoGigabytes)
        {
            var data = File.ReadAllBytes(filePath);
            return new ModelWeights(data);
        }

        // For files >2GB, use memory-mapped file
        var memoryMappedFile = MemoryMappedFile.CreateFromFile(
            filePath,
            FileMode.Open,
            null,
            0,
            MemoryMappedFileAccess.Read);

        var accessor = memoryMappedFile.CreateViewAccessor(
            0,
            0,
            MemoryMappedFileAccess.Read);

        return new ModelWeights(memoryMappedFile, accessor, fileLength);
    }

    /// <summary>
    /// Creates a ModelWeights instance from a byte array.
    /// This method is primarily intended for testing purposes.
    /// </summary>
    /// <param name="data">The byte array containing weight data.</param>
    /// <returns>A new ModelWeights instance.</returns>
    /// <exception cref="ArgumentNullException">Thrown if data is null.</exception>
    public static ModelWeights FromByteArray(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        return new ModelWeights(data);
    }

    /// <summary>
    /// Reads a byte from the weight data at the specified position.
    /// </summary>
    /// <param name="position">The position to read from.</param>
    /// <returns>The byte value at the specified position.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the instance has been disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if position is out of range.</exception>
    public byte ReadByte(long position)
    {
        ThrowIfDisposed();

        if (position < 0 || position >= _length)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        if (_smallData != null)
        {
            return _smallData[position];
        }

        return _accessor.ReadByte(position);
    }

    /// <summary>
    /// Reads a sequence of bytes from the weight data.
    /// </summary>
    /// <param name="position">The position to start reading from.</param>
    /// <param name="count">The number of bytes to read.</param>
    /// <returns>A byte array containing the read data.</returns>
    /// <exception cref="ObjectDisposedException">Thrown if the instance has been disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if position or count is out of range.</exception>
    public byte[] ReadBytes(long position, int count)
    {
        ThrowIfDisposed();

        if (position < 0 || position >= _length)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        if (count < 0 || position + count > _length)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        var buffer = new byte[count];

        if (_smallData != null)
        {
            Array.Copy(_smallData, position, buffer, 0, count);
        }
        else
        {
            _accessor.ReadArray(position, buffer, 0, count);
        }

        return buffer;
    }

    /// <summary>
    /// Gets the underlying byte array if this instance uses in-memory storage.
    /// For memory-mapped files, reads all data into a new byte array.
    /// </summary>
    /// <returns>The byte array containing all weight data.</returns>
    public byte[] ToByteArray()
    {
        ThrowIfDisposed();

        if (_smallData != null)
        {
            return _smallData;
        }

        // For memory-mapped files, read all data into a byte array
        var buffer = new byte[_length];
        _accessor.ReadArray(0, buffer, 0, (int)_length);
        return buffer;
    }

    /// <summary>
    /// Disposes the ModelWeights instance and releases associated resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (_accessor != null)
        {
            _accessor.Dispose();
        }

        if (_memoryMappedFile != null)
        {
            _memoryMappedFile.Dispose();
        }

        _disposed = true;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ModelWeights));
        }
    }
}
