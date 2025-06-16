module GNNGraphsCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNGraphs
using GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 
using SparseArrays

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

# Query 

GNNGraphs._rand_dense_vector(A::CUMAT_T) = CUDA.randn(size(A, 1))
function GNNGraphs.adjacency_matrix(g::GNNGraph{<:CUMAT_T}, T::DataType = eltype(g);
                                 dir = :out, weighted = true)
    @assert dir ∈ [:in, :out]
    A = g.graph
    A = T != eltype(A) ? T.(A) : A
    return dir == :out ? A : A'
end

# Transform

GNNGraphs.dense_zeros_like(a::CUMAT_T, T::Type, sz = size(a)) = CUDA.zeros(T, sz)


# Utils

GNNGraphs.iscuarray(x::AnyCuArray) = true

function GNNGraphs.binarize(Mat::CUSPARSE.CuSparseMatrixCSC)
    @debug "Binarizing CuSparseMatrixCSC"
    bin_vals = fill!(similar(nonzeros(Mat), Bool), true)
    return CUSPARSE.CuSparseMatrixCSC(Mat.colPtr, rowvals(Mat), bin_vals, size(Mat))
end
function GNNGraphs.binarize(Mat::CUSPARSE.CuSparseMatrixCSC, T::DataType)
    @debug "Binarizing CuSparseMatrixCSC with type $(T)"
    bin_vals = fill!(similar(nonzeros(Mat)), one(T))
    # Binarize a CuSparseMatrixCSC by setting all nonzero values to one(T)
    return CUSPARSE.CuSparseMatrixCSC(Mat.colPtr, rowvals(Mat), bin_vals, size(Mat))
end


function sort_edge_index(u::AnyCuArray, v::AnyCuArray)
    dev = get_device(u)
    cdev = cpu_device()
    u, v = u |> cdev, v |> cdev
    #TODO proper cuda friendly implementation
    sort_edge_index(u, v) |> dev
end


end #module
