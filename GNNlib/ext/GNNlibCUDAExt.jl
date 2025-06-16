module GNNlibCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs
using GNNGraphs: GNNGraph, COO_T, SPARSE_T

###### PROPAGATE SPECIALIZATIONS ####################

## COPY_XJ 

## QUESTION_DF: this could probably be avoided alltogether, just 'fix' the default propagate method to correctly cast adjmat 
# function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
#         xi, xj::AnyCuMatrix, e)
#     @debug "Using CUDA copy_xj propagate specialization, $(typeof(g))"
#     A = adjacency_matrix(g, eltype(xj); weighted = false)
#     xj * A
# end

## E_MUL_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(e_mul_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyCuMatrix, e::AbstractVector)
    propagate((xi, xj, e) -> e_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

## W_MUL_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(w_mul_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyCuMatrix, e::Nothing)
    propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

# function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph, ::typeof(mean), xi, xj::AbstractMatrix, e)
#     A = adjacency_matrix(g, weighted=false)
#     D = compute_degree(A)
#     return xj * A * D
# end

# # Zygote bug. Error with sparse matrix without nograd
# compute_degree(A) = Diagonal(1f0 ./ vec(sum(A; dims=2)))

# Flux.Zygote.@nograd compute_degree

end #module
