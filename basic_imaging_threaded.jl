using Pkg; Pkg.activate(@__DIR__)
# Really only need this the first time
Pkg.instantiate()
using NFFT
using LinearAlgebra
using BenchmarkTools
import Comrade
using Zygote

using Polyester
using Folds
"""
    ObservedNUFT

Container type for a non-uniform Fourier transform (NUFT).
This stores the uv-positions that the model will be sampled at in the Fourier domain,
allowing certain transformtion factors (e.g., NUFT matrix) to be cached.

This is an internal type, an end user should instead create this using [`NFFTAlg`](@ref NFFTAlg)
or [`DFTAlg`](@ref DFTAlg).
"""
struct ObservedNUFT{A<:Comrade.NUFT, T} <: Comrade.NUFT
    """
    Which NUFT algorithm to use (e.g. NFFTAlg or DFTAlg)
    """
    alg::A
    """
    uv positions of the NUFT transform. This is used for precomputation.
    """
    uv::T
end


struct DFTAlg <: Comrade.NUFT end

"""
    DFTAlg(u::AbstractArray, v::AbstractArray)

Create an algorithm object using the direct Fourier transform object using the uv positions
`u`, `v` allowing for a more efficient transform.
"""
function DFTAlg(u::AbstractArray, v::AbstractArray)
    uv = Matrix{eltype(u)}(undef, 2, length(u))
    uv[1,:] .= u
    uv[2,:] .= v
    return ObservedNUFT(DFTAlg(), uv)
end


function plan_nuft(alg::ObservedNUFT{<:DFTAlg}, img)
    uv = alg.uv
    xitr, yitr = Comrade.imagepixels(img)
    dft = similar(img, Complex{eltype(img)}, (size(uv, 2), size(img)...))
    @fastmath for i in axes(img,2), j in axes(img,1), k in axes(uv,2)
        u = uv[1,k]
        v = uv[2,k]
        dft[k, j, i] = cispi(-2(u*xitr[i] + v*yitr[j]))
    end
    # reshape to a matrix so we can take advantage of an easy BLAS call
    return reshape(dft, size(uv,2), :)
end


"""
    create_cache(alg::ObservedNUFT, plan , phases, img)

Create a cache for the DFT algorithm with precomputed `plan`, `phases` and `img`.
This is an internal version.
"""
function create_cache(alg::ObservedNUFT{<:DFTAlg}, plan, phases, img)
    return NUFTCache(alg, plan, phases, img.pulse, reshape(img.img, :))
end


function dft(dft_matrix::AbstractMatrix, img::AbstractMatrix)
    dft_matrix*reshape(img, :)
end

struct TestPost{A, V, F, S}
    dft_mat::A
    vis::V
    fov::F
    sigma::S
end

function TestPost(u, v, vis, sigma, fov, npix)
    alg = DFTAlg(u, v)
    img = Comrade.IntensityMap(similar(u, (npix, npix)), fov, fov)
    dft_mat = plan_nuft(alg, img)
    return TestPost(dft_mat, vis, fov, sigma)
end

function (m::TestPost)(I)
    vmodel = dft(m.dft_mat, I)

    # with Polyester.jl for multithreading
    results = Array{Float32}(undef, length(vmodel))
    @batch for i in eachindex(vmodel, m.vis, m.sigma)
        results[i] = -abs2( (vmodel[i] - m.vis[i])/m.sigma[i] )
    end
    return Folds.sum(results)/2
end

image_sizes = [2,4,8,16,32,64,128,256]
data_sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

benchmark_results = Array{BenchmarkTools.Trial}(undef, length(image_sizes), length(data_sizes))

for i in eachindex(image_sizes)
    for j in eachindex(data_sizes)
        try
            im_size = image_sizes[i]
            data_size = data_sizes[j]

            nvis = data_size
            u = randn(Float32, nvis)
            v = randn(Float32, nvis)

            sigma = fill(0.01, nvis)
            vis = complex.(exp.(-2π^2. *(u.^2 .+ v.^2))) .+ sigma.*randn(nvis)

            post = TestPost(u, v, vis, sigma, 10.0, im_size)
            data = rand(Float32, im_size, im_size)
            benchmark_results[i, j] = @benchmark $(post)($data)
        catch e
            println(e) # we expect to get out of memory errors for the larger problem sizes above on most hardware
            continue
        end
    end
end

