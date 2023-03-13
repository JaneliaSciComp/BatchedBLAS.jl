module BatchedBLAS

using CUDA

export batched_dot!
export batched_gemv!, batched_symv!, batched_spmv!
export batched_ger!, batched_syr!, batched_spr!

const IntOrFloat = Union{Integer,AbstractFloat}

# https://github.com/JuliaLang/julia/issues/40469
maybe_cast(::Type, x) = x
maybe_cast(::Type{T}, x::AbstractFloat) where T<:Integer =
        round(T, clamp(x, typemin(T), typemax(T)))

function configurator(config, dim1)
    xthreads = min(32, dim1)
    xblocks = cld(dim1, xthreads)
    return (xthreads,), (xblocks,)
end

function configurator(config, dim1, dim2)
    xthreads = min(32, dim1)
    ythreads = min(fld(config.threads, xthreads), cld(dim1*dim2, xthreads))
    xblocks = cld(dim1, xthreads)
    yblocks = cld(dim2, ythreads)
    return (xthreads, ythreads), (xblocks, yblocks)
end

"""
    batched_dot!(o, x, y)

In-place batched vector-vector multiplication, equivalent to `o[k] =
transpose(x[:,k]) * y[:,k]` for all `k`.  All inputs can have eltypes of
either AbstractFloats or Integers.
"""
function batched_dot!(o::CuVector{To}, x::CuMatrix{Tx}, y::CuMatrix{Ty}) where {
                          To<:IntOrFloat, Tx<:IntOrFloat, Ty<:IntOrFloat}

    function kernel(::Type{T}, o, x, y) where T
        k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if k<=size(x,2)
            tmp = T(0)
            for i=1:size(x,1)
                tmp += x[i,k] * T(y[i,k])
            end
            o[k] = maybe_cast(To, tmp)
        end
        return nothing
    end

    T = promote_type(To, Tx, Ty)
    kernel = @cuda name="batched_dot!" launch=false kernel(T, o, x, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(o,1))
    kernel(T, o, x, y; threads=threads, blocks=blocks)
end

"""
    batched_gemv!(tA, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent
to `y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all
`k`. `A` can optionally be transposed with `tA` as `N`, `T`, or `C`.
All other inputs can have eltypes of either AbstractFloats or Integers.
`alpha` and `beta` can also be scalars.
"""
function batched_gemv!(tA::AbstractChar,
                       alpha::Talpha, A::CuArray{TA,3}, x::CuMatrix{Tx},
                       beta::Tbeta, y::CuMatrix{Ty}) where {
                           Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                           TA<:IntOrFloat, Tx<:IntOrFloat,
                           Tbeta<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                           Ty<:IntOrFloat}

    function kernel(::Type{T}, tA, alpha, A, x, beta, y) where T
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if tA=='N'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[i,j,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif tA=='T'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[j,i,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif tA=='C'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += adjoint(A[j,i,k]) * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`tA` should be 'N', 'T', or 'C'"))
        end
        return nothing
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel = @cuda name="batched_gemv!" launch=false kernel(T, tA, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(y,1), size(y,2))
    kernel(T, tA, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
    batched_symv!(uplo, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`.  `A`
is assumed to be symmetric.  Only the `uplo` (either 'U' or 'L') triangle of
`A` is used.  All other inputs can have eltypes of either AbstractFloats
or Integers.  `alpha` and `beta` can also be scalars.
"""
function batched_symv!(uplo::AbstractChar,
                       alpha::Talpha, A::CuArray{TA,3}, x::CuMatrix{Tx},
                       beta::Tbeta, y::CuMatrix{Ty}) where {
                           Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                           TA<:IntOrFloat, Tx<:IntOrFloat,
                           Tbeta<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                           Ty<:IntOrFloat}

    function kernel(::Type{T}, uplo, alpha, A, x, beta, y) where T
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    tmp += A[ijmin,ijmax,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif uplo=='L'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    tmp += A[ijmax,ijmin,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
        return nothing
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel = @cuda name="batched_symv!" launch=false kernel(T, uplo, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(y,1), size(y,2))
    kernel(T, uplo, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
  batched_spmv!(ul, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`.
`A` must be symmetric, and `uplo` specifies whether the upper ('U') or lower
('L') triangle was packed.  All other inputs can have eltypes of either
AbstractFloats or Integers.  `alpha` and `beta` can also be scalars.
"""
function batched_spmv!(uplo::AbstractChar,
                       alpha::Talpha, A::CuMatrix{TA}, x::CuMatrix{Tx},
                       beta::Tbeta, y::CuMatrix{Ty}) where {
                         Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                         TA<:IntOrFloat, Tx<:IntOrFloat,
                         Tbeta<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                         Ty<:IntOrFloat}

    function kernel(::Type{T}, uplo, alpha, A, x, beta, y) where T
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmin+(ijmax*(ijmax-1))>>1
                    tmp += A[h,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmax+((2n-ijmin)*(ijmin-1))>>1
                    tmp += A[h,k] * T(x[j,k])
                end
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                thisbeta = Tbeta<:CuVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
        return nothing
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel = @cuda name="batched_spmv!" launch=false kernel(T, uplo, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(y,1), size(y,2))
    kernel(T, uplo, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
    batched_ger!(alpha, x, y, A)

In-place rank-1 update of matrix `A` with vectors `x` and `y` as `alpha[k] *
x[:,k] * transpose(y[:,k]) + A[:,:,k]` for all `k`.  All inputs can have
eltypes of either AbstractFloats or Integers.  `alpha` can also be a scalar.
"""
function batched_ger!(alpha::Talpha, x::CuMatrix{Tx}, y::CuMatrix{Ty}, A::CuArray{TA,3}) where {
                          Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, Ty<:IntOrFloat, TA<:IntOrFloat}

    function kernel(alpha, x, y, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        @inbounds if k<=size(x,2) && i<=size(x,1)
            for j=1:size(x,1)
                thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * y[j,k])
            end
        end
    end

    kernel = @cuda name="batched_ger_vector!" launch=false kernel(alpha, x, y, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(x,1), size(x,2))
    kernel(alpha, x, y, A; threads=threads, blocks=blocks)
end

"""
    batched_syr!(uplo, alpha, x, A)

In-place rank-1 update of symmetric matrix `A` with vector `x` as `alpha[k]
* x[:,k] * transpose(x[:,k]) + A[:,:,k]` for all `k`.  `A` is assumed to
be symmetric.  Only the `uplo` (either 'U' or 'L') triangle of `A` is used.
All other inputs can have eltypes of either AbstractFloats or Integers.
`alpha` can also be a scalar.
"""
function batched_syr!(uplo::AbstractChar,
                      alpha::Talpha, x::CuMatrix{Tx}, A::CuArray{TA,3}) where {
                          Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, TA<:IntOrFloat}

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                    A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        elseif uplo=='L'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                    A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_syr_vector!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(x,1), size(x,2))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

"""
    batched_spr!(uplo, alpha, x, A)

In-place rank-1 update of packed symmetric matrix `A` with vector `x`
as `alpha[k] * x[:,k] * transpose(x[:,k]) + A[:,:,k]` for all `k`.  `A`
must be symmetric, and `uplo` specifies whether the upper ('U') or lower
('L') triangle was packed.  All other inputs can have eltypes of either
AbstractFloats or Integers.  `alpha` can also be a scalar.
"""
function batched_spr!(uplo::AbstractChar,
                      alpha::Talpha, x::CuMatrix{Tx}, A::CuMatrix{TA}) where {
                          Talpha<:Union{IntOrFloat, CuVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, TA<:IntOrFloat}

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    h = i+(j*(j-1))>>1
                    thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                    A[h,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    h = i+((2n-j)*(j-1))>>1
                    thisalpha = Talpha<:CuVector ? alpha[k] : alpha
                    A[h,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_spr_vector!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, size(x,1), size(x,2))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

end
