module BatchedBLAS

using GPUArrays, KernelAbstractions

export batched_dot!
export batched_gemv!, batched_symv!, batched_spmv!
export batched_ger!, batched_syr!, batched_spr!

const IntOrFloat = Union{Integer,AbstractFloat}

# https://github.com/JuliaLang/julia/issues/40469
maybe_cast(::Type, x) = x
maybe_cast(::Type{T}, x::AbstractFloat) where T<:Integer =
        round(T, clamp(x, typemin(T), typemax(T)))

const MAX_THREADS_PER_BLOCK=1024
const MAX_XTHREADS=32

configurator(dim1) = min(MAX_XTHREADS, dim1)

function configurator(dim1, dim2)
    xthreads = min(MAX_XTHREADS, dim1)
    ythreads = min(fld(MAX_THREADS_PER_BLOCK, xthreads), cld(dim1*dim2, xthreads))
    return xthreads, ythreads
end

# from NNlib.jl
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)
storage_typejoin(A, Bs...) = Base.promote_typejoin(storage_type(A), storage_typejoin(Bs...))
storage_typejoin(A) = storage_type(A)

"""
    batched_dot!(o, x, y)

In-place batched vector-vector multiplication, equivalent to `o[k] =
transpose(x[:,k]) * y[:,k]` for all `k`.  All inputs can have eltypes of either
AbstractFloats or Integers.
"""
batched_dot!(o::AbstractVector, x::AbstractMatrix, y::AbstractMatrix) =
        _batched_dot!(storage_typejoin(o,x,y), o, x, y)

function _batched_dot!(::Type{<:AbstractGPUArray},
                       o::AbstractArray{To,1}, x::AbstractArray{Tx,2}, y::AbstractArray{Ty,2}) where {
                       To<:IntOrFloat, Tx<:IntOrFloat, Ty<:IntOrFloat}

    allequal((length(o), size(x,2), size(y,2))) ||
            throw(DimensionMismatch("Last dimension of `o`, `x`, and `y` must be the same length!  \
                                     size(o)==$(size(o)), size(x)==$(size(x)), size(y)==$(size(y))"))
    size(x,1) == size(y,1) ||
            throw(DimensionMismatch("First dimension of `x` and `y` must be the same length!  \
                                     size(x)==$(size(x)), size(y)==$(size(y))"))

    @kernel function kernel(::Type{T}, o, x, y) where T
        k = @index(Global)

        @inbounds begin
            tmp = T(0)
            for i=1:size(x,1)
                tmp += x[i,k] * T(y[i,k])
            end
            o[k] = maybe_cast(To, tmp)
        end
    end

    T = promote_type(To, Tx, Ty)
    kernel(get_backend(o),configurator(length(o)))(T, o, x, y; ndrange=length(o))
end

"""
    batched_gemv!(tA, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`. `A` can
optionally be transposed with `tA` as `N`, `T`, or `C`.  All other inputs
can have eltypes of either AbstractFloats or Integers.  `alpha` and `beta` can
also be scalars.
"""
batched_gemv!(tA::AbstractChar, alpha, A::AbstractArray, x::AbstractMatrix,
              beta, y::AbstractMatrix) =
        _batched_gemv!(storage_typejoin(A,x,y), tA, alpha, A, x, beta, y)

function _batched_gemv!(::Type{<:AbstractGPUArray},
                        tA::AbstractChar,
                        alpha::Talpha, A::AbstractArray{TA,3}, x::AbstractArray{Tx,2},
                        beta::Tbeta, y::AbstractArray{Ty,2}) where {
                            Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            TA<:IntOrFloat, Tx<:IntOrFloat,
                            Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            Ty<:IntOrFloat}

    allequal((size(A,3), size(x,2), size(y,2))) ||
            throw(DimensionMismatch("Last dimension of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))
    allequal((size(A,1), size(A,2), size(x,1), size(y,1))) ||
            throw(DimensionMismatch("Leading dimensions of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))

    @kernel function kernel(::Type{T}, tA, alpha, A, x, beta, y) where T
        i, k = @index(Global, NTuple)

        if tA=='N'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[i,j,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif tA=='T'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[j,i,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif tA=='C'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += adjoint(A[j,i,k]) * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`tA` should be 'N', 'T', or 'C'"))
        end
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel(get_backend(y),configurator(size(y)...))(T, tA, alpha, A, x, beta, y; ndrange=size(y))
end

"""
    batched_symv!(uplo, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`.  `A` is
assumed to be symmetric.  Only the `uplo` (either 'U' or 'L') triangle of `A` is
used.  All other inputs can have eltypes of either AbstractFloats or
Integers.  `alpha` and `beta` can also be scalars.
"""
batched_symv!(uplo::AbstractChar,
              alpha, A::AbstractArray, x::AbstractMatrix,
              beta, y::AbstractMatrix) =
        _batched_symv!(storage_typejoin(A,x,y), uplo, alpha, A, x, beta, y)

function _batched_symv!(::Type{<:AbstractGPUArray},
                        uplo::AbstractChar,
                        alpha::Talpha, A::AbstractArray{TA,3}, x::AbstractArray{Tx,2},
                        beta::Tbeta, y::AbstractArray{Ty,2}) where {
                            Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            TA<:IntOrFloat, Tx<:IntOrFloat,
                            Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            Ty<:IntOrFloat}

    allequal((size(A,3), size(x,2), size(y,2))) ||
            throw(DimensionMismatch("Last dimension of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))
    allequal((size(A,1), size(A,2), size(x,1), size(y,1))) ||
            throw(DimensionMismatch("Leading dimensions of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))

    @kernel function kernel(::Type{T}, uplo, alpha, A, x, beta, y) where T
        i, k = @index(Global, NTuple)

        if uplo=='U'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    tmp += A[ijmin,ijmax,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif uplo=='L'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    tmp += A[ijmax,ijmin,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel(get_backend(y),configurator(size(y)...))(T, uplo, alpha, A, x, beta, y; ndrange=size(y))
end

"""
    batched_spmv!(ul, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`.  `A`
must be packed symmetric, and `uplo` specifies whether the upper ('U') or lower
('L') triangle was packed.  All other inputs can have eltypes of either
AbstractFloats or Integers.  `alpha` and `beta` can also be scalars.
"""
batched_spmv!(uplo::AbstractChar,
              alpha, A::AbstractMatrix, x::AbstractMatrix,
              beta, y::AbstractMatrix) =
        _batched_spmv!(storage_typejoin(A,x,y), uplo, alpha, A, x, beta, y)

function _batched_spmv!(::Type{<:AbstractGPUArray},
                        uplo::AbstractChar,
                        alpha::Talpha, A::AbstractArray{TA,2}, x::AbstractArray{Tx,2},
                        beta::Tbeta, y::AbstractArray{Ty,2}) where {
                            Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            TA<:IntOrFloat, Tx<:IntOrFloat,
                            Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                            Ty<:IntOrFloat}

    allequal((size(A,2), size(x,2), size(y,2))) ||
            throw(DimensionMismatch("Last dimension of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))
    size(x,1) == size(y,1) && size(A,1) == (size(x,1)*(size(x,1)+1))>>1 ||
            throw(DimensionMismatch("Leading dimensions of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))

    @kernel function kernel(::Type{T}, uplo, alpha, A, x, beta, y) where T
        i, k = @index(Global, NTuple)

        if uplo=='U'
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmin+(ijmax*(ijmax-1))>>1
                    tmp += A[h,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds begin
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmax+((2n-ijmin)*(ijmin-1))>>1
                    tmp += A[h,k] * T(x[j,k])
                end
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                thisbeta = Tbeta<:AbstractGPUVector ? beta[k] : beta
                y[i,k] = maybe_cast(Ty, thisalpha*tmp + thisbeta*y[i,k])
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    T = promote_type(eltype(Talpha), TA, Tx, eltype(Tbeta), Ty)
    kernel(get_backend(y),configurator(size(y)...))(T, uplo, alpha, A, x, beta, y; ndrange=size(y))
end

"""
    batched_ger!(alpha, x, y, A)

In-place rank-1 update of matrix `A` with vectors `x` and `y` as
`alpha[k] * x[:,k] * transpose(y[:,k]) + A[:,:,k]` for all `k`.  All
inputs can have eltypes of either AbstractFloats or Integers.  `alpha`
can also be a scalar.
"""
batched_ger!(alpha, x::AbstractMatrix, y::AbstractMatrix, A::AbstractArray) =
        _batched_ger!(storage_typejoin(x,y,A), alpha, x, y, A)

function _batched_ger!(::Type{<:AbstractGPUArray},
                       alpha::Talpha, x::AbstractArray{Tx,2}, y::AbstractArray{Ty,2}, A::AbstractArray{TA,3}) where {
                           Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           Tx<:IntOrFloat, Ty<:IntOrFloat, TA<:IntOrFloat}

    allequal((size(A,3), size(x,2), size(y,2))) ||
            throw(DimensionMismatch("Last dimension of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))
    allequal((size(A,1), size(A,2), size(x,1), size(y,1))) ||
            throw(DimensionMismatch("Leading dimensions of `A`, `x`, and `y` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x)), size(y)==$(size(y))"))

    @kernel function kernel(alpha, x, y, A)
        i, k = @index(Global, NTuple)

        @inbounds begin
            for j=1:size(x,1)
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * y[j,k])
            end
        end
    end

    kernel(get_backend(A),configurator(size(x)...))(alpha, x, y, A; ndrange=size(x))
end

"""
    batched_syr!(uplo, alpha, x, A)

In-place rank-1 update of symmetric matrix `A` with vector `x` as `alpha[k] *
x[:,k] * transpose(x[:,k]) + A[:,:,k]` for all `k`.  `A` is assumed to be
symmetric.  Only the `uplo` (either 'U' or 'L') triangle of `A` is used.  All
other inputs can have eltypes of either AbstractFloats or Integers.  `alpha`
can also be a scalar.
"""
batched_syr!(uplo::AbstractChar, alpha, x::AbstractMatrix, A::AbstractArray) =
        _batched_syr!(storage_typejoin(x,A), uplo, alpha, x, A)

function _batched_syr!(::Type{<:AbstractGPUArray},
                       uplo::AbstractChar,
                       alpha::Talpha, x::AbstractArray{Tx,2}, A::AbstractArray{TA,3}) where {
                           Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           Tx<:IntOrFloat, TA<:IntOrFloat}

    size(A,3) == size(x,2) ||
            throw(DimensionMismatch("Last dimension of `A` and `x` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x))"))
    allequal((size(A,1), size(A,2), size(x,1))) ||
            throw(DimensionMismatch("Leading dimensions of `A` and `x` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x))"))

    @kernel function kernel(uplo, alpha, x, A)
        i, k = @index(Global, NTuple)

        if uplo=='U'
            @inbounds begin
                for j=size(x,1):-1:1
                    j<i && break
                    thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                    A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        elseif uplo=='L'
            @inbounds begin
                for j=1:size(x,1)
                    j>i && break
                    thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                    A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel(get_backend(A),configurator(size(x)...))(uplo, alpha, x, A; ndrange=size(x))
end

"""
    batched_spr!(uplo, alpha, x, A)

In-place rank-1 update of packed symmetric matrix `A` with vector `x` as
`alpha[k] * x[:,k] * transpose(x[:,k]) + A[:,:,k]` for all `k`.  `A` must be
symmetric, and `uplo` specifies whether the upper ('U') or lower ('L') triangle
was packed.  All other inputs can have eltypes of either AbstractFloats or
Integers.  `alpha` can also be a scalar.
"""
batched_spr!(uplo::AbstractChar, alpha, x::AbstractMatrix, A::AbstractMatrix) =
        _batched_spr!(storage_typejoin(x,A), uplo, alpha, x, A)

function _batched_spr!(::Type{<:AbstractGPUArray},
                       uplo::AbstractChar,
                       alpha::Talpha, x::AbstractArray{Tx,2}, A::AbstractArray{TA,2}) where {
                           Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           Tx<:IntOrFloat, TA<:IntOrFloat}

    size(A,2) == size(x,2) ||
            throw(DimensionMismatch("Last dimension of `A` and `x` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x))"))
    size(A,1) == (size(x,1)*(size(x,1)+1))>>1 ||
            throw(DimensionMismatch("Leading dimensions of `A` and `x` must be the same length!  \
                                     size(A)==$(size(A)), size(x)==$(size(x))"))

    @kernel function kernel(uplo, alpha, x, A)
        i, k = @index(Global, NTuple)

        if uplo=='U'
            @inbounds begin
                for j=size(x,1):-1:1
                    j<i && break
                    h = i+(j*(j-1))>>1
                    thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                    A[h,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds begin
                for j=1:size(x,1)
                    j>i && break
                    h = i+((2n-j)*(j-1))>>1
                    thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                    A[h,k] += maybe_cast(TA, thisalpha * x[i,k] * x[j,k])
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel(get_backend(A),configurator(size(x)...))(uplo, alpha, x, A; ndrange=size(x))
end

end
