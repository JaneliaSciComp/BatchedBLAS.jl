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

"""
    batched_dot!(o, x, y)

In-place batched vector-vector multiplication, equivalent to
`o[k] = transpose(x[:,k]) * y[:,k]` for all `k`.  All inputs
can have eltypes of either AbstractFloats or Integers.
"""
function batched_dot!(o::AbstractGPUVector{To}, x::AbstractGPUMatrix{Tx}, y::AbstractGPUMatrix{Ty};
                      backend=get_backend(o)) where {
                      To<:IntOrFloat, Tx<:IntOrFloat, Ty<:IntOrFloat}

    @kernel function kernel(::Type{T}, o, @Const(x), @Const(y)) where T
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
    kernel(backend)(T, o, x, y; ndrange=length(o))
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
                       alpha::Talpha, A::AbstractGPUArray{TA,3}, x::AbstractGPUMatrix{Tx},
                       beta::Tbeta, y::AbstractGPUMatrix{Ty};
                       backend=get_backend(A)) where {
                           Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           TA<:IntOrFloat, Tx<:IntOrFloat,
                           Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           Ty<:IntOrFloat}

    @kernel function kernel(::Type{T}, @Const(tA), @Const(alpha), @Const(A), @Const(x), @Const(beta), y) where T
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
    kernel(backend)(T, tA, alpha, A, x, beta, y; ndrange=size(y))
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
                       alpha::Talpha, A::AbstractGPUArray{TA,3}, x::AbstractGPUMatrix{Tx},
                       beta::Tbeta, y::AbstractGPUMatrix{Ty};
                       backend=get_backend(A)) where {
                           Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           TA<:IntOrFloat, Tx<:IntOrFloat,
                           Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                           Ty<:IntOrFloat}

    @kernel function kernel(::Type{T}, @Const(uplo), @Const(alpha), @Const(A), @Const(x), @Const(beta), y) where T
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
    kernel(backend)(T, uplo, alpha, A, x, beta, y; ndrange=size(y))
end

"""
    batched_spmv!(ul, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha[k] * A[:,:,k] * x[:,k] + beta[k] * y[:,k]` for all `k`.
`A` must be packed symmetric, and `uplo` specifies whether the upper ('U')
or lower ('L') triangle was packed.  All other inputs can have eltypes of
either AbstractFloats or Integers.  `alpha` and `beta` can also be scalars.
"""
function batched_spmv!(uplo::AbstractChar,
                       alpha::Talpha, A::AbstractGPUMatrix{TA}, x::AbstractGPUMatrix{Tx},
                       beta::Tbeta, y::AbstractGPUMatrix{Ty};
                       backend=get_backend(A)) where {
                         Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                         TA<:IntOrFloat, Tx<:IntOrFloat,
                         Tbeta<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                         Ty<:IntOrFloat}

    @kernel function kernel(::Type{T}, @Const(uplo), @Const(alpha), @Const(A), @Const(x), @Const(beta), y) where T
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
            @inbounds begin
                n = round(Int, (sqrt(8*size(A,1))-1)/2)
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
    kernel(backend)(T, uplo, alpha, A, x, beta, y; ndrange=size(y))
end

"""
    batched_ger!(alpha, x, y, A)

In-place rank-1 update of matrix `A` with vectors `x` and `y` as
`alpha[k] * x[:,k] * transpose(y[:,k]) + A[:,:,k]` for all `k`.  All
nputs can have eltypes of either AbstractFloats or Integers.  `alpha`
can also be a scalar.
"""
function batched_ger!(alpha::Talpha,
                      x::AbstractGPUMatrix{Tx}, y::AbstractGPUMatrix{Ty},
                      A::AbstractGPUArray{TA,3};
                      backend=get_backend(A)) where {
                          Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, Ty<:IntOrFloat, TA<:IntOrFloat}

    @kernel function kernel(@Const(alpha), @Const(x), @Const(y), A)
        i, k = @index(Global, NTuple)

        @inbounds begin
            for j=1:size(x,1)
                thisalpha = Talpha<:AbstractGPUVector ? alpha[k] : alpha
                A[i,j,k] += maybe_cast(TA, thisalpha * x[i,k] * y[j,k])
            end
        end
    end

    kernel(backend)(alpha, x, y, A; ndrange=size(x))
end

"""
    batched_syr!(uplo, alpha, x, A)

In-place rank-1 update of symmetric matrix `A` with vector `x` as
`alpha[k] * x[:,k] * transpose(x[:,k]) + A[:,:,k]` for all `k`.  `A` is
assumed to be symmetric.  Only the `uplo` (either 'U' or 'L') triangle of
`A` is used.  All other inputs can have eltypes of either AbstractFloats
or Integers.  `alpha` can also be a scalar.
"""
function batched_syr!(uplo::AbstractChar,
                      alpha::Talpha, x::AbstractGPUMatrix{Tx}, A::AbstractGPUArray{TA,3};
                      backend=get_backend(A)) where {
                          Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, TA<:IntOrFloat}

    @kernel function kernel(@Const(uplo), @Const(alpha), @Const(x), A)
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

    kernel(backend)(uplo, alpha, x, A; ndrange=size(x))
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
                      alpha::Talpha, x::AbstractGPUMatrix{Tx}, A::AbstractGPUMatrix{TA};
                      backend=get_backend(A)) where {
                          Talpha<:Union{IntOrFloat, AbstractGPUVector{<:IntOrFloat}},
                          Tx<:IntOrFloat, TA<:IntOrFloat}

    @kernel function kernel(@Const(uplo), @Const(alpha), @Const(x), A)
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
            @inbounds begin
                n = round(Int, (sqrt(8*size(A,1))-1)/2)
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

    kernel(backend)(uplo, alpha, x, A; ndrange=size(x))
end

end
