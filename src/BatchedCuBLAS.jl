module BatchedCuBLAS

using CUDA

export batched_dot!
export batched_gemv!, batched_symv!, batched_spmv!
export batched_ger!, batched_syr!, batched_spr!

function configurator(config, dims)
    if length(dims)==1
        xthreads = min(32, dims[1])
        xblocks = cld(dims[1], xthreads)
        return (xthreads,), (xblocks,)
    elseif length(dims)==2
        xthreads = min(32, dims[1])
        ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
        xblocks = cld(dims[1], xthreads)
        yblocks = cld(dims[2], ythreads)
        return (xthreads, ythreads), (xblocks, yblocks)
    end
end

"""
    batched_dot!(o, x, y)

In-place batched vector-vector multiplication, equivalent to `o[:,k] =
transpose(x[:,k]) * y[:,k]` for all `k`.
"""
function batched_dot!(o::CuVector{T}, x::CuMatrix{T}, y::CuMatrix{T}) where T

    function kernel(o, x, y)
        k = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if k<=size(x,2)
            o[k] = 0
            for i=1:size(x,1)
                o[k] += x[i,k] * y[i,k]
            end
        end
    end

    kernel = @cuda name="batched_dot!" launch=false kernel(o, x, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(o,1),))
    kernel(o, x, y; threads=threads, blocks=blocks)
end

"""
    batched_gemv!(tA, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha*A[:,:,k]*x[:,k] + beta*y[:,k]` for all `k`. `A` can optionally
be transposed with `tA` as `N`, `T`, or `C`.  `alpha` and `beta` are scalars.
"""
function batched_gemv!(tA::AbstractChar, alpha::T, A::CuArray{T},
                       x::CuMatrix{T}, beta::T, y::CuMatrix{T}) where T

    function kernel(tA, alpha, A, x, beta, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if tA=='N'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[i,j,k] * x[j,k]
                end
                y[i,k] = alpha*tmp + beta*y[i,k]
            end
        elseif tA=='T'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += A[j,i,k] * x[j,k]
                end
                y[i,k] = alpha*tmp + beta*y[i,k]
            end
        elseif tA=='C'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    tmp += adjoint(A[j,i,k]) * x[j,k]
                end
                y[i,k] = alpha*tmp + beta*y[i,k]
            end
        else
            throw(ArgumentError("`tA` should be 'N', 'T', or 'C'"))
        end
        return nothing
    end

    kernel = @cuda name="batched_gemv!" launch=false kernel(tA, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(y,1),size(y,2)))
    kernel(tA, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
    batched_symv!(uplo, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent to
`y[:,k] = alpha*A[:,:,k]*x[:,k] + beta*y[:,k]` for all `k`.  `A` is assumed
to be symmetric.  Only the `uplo` (either 'U' or 'L') triangle of `A` is used.
`alpha` and `beta` are scalars.
"""
function batched_symv!(uplo::AbstractChar, alpha::T, A::CuArray{T},
                       x::CuMatrix{T}, beta::T, y::CuMatrix{T}) where T

    function kernel(uplo, alpha, A, x, beta, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            if beta==0.0
                @inbounds if k<=size(y,2) && i<=size(y,1)
                    tmp = T(0)
                    for j=1:size(x,1)
                        ijmin,ijmax = minmax(i,j)
                        tmp += A[ijmin,ijmax,k] * x[j,k]
                    end
                    y[i,k] = alpha*tmp
                end
            else
                @inbounds if k<=size(y,2) && i<=size(y,1)
                    tmp = T(0)
                    for j=1:size(x,1)
                        ijmin,ijmax = minmax(i,j)
                        tmp += A[ijmin,ijmax,k] * x[j,k]
                    end
                    y[i,k] = alpha*tmp + beta*y[i,k]
                end
            end
        elseif uplo=='L'
            if beta==0.0
                @inbounds if k<=size(y,2) && i<=size(y,1)
                    tmp = T(0)
                    for j=1:size(x,1)
                        ijmin,ijmax = minmax(i,j)
                        tmp += A[ijmax,ijmin,k] * x[j,k]
                    end
                    y[i,k] = alpha*tmp
                end
            else
                @inbounds if k<=size(y,2) && i<=size(y,1)
                    tmp = T(0)
                    for j=1:size(x,1)
                        ijmin,ijmax = minmax(i,j)
                        tmp += A[ijmax,ijmin,k] * x[j,k]
                    end
                    y[i,k] = alpha*tmp + beta*y[i,k]
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
        return nothing
    end

    kernel = @cuda name="batched_symv!" launch=false kernel(uplo, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(y,1),size(y,2)))
    kernel(uplo, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
  spmv!(ul, alpha, A, x, beta, y)

In-place batched matrix-vector multiplication and addition, equivalent
to `y[:,k] = alpha*A[:,:,k]*x[:,k] + beta*y[:,k]` for all `k`.  `uplo`
specifies whether the upper ('U') or lower ('L') triangle was packed.
`A` is assumed to be symmetric and packed.  `alpha` and `beta` are scalars.
"""
function batched_spmv!(uplo::AbstractChar, alpha::T, A::CuMatrix{T},
                       x::CuMatrix{T}, beta::T, y::CuMatrix{T}) where T

    function kernel(uplo, alpha, A, x, beta, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmin+(ijmax*(ijmax-1))>>1
                    tmp += A[h,k] * x[j,k]
                end
                y[i,k] = alpha*tmp + beta*y[i,k]
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds if k<=size(y,2) && i<=size(y,1)
                tmp = T(0)
                for j=1:size(x,1)
                    ijmin,ijmax = minmax(i,j)
                    h = ijmax+((2n-ijmin)*(ijmin-1))>>1
                    tmp += A[h,k] * x[j,k]
                end
                y[i,k] = alpha*tmp + beta*y[i,k]
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
        return nothing
    end

    kernel = @cuda name="batched_spmv!" launch=false kernel(uplo, alpha, A, x, beta, y)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(y,1),size(y,2)))
    kernel(uplo, alpha, A, x, beta, y; threads=threads, blocks=blocks)
end

"""
    ger!(alpha, x, y, A)

In-place rank-1 update of matrix `A` with vectors `x` and `y` as
`alpha[k]*x[:,k]*transpose(y[:,k]) + A[:,:,k]`.`alpha` can be also be
a scalar.
"""
function batched_ger!(alpha::T, x::CuMatrix{T}, y::CuMatrix{T}, A::CuArray{T}) where T

    function kernel(alpha, x, y, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        @inbounds if k<=size(x,2) && i<=size(x,1)
            for j=size(x,1):-1:1
                A[i,j,k] += alpha * x[i,k] * y[j,k]
            end
        end
    end

    kernel = @cuda name="batched_ger_scalar!" launch=false kernel(alpha, x, y, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(alpha, x, y, A; threads=threads, blocks=blocks)
end

function batched_ger!(alpha::CuVector{T}, x::CuMatrix{T}, y::CuMatrix{T}, A::CuArray{T}) where T

    function kernel(alpha, x, y, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        @inbounds if k<=size(x,2) && i<=size(x,1)
            for j=1:size(x,1)
                A[i,j,k] += alpha[k] * x[i,k] * y[j,k]
            end
        end
    end

    kernel = @cuda name="batched_ger_vector!" launch=false kernel(alpha, x, y, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(alpha, x, y, A; threads=threads, blocks=blocks)
end

"""
    syr!(uplo, alpha, x, A)

In-place rank-1 update of symmetric matrix `A` with vector `x` as
`alpha[k]*x[:,k]*transpose(x[:,k]) + A[:,:,k]`.  Only the `uplo` (either
'U' or 'L') triangle of `A` is used.  `alpha` can be also be a scalar.
"""
function batched_syr!(uplo::AbstractChar, alpha::T, x::CuMatrix{T}, A::CuArray{T}) where T

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    A[i,j,k] += alpha * x[i,k] * x[j,k]
                end
            end
        elseif uplo=='L'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    A[i,j,k] += alpha * x[i,k] * x[j,k]
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_syr_scalar!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

function batched_syr!(uplo::AbstractChar, alpha::CuVector{T},
                      x::CuMatrix{T}, A::CuArray{T}) where T

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    A[i,j,k] += alpha[k] * x[i,k] * x[j,k]
                end
            end
        elseif uplo=='L'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    A[i,j,k] += alpha[k] * x[i,k] * x[j,k]
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_syr_vector!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

"""
    spr!(uplo, alpha, x, A)

In-place rank-1 update of packed symmetric matrix `A` with vector `x` as
`alpha[k]*x[:,k]*transpose(x[:,k]) + A[:,:,k]`.  `uplo` specifies whether
the upper ('U') or lower ('L') triangle was packed.  `alpha` can be also be
a scalar.
"""
function batched_spr!(uplo::AbstractChar, alpha::T, x::CuMatrix{T}, A::CuMatrix{T}) where T

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    h = i+(j*(j-1))>>1
                    A[h,k] += alpha * x[i,k] * x[j,k]
                end
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    h = i+((2n-j)*(j-1))>>1
                    A[h,k] += alpha * x[i,k] * x[j,k]
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_spr_scalar!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

function batched_spr!(uplo::AbstractChar, alpha::CuVector{T},
                      x::CuMatrix{T}, A::CuMatrix{T}) where T

    function kernel(uplo, alpha, x, A)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        k = threadIdx().y + (blockIdx().y - 1) * blockDim().y

        if uplo=='U'
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=size(x,1):-1:1
                    j<i && break
                    h = i+(j*(j-1))>>1
                    A[h,k] += alpha[k] * x[i,k] * x[j,k]
                end
            end
        elseif uplo=='L'
            n = round(Int, (sqrt(8*size(A,1))-1)/2)
            @inbounds if k<=size(x,2) && i<=size(x,1)
                for j=1:size(x,1)
                    j>i && break
                    h = i+((2n-j)*(j-1))>>1
                    A[h,k] += alpha[k] * x[i,k] * x[j,k]
                end
            end
        else
            throw(ArgumentError("`uplo` should be 'U' or 'L'"))
        end
    end

    kernel = @cuda name="batched_spr_vector!" launch=false kernel(uplo, alpha, x, A)
    config = launch_configuration(kernel.fun)
    threads, blocks = configurator(config, (size(x,1),size(x,2)))
    kernel(uplo, alpha, x, A; threads=threads, blocks=blocks)
end

end
