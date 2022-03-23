using BatchedBLAS, Test, CUDA, LinearAlgebra, SymmetricFormats

L=16; N=4
A = reshape(1.0:L*L*N, L,L,N);
APU = hcat([SymmetricPacked(x, :U).tri for x in eachslice(A, dims=3)]...);
APL = hcat([SymmetricPacked(x, :L).tri for x in eachslice(A, dims=3)]...);
x = reshape(1.0:L*N, L,N);
y = reshape(L*N:-1:1.0, L,N);
o = collect(1.0:N);
alpha = beta = 0.5
alpha1 = range(0,1,N)
cualpha1 = CuArray(alpha1)

test_equality(::Type{<:Integer}, cpu, gpu) = @test maximum(abs.(cpu-Array(gpu))) < 1
test_equality(::Type{<:AbstractFloat}, cpu, gpu) = @test isapprox(cpu, Array(gpu))

test_types = (Float64, Int32)
@testset "$TAo, $Tx, $Ty" for TAo in test_types, Tx in test_types, Ty in test_types

    cuA = CuArray{TAo}(A)
    cuAPU = CuArray{TAo}(APU)
    cuAPL = CuArray{TAo}(APL)
    cux = CuArray{Tx}(x)
    cuy = CuArray{Ty}(y)
    cuo = CuArray{TAo}(o)

    o_cpu=copy(o);
    @views for i=1:N
        o_cpu[i] = x[:,i]' * y[:,i]
    end
    o_gpu=copy(cuo);
    batched_dot!(o_gpu, cux, cuy)
    test_equality(TAo, o_cpu, o_gpu)

    y_cpu=copy(y);
    for i=1:N
        BLAS.gemv!('N', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_gemv!('N', alpha, cuA, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.gemv!('T', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_gemv!('T', alpha, cuA, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.gemv!('C', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_gemv!('C', alpha, cuA, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.symv!('U', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_symv!('U', alpha, cuA, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.symv!('L', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_symv!('L', alpha, cuA, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.spmv!('U', alpha, APU[:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_spmv!('U', alpha, cuAPU, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    y_cpu=copy(y);
    for i=1:N
        BLAS.spmv!('L', alpha, APL[:,i], x[:,i], beta, @view y_cpu[:,i])
    end
    y_gpu=copy(cuy);
    batched_spmv!('L', alpha, cuAPL, cux, beta, y_gpu)
    test_equality(Ty, y_cpu, Array(y_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.ger!(alpha, x[:,i], y[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_ger!(alpha, cux, cuy, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.ger!(alpha1[i], x[:,i], y[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_ger!(cualpha1, cux, cuy, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.syr!('U', alpha, x[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_syr!('U', alpha, cux, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.syr!('L', alpha, x[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_syr!('L', alpha, cux, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.syr!('U', alpha1[i], x[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_syr!('U', cualpha1, cux, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    A_cpu=copy(A);
    for i=1:N
        BLAS.syr!('L', alpha1[i], x[:,i], @view A_cpu[:,:,i])
    end
    A_gpu=copy(cuA);
    batched_syr!('L', cualpha1, cux, A_gpu)
    test_equality(TAo, A_cpu, Array(A_gpu))

    APU_cpu=copy(APU);
    for i=1:N
        spr!('U', alpha, x[:,i], @view APU_cpu[:,i])
    end
    APU_gpu=copy(cuAPU);
    batched_spr!('U', alpha, cux, APU_gpu)
    test_equality(TAo, APU_cpu, Array(APU_gpu))

    APL_cpu=copy(APL);
    for i=1:N
        spr!('L', alpha, x[:,i], @view APL_cpu[:,i])
    end
    APL_gpu=copy(cuAPL);
    batched_spr!('L', alpha, cux, APL_gpu)
    test_equality(TAo, APL_cpu, Array(APL_gpu))

    APU_cpu=copy(APU);
    for i=1:N
        spr!('U', alpha1[i], x[:,i], @view APU_cpu[:,i])
    end
    APU_gpu=copy(cuAPU);
    batched_spr!('U', cualpha1, cux, APU_gpu)
    test_equality(TAo, APU_cpu, Array(APU_gpu))

    APL_cpu=copy(APL);
    for i=1:N
        spr!('L', alpha1[i], x[:,i], @view APL_cpu[:,i])
    end
    APL_gpu=copy(cuAPL);
    batched_spr!('L', cualpha1, cux, APL_gpu)
    test_equality(TAo, APL_cpu, Array(APL_gpu))
end
