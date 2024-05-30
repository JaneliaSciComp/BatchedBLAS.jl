using BatchedBLAS, Test, CUDA, LinearAlgebra, LinearAlgebra.BLAS, SymmetricFormats

L=16; N=4
A = reshape(1.0:L*L*N, L,L,N);
APU = hcat([SymmetricPacked(x, :U).tri for x in eachslice(A, dims=3)]...);
APL = hcat([SymmetricPacked(x, :L).tri for x in eachslice(A, dims=3)]...);
x = reshape(1.0:L*N, L,N);
y = reshape(L*N:-1:1.0, L,N);
o = collect(1.0:N);
alpha_scalar = beta_scalar = 0.5
alpha_vector = range(0,1, length=N)
beta_vector = range(1,0, length=N)
cualpha_vector = CuArray(alpha_vector)
cubeta_vector = CuArray(beta_vector)

test_equality(::Type{<:Integer}, cpu, gpu) = @test maximum(abs.(cpu-Array(gpu))) < 1
test_equality(::Type{<:AbstractFloat}, cpu, gpu) = @test isapprox(cpu, Array(gpu))

test_types = (Float64, Int32)
@testset "A=$TAo, x=$Tx, y=$Ty, α=$Talpha, β=$Tbeta" for
         TAo in test_types, Tx in test_types, Ty in test_types,
         Talpha in (Float64, Vector{Float64}), Tbeta in (Float64, Vector{Float64})

    cuA = CuArray{TAo}(A)
    cuAPU = CuArray{TAo}(APU)
    cuAPL = CuArray{TAo}(APL)
    cux = CuArray{Tx}(x)
    cuy = CuArray{Ty}(y)
    cuo = CuArray{TAo}(o)

    @testset "dot!" begin
        o_cpu=copy(o);
        @views for i=1:N
            o_cpu[i] = x[:,i]' * y[:,i]
        end
        o_gpu=copy(cuo);
        batched_dot!(o_gpu, cux, cuy)
        test_equality(TAo, o_cpu, o_gpu)
    end

    @testset "gemv!" for trans in ['N', 'T', 'C']
        y_cpu=copy(y);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            gemv!(trans, thisalpha, A[:,:,i], x[:,i], thisbeta, @view y_cpu[:,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector : beta_scalar
        batched_gemv!(trans, thisalpha, cuA, cux, thisbeta, y_gpu)
        test_equality(Ty, y_cpu, Array(y_gpu))
    end

    @testset "symv! & spmv!" for uplo in ['U', 'L']
        y_cpu=copy(y);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            symv!(uplo, thisalpha, A[:,:,i], x[:,i], thisbeta, @view y_cpu[:,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector : beta_scalar
        batched_symv!(uplo, thisalpha, cuA, cux, thisbeta, y_gpu)
        test_equality(Ty, y_cpu, Array(y_gpu))

        y_cpu=copy(y);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            spmv!(uplo, thisalpha, (uplo=='U' ? APU : APL)[:,i], x[:,i], thisbeta, @view y_cpu[:,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector : beta_scalar
        batched_spmv!(uplo, thisalpha, (uplo=='U' ? cuAPU : cuAPL), cux, thisbeta, y_gpu)
        test_equality(Ty, y_cpu, Array(y_gpu))
    end

    @testset "ger!" begin
        A_cpu=copy(A);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            ger!(thisalpha, x[:,i], y[:,i], @view A_cpu[:,:,i])
        end
        A_gpu=copy(cuA);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        batched_ger!(thisalpha, cux, cuy, A_gpu)
        test_equality(TAo, A_cpu, Array(A_gpu))
    end

    @testset "syr! & spr!" for uplo in ['U', 'L']
        A_cpu=copy(A);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            syr!(uplo, thisalpha, x[:,i], @view A_cpu[:,:,i])
        end
        A_gpu=copy(cuA);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        batched_syr!(uplo, thisalpha, cux, A_gpu)
        test_equality(TAo, A_cpu, Array(A_gpu))

        AP_cpu=copy(uplo=='U' ? APU : APL);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            spr!(uplo, thisalpha, x[:,i], @view AP_cpu[:,i])
        end
        AP_gpu=copy(uplo=='U' ? cuAPU : cuAPL);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        batched_spr!(uplo, thisalpha, cux, AP_gpu)
        test_equality(TAo, AP_cpu, Array(AP_gpu))
    end

end
