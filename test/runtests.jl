using BatchedBLAS, Test, CUDA, LinearAlgebra, LinearAlgebra.BLAS, SymmetricFormats

L=16; N=4
A = reshape(1.0:L*L*N, L,L,N);
APU = [SymmetricPacked(x, :U) for x in eachslice(A, dims=3)];
APL = [SymmetricPacked(x, :L) for x in eachslice(A, dims=3)];
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
    cuAPU = CuArray{TAo}(hcat([x.tri for x in APU]...))
    cuAPL = CuArray{TAo}(hcat([x.tri for x in APL]...))
    cux = CuArray{Tx}(x)
    cuy = CuArray{Ty}(y)
    cuo = CuArray{TAo}(o)

    @testset "dot!" begin
        o_cpu=copy(o);
        @views for i=1:N
            @views o_cpu[i] = dot(x[:,i], y[:,i])
        end
        o_gpu=copy(cuo);
        batched_dot!(o_gpu, cux, cuy)
        test_equality(TAo, o_cpu, o_gpu)

        o_cpu=copy(o);
        @views for i=2:N-1
            @views o_cpu[i] = dot(x[2:end-1,i],  y[2:end-1,i])
        end
        o_gpu=copy(cuo);
        @views batched_dot!(o_gpu[2:end-1], cux[2:end-1,2:end-1], cuy[2:end-1,2:end-1])
        test_equality(TAo, o_cpu[2:end-1], o_gpu[2:end-1])
    end

    @testset "gemv!" for trans in ('N', 'T', 'C')
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

        y_cpu=copy(y);
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            gemv!(trans, thisalpha, A[2:end-1,2:end-1,i], x[2:end-1,i], thisbeta, @view y_cpu[2:end-1,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector[2:end-1] : beta_scalar
        @views batched_gemv!(trans, thisalpha, cuA[2:end-1,2:end-1,2:end-1], cux[2:end-1,2:end-1],
                             thisbeta, y_gpu[2:end-1,2:end-1])
        test_equality(Ty, y_cpu[2:end-1,2:end-1], Array(y_gpu[2:end-1,2:end-1]))
    end

    @testset "symv!" for uplo in ('U', 'L')
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
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            symv!(uplo, thisalpha, A[2:end-1,2:end-1,i], x[2:end-1,i], thisbeta, @view y_cpu[2:end-1,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector[2:end-1] : beta_scalar
        @views batched_symv!(uplo, thisalpha, cuA[2:end-1,2:end-1,2:end-1], cux[2:end-1,2:end-1],
                             thisbeta, y_gpu[2:end-1,2:end-1])
        test_equality(Ty, y_cpu[2:end-1,2:end-1], Array(y_gpu[2:end-1,2:end-1]))
    end

    @testset "spmv!" for uplo in ('U', 'L')
        y_cpu=copy(y);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            spmv!(uplo, thisalpha, (uplo=='U' ? APU : APL)[i].tri, x[:,i], thisbeta, @view y_cpu[:,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector : beta_scalar
        batched_spmv!(uplo, thisalpha, (uplo=='U' ? cuAPU : cuAPL), cux, thisbeta, y_gpu)
        test_equality(Ty, y_cpu, Array(y_gpu))

        y_cpu=copy(y);
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            thisbeta = Tbeta<:Vector ? beta_vector[i] : beta_scalar
            AP = SymmetricPacked((uplo=='U' ? APU : APL)[i][2:end-1,2:end-1], Symbol(uplo))
            spmv!(uplo, thisalpha, AP.tri, x[2:end-1,i], thisbeta, @view y_cpu[2:end-1,i])
        end
        y_gpu=copy(cuy);
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        thisbeta = Tbeta<:Vector ? cubeta_vector[2:end-1] : beta_scalar
        cuAP = CuArray{TAo}(hcat([SymmetricPacked(x[2:end-1,2:end-1], Symbol(uplo)).tri
                                  for x in (uplo=='U' ? APU : APL)]...))
        @views batched_spmv!(uplo, thisalpha, cuAP[:,2:end-1], cux[2:end-1,2:end-1],
                             thisbeta, y_gpu[2:end-1,2:end-1])
        test_equality(Ty, y_cpu[2:end-1,2:end-1], Array(y_gpu[2:end-1,2:end-1]))
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

        A_cpu=copy(A);
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            ger!(thisalpha, x[2:end-1,i], y[2:end-1,i], @view A_cpu[2:end-1,2:end-1,i])
        end
        A_gpu=copy(cuA);
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        @views batched_ger!(thisalpha, cux[2:end-1,2:end-1], cuy[2:end-1,2:end-1], A_gpu[2:end-1,2:end-1,2:end-1])
        test_equality(TAo, A_cpu[2:end-1,2:end-1,2:end-1], Array(A_gpu[2:end-1,2:end-1,2:end-1]))
    end

    @testset "syr!" for uplo in ('U', 'L')
        A_cpu=copy(A);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            syr!(uplo, thisalpha, x[:,i], @view A_cpu[:,:,i])
        end
        A_gpu=copy(cuA);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        batched_syr!(uplo, thisalpha, cux, A_gpu)
        test_equality(TAo, A_cpu, Array(A_gpu))

        A_cpu=copy(A);
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            syr!(uplo, thisalpha, x[2:end-1,i], @view A_cpu[2:end-1,2:end-1,i])
        end
        A_gpu=copy(cuA);
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        @views batched_syr!(uplo, thisalpha, cux[2:end-1,2:end-1], A_gpu[2:end-1,2:end-1,2:end-1])
        test_equality(TAo, A_cpu, Array(A_gpu))
    end

    @testset "spr!" for uplo in ('U', 'L')
        AP_cpu=deepcopy(uplo=='U' ? APU : APL);
        for i=1:N
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            spr!(uplo, thisalpha, x[:,i], AP_cpu[i].tri)
        end
        AP_gpu=copy(uplo=='U' ? cuAPU : cuAPL);
        thisalpha = Talpha<:Vector ? cualpha_vector : alpha_scalar
        batched_spr!(uplo, thisalpha, cux, AP_gpu)
        test_equality(TAo, hcat([x.tri for x in AP_cpu]...), Array(AP_gpu))

        AP_cpu = [SymmetricPacked(x[2:end-1,2:end-1], Symbol(uplo))
                  for x in (uplo=='U' ? APU : APL)]
        for i=2:N-1
            thisalpha = Talpha<:Vector ? alpha_vector[i] : alpha_scalar
            spr!(uplo, thisalpha, x[2:end-1,i], AP_cpu[i].tri)
        end
        AP_gpu = CuArray{TAo}(hcat([SymmetricPacked(x[2:end-1,2:end-1], Symbol(uplo)).tri
                                    for x in (uplo=='U' ? APU : APL)]...))
        thisalpha = Talpha<:Vector ? cualpha_vector[2:end-1] : alpha_scalar
        @views batched_spr!(uplo, thisalpha, cux[2:end-1,2:end-1], AP_gpu[:,2:end-1])
        test_equality(TAo, hcat([x.tri for x in AP_cpu]...), Array(AP_gpu))
    end

end
