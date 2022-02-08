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

o_cpu=copy(o);
for i=1:N
    o_cpu[i] = x[:,i]' * y[:,i]
end
o_gpu=CuArray(o);
batched_dot!(o_gpu, CuArray(x), CuArray(y))
@test isapprox(o_cpu, Array(o_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.gemv!('N', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_gemv!('N', alpha, CuArray(A), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.gemv!('T', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_gemv!('T', alpha, CuArray(A), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.gemv!('C', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_gemv!('C', alpha, CuArray(A), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.symv!('U', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_symv!('U', alpha, CuArray(A), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.symv!('L', alpha, A[:,:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_symv!('L', alpha, CuArray(A), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.spmv!('U', alpha, APU[:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_spmv!('U', alpha, CuArray(APU), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

y_cpu=copy(y);
for i=1:N
    BLAS.spmv!('L', alpha, APL[:,i], x[:,i], beta, @view y_cpu[:,i])
end
y_gpu=CuArray(y);
batched_spmv!('L', alpha, CuArray(APL), CuArray(x), beta, y_gpu)
@test isapprox(y_cpu, Array(y_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.ger!(alpha, x[:,i], y[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_ger!(alpha, CuArray(x), CuArray(y), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.ger!(alpha1[i], x[:,i], y[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_ger!(CuArray(alpha1), CuArray(x), CuArray(y), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.syr!('U', alpha, x[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_syr!('U', alpha, CuArray(x), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.syr!('L', alpha, x[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_syr!('L', alpha, CuArray(x), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.syr!('U', alpha1[i], x[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_syr!('U', CuArray(alpha1), CuArray(x), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

A_cpu=copy(A);
for i=1:N
    BLAS.syr!('L', alpha1[i], x[:,i], @view A_cpu[:,:,i])
end
A_gpu=CuArray(A);
batched_syr!('L', CuArray(alpha1), CuArray(x), A_gpu)
@test isapprox(A_cpu, Array(A_gpu))

APU_cpu=copy(APU);
for i=1:N
    spr!('U', alpha, x[:,i], @view APU_cpu[:,i])
end
APU_gpu=CuArray(APU);
batched_spr!('U', alpha, CuArray(x), APU_gpu)
@test isapprox(APU_cpu, Array(APU_gpu))

APL_cpu=copy(APL);
for i=1:N
    spr!('L', alpha, x[:,i], @view APL_cpu[:,i])
end
APL_gpu=CuArray(APL);
batched_spr!('L', alpha, CuArray(x), APL_gpu)
@test isapprox(APL_cpu, Array(APL_gpu))

APU_cpu=copy(APU);
for i=1:N
    spr!('U', alpha1[i], x[:,i], @view APU_cpu[:,i])
end
APU_gpu=CuArray(APU);
batched_spr!('U', CuArray(alpha1), CuArray(x), APU_gpu)
@test isapprox(APU_cpu, Array(APU_gpu))

APL_cpu=copy(APL);
for i=1:N
    spr!('L', alpha1[i], x[:,i], @view APL_cpu[:,i])
end
APL_gpu=CuArray(APL);
batched_spr!('L', CuArray(alpha1), CuArray(x), APL_gpu)
@test isapprox(APL_cpu, Array(APL_gpu))
