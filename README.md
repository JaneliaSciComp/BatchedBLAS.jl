Nvidia only provides batched versions of the BLAS `gemv`, `gemm`, and `trsm`
functions.  Further, they only support floating point, and alpha and beta
can only be scalars.  BatchedBLAS.jl extends support for batched arrays by
(currently) providing batched versions of `dot`, `gemv`, `symv`, `spmv`,
`ger`, `syr`, and `spr` that work with arrays of AbstractFloats and Integers,
and scaling coefficients which can be scalars or Vectors.

In addition to the type flexibility, there is a performance benefit for rank-1
updates as execution times for `ger`, `syr`, and `spr` are faster than the
equivalent batched `gemm` for the range of parameters tested.  `dot` is also
faster for small matrices.  Benchmarks on an H100 follow.  The dashed lines are
for the transposed version of `gemv` and the upper-triangle versions of all
other functions.  Lower numbers are better.

![benchmarks](/bench/bench.svg)

Example usage:

```
julia> using CUDA, SymmetricFormats, BatchedBLAS

julia> L = 4  # the matrix size
4

julia> N = 6  # the batch dimension
6

julia> A = reshape(1:L*L*N, L, L, N)
4×4×6 reshape(::UnitRange{Int64}, 4, 4, 6) with eltype Int64:
[:, :, 1] =
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

[:, :, 2] =
 17  21  25  29
 18  22  26  30
 19  23  27  31
 20  24  28  32

[:, :, 3] =
 33  37  41  45
 34  38  42  46
 35  39  43  47
 36  40  44  48

[:, :, 4] =
 49  53  57  61
 50  54  58  62
 51  55  59  63
 52  56  60  64

[:, :, 5] =
 65  69  73  77
 66  70  74  78
 67  71  75  79
 68  72  76  80

[:, :, 6] =
 81  85  89  93
 82  86  90  94
 83  87  91  95
 84  88  92  96

julia> SP = CuArray{Float64}(undef, packedsize(A), N);

julia> for i in 1:N
           SP[:,i] = SymmetricPacked(view(A,:,:,i), :U).tri
       end

julia> SP
10×6 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
  1.0  17.0  33.0  49.0  65.0  81.0
  5.0  21.0  37.0  53.0  69.0  85.0
  6.0  22.0  38.0  54.0  70.0  86.0
  9.0  25.0  41.0  57.0  73.0  89.0
 10.0  26.0  42.0  58.0  74.0  90.0
 11.0  27.0  43.0  59.0  75.0  91.0
 13.0  29.0  45.0  61.0  77.0  93.0
 14.0  30.0  46.0  62.0  78.0  94.0
 15.0  31.0  47.0  63.0  79.0  95.0
 16.0  32.0  48.0  64.0  80.0  96.0

julia> x = CuArray{Float64}(reshape(1:L*N, L, N))
4×6 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 1.0  5.0   9.0  13.0  17.0  21.0
 2.0  6.0  10.0  14.0  18.0  22.0
 3.0  7.0  11.0  15.0  19.0  23.0
 4.0  8.0  12.0  16.0  20.0  24.0

julia> batched_spr!('U', 1.0, x, SP)

julia> SP
10×6 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
  2.0  42.0  114.0  218.0  354.0  522.0
  7.0  51.0  127.0  235.0  375.0  547.0
 10.0  58.0  138.0  250.0  394.0  570.0
 12.0  60.0  140.0  252.0  396.0  572.0
 16.0  68.0  152.0  268.0  416.0  596.0
 20.0  76.0  164.0  284.0  436.0  620.0
 17.0  69.0  153.0  269.0  417.0  597.0
 22.0  78.0  166.0  286.0  438.0  622.0
 27.0  87.0  179.0  303.0  459.0  647.0
 32.0  96.0  192.0  320.0  480.0  672.0

julia> SymmetricPacked(SP[:,1])
4×4 SymmetricPacked{Float64, CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}:
  2.0   7.0  12.0  17.0
  7.0  10.0  16.0  22.0
 12.0  16.0  20.0  27.0
 17.0  22.0  27.0  32.0

julia> SymmetricPacked(A[:,:,1] .+ Array(x[:,1]*transpose(x[:,1])))
4×4 SymmetricPacked{Float64, Matrix{Float64}}:
  2.0   7.0  12.0  17.0
  7.0  10.0  16.0  22.0
 12.0  16.0  20.0  27.0
 17.0  22.0  27.0  32.0
```
