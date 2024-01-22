using LinearAlgebra, BatchedBLAS, NNlib, SymmetricFormats, BenchmarkTools, DataFrames, Gadfly, JLD2
using KernelAbstractions, CUDA

macro belapsed_median(args...)
    esc(:(time(median(@benchmark $(args...))) / 1e9))
end


function doit(L,N)
    x2 = CuArray(rand(L,N));
    x3 = CuArray(rand(L,1,N));
    y2 = CuArray(rand(L,N));
    y3 = CuArray(rand(L,1,N));
    o1 = CuArray(rand(N));
    o3 = CuArray(rand(1,1,N));

    tbgemm = @belapsed_median CUDA.@sync batched_mul!($o3, batched_transpose($x3), $y3)
    tbdot = @belapsed_median CUDA.@sync batched_dot!($o1, $x2, $y2)

    CUDA.unsafe_free!.((x2, x3, y2, y3, o1, o3))
    CUDA.memory_status()

    return tbgemm, tbdot
end

df_b_dot = DataFrame(func = String[], N = Int[], L = Int[], elapsed_time = Float64[])
              
@info "batched_dot"
N=32768
for L=[32, 64, 128, 256]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_dot, ("bgemm", N, L, times[1]))
    push!(df_b_dot, ("bdot",  N, L, times[2]))
end

L=128
for N=[8192, 16384, 65536, 131072]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_dot, ("bgemm", N, L, times[1]))
    push!(df_b_dot, ("bdot",  N, L, times[2]))
end


function doit(L,N)
    _A = rand(L,L,N);
    A3 = CuArray(_A);
    x2 = CuArray(rand(L,N));
    x3 = CuArray(rand(L,1,N));
    y2 = CuArray(rand(L,N));
    y3 = CuArray(rand(L,1,N));

    tbgemm = @belapsed_median CUDA.@sync batched_mul!($y3, $A3, $x3)

    tbgemvn = @belapsed_median CUDA.@sync batched_gemv!('N', 1.0, $A3, $x2, 0.0, $y2)
    tbgemvt = @belapsed_median CUDA.@sync batched_gemv!('T', 1.0, $A3, $x2, 0.0, $y2)

    tbsymvu = @belapsed_median CUDA.@sync batched_symv!('U', 1.0, $A3, $x2, 0.0, $y2)
    tbsymvl = @belapsed_median CUDA.@sync batched_symv!('L', 1.0, $A3, $x2, 0.0, $y2)

    AP = CuArray(hcat([SymmetricPacked(x, :U).tri for x in eachslice(_A, dims=3)]...));
    tbspmvu = @belapsed_median CUDA.@sync batched_spmv!('U', 1.0, $AP, $x2, 0.0, $y2)
    AP = CuArray(hcat([SymmetricPacked(x, :L).tri for x in eachslice(_A, dims=3)]...));
    tbspmvl = @belapsed_median CUDA.@sync batched_spmv!('L', 1.0, $AP, $x2, 0.0, $y2)

    CUDA.unsafe_free!.((A3, AP, x2, x3, y2, y3))
    CUDA.memory_status()

    return tbgemm, tbgemvn, tbgemvt, tbsymvu, tbsymvl, tbspmvu, tbspmvl
end

df_b_mv = DataFrame(func = String[], N = Int[], L = Int[], elapsed_time = Float64[])
              
@info "batched_{gemv,symv,spmv}"
N=32768
for L=[32, 64, 128, 256]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_mv, ("bgemm",  N, L, times[1]))
    push!(df_b_mv, ("bgemvn", N, L, times[2]))
    push!(df_b_mv, ("bgemvt", N, L, times[3]))
    push!(df_b_mv, ("bsymvu", N, L, times[4]))
    push!(df_b_mv, ("bsymvl", N, L, times[5]))
    push!(df_b_mv, ("bspmvu", N, L, times[6]))
    push!(df_b_mv, ("bspmvl", N, L, times[7]))
end

L=128
for N=[8192, 16384, 65536, 131072]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_mv, ("bgemm",  N, L, times[1]))
    push!(df_b_mv, ("bgemvn", N, L, times[2]))
    push!(df_b_mv, ("bgemvt", N, L, times[3]))
    push!(df_b_mv, ("bsymvu", N, L, times[4]))
    push!(df_b_mv, ("bsymvl", N, L, times[5]))
    push!(df_b_mv, ("bspmvu", N, L, times[6]))
    push!(df_b_mv, ("bspmvl", N, L, times[7]))
end


function doit(L,N)
    _A = rand(L,L,N);
    A3 = CuArray(_A);
    x2 = CuArray(rand(L,N));
    x3 = CuArray(rand(L,1,N));
    y2 = CuArray(rand(L,N));
    y3 = CuArray(rand(L,1,N));

    tbgemm = @belapsed_median CUDA.@sync batched_mul!($A3, $x3, batched_transpose($x3), -1.0, 1.0)

    tbger = @belapsed_median CUDA.@sync batched_ger!(-1.0, $x2, $y2, $A3)

    tbsyru = @belapsed_median CUDA.@sync batched_syr!('U', -1.0, $x2, $A3)
    tbsyrl = @belapsed_median CUDA.@sync batched_syr!('L', -1.0, $x2, $A3)

    AP = CuArray(hcat([SymmetricPacked(x, :U).tri for x in eachslice(_A, dims=3)]...));
    tbspru = @belapsed_median CUDA.@sync batched_spr!('U', -1.0, $x2, $AP)
    AP = CuArray(hcat([SymmetricPacked(x, :L).tri for x in eachslice(_A, dims=3)]...));
    tbsprl = @belapsed_median CUDA.@sync batched_spr!('L', -1.0, $x2, $AP)

    CUDA.unsafe_free!.((A3, AP, x2, x3, y2, y3))
    CUDA.memory_status()

    return tbgemm, tbger, tbsyru, tbsyrl, tbspru, tbsprl
end

df_b_r = DataFrame(func = String[], N = Int[], L = Int[], elapsed_time = Float64[])
              
@info "batched_{ger,syr,spr}"
N=32768
for L=[32, 64, 128, 256]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_r, ("bgemm", N, L, times[1]))
    push!(df_b_r, ("bger",  N, L, times[2]))
    push!(df_b_r, ("bsyru", N, L, times[3]))
    push!(df_b_r, ("bsyrl", N, L, times[4]))
    push!(df_b_r, ("bspru", N, L, times[5]))
    push!(df_b_r, ("bsprl", N, L, times[6]))
end

L=128
for N=[8192, 16384, 65536, 131072]
    @info string("N=", N, "; L=", L)
    times = doit(L, N)
    push!(df_b_r, ("bgemm", N, L, times[1]))
    push!(df_b_r, ("bger",  N, L, times[2]))
    push!(df_b_r, ("bsyru", N, L, times[3]))
    push!(df_b_r, ("bsyrl", N, L, times[4]))
    push!(df_b_r, ("bspru", N, L, times[5]))
    push!(df_b_r, ("bsprl", N, L, times[6]))
end


save("bench.jld2", "df_b_dot", df_b_dot, "df_b_mv", df_b_mv, "df_b_r", df_b_r)
#=
df_b_dot = load("bench.jld2", "df_b_dot")
df_b_mv = load("bench.jld2", "df_b_mv")
df_b_r = load("bench.jld2", "df_b_r")
=#


theme = Theme(highlight_width=0mm, point_size=0.6mm, line_style=[:solid, :dot]);

gd = groupby(df_b_dot, [:N, :L])
df_b_dot2 = combine(gd, :func => identity, :elapsed_time => x->x./x[1])

Nbdotplot = plot(df_b_dot2[df_b_dot2.L.==2^7, :],
             x=:N, y=:elapsed_time_function, color=:func_identity,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.colorkey(title=""),
             Guide.title("L=2<sup>7</sup>"),
             Guide.xlabel("N", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

Lbdotplot = plot(df_b_dot2[df_b_dot2.N.==2^15, :],
             x=:L, y=:elapsed_time_function, color=:func_identity,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.xticks(orientation=:horizontal),
             Guide.colorkey(title=""),
             Guide.title("N=2<sup>15</sup>"),
             Guide.xlabel("L", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

df_b_mv[!,:func2] .= [x[end] in ['t', 'u'] ? 2 : 1 for x in df_b_mv[!,:func]]
df_b_mv[!,:func] .=  [rstrip(x, ['u', 'l', 'n', 't']) for x in df_b_mv[!,:func]]

gd = groupby(df_b_mv, [:N, :L])
df_b_mv2 = transform(gd, :elapsed_time => x->x./x[1])

Nbmvplot = plot(df_b_mv2[df_b_mv2.L.==2^7, :],
             x=:N, y=:elapsed_time_function, color=:func, linestyle=:func2,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.colorkey(title=""),
             Guide.title("L=2<sup>7</sup>"),
             Guide.xlabel("N", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

Lbmvplot = plot(df_b_mv2[df_b_mv2.N.==2^15, :],
             x=:L, y=:elapsed_time_function, color=:func, linestyle=:func2,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.xticks(orientation=:horizontal),
             Guide.colorkey(title=""),
             Guide.title("N=2<sup>15</sup>"),
             Guide.xlabel("L", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

df_b_r[!,:func2] .= [x[end]=='u' ? 2 : 1 for x in df_b_r[!,:func]]
df_b_r[!,:func] .=  [rstrip(x, ['u', 'l']) for x in df_b_r[!,:func]]

gd = groupby(df_b_r, [:N, :L])
df_b_r2 = transform(gd, :elapsed_time => x->x./x[1])

Nbrplot = plot(df_b_r2[df_b_r2.L.==2^7, :],
             x=:N, y=:elapsed_time_function, color=:func, linestyle=:func2,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.colorkey(title=""),
             Guide.title("L=2<sup>7</sup>"),
             Guide.xlabel("N", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

Lbrplot = plot(df_b_r2[df_b_r2.N.==2^15, :],
             x=:L, y=:elapsed_time_function, color=:func, linestyle=:func2,
             Scale.x_log2,
             Geom.line, Geom.point,
             Guide.xticks(orientation=:horizontal),
             Guide.colorkey(title=""),
             Guide.title("N=2<sup>15</sup>"),
             Guide.xlabel("L", orientation=:horizontal),
             Guide.ylabel("time re. bgemm", orientation=:vertical),
             theme);

gridstack([Nbdotplot Nbmvplot Nbrplot;
           Lbdotplot Lbmvplot Lbrplot]) |> SVG("bench.svg", 21cm, 12cm)
