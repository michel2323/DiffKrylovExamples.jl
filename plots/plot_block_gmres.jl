import PythonPlot
using Plots
# using PyPlot
using CSV
using DataFrames
using LaTeXStrings


pythonplot()
PythonPlot.matplotlib.rcParams["text.usetex"] = true
PythonPlot.matplotlib.rcParams["font.family"] = "serif"
PythonPlot.matplotlib.rcParams["font.size"] = 12
folder="results"
ncases=64

title = L"\mbox{Block-GMRES}"
df_iters = CSV.read("$folder/iters_block_gmres.csv", DataFrame)
df_stats = CSV.read("$folder/stats_block_gmres.csv", DataFrame)

success = "solution good enough given atol and rtol"
fstats = [df_stats[!,:gmres]]
push!(fstats, df_stats[!,:bgmres8])
push!(fstats, df_stats[!,:bgmres16])
push!(fstats, df_stats[!,:bgmres32])

fiters = [df_iters[!,:gmres]]
push!(fiters, df_iters[!,:bgmres8])
push!(fiters, df_iters[!,:bgmres16])
push!(fiters, df_iters[!,:bgmres32])

sidx = []
for fstat in fstats
    push!(sidx, findall(x -> x == success, fstat))
end
fiters .= getindex.(fiters, sidx)
sort!.(fiters)
itermax = 10

fsuccess = [zeros(Int, itermax) for i in 1:4]
for (k,fiter) in enumerate(fiters)
    for i in fiter
        for j in i:itermax
            fsuccess[k][j] += 1
        end
    end
end

p = plot(size=(500,250))
plot!(p, fsuccess[1], label="GMRES", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=1, legend=:bottomright)
plot!(p, fsuccess[2], label="BGMRES8", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=1)
plot!(p, fsuccess[3], label="BGMRES16", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=1)
plot!(p, fsuccess[4], label="BGMRES32", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=1)
savefig(p, "results/plot_blockgmres.png")
savefig(p, "results/plot_blockgmres.pdf")
