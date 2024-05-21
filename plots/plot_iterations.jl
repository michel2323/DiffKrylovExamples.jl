import PythonPlot
using Plots
# using PyPlot
using CSV
using DataFrames
using LaTeXStrings


pythonplot()
mycolor=:orange
nopmycolor=:red
PythonPlot.matplotlib.rcParams["text.usetex"] = true
PythonPlot.matplotlib.rcParams["font.family"] = "serif"
PythonPlot.matplotlib.rcParams["font.size"] = 12
folder="results"
ncases=64
title = L"""
GMRES $\alpha$
"""

@show String(title)
cases = [
    ["1e-6", "GMRES", "GmresSolver", false, :orange, "1e^{-6}"],
    ["1e-6", "GMRES(30)", "GmresSolver", true, :blue, "1e^{-6}"],
    ["1e-6", "BiCGSTAB", "BicgstabSolver", false, :green, "1e^{-6}"],
    ["1e-6", "QMR", "QmrSolver", false, :green, "1e^{-6}"],
    ["1e-6", "Bilq", "BilqSolver", false, :green, "1e^{-6}"]
]
ps = []
for (zoom,zend) in [(false, ""), (true, "z")]
for case in cases
    stol = case[1]
    series = case[2]
    solver = case[3]
    restart = case[4]
    color = case[5]
    ltol = case[6]
    tol = parse(Float64, case[1])

    title = L"%$series $ $"
    df = CSV.read("$folder/1.0e-10/iters_$(solver)_$restart.csv", DataFrame)

    fiter = df[!,:nit_P]
    riter = df[!,:adj_nit_P]

    fstats = df[!,:Pres]
    rstats = df[!,:adj_Pres]

    noPfiter = df[!,:nit_noP]
    noPriter = df[!,:adj_nit_noP]

    noPfstats = df[!,:res]
    noPrstats = df[!,:adj_res]
    @show riter 
    @show rstats 
    fiter = fiter[findall(x -> x < tol, fstats)]
    riter = riter[findall(x -> x < tol, rstats)]

    noPfiter = noPfiter[findall(x -> x < tol, noPfstats)]
    noPriter = noPriter[findall(x -> x < tol, noPrstats)]

    sort!(fiter)
    sort!(riter)
    sort!(noPfiter)
    sort!(noPriter)

    # itermax = max(maximum(fiter), maximum(riter))
    itermax=0
    @show zoom
    @show fiter, riter
    if zoom
        itermax = 10
    else
        itermax = max(maximum(fiter), maximum(riter), maximum(noPfiter), maximum(noPriter))
    end

    fsuccess = zeros(Int, itermax)
    rsuccess = zeros(Int, itermax)
    noPfsuccess = zeros(Int, itermax)
    noPrsuccess = zeros(Int, itermax)

    for i in fiter
        for j in i:itermax
            fsuccess[j] += 1
        end
    end

    for i in riter
        for j in i:itermax
            rsuccess[j] += 1
        end
    end

    for i in noPfiter
        for j in i:itermax
            noPfsuccess[j] += 1
        end
    end

    for i in noPriter
        for j in i:itermax
            noPrsuccess[j] += 1
        end
    end
    # precond

    # yticks=collect(range(1,10))
    p = plot(size=(250,250))
    # PythonPlot.figure(figsize=(7.5,2.5))
    # plot!(p, fsuccess, label="Original", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=color)
    # plot!(p, rsuccess, label="Adjoint", lw=2, ylim=(1,ncases), linecolor=color, linestyle=:dash)
    plot!(p, fsuccess, label="Original", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=:darkblue)
    plot!(p, rsuccess, label="Adjoint", lw=2, ylim=(1,ncases), linecolor=:red)
    plot!(p, noPfsuccess, label="Original no P", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=:orange)
    plot!(p, noPrsuccess, label="Adjoint no P", lw=2, ylim=(1,ncases), linecolor=:green)
    push!(ps, p)
    savefig(p, "results/plot_$(solver)_$(restart)_$(tol)$(zend).pdf")
end
end
for p in ps
    display(p)
end
# for (i,p) in enumerate(ps)
#     savefig(p, "results/plot_$i.png")
# end