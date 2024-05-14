using Plots
using CSV
using DataFrames

folder="results"
ncases=64

cases = [
    ["1e-8", "GMRES", "GmresSolver", false, :red],
    ["1e-8", "GMRES Restart", "GmresSolver", true, :blue],
    ["1e-8", "BiCGSTAB", "BicgstabSolver", false, :green]
]
ps = []
for case in cases
    stol = case[1]
    series = case[2]
    solver = case[3]
    restart = case[4]
    color = case[5]
    tol = parse(Float64, case[1])

    title = "$series solver with tol=$stol"
    df = CSV.read("$folder/(1.0e-10, 1.0e-10)/iters_$(solver)_$restart.csv", DataFrame)

    fiter = df[!,:nit_P]
    riter = df[!,:adj_nit_P]

    fstats = df[!,:Pres]
    rstats = df[!,:adj_Pres]

    noPfiter = df[!,:nit_noP]
    noPriter = df[!,:adj_nit_noP]

    noPfstats = df[!,:res]
    noPrstats = df[!,:adj_res]

    fiter = fiter[findall(x -> x < tol, fstats)]
    riter = riter[findall(x -> x < tol, rstats)]

    noPfiter = noPfiter[findall(x -> x < tol, noPfstats)]
    noPriter = noPriter[findall(x -> x < tol, noPrstats)]

    sort!(fiter)
    sort!(riter)
    sort!(noPfiter)
    sort!(noPriter)

    # itermax = max(maximum(fiter), maximum(riter))
    # itermax = max(maximum(fiter), maximum(riter), maximum(noPfiter), maximum(noPriter))
    itermax = 10

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
    p = plot()
    # plot!(p, fsuccess, label="Original", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=color)
    # plot!(p, rsuccess, label="Adjoint", lw=2, ylim=(1,ncases), linecolor=color, linestyle=:dash)
    plot!(p, fsuccess, label="Original", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=:blue)
    plot!(p, rsuccess, label="Adjoint", lw=2, ylim=(1,ncases), linecolor=:blue, linestyle=:dash)
    plot!(p, noPfsuccess, label="Original no P", xlabel="Iterations", ylim=(1,ncases), ylabel="Successes", title=title, lw=2, linecolor=:red)
    plot!(p, noPrsuccess, label="Adjoint no P", lw=2, ylim=(1,ncases), linecolor=:red, linestyle=:dash)
    push!(ps, p)
    savefig(p, "results/plot_$(solver)_$(restart)_$(tol)_zoom.pdf")
end
display(ps[1])
display(ps[2])
display(ps[3])
# for (i,p) in enumerate(ps)
#     savefig(p, "results/plot_$i.png")
# end