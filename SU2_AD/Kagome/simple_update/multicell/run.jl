# Kagome SU(2) SU runner: julia --project=. run.jl --cell sqrt3xsqrt3 --Dmax 6
using TensorKit, Random, LinearAlgebra, Dates, Printf
if !isdefined(@__MODULE__,:KagomeSU2Cells)
    include(joinpath(@__DIR__,"..","..","..","src","bosonic","kagome",
                     "simple_update_cells.jl"))
end
using .KagomeSU2Cells

function options(args)
    values = Dict("cell"=>"sqrt3xsqrt3", "Dmax"=>"3", "seed"=>"1234",
                  "threads"=>"10", "virtual"=>"0:1,1/2:1", "out"=>"",
                  "resume"=>"", "J1"=>"1.0")
    smoke = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--smoke"
            smoke = true
        elseif arg == "--help"
            println("--cell sqrt3xsqrt3|3x3|1x1 --Dmax N --virtual 0:1,1/2:1 ",
                    "--seed N --threads 10 --J1 1 --out DIRECTORY --resume CHECKPOINT --smoke")
            return nothing
        else
            startswith(arg,"--") || error("unexpected argument: $arg")
            key = arg[3:end]
            haskey(values,key) || error("unknown option: $arg")
            i < length(args) || error("missing value for $arg")
            i += 1
            values[key] = args[i]
        end
        i += 1
    end
    return values,smoke
end

function virtual_space(text)
    pairs = Pair{Rational{Int},Int}[]
    for item in split(text,',')
        spin,multiplicity = split(item,':')
        parts = split(spin,'/')
        j = length(parts) == 1 ? parse(Int,parts[1])//1 :
            length(parts) == 2 ? parse(Int,parts[1])//parse(Int,parts[2]) : error("invalid spin")
        n = parse(Int,multiplicity)
        j >= 0 && denominator(2j) == 1 && n > 0 || error("invalid SU(2) irrep/multiplicity")
        any(p->first(p)==j,pairs) && error("duplicate spin in --virtual")
        push!(pairs,j=>n)
    end
    return SU2Space(pairs...)
end

function main(args=ARGS)
    parsed = options(args)
    isnothing(parsed) && return
    opts,smoke = parsed
    pattern = opts["cell"] == "sqrt3xsqrt3" ? :sqrt3xsqrt3 : :rectangular
    opts["cell"] in ("sqrt3xsqrt3","3x3","1x1") || error("unsupported --cell")
    length_side = opts["cell"] == "1x1" ? 1 : 3
    geometry = KagomeSUCellGeometry(length_side,length_side;pattern)
    Dmax = parse(Int,opts["Dmax"])
    seed = parse(Int,opts["seed"])
    threads = parse(Int,opts["threads"])
    Dmax > 0 && threads > 0 || error("Dmax and threads must be positive")
    BLAS.set_num_threads(threads)
    J1 = parse(Float64,opts["J1"])
    isfinite(J1) && J1 != 0 || error("J1 must be finite and nonzero")
    schedule = smoke ? [(dt=.001,tau=.001),(dt=.0005,tau=.0005)] :
        [(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),
         (dt=.001,tau=2.),(dt=.0005,tau=1.)]
    out = isempty(opts["out"]) ? joinpath(@__DIR__,"results",
        "$(opts["cell"])_D$(Dmax)_seed$(seed)_$(Dates.format(now(),"yyyymmdd_HHMMSS"))") :
        abspath(opts["out"])
    ispath(out) && error("output already exists; use a NEW --out directory (also for --resume)")
    completed = 0
    if isempty(opts["resume"])
        Vv = virtual_space(opts["virtual"])
        dim(Vv) <= Dmax || error("initial virtual dimension exceeds Dmax")
        state = initialize_cells(geometry,Vv;rng=MersenneTwister(seed))
        H = heisenberg_triangle(;J1)
    else
        state,data = load_checkpoint(opts["resume"])
        state.geometry.site_map == geometry.site_map || error("resume cell mismatch")
        metadata = data["metadata"]
        metadata["Dmax"] == Dmax || error("resume Dmax mismatch")
        metadata["J1"] == J1 || error("resume J1 mismatch")
        metadata["schedule"] == schedule || error("resume schedule mismatch")
        completed = metadata["completed_stages"]
        seed = metadata["seed"]
        H = data["H"]
    end
    mkpath(out)
    metadata = Dict{String,Any}("cell"=>opts["cell"],"Dmax"=>Dmax,"seed"=>seed,
        "J1"=>J1,"schedule"=>schedule,"completed_stages"=>completed,
        "julia"=>string(VERSION),"TensorKit"=>string(pkgversion(TensorKit)),
        "smoke_test"=>smoke,"parent_checkpoint"=>opts["resume"])
    save_checkpoint(joinpath(out,"initial.jld2"),state,H;metadata)
    println("cell=$(opts["cell"]), independent PESS=$(length(state.cells)), ",
            "physical spins=$(3length(state.cells)), Dmax=$Dmax, BLAS threads=$threads")
    println("h=0 SU(2) calculation; no CTMRG is launched by this runner.")
    open(joinpath(out,"progress.txt"),"w") do log
        for k in completed+1:length(schedule)
            stage = schedule[k]
            println("stage $k/$(length(schedule)): dt=$(stage.dt), tau=$(stage.tau), ",
                    "steps=$(round(Int,stage.tau/stage.dt))"); flush(stdout)
            start = time()
            callback = (s,step,total)->begin
                if step == 1 || step % 100 == 0 || step == total
                    println("  step $step/$total"); flush(stdout)
                end
            end
            state = itebd_cells(state,H,stage.tau,stage.dt;Dmax,callback)
            metadata["completed_stages"] = k
            elapsed = time()-start
            save_checkpoint(joinpath(out,@sprintf("stage_%02d.jld2",k)),state,H;metadata)
            println(log,"stage=$k dt=$(stage.dt) tau=$(stage.tau) seconds=$elapsed")
            flush(log)
            println("stage $k saved; seconds=$elapsed"); flush(stdout)
        end
    end
    final_path = save_checkpoint(joinpath(out,"final.jld2"),state,H;metadata)
    # Explicitly labelled diagnostics: these are not CTMRG energies.
    estimates = su_energy_estimates(state,H)
    open(joinpath(out,"su_diagnostics.txt"),"w") do log
        println(log,"Local SU lambda-environment estimates; NOT CTMRG/variational energies.")
        println(log,"up_triangle = ",estimates.up)
        println(log,"down_triangle = ",estimates.down)
        println(log,"per_site_estimate = ",estimates.per_site)
        for item in bond_report(state)
            println(log,item)
        end
    end
    println("saved: ",final_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
