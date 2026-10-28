# Used by run.jl --symmetry u1 for neutral or charged rectangular / sqrt3 cells.
if !isdefined(@__MODULE__,:KagomeU1Cells)
    include(joinpath(@__DIR__,"..","..","..","src","bosonic","kagome","simple_update_u1_cells.jl"))
end
using .KagomeU1Cells
using JLD2

function run_u1_options(opts,smoke;schedule_override=nothing)
    cell=opts["cell"];cell in ("1x1","3x3","sqrt3xsqrt3") || error("unsupported cell")
    size=cell=="1x1" ? 1 : 3
    pattern=cell=="sqrt3xsqrt3" ? :sqrt3xsqrt3 : :rectangular
    Dmax=parse(Int,opts["Dmax"]);seed=parse(Int,opts["seed"])
    threads=parse(Int,opts["threads"])
    Dmax>0 && threads>0 || error("Dmax and threads must be positive")
    BLAS.set_num_threads(threads)
    J1=parse(Float64,opts["J1"])
    isfinite(J1) && J1!=0 || error("J1 must be finite and nonzero")
    schedule=!isnothing(schedule_override) ? schedule_override : smoke ? [(dt=.001,tau=.001),(dt=.0005,tau=.0005)] :
        [(dt=.03,tau=40.),(dt=.01,tau=20.),(dt=.003,tau=5.),(dt=.001,tau=2.),(dt=.0005,tau=1.)]
    out=isempty(opts["out"]) ? joinpath(@__DIR__,"results","u1_$(cell)_D$(Dmax)_$(Dates.format(now(),"yyyymmdd_HHMMSS"))") : abspath(opts["out"])
    ispath(out) && error("use a NEW output directory")
    completed=0
    isempty(opts["init"]) || isempty(opts["resume"]) || error("choose --init or --resume")
    if isempty(opts["resume"])
        if isempty(opts["init"])
            g=KagomeSUCellGeometry(size,size;pattern)
            source=initialize_cells(g,virtual_space(opts["virtual"]);rng=MersenneTwister(seed))
            state=to_u1(source)
        else
            data=JLD2.load(opts["init"])
            if data["format"]=="kagome_u1_cells_v1"
                state=first(load_u1_checkpoint(opts["init"]))
            elseif data["format"]=="kagome_su2_charged_sqrt3_v1"
                source=first(KagomeU1Cells.SU2.load_charged_checkpoint(opts["init"]))
                state=to_u1(source;component=parse(Int,opts["component"]))
            elseif data["format"]=="kagome_su2_cells_v1"
                state=to_u1(data["state"])
            else
                error("unsupported --init checkpoint")
            end
            pattern==:rectangular && size==3 && state.geometry.pattern==:sqrt3xsqrt3 &&
                (state=expand_u1_rectangular(state))
            state.geometry.pattern==pattern && state.geometry.Lx==size && state.geometry.Ly==size || error("initial cell mismatch")
        end
        H=u1_heisenberg_triangle(;J1)
    else
        state,data=load_u1_checkpoint(opts["resume"])
        metadata=data["metadata"]
        metadata["schedule"]==schedule && metadata["Dmax"]==Dmax && metadata["J1"]==J1 || error("resume settings mismatch")
        state.geometry.pattern==pattern && state.geometry.Lx==size && state.geometry.Ly==size || error("resume cell mismatch")
        completed=metadata["completed_stages"];seed=metadata["seed"];H=data["H"]
    end
    mkpath(out)
    metadata=Dict{String,Any}("symmetry"=>"u1","cell"=>cell,"Dmax"=>Dmax,"J1"=>J1,
        "seed"=>seed,"schedule"=>schedule,"completed_stages"=>completed,
        "initialization"=>isempty(opts["resume"]) ?
            (isempty(opts["init"]) ? "exact SU2 restriction" : abspath(opts["init"])) :
            get(metadata,"initialization","loaded U1 checkpoint"),
        "parent_checkpoint"=>opts["resume"],"smoke_test"=>smoke)
    save_u1_checkpoint(joinpath(out,"initial.jld2"),state,H;metadata)
    for k in completed+1:length(schedule)
        stage=schedule[k]
        println("U1 stage $k/$(length(schedule)): dt=$(stage.dt), tau=$(stage.tau)");flush(stdout)
        state=itebd_u1(state,H,stage.tau,stage.dt;Dmax)
        metadata["completed_stages"]=k
        save_u1_checkpoint(joinpath(out,@sprintf("stage_%02d.jld2",k)),state,H;metadata)
    end
    save_u1_checkpoint(joinpath(out,"final.jld2"),state,H;metadata)
    open(joinpath(out,"u1_state.toml"),"w") do io
        TOML.print(io,export_u1_portable(state);sorted=true)
    end
    println("U1 state saved: ",out)
    return state
end
