function measure_state(state,model,settings)
    A = convert_iPESS_to_iPEPS(state)
    U = Opt._fusion(projected_spin_space())
    measure_heisenberg(A,U;J=model.J,h=model.h,chi=settings["chi"],
        maxiter=settings["iterations"],tol=get(settings,"tol",1e-7),
        projector_tol=get(settings,"projector_tol",1e-10),verbose=get(settings,"verbose",false))
end

function write_measurement(path,state,model,settings)
    r = measure_state(state,model,settings)
    keys = (:energy_per_site,:exchange_per_site,:zeeman_per_site,:magnetization_per_site,
            :mz_over_saturation,:max_imag,:chi,:iterations,:ctm_error,:converged,:J,:h)
    record = Dict(string(key)=>getproperty(r,key) for key in keys)
    open(io->TOML.print(io,record),path,"w")
    jldsave(splitext(path)[1]*"_observables.jld2";up_bonds=r.up_bonds,down_bonds=r.down_bonds,
            magnetization=r.magnetization,J=r.J,h=r.h)
    println(basename(path)," E/site=",r.energy_per_site," Mz/Msat=",r.mz_over_saturation,
            " CTM error=",r.ctm_error," converged=",r.converged)
    return record
end

"""Start an independent SU, FU or AD run from a projected native iPESS.
MAT input is projected once. A projected JLD2 checkpoint can serve as a seed
at the same or a new field. A new optimizer trajectory is started in both cases.
"""
function run_spin_optimization(method,path)
    method in ("su","fu","ad") || throw(ArgumentError("expected su, fu or ad"))
    path = abspath(path)
    cfg = TOML.parsefile(path)
    job = cfg["job"]
    input = normpath(joinpath(dirname(path),job["input"]))
    state = if endswith(lowercase(input),".mat")
        first(import_gaussian_ipess(input;gutzwiller=true))
    elseif endswith(lowercase(input),".jld2")
        first(load_projected_checkpoint(input))
    else
        throw(ArgumentError("input must be a native iPESS MAT file or projected JLD2 checkpoint"))
    end
    model = HeisenbergField(cfg["model"]["J"],cfg["model"]["h"])
    Dmax = job["D_max"]
    KF.check_state(state;Dmax)
    BLAS.set_num_threads(get(job,"blas_threads",1))
    # mkdir reserves a fresh method directory, also for simultaneous jobs.
    parent = normpath(joinpath(dirname(path),job["output"]))
    mkpath(parent)
    out = joinpath(parent,method)
    ispath(out) && error("Output already exists: $out. Select a fresh job.output.")
    mkdir(out)
    cp(path,joinpath(out,"config.toml"))
    source_hash = bytes2hex(open(sha256,input))
    parameters = Dict("model"=>"heisenberg_z","J"=>model.J,"h"=>model.h)
    checkpoint = joinpath(out,"state.jld2")
    function save(s,step)
        temp = checkpoint*".tmp"
        save_projected_checkpoint(temp,s;parameters,
            metadata=Dict("method"=>method,"step"=>step,"input_sha256"=>source_hash))
        Base.Filesystem.rename(temp,checkpoint)
    end
    save(state,0)
    initial = write_measurement(joinpath(out,"initial.toml"),state,model,cfg["measurement"])
    steps = 0
    status = "completed_schedule"
    open(joinpath(out,"history.tsv"),"w") do io
        if method=="ad"
            opt = cfg["ad"]
            println(io,"iteration\tobjective_per_site\tgradient_norm\tstep")
            on_step = function(s,r)
                println(io,join((r.iteration,r.energy,r.gradient_norm,r.step),'\t'))
                flush(io)
                save(s,r.iteration)
            end
            result = optimize_variational(state,model;chi=opt["chi"],ctm_iterations=opt["iterations"],
                maxiter=opt["maxiter"],step_size=get(opt,"step_size",0.2),
                memory=get(opt,"memory",0),grad_tol=get(opt,"grad_tol",1e-6),
                projector_tol=get(opt,"projector_tol",1e-10),callback=on_step)
            state,steps,status = result.state,last(result.history).iteration,string(result.status)
        else
            println(io,method=="su" ? "step\tdiscarded_weight" : "stage\tstep\torientation\tx\ty\tfidelity\tloss\tmetric_error")
            for (stage,schedule) in enumerate(cfg["schedule"])
                tau,dt = schedule["tau"],schedule["dt"]
                if method=="su"
                    on_sweep = function(s,r)
                        steps += 1
                        println(io,steps,'\t',r.discarded_weight)
                    end
                    itebd!(state,model,tau,dt,Dmax;trun_tol=get(job,"trun_tol",1e-10),callback=on_sweep)
                else
                    opt = cfg["fu"]
                    on_triangle = function(s,r)
                        println(io,join((stage,r.step,r.orientation,r.x,r.y,r.fidelity,
                            r.loss,r.environment.extraction_error),'\t'))
                    end
                    full_update!(state,model,tau,dt,Dmax;chi=opt["chi"],
                        ctm_iterations=opt["iterations"],n_sweep=opt["n_sweep"],
                        ctm_tol=get(opt,"tol",1e-7),projector_tol=get(opt,"projector_tol",1e-10),
                        trun_tol=get(job,"trun_tol",1e-10),use_gauge=get(opt,"use_gauge",true),
                        callback=on_triangle)
                    steps += round(Int,tau/dt)
                end
                flush(io)
                save(state,steps)
            end
        end
    end
    save(state,steps)
    final = write_measurement(joinpath(out,"final.toml"),state,model,cfg["measurement"])
    open(io->TOML.print(io,Dict("method"=>method,"status"=>status,"steps"=>steps,
        "input_sha256"=>source_hash)),joinpath(out,"run.toml"),"w")
    return (;state,initial,final,checkpoint,status)
end
