isdefined(Main,:PAIR_OUTPUT) || include("common.jl")
include("validate.jl")
const PAIR_SU_SCHEDULE = ((8.0,0.1),(4.0,0.02),(3.0,0.005),(1.0,0.001))
const PAIR_FU_SCHEDULE = ((0.5,0.05),(0.2,0.02),(0.05,0.005))

function choose_measurement(records)
    valid=filter(r -> r["ctm_converged"],records)
    isempty(valid) && error("no converged CTM measurement")
    valid[argmax([(r["chi"],-r["projector_tol"]) for r in valid])]
end

function pairing_problem(state,parameters,measured,directory;energy_tolerance=2e-5)
    v,_=pack_state(state)
    attempts=[]
    for chi in (12,16),cutoff in (1e-8,1e-7,1e-6)
        energies=[variational_energy(v,VariationalProblem(state,parameters;chi,ctm_iterations=n,projector_tol=cutoff)) for n in (4,8)]
        iterations=abs(energies[2]-energies[1])<1e-7 ? 4 : 8
        problem=VariationalProblem(state,parameters;chi,ctm_iterations=iterations,projector_tol=cutoff)
        report=check_gradient(v,problem;epsilon=1e-5)
        record=merge(asdict(report),Dict("chi"=>chi,"ctm_iterations"=>iterations,"projector_tol"=>cutoff,
            "difference_from_measured"=>report.energy-measured,"sweep_energy_difference"=>energies[2]-energies[1],
            "energy_tolerance"=>energy_tolerance))
        push!(attempts,record)
        note(directory,"AD_CHECK $record")
        write_toml(joinpath(directory,"ad_initial_checks.toml"),Dict("attempts"=>attempts))
        if (report.relative_error<1e-3 || report.absolute_error<1e-7) && abs(report.energy-measured)<energy_tolerance
            return (;chi,iterations,cutoff,problem)
        end
    end
    error("no AD setting passed both gradient and energy checks")
end

function run_pairing(delta;ad_maxiter=50,su_only=false)
    directory=joinpath(PAIR_OUTPUT,pair_name(delta)); mkpath(directory)
    parameters=Dict{String,Any}("t_up"=>1.0,"t_dn"=>1.0,"U"=>0.0,"μ"=>-1.0,"pairing"=>delta,"centered"=>true)
    references=[pairing_reference(;delta,mesh=n) for n in (300,600)]
    @assert abs(references[1]["energy_per_site"]-references[2]["energy_per_site"])<1e-9
    exact=references[end]
    write_toml(joinpath(directory,"settings.toml"),Dict("parameters"=>parameters,"reference"=>references,
        "D_max"=>4,"seed"=>888,"su_schedule_tau_dt"=>[collect(s) for s in PAIR_SU_SCHEDULE],
        "fu_schedule_tau_dt"=>[collect(s) for s in PAIR_FU_SCHEDULE],"ad_maxiter"=>ad_maxiter,
        "initialization"=>"FU and AD independently start from the same SU checkpoint"))
    note(directory,"BEGIN delta=$delta exact=$exact")
    sufile=joinpath(directory,"su.jld2")
    if isfile(sufile)
        su,saved,_=load_checkpoint(sufile); @assert saved==parameters
    else
        su=initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(888))
        history=[]
        previous=Inf
        for (stage,(tau,dt)) in enumerate(vcat(collect(PAIR_SU_SCHEDULE),fill((1.0,0.001),3)))
            elapsed=@elapsed itebd!(su,parameters,tau,dt,4;trun_tol=1e-12)
            measured=pair_measure(directory,"su_stage$(stage)",su,parameters,exact;settings=((16,1e-8),))
            energy=measured[1]["energy"]
            record=(;stage,tau,dt,energy,energy_change=energy-previous,seconds=elapsed)
            push!(history,record); note(directory,"SU $record sectors=$(sectors(su))")
            save_checkpoint(joinpath(directory,"su_latest.jld2"),su;parameters,metadata=Dict("history"=>history))
            stage>=5 && abs(energy-previous)<2e-7 && break
            previous=energy
        end
        save_checkpoint(sufile,su;parameters,metadata=Dict("method"=>"SU","history"=>history,"sectors"=>sectors(su)))
    end
    su_measured=choose_measurement(pair_measure(directory,"su",su,parameters,exact))
    su_only && return su

    fufile=joinpath(directory,"fu.jld2")
    if isfile(fufile)
        fu,saved,_=load_checkpoint(fufile); @assert saved==parameters
    else
        fu=deepcopy(su); history=[]
        for (stage,(tau,dt)) in enumerate(PAIR_FU_SCHEDULE)
            counter=Ref(0)
            callback=function(s,record)
                counter[]+=1
                counter[]%3==0 && note(directory,"FU stage=$stage step=$(record.step) fidelity=$(record.fidelity) CTM_error=$(record.environment.ctm_error)")
            end
            records=full_update!(fu,parameters,tau,dt,4;chi=12,ctm_iterations=60,ctm_tol=1e-7,
                projector_tol=1e-8,n_sweep=8,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
            append!(history,records)
            save_checkpoint(joinpath(directory,"fu_latest.jld2"),fu;parameters,metadata=Dict("stage"=>stage,"history"=>history))
            pair_measure(directory,"fu_stage$(stage)",fu,parameters,exact;settings=((16,1e-8),))
        end
        save_checkpoint(fufile,fu;parameters,metadata=Dict("method"=>"FU","history"=>history,"chi"=>12,"projector_tol"=>1e-8,"sectors"=>sectors(fu)))
    end
    pair_measure(directory,"fu",fu,parameters,exact)

    adfile=joinpath(directory,"ad.jld2")
    if isfile(adfile)
        ad,saved,_=load_checkpoint(adfile); @assert saved==parameters
    else
        # Limit the optimization objective's bias to 0.1% of the SU energy
        # error, with the original absolute floor for the more accurate states.
        # Final comparisons always use independent stricter CTM measurements.
        energy_tolerance=max(2e-5,1e-3*abs(su_measured["energy_error"]))
        note(directory,"AD objective energy tolerance=$energy_tolerance")
        settings=pairing_problem(su,parameters,su_measured["energy"],directory;energy_tolerance)
        callback=function(s,record)
            note(directory,"AD $record")
            record.iteration%10==0 && save_checkpoint(joinpath(directory,"ad_latest.jld2"),s;parameters,metadata=Dict("record"=>record))
        end
        result=optimize_variational(su,parameters;chi=settings.chi,ctm_iterations=settings.iterations,
            projector_tol=settings.cutoff,maxiter=ad_maxiter,memory=8,step_size=1.0,grad_tol=1e-7,callback)
        ad=result.state
        note(directory,"AD finished status=$(result.status) E=$(result.energy) gradient_norm=$(result.gradient_norm)")
        save_checkpoint(adfile,ad;parameters,metadata=Dict("method"=>"AD","history"=>result.history,"status"=>String(result.status),
            "chi"=>settings.chi,"ctm_iterations"=>settings.iterations,"projector_tol"=>settings.cutoff,
            "initial_objective_energy_tolerance"=>energy_tolerance,"sectors"=>sectors(ad)))
        v,_=pack_state(ad)
        report=check_gradient(v,settings.problem;epsilon=1e-5)
        note(directory,"AD_FINAL_GRADIENT $report")
        write_toml(joinpath(directory,"ad_final_gradient.toml"),asdict(report))
    end
    pair_measure(directory,"ad",ad,parameters,exact)
    validate_pair_final(delta)
    note(directory,"COMPLETE delta=$delta")
end

if abspath(PROGRAM_FILE)==@__FILE__
    for delta in (isempty(ARGS) ? [2.0,1.0,0.5] : parse.(Float64,ARGS))
        run_pairing(delta)
        GC.gc()
    end
end
