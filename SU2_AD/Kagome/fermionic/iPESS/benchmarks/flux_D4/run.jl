isdefined(Main,:PAIR_OUTPUT) || include("../pairing_D4/common.jl")
isdefined(Main,:flux_reference) || include("reference.jl")
using JSON
const FLUX_OUTPUT=normpath(joinpath(@__DIR__,"..","..","results","flux_D4"))
mkpath(FLUX_OUTPUT)
flux_parameters()=Dict{String,Any}("t_up"=>flux_hoppings(),"t_dn"=>flux_hoppings(),"U"=>0.0,"μ"=>-sqrt(3)/2)
function flux_measure(label,state;directory=FLUX_OUTPUT,settings=((16,1e-8),(32,1e-8),(32,1e-10)))
    pair_measure(directory,label,state,flux_parameters(),flux_reference();settings)
end
function best_flux_measurement(rows)
    good=filter(r->r["ctm_converged"],rows)
    isempty(good) && error("no converged CTM measurement")
    good[argmax([(r["chi"],-r["projector_tol"]) for r in good])]
end
function check_flux_bands()
    path=joinpath(FLUX_OUTPUT,"bands.json")
    isfile(path) || error("run bands.py first")
    bands=JSON.parsefile(path)
    for r in bands["checks"]
        @assert maximum(abs.(r["chern"].-[-1,0,1]))<1e-9
        @assert r["flat_band_max_abs"]<1e-12
        @assert abs(r["gap"]-sqrt(3))<1e-12
    end
    exact=flux_reference()
    @assert abs(exact["energy_per_site"]-bands["grand_energy_per_site"])<1e-10
    @assert abs(exact["energy_per_site"]-flux_reference(mesh=300)["energy_per_site"])<1e-10
    exact
end
function run_flux_su(;seed=888,directory=FLUX_OUTPUT)
    mkpath(directory)
    exact=check_flux_bands(); p=flux_parameters()
    schedule=vcat([(8.,.1),(4.,.02),(3.,.005),(1.,.001)],fill((1.,.001),10))
    write_toml(joinpath(directory,"settings.toml"),Dict("D_max"=>4,"seed"=>seed,
        "t_up_real"=>[1.,1.,0.],"t_up_imag"=>[0.,0.,-1.],"t_dn_equals_t_up"=>true,
        "U"=>0.0,"mu"=>p["μ"],"triangle_flux"=>pi/2,"hexagon_flux_mod_2pi"=>Float64(pi),
        "reference"=>exact,"spin_degeneracy"=>2,"su_schedule_tau_dt"=>[collect(t) for t in schedule]))
    checkpoint=joinpath(directory,"su.jld2")
    if isfile(checkpoint)
        s,saved,metadata=load_checkpoint(checkpoint); @assert saved==p
        history=metadata["history"]
        for stage in length(history)+1:14
            abs(last(history).energy_change)<2e-7 && break
            previous=last(history).energy
            note(directory,"SU CONTINUE stage=$stage tau=1 dt=0.001")
            itebd!(s,p,1.,.001,4;trun_tol=1e-12)
            row=best_flux_measurement(flux_measure("su_stage$stage",s;directory,settings=((16,1e-8),)))
            push!(history,(;stage,tau=1.,dt=.001,energy=row["energy"],energy_change=row["energy"]-previous))
        end
        save_checkpoint(checkpoint,s;parameters=p,metadata=Dict("method"=>"SU","seed"=>seed,"history"=>history,"sectors"=>sectors(s)))
    else
        s=initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(seed))
        history=[]; previous=Inf
        for (stage,(tau,dt)) in enumerate(schedule)
            note(directory,"SU BEGIN stage=$stage tau=$tau dt=$dt")
            elapsed=@elapsed itebd!(s,p,tau,dt,4;trun_tol=1e-12)
            row=best_flux_measurement(flux_measure("su_stage$stage",s;directory,settings=((16,1e-8),)))
            record=(;stage,tau,dt,energy=row["energy"],energy_change=row["energy"]-previous,seconds=elapsed)
            push!(history,record); note(directory,"SU $record sectors=$(sectors(s))")
            save_checkpoint(joinpath(directory,"su_latest.jld2"),s;parameters=p,metadata=Dict("history"=>history))
            stage>=5 && abs(record.energy_change)<2e-7 && break
            previous=row["energy"]
        end
        save_checkpoint(checkpoint,s;parameters=p,metadata=Dict("method"=>"SU","history"=>history,"sectors"=>sectors(s)))
    end
    flux_measure("su",s;directory)
    note(directory,"SU COMPLETE")
    s
end

function run_flux_fu(;directory=FLUX_OUTPUT,refine_tau=0.0)
    p=flux_parameters(); su,saved,_=load_checkpoint(joinpath(directory,"su.jld2")); @assert saved==p
    file=joinpath(directory,"fu.jld2")
    if isfile(file)
        fu,saved,_=load_checkpoint(file); @assert saved==p
    else
        fu=deepcopy(su); history=[]; stages=[]; previous=Inf
        schedule=[(.5,.05),(.4,.02),(.2,.005),(.1,.005),(.1,.005),(.1,.005)]
        for (stage,(tau,dt)) in enumerate(schedule)
            counter=Ref(0)
            callback=function(s,r)
                counter[]+=1
                counter[]%6==0 && note(directory,"FU stage=$stage step=$(r.step) fidelity=$(r.fidelity) CTM_error=$(r.environment.ctm_error)")
            end
            records=full_update!(fu,p,tau,dt,4;chi=16,ctm_iterations=80,ctm_tol=1e-8,
                projector_tol=1e-8,n_sweep=12,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
            append!(history,records)
            row=best_flux_measurement(flux_measure("fu_stage$stage",fu;directory,settings=((16,1e-8),)))
            record=(;stage,tau,dt,energy=row["energy"],energy_change=row["energy"]-previous)
            push!(stages,record); note(directory,"FU_STAGE $record")
            save_checkpoint(joinpath(directory,"fu_stage$stage.jld2"),fu;parameters=p,
                metadata=Dict("history"=>history,"stages"=>stages,"sectors"=>sectors(fu)))
            stage>=4 && abs(record.energy_change)<2e-7 && break
            previous=row["energy"]
        end
        if refine_tau>0
            # End with the same dt as SU; preserve the coarser result too.
            save_checkpoint(joinpath(directory,"fu_before_fine.jld2"),fu;parameters=p,metadata=Dict("history"=>history,"stages"=>stages,"sectors"=>sectors(fu)))
            counter=Ref(0)
            callback=function(s,r)
                counter[]+=1
                counter[]%6==0 && note(directory,"FU fine step=$(r.step) fidelity=$(r.fidelity)")
            end
            records=full_update!(fu,p,refine_tau,.001,4;chi=16,ctm_iterations=80,ctm_tol=1e-8,
                projector_tol=1e-8,n_sweep=12,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
            append!(history,records)
            row=best_flux_measurement(flux_measure("fu_fine",fu;directory,settings=((16,1e-8),)))
            record=(;stage=length(stages)+1,tau=refine_tau,dt=.001,energy=row["energy"],energy_change=row["energy"]-last(stages).energy)
            push!(stages,record); note(directory,"FU_FINE $record")
        end
        save_checkpoint(file,fu;parameters=p,metadata=Dict("method"=>"FU","history"=>history,"stages"=>stages,
            "chi"=>16,"n_sweep"=>12,"projector_tol"=>1e-8,"sectors"=>sectors(fu)))
    end
    flux_measure("fu",fu;directory)
    note(directory,"FU COMPLETE")
    fu
end

function flux_ad_settings(su,p;directory=FLUX_OUTPUT)
    v,_=pack_state(su); attempts=[]
    rows=TOML.parsefile(joinpath(directory,"su_measurements.toml"))["measurements"]
    measured=best_flux_measurement(rows)
    energy_tolerance=max(2e-5,.001*abs(measured["energy_error"]))
    for chi in (16,24,32),cutoff in (1e-6,1e-7,1e-8,1e-5,1e-4)
        e=[variational_energy(v,VariationalProblem(su,p;chi,ctm_iterations=n,projector_tol=cutoff)) for n in (4,8,12)]
        n=abs(e[2]-e[1])<1e-7 && abs(e[3]-e[2])<1e-7 ? 4 : (abs(e[3]-e[2])<1e-7 ? 8 : 12)
        problem=VariationalProblem(su,p;chi,ctm_iterations=n,projector_tol=cutoff)
        report=check_gradient(v,problem;epsilon=1e-5)
        row=merge(asdict(report),Dict("chi"=>chi,"ctm_iterations"=>n,"projector_tol"=>cutoff,
            "sweep_energies"=>e,"energy_tolerance"=>energy_tolerance,
            "difference_from_measured"=>report.energy-measured["energy"]))
        push!(attempts,row); note(directory,"AD_CHECK $row")
        write_toml(joinpath(directory,"ad_initial_checks.toml"),Dict("attempts"=>attempts))
        if (report.relative_error<1e-3 || report.absolute_error<1e-7) && abs(row["difference_from_measured"])<energy_tolerance
            return (;chi,n,cutoff,problem)
        end
    end
    error("no AD setting passed gradient and energy checks")
end

function run_flux_ad(;maxiter=150,directory=FLUX_OUTPUT)
    p=flux_parameters(); su,saved,_=load_checkpoint(joinpath(directory,"su.jld2")); @assert saved==p
    file=joinpath(directory,"ad.jld2")
    if isfile(file)
        ad,saved,_=load_checkpoint(file); @assert saved==p
    else
        settings=flux_ad_settings(su,p;directory)
        callback=function(s,r)
            note(directory,"AD $r")
            r.iteration%10==0 && save_checkpoint(joinpath(directory,"ad_latest.jld2"),s;parameters=p,metadata=Dict("record"=>r))
            r.iteration in (50,100) && save_checkpoint(joinpath(directory,"ad_iter$(r.iteration).jld2"),s;parameters=p,metadata=Dict("record"=>r))
        end
        result=optimize_variational(su,p;chi=settings.chi,ctm_iterations=settings.n,projector_tol=settings.cutoff,
            maxiter,memory=8,step_size=1.0,grad_tol=1e-7,callback)
        ad=result.state
        save_checkpoint(file,ad;parameters=p,metadata=Dict("method"=>"AD","history"=>result.history,"status"=>String(result.status),
            "chi"=>settings.chi,"ctm_iterations"=>settings.n,"projector_tol"=>settings.cutoff,"sectors"=>sectors(ad)))
        note(directory,"AD FINISHED status=$(result.status) E=$(result.energy) gradient_norm=$(result.gradient_norm)")
        v,_=pack_state(ad); report=check_gradient(v,settings.problem;epsilon=1e-5)
        write_toml(joinpath(directory,"ad_final_gradient.toml"),asdict(report)); note(directory,"AD_FINAL_GRADIENT $report")
    end
    flux_measure("ad",ad;directory)
    note(directory,"AD COMPLETE")
    ad
end

if abspath(PROGRAM_FILE)==@__FILE__
    seed=isempty(ARGS) ? 888 : parse(Int,only(ARGS))
    directory=seed==888 ? FLUX_OUTPUT : joinpath(FLUX_OUTPUT,"seed$seed")
    run_flux_su(;seed,directory)
    run_flux_fu(;directory,refine_tau=seed==999 ? .05 : 0.0)
    run_flux_ad(;directory)
end
