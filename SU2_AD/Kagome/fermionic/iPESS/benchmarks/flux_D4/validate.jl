isdefined(Main,:FLUX_OUTPUT) || include("run.jl")
function validate_flux(;directory=FLUX_OUTPUT)
    checks=[]
    for method in ("su","fu","ad")
        s,p,_=load_checkpoint(joinpath(directory,"$method.jld2"))
        saved=load(joinpath(directory,"$(method)_chi32_tol1.0e-10_ob.jld2"))
        original=saved["observables"]
        reversed=measure_ctmrg(reverse_triangles(s),p;chi=32,maxiter=80,tol=1e-8,trun_tol=1e-10,verbose=false)
        row=Dict("method"=>method,"check"=>"opposite_triangle_blocking",
            "energy_difference"=>reversed.observables.energy_per_site-original.energy_per_site,
            "density_difference"=>reversed.observables.density-original.density,
            "original_converged"=>saved["record"]["ctm_converged"],"check_converged"=>reversed.converged)
        push!(checks,row); note(directory,"VALIDATE $row")
        if method=="ad"
            angles=(.31,-.27,.83); gauged=deepcopy(s); gp=copy(p)
            for site in 1:3
                P=space(s.B[1,1,site],3)
                U=TensorMap(diagm(cis.(angles[site].*[0,2,1,1])),P,P)
                gauged.B[1,1,site]=KagomeFermionicZ2._leg_map(s.B[1,1,site],U,3)
            end
            phases=Tuple(cis(angles[i]-angles[j]) for (i,j) in ((1,2),(2,3),(3,1)))
            for key in ("t_up","t_dn"); gp[key]=p[key].*phases; end
            result=measure_ctmrg(gauged,gp;chi=32,maxiter=80,tol=1e-8,trun_tol=1e-10,verbose=false)
            row=Dict("method"=>method,"check"=>"onsite_gauge_covariance",
                "energy_difference"=>result.observables.energy_per_site-original.energy_per_site,
                "density_difference"=>result.observables.density-original.density,
                "original_converged"=>saved["record"]["ctm_converged"],"check_converged"=>result.converged)
            push!(checks,row); note(directory,"VALIDATE $row")
        end
    end
    write_toml(joinpath(directory,"validation.toml"),Dict("checks"=>checks))
    for iteration in (50,100)
        file=joinpath(directory,"ad_iter$iteration.jld2")
        if isfile(file) && !isfile(joinpath(directory,"ad_iter$(iteration)_measurements.toml"))
            s,_,_=load_checkpoint(file)
            flux_measure("ad_iter$iteration",s;directory,settings=((32,1e-10),))
        end
    end
    _,_,metadata=load_checkpoint(joinpath(directory,"fu.jld2"))
    stages=metadata["stages"]
    best=stages[argmin([s.energy for s in stages])]
    file=joinpath(directory,"fu_stage$(best.stage).jld2")
    if isfile(file)
        state,_,_=load_checkpoint(file)
        flux_measure("fu_best_stage",state;directory,settings=((32,1e-10),))
        write_toml(joinpath(directory,"fu_best_stage_selection.toml"),asdict(best))
    end
    checks
end
