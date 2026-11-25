isdefined(Main,:PAIR_OUTPUT) || include("common.jl")

function validate_pair_final(delta)
    directory=joinpath(PAIR_OUTPUT,pair_name(delta))
    records=[]
    for method in ("su","ad")
        state,parameters,_=load_checkpoint(joinpath(directory,"$method.jld2"))
        saved=load(joinpath(directory,"$(method)_chi32_tol1.0e-10_ob.jld2"))
        original=saved["observables"]
        mirrored=measure_ctmrg(reverse_triangles(state),parameters;chi=32,maxiter=80,tol=1e-8,trun_tol=1e-10,verbose=false)
        row=Dict("method"=>method,"energy_difference"=>mirrored.observables.energy_per_site-original.energy_per_site,
            "pair_difference_abs"=>abs(mirrored.observables.pair_amplitude-original.pair_amplitude),
            "density_difference"=>mirrored.observables.density-original.density,
            "original_converged"=>saved["record"]["ctm_converged"],"reflected_converged"=>mirrored.converged,
            "reflected_ctm_error"=>mirrored.error,"chi"=>32,"projector_tol"=>1e-10)
        push!(records,row)
        note(directory,"REFLECTION $row")
    end
    write_toml(joinpath(directory,"reflection_validation.toml"),Dict("checks"=>records))
    records
end
