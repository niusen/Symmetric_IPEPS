# Include after importing KagomeGFPEPSValidation in a persistent Julia session.
function validate_saved(family,D,chi;cutoff=1e-9,maxiter=60,tol=1e-9,warmstart=nothing,save_environment=false,label="",update_method="continuous_update",gauge_sweeps=0,directory_override=nothing,output_directory=nothing)
    module_=KagomeGFPEPSValidation
    directory=isnothing(directory_override) ? joinpath(KAGOME_WORKSPACE,"Kagome/fermionic/iPESS/results/gfpeps_validation/$(family)_D$D") : abspath(directory_override)
    output=isnothing(output_directory) ? directory : abspath(output_directory)
    mkpath(output)
    if family=="ipess"
        state,fields=module_.import_gaussian_ipess(joinpath(directory,"tensors.mat"))
        triangle=convert(Array,triangle_wavefunction(state,1,1,:up;weighted=false))
        write(joinpath(output,"triangle_native.bin"),triangle)
        if D<=8
            block=build_A_kagome(state,1,1)[1]
            for (i,j) in ((1,2),(1,3),(2,3))
                block=KagomeFermionicZ2._crossing(block,i,j)
            end
            write(joinpath(output,"block_native.bin"),convert(Array,block))
        end
        A=convert_iPESS_to_iPEPS(state)
        _,_,U=build_A_kagome(state,1,1)
        jldsave(joinpath(output,"state.jld2");state,fields,
            physical_basis="b=(up,down-hole); (vac,up+hole,up,hole)")
    elseif family=="ipeps"
        A,U,fields=module_.import_gaussian_ipeps(joinpath(directory,"tensors.mat"))
        jldsave(joinpath(output,"state.jld2");A,U,fields,
            physical_basis="b=(up,down-hole); sites B,A,C")
    else
        error("unknown family")
    end
    gauge_info=nothing
    if gauge_sweeps>0
        A,gauge_info=module_.balance_virtual_gauge(A;sweeps=gauge_sweeps)
        println("Exact virtual gauge conditioning: ",gauge_info);flush(stdout)
    end
    start=time()
    CTM0=if isnothing(warmstart)
        nothing
    else
        saved=load(joinpath(output,warmstart))
        haskey(saved,"CTM") ? saved["CTM"] : saved["result"].CTM
    end
    C=KagomeFermionicCTM.Legacy
    compact=isdefined(C,:compact_double_layer_enabled) && C.compact_double_layer_enabled[]
    tag="ctm_chi$(chi)_cut$(cutoff)"*(compact ? "_compact" : "")
    update_method=="continuous_update" || (tag*="_"*update_method)
    gauge_sweeps==0 || (tag*="_gauge$(gauge_sweeps)")
    isempty(label) || (tag*="_"*label)
    environment_path=joinpath(output,tag*"_environment.jld2")
    options=(;chi,maxiter,tol,projector_tol=cutoff,CTM0,update_method)
    save_environment && (options=merge(options,(;environment_path)))
    isdefined(C,:gfpeps_projector_ranks) && empty!(C.gfpeps_projector_ranks)
    result=module_.measure_gaussian_cell(A,U,fields;options...)
    report=module_.compare_reference(result,joinpath(directory,"reference_N192.mat"))
    report["elapsed_seconds"]=time()-start
    report["requested_ctm_tolerance"]=tol
    report["initial_environment"]=isnothing(warmstart) ? "PBC" : warmstart
    report["blas_threads"]=BLAS.get_num_threads()
    report["julia_version"]=string(VERSION)
    report["tensorkit_version"]=string(Base.pkgversion(TensorKit))
    report["detailed_observables"]=true
    report["double_layer_implementation"]=compact ? "compact_equivalent" : "shared_original"
    report["label"]=label
    report["update_method"]=update_method
    report["gauge_sweeps"]=gauge_sweeps
    if isdefined(C,:gfpeps_projector_ranks) && !isempty(C.gfpeps_projector_ranks)
        ranks=C.gfpeps_projector_ranks
        report["projector_min_retained_rank"]=minimum(ranks)
        report["projector_max_retained_rank"]=maximum(ranks)
        report["projector_fraction_rank_below_chi"]=count(<(chi),ranks)/length(ranks)
    end
    if !isnothing(gauge_info)
        report["gauge_identity_residual"]=gauge_info.identity_residual
        report["max_gauge_condition"]=gauge_info.max_gauge_condition
    end
    open(io->TOML.print(io,report),joinpath(output,tag*".toml"),"w")
    jldsave(joinpath(output,tag*".jld2");result)
    println(report);flush(stdout)
    GC.gc()
    report
end

function run_spin_energy(family,D,chi;cutoff=1e-10,maxiter=100,tol=1e-8,warmstart=nothing,label="",gauge_sweeps=0,update_method="continuous_update",target_mz_over_saturation=1/9,magnetization_tolerance=1e-6,directory_override=nothing,output_directory=nothing)
    directory=isnothing(directory_override) ? joinpath(KAGOME_WORKSPACE,"Kagome/fermionic/iPESS/results/gfpeps_validation/$(family)_D$D") : abspath(directory_override)
    output=isnothing(output_directory) ? directory : abspath(output_directory)
    mkpath(output)
    audit_path=joinpath(directory,"spin_sector_audit.toml")
    source_audit=isfile(audit_path) ? TOML.parsefile(audit_path) : nothing
    if !isnothing(source_audit) && source_audit["bulk_Nb_variance_per_magnetic_cell"]>1e-8
        @warn "Source Gaussian state mixes total-Sz sectors; local Gutzwiller projection does not fix the magnetization." family D
    end
    if family=="ipess"
        state,fields=import_gaussian_ipess(joinpath(directory,"tensors.mat");gutzwiller=true)
        A=convert_iPESS_to_iPEPS(state);_,_,U=build_A_kagome(state,1,1)
    elseif family=="ipeps"
        A,U,fields=import_gaussian_ipeps(joinpath(directory,"tensors.mat");gutzwiller=true)
    else
        error("unknown family")
    end
    if gauge_sweeps>0
        A,gauge_info=KagomeGFPEPSValidation.balance_virtual_gauge(A;sweeps=gauge_sweeps)
        println("Gauge: ",gauge_info);flush(stdout)
    end
    tag="heisenberg_chi$(chi)_cut$(cutoff)"*(gauge_sweeps==0 ? "" : "_gauge$(gauge_sweeps)")*(isempty(label) ? "" : "_"*label)
    update_method=="continuous_update" || (tag*="_"*update_method)
    env=if isnothing(warmstart)
        nothing
    else
        saved=load(joinpath(output,warmstart))
        haskey(saved,"CTM") ? saved["CTM"] : saved["result"].CTM
    end
    start=time()
    empty!(KagomeFermionicCTM.Legacy.gfpeps_projector_ranks)
    r=measure_heisenberg(A,U;chi,maxiter,tol,projector_tol=cutoff,CTM0=env,update_method,
        environment_path=joinpath(output,tag*"_environment.jld2"))
    report=Dict(string(k)=>getproperty(r,k) for k in (:energy_per_site,:energy_per_bond,
        :magnetization_per_site,:uniform_magnitude,:mean_local_magnitude,:mz_over_saturation,
        :max_imag,:J,:chi,:projector_tol,:iterations,:ctm_error,:converged))
    merge!(report,Dict("family"=>family,"D"=>D,"elapsed_seconds"=>time()-start,
        "requested_tolerance"=>tol,"initial_environment"=>something(warmstart,"PBC"),
        "source_fields_metadata"=>fields,"reoptimized"=>false,"model"=>"J sum_nn S_i dot S_j; h=0; no constant shift"))
    report["source_directory"]=abspath(directory)
    report["gauge_sweeps"]=gauge_sweeps
    report["update_method"]=update_method
    if !isnothing(target_mz_over_saturation)
        report["target_mz_over_saturation"]=target_mz_over_saturation
        report["magnetization_tolerance"]=magnetization_tolerance
        report["magnetization_within_target_tolerance"]=abs(r.mz_over_saturation-target_mz_over_saturation)<=magnetization_tolerance
        if !report["magnetization_within_target_tolerance"]
            @warn "Measured magnetization is outside the requested numerical tolerance; use the source-sector audit and chi convergence to distinguish source mixing from contraction error." measured=r.mz_over_saturation target=target_mz_over_saturation
        end
    end
    report["spin_sector_audit_available"]=!isnothing(source_audit)
    if !isnothing(source_audit)
        report["source_b_number_variance_per_cell"]=source_audit["bulk_Nb_variance_per_magnetic_cell"]
        report["source_sector_audit_sha256"]=source_audit["source_sha256"]
    end
    ranks=KagomeFermionicCTM.Legacy.gfpeps_projector_ranks
    if !isempty(ranks)
        report["projector_min_retained_rank"]=minimum(ranks)
        report["projector_max_retained_rank"]=maximum(ranks)
    end
    open(io->TOML.print(io,report),joinpath(output,tag*".toml"),"w")
    jldsave(joinpath(output,tag*".jld2");result=r)
    println(report);flush(stdout);GC.gc()
    report
end
