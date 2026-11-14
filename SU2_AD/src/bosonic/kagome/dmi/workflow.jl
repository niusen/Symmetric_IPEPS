function save_checkpoint(path,s;J=1.,Dz=0.,h=.38,metadata=Dict{String,Any}())
    # Distinct format prevents old Heisenberg-only workflows silently treating
    # a DMI checkpoint as an unmodified Heisenberg state/measurement.
    target=abspath(path);mkpath(dirname(target));temporary=tempname(dirname(target))
    try
        JLD2.jldsave(temporary;format="kagome_dmi_v1",state=s,J,Dz,h,metadata)
        mv(temporary,target;force=true)
        if s isa SU.U1KagomeState
            export_u1_seed(splitext(target)[1]*".seed.toml",s;
                metadata=merge(Dict{String,Any}("J"=>J,"Dz"=>Dz,"h"=>h,"model"=>"kagome_dmi_v1"),metadata))
        end
    finally
        isfile(temporary) && rm(temporary)
    end
    target
end
function load_checkpoint(path)
    portable=endswith(lowercase(path),".toml") ? path : splitext(path)[1]*".seed.toml"
    if isfile(portable)
        data=TOML.parsefile(portable)
        data["format"]=="kagome_u1_seed_v1" || error("expected a full U1 seed including SU lambdas")
        return load_u1_seed(portable),data
    end
    data=JLD2.load(path)
    data["format"] in ("kagome_dmi_v1","kagome_u1_cells_v1","kagome_dense_native_v1") || error("unsupported state")
    data["state"],data
end
function run(cfg)
    BLAS.set_num_threads(cfg.threads);N.prepare_device!("cpu")
    ispath(cfg.out) && error("choose a new output directory")
    if isempty(cfg.input)
        cfg.representation==:dense || error("U1 requires an existing U1 seed with a chosen charge sector")
        s=random_dense_state(cfg.D,cfg.seed;Lx=cfg.Lx,Ly=cfg.Ly,pattern=cfg.pattern)
    else
        s,_=load_checkpoint(cfg.input)
        if cfg.representation==:dense
            s=N.to_dense(s) # retains the input sector; not an unrestricted random state
        elseif !(s isa SU.U1KagomeState)
            error("U1 mode requires a U1 checkpoint")
        end
    end
    mkpath(cfg.out);start=time()
    save_checkpoint(joinpath(cfg.out,"initial.jld2"),s;J=cfg.J,Dz=cfg.Dz,h=cfg.h)
    evolve!(s;schedule=cfg.schedule,Dmax=cfg.D,J=cfg.J,Dz=cfg.Dz,h=cfg.h,trun_tol=cfg.trun_tol,
        callback=(s,stage,step,n)->begin
            if step%cfg.progress_every==0 || step==n
                println("DMI SU stage=$stage step=$step/$n seconds=$(time()-start)");flush(stdout)
                save_checkpoint(joinpath(cfg.out,"latest.jld2"),s;J=cfg.J,Dz=cfg.Dz,h=cfg.h,
                    metadata=Dict("stage"=>stage,"step"=>step,"complete"=>false))
            end
        end)
    save_checkpoint(joinpath(cfg.out,"final.jld2"),s;J=cfg.J,Dz=cfg.Dz,h=cfg.h)
    result=measure(s;J=cfg.J,Dz=cfg.Dz,h=cfg.h,chi=cfg.chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance)
    N.atomic_toml(joinpath(cfg.out,"observables.toml"),result)
    N.write_json(joinpath(cfg.out,"observables.json"),result)
    JLD2.jldsave(joinpath(cfg.out,"observables.jld2");result)
    sites=NamedTuple[];bonds=NamedTuple[]
    for c in result["cells"]
        for i in 1:3
            push!(sites,(x=c["x"],y=c["y"],site=i,Sx=c["Sx"][i],Sy=c["Sy"][i],Sz=c["Sz"][i]))
            for orientation in ("up","down")
                push!(bonds,(x=c["x"],y=c["y"],triangle=orientation,bond=i,
                    SS=c["bonds_$(orientation)_SS"][i],kappa_CCW=c["kappa_$(orientation)_CCW"][i],
                    energy=c["bonds_$(orientation)_energy"][i]))
            end
        end
    end
    N.write_csv(joinpath(cfg.out,"site_magnetization.csv"),sites)
    N.write_csv(joinpath(cfg.out,"bond_energies.csv"),bonds)
    println("DMI FINAL E=",result["energy_per_spin"]," m=",result["m_normalized"])
    result
end
