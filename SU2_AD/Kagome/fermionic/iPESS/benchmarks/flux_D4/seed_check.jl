include("run.jl")
function check_flux_seed(seed)
    directory=joinpath(FLUX_OUTPUT,"seed$seed"); mkpath(directory)
    p=flux_parameters(); exact=check_flux_bands()
    if isfile(joinpath(directory,"su_measurements.toml")) && isfile(joinpath(directory,"su.jld2"))
        _,saved,_=load_checkpoint(joinpath(directory,"su.jld2")); @assert saved==p
        note(directory,"REUSE completed seed=$seed")
        return
    end
    s=initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(seed))
    history=[]; previous=Inf
    for (stage,(tau,dt)) in enumerate(vcat([(8.,.1),(4.,.02),(3.,.005),(1.,.001)],fill((1.,.001),5)))
        note(directory,"SU BEGIN seed=$seed stage=$stage")
        itebd!(s,p,tau,dt,4;trun_tol=1e-12)
        row=only(pair_measure(directory,"stage$stage",s,p,exact;settings=((16,1e-8),)))
        row["ctm_converged"] || error("seed measurement not converged")
        record=(;stage,tau,dt,energy=row["energy"],energy_change=row["energy"]-previous)
        push!(history,record)
        save_checkpoint(joinpath(directory,"su_latest.jld2"),s;parameters=p,metadata=Dict("seed"=>seed,"history"=>history))
        stage>=5 && abs(record.energy_change)<2e-7 && break
        previous=row["energy"]
    end
    save_checkpoint(joinpath(directory,"su.jld2"),s;parameters=p,metadata=Dict("seed"=>seed,"history"=>history,"sectors"=>sectors(s)))
    pair_measure(directory,"su",s,p,exact;settings=((32,1e-10),))
    note(directory,"SEED COMPLETE seed=$seed")
end
if abspath(PROGRAM_FILE)==@__FILE__
    for seed in (777,999)
        check_flux_seed(seed)
        GC.gc()
    end
end
