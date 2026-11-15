using Test, TensorKit, LinearAlgebra, TOML, SHA
include(joinpath(@__DIR__,"..","..","..","..","..","src","bosonic","kagome","dmi","plateau_continuation.jl"))
const C=KagomeDMIPlateauContinuation
const K=C.K
println("Loaded native Julia DMI modules; preparing the tiny product fixture.");flush(stdout)

mktempdir() do tmp
    # Exact nine-spin M=1/9 PRODUCT fixture, not a ground-state/platform result.
    # Tests never load the server's optimized seed or run a production schedule.
    seed=K.product_120(;Lx=3,Ly=3,pattern=:sqrt3xsqrt3)
    for (r,c) in enumerate(seed.cells),(i,name) in enumerate(C.BS.B_FIELDS)
        t=getfield(c,name);a=zeros(ComplexF64,1,1,2)
        a[1,1,3(r-1)+i<=5 ? 1 : 2]=1
        setfield!(c,name,TensorMap(a,codomain(t),domain(t)))
    end
    statefile=joinpath(tmp,"seed.jld2");obsfile=joinpath(tmp,"seed.toml")
    @test C.check_dense(seed;Dmax=1)==1
    K.save_checkpoint(statefile,seed;J=1.,Dz=.02,h=-.32)
    restored,_=K.load_checkpoint(statefile)
    @test norm(restored.cells[1].B_a-seed.cells[1].B_a)==0
    println("Dense fixture checkpoint roundtrip passed; measuring D=1 only.");flush(stdout)
    obs=K.measure(seed;J=1.,Dz=.02,h=-.32,chi=2,iterations=8,verbose=false)
    merge!(obs,Dict("D"=>1,"h_physical"=>.32,"m_normalized"=>2obs["mean_Sz"],
        "valid_measurement"=>obs["observables_hermiticity_ok"],"trial"=>1,"seed"=>1))
    @test obs["m_normalized"]≈1/9 atol=1e-12
    @test obs["measurement_reliable"]
    C.N.atomic_toml(obsfile,obs)
    cfg=(mode=:D_chain,dims=[1,2],Dz_values=[.02],fields=[.30,.38],
        input=statefile,seed_observables=obsfile,seed_D=1,seed_Dz=.02,seed_h=.32,
        seed_magnetization=1/9,seed_m_tolerance=1e-5,threads=2,trun_tol=1e-8,
        chi=2,ctm_max=2,ctm_tol=1e-6,schedule=[(dt=.01,tau=.01)],out=joinpath(tmp,"Dchain"))

    @testset "Plan inheritance and signed Dz" begin
        production=merge(cfg,(seed_D=2,dims=collect(2:10),fields=[.28,.32,.36,.38,.4,.44,.48]))
        nodes=C.plan(production)
        @test length(nodes)==63
        @test count(n->n.parent==0,nodes)==7
        @test all(n.parent==0 || (nodes[n.parent].D==n.D-1 && nodes[n.parent].h==n.h) for n in nodes)
        dzcfg=merge(production,(mode=:Dz_chain,dims=[2],Dz_values=[-.1,-.075,-.05,-.03,-.02,-.01,0.,.01,.02,.03,.05,.075,.1]))
        dzplan=C.plan(dzcfg)
        @test length(dzplan)==91
        @test all(n.parent==0 || dzplan[n.parent].h==n.h for n in dzplan)
        for n in dzplan
            n.Dz==-.02 || continue
            @test dzplan[n.parent].Dz==-.01
        end
        Hplus=convert(Array,K.triangle_hamiltonian(;J=1.,Dz=.1,h=-.38,representation=:dense))
        Hminus=convert(Array,K.triangle_hamiltonian(;J=1.,Dz=-.1,h=-.38,representation=:dense))
        @test norm(Hminus-conj.(Hplus))<1e-12
        @test_throws ErrorException C.plan(merge(production,(dims=[2,4],)))
        @test_throws ErrorException C.run(merge(cfg,(input=joinpath(tmp,"missing.jld2"),out=joinpath(tmp,"must_not_exist"))))
        @test !ispath(joinpath(tmp,"must_not_exist"))
    end

    function check_outputs(job)
        for n in C.plan(job)
            folder=joinpath(job.out,n.id)
            @test all(isfile(joinpath(folder,name)) for name in ("initial.jld2","stage.jld2","final.jld2",
                "continuation_result.toml","site_magnetization.csv","bond_energies.csv","triangle_observables.csv"))
            meta=TOML.parsefile(joinpath(folder,"initialization.toml"))
            parent,_=K.load_checkpoint(meta["source"])
            initial,_=K.load_checkpoint(joinpath(folder,"initial.jld2"))
            @test all(norm(getfield(a,name)-getfield(b,name))<1e-14 for (a,b) in zip(parent.cells,initial.cells) for name in (:T_u,:T_d,C.BS.B_FIELDS...))
            r=TOML.parsefile(joinpath(folder,"continuation_result.toml"))
            @test r["initial_noise"]==0 && !r["h_continuation"]
            @test r["actual_max_bond_dimension"]<=n.D
            lines=readlines(joinpath(folder,"triangle_observables.csv"))
            header=replace.(split(lines[1],','),"\""=>"")
            column=findfirst(==("energy"),header)
            energy=sum(parse(Float64,replace(split(line,',')[column],"\""=>"")) for line in lines[2:end])
            @test energy/(3length(r["cells"]))≈r["energy_per_spin"] atol=1e-10
        end
        @test isfile(joinpath(job.out,"COMPLETED.txt")) && isfile(joinpath(job.out,"progress.csv"))
    end

    @testset "Small D continuation, original ket and resume" begin
        C.run(cfg);check_outputs(cfg)
        bigger=only(filter(n->n.D==2 && n.h==.38,C.plan(cfg)))
        result=TOML.parsefile(joinpath(cfg.out,bigger.id,"continuation_result.toml"))
        @test result["initial_max_bond_dimension"]==1
        @test result["actual_max_bond_dimension"]==2
        finals=[joinpath(cfg.out,n.id,"final.jld2") for n in C.plan(cfg)]
        hashes=sha256.(read.(finals));C.run(cfg)
        @test sha256.(read.(finals))==hashes
        @test_throws ErrorException C.run(merge(cfg,(chi=3,)))
    end
    @testset "Small signed-Dz continuation" begin
        job=merge(cfg,(mode=:Dz_chain,dims=[1],Dz_values=[-.02,0.,.02,.05],fields=[.38],out=joinpath(tmp,"Dzchain")))
        C.run(job);check_outputs(job)
        nodes=C.plan(job)
        negative=only(filter(n->n.Dz==-.02,nodes))
        @test nodes[negative.parent].Dz==0.
        @test all(n.parent==0 || nodes[n.parent].h==n.h for n in nodes)
    end
    println("ALL SMALL PLATEAU CONTINUATION CHECKS PASSED; only D=1/2 one-step fixtures were evolved.")
end
