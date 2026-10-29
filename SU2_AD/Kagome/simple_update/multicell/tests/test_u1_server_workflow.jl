using Test
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","native_measurement","u1_server_workflow.jl"))
const U1_SEED=joinpath(@__DIR__,"..","data","u1_D4_optimized_seed.toml")
const U1_REF=joinpath(@__DIR__,"..","data","dense_reference_h038.csv")
BLAS.set_num_threads(10)

function test_u1_seed_transport()
    @testset "portable U1 seed: physical state, duals, axis ordering and lambdas" begin
        s=load_u1_seed(U1_SEED)
        @test ServerU1.check_u1(s;Dmax=4)===s
        oracle=TOML.parsefile(U1_SEED)["metadata"]["finite_nine_spin_validation"]
        obs=ServerU1.u1_torus_observables(s;h=oracle["h"])
        @test abs(obs.field_energy_per_site-oracle["total_energy_per_spin"])<1e-11
        @test maximum(abs.(obs.site_Sz-oracle["site_Sz"]))<1e-11
        @test abs(obs.magnetization_normalized-1/9)<1e-12
        @test obs.variance_Sz<1e-12
        for c in s.cells,name in fieldnames(ServerBase.KagomeSUTensors)
            t=getfield(c,name);isnothing(t) && continue
            d=u1_seed_tensor_data(t)
            # Reverse charge blocks, retaining the basis order WITHIN each multiplicity.
            # Restoration must recover the same invariant TensorMap coefficients.
            A=convert(Array,t);orders=[vcat([findall(==(q),axis) for q in reverse(unique(axis))]...) for axis in d["axis_charges"]]
            A=A[orders...];d["real"]=real.(vec(A));d["imag"]=imag.(vec(A))
            d["axis_charges"]=[q[order] for (q,order) in zip(d["axis_charges"],orders)]
            @test norm(u1_seed_restore_tensor(d)-t)<1e-12
        end
        mktempdir() do folder
            file=joinpath(folder,"seed.toml");export_u1_seed(file,s)
            copy=load_u1_seed(file)
            for (a,b) in zip(s.cells,copy.cells),name in fieldnames(ServerBase.KagomeSUTensors)
                x,y=getfield(a,name),getfield(b,name)
                @test isnothing(x) ? isnothing(y) : norm(x-y)<1e-12
            end
        end
    end
end

u1_smoke_cfg(out)=(dims=[4,5],h=.38,chi=2,threads=10,device="cpu",ctm_iterations=2,
    ctm_tolerance=1e-6,schedule=[(dt=.001,tau=.002)],truncation_tolerance=1e-6,
    checkpoint_every=1,progress_every=1,initial_state=U1_SEED,dense_reference=U1_REF,out=out)

function test_u1_server_body_resume()
    @testset "U1 body checkpoint resumes without an extra half-step" begin
        mktempdir() do folder
            cfg=u1_smoke_cfg(joinpath(folder,"unused"));s=load_u1_seed(U1_SEED)
            H=ServerU1.u1_heisenberg_triangle();stage=only(cfg.schedule)
            reference=ServerU1.itebd_u1(s,H,stage.tau,stage.dt;Dmax=4)
            body=deepcopy(s);gate,half=ServerU1.Kernels.trotter_gate(H,stage.dt)
            ServerU1._sweep!(body,half,:dn;Dmax=4,trun_tol=1e-6)
            ServerU1._sweep!(body,gate,:up;Dmax=4,trun_tol=1e-6)
            ServerU1._sweep!(body,gate,:dn;Dmax=4,trun_tol=1e-6)
            m=Dict{String,Any}("plan_id"=>"body-test","Dmax"=>4,"completed_stages"=>0,
                "stage_step"=>1,"phase"=>"body","stage_seconds"=>0.0,"history"=>Dict[])
            checkpoint=joinpath(folder,"D4");mkpath(checkpoint)
            ServerU1.save_u1_checkpoint(joinpath(checkpoint,"latest.jld2"),body,H;metadata=m)
            final=u1_server_evolve_D(cfg,4,U1_SEED,checkpoint,"body-test")
            result,data=ServerU1.load_u1_checkpoint(final)
            a=vec(ServerU1.u1_torus_wavefunction(reference));b=vec(ServerU1.u1_torus_wavefunction(result))
            @test abs(dot(a,b))/(norm(a)*norm(b))>1-1e-12
            @test data["metadata"]["completed_stages"]==1
            @test data["metadata"]["phase"]=="stage_complete"
            @test data["metadata"]["history"][1]["steps"]==2
            digest=bytes2hex(sha256(read(final)))
            u1_server_evolve_D(cfg,4,U1_SEED,checkpoint,"body-test")
            @test digest==bytes2hex(sha256(read(final)))
        end
    end
end

function test_u1_server_end_to_end()
    @testset "U1 D4 -> D5: tiny CTMRG, collected observables and completed resume" begin
        mktempdir() do folder
            cfg=u1_smoke_cfg(joinpath(folder,"job"));run_u1_server(cfg)
            rows=TOML.parsefile(joinpath(cfg.out,"summary.toml"))["results"]
            @test [r["D"] for r in rows]==[4,5]
            for r in rows
                @test r["ctm_iterations"]<=2
                @test r["actual_D_max"]<=r["D"]
                @test isfinite(r["energy_per_spin"])
                @test haskey(r,"delta_E_u1_minus_dense")
                @test !haskey(r,"delta_E_su2_minus_dense")
            end
            @test rows[2]["actual_D_max"]==5
            for name in ("summary.csv","summary.json","summary.txt","site_magnetization.csv",
                         "bond_energies.csv","triangle_observables.csv","virtual_spaces.csv")
                @test isfile(joinpath(cfg.out,name))
            end
            final=joinpath(cfg.out,"D5","final.jld2");_,data=ServerU1.load_u1_checkpoint(final)
            @test data["metadata"]["initial_source_sha256"]==bytes2hex(sha256(read(joinpath(cfg.out,"D4","final.jld2"))))
            files=[joinpath(cfg.out,"D$D",tail...) for D in (4,5)
                   for tail in (("final.jld2",),("ctm","observables.toml"))]
            digests=[bytes2hex(sha256(read(f))) for f in files]
            run_u1_server(cfg)
            @test digests==[bytes2hex(sha256(read(f))) for f in files]
            @test_throws ErrorException run_u1_server(merge(cfg,(chi=3,)))
            legacy=server_collect_outputs(cfg.out;reference_file=cfg.dense_reference)
            @test all(haskey(r,"delta_E_su2_minus_dense") for r in legacy)
        end
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    test_u1_seed_transport()
    test_u1_server_body_resume()
    test_u1_server_end_to_end()
    println("U1 server checks passed; Julia ",VERSION," / TensorKit ",pkgversion(TensorKit))
end
