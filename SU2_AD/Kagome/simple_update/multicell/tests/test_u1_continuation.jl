using Test
# Overrides are for testing staged files before installing them in the repo.
root=get(ENV,"KAGOME_TEST_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..")))
include(joinpath(root,"src","bosonic","kagome","native_measurement","u1_server_workflow.jl"))
include(get(ENV,"KAGOME_TEST_CONTINUATION_FILE",
    joinpath(root,"src","bosonic","kagome","native_measurement","u1_continuation_workflow.jl")))
const CELL_ROOT=joinpath(root,"Kagome","simple_update","multicell")
const KAGOME_SETTINGS_ONLY=true
include(get(ENV,"KAGOME_TEST_ENTRY_FILE",joinpath(CELL_ROOT,"run_u1_D10_D15_server.jl")))
BLAS.set_num_threads(10)

@testset "saved D10 source and unchanged production settings; no D10 evolution" begin
    cfg=u1_continuation_settings
    @test cfg.dims==collect(11:15)
    @test cfg.source_D==10 && cfg.h==.38 && cfg.chi==40 && cfg.device=="cpu"
    @test cfg.ctm_iterations==50 && cfg.ctm_tolerance==1e-6
    @test sum(round(Int,s.tau/s.dt) for s in cfg.schedule)==9000
    source=joinpath(CELL_ROOT,"results","u1_from_D4_D4_D10_h038_chi40","D10","final.jld2")
    before=bytes2hex(sha256(read(source)))
    mktempdir() do tmp
        plan=u1_continuation_plan(merge(cfg,(initial_state=source,out=joinpath(tmp,"run"))))
        @test plan["initial_sha256"]==before==cfg.initial_sha256
        @test plan["source_D"]==10
        @test !isdir(joinpath(tmp,"run")) # plan is read-only
    end
    @test bytes2hex(sha256(read(source)))==before
end

@testset "tiny saved D4 -> D5 -> D6; full observables, lineage and restart" begin
    mktempdir() do tmp
        seed=load_u1_seed(joinpath(CELL_ROOT,"data","u1_D4_optimized_seed.toml"))
        source=joinpath(tmp,"source_D4.jld2")
        ServerU1.save_u1_checkpoint(source,seed,ServerU1.u1_heisenberg_triangle();
            metadata=Dict("Dmax"=>4,"h"=>.38))
        source_hash=bytes2hex(sha256(read(source)))
        cfg=merge(u1_continuation_settings,(source_D=4,dims=[5,6],initial_state=source,
            initial_sha256=source_hash,out=joinpath(tmp,"run"),chi=2,ctm_iterations=2,
            schedule=[(dt=.001,tau=.002)],checkpoint_every=1,progress_every=1))
        run_u1_continuation(cfg)
        @test bytes2hex(sha256(read(source)))==source_hash
        @test !isdir(joinpath(cfg.out,"D4"))
        rows=TOML.parsefile(joinpath(cfg.out,"summary.toml"))["results"]
        @test [r["D"] for r in rows]==[5,6]
        @test all(isfinite(r["energy_per_spin"]) for r in rows)
        for D in cfg.dims
            input=D==5 ? source : joinpath(cfg.out,"D5","final.jld2")
            initial,data=ServerU1.load_u1_checkpoint(joinpath(cfg.out,"D$D","initial.jld2"))
            parent=load_u1_seed(input)
            @test data["metadata"]["initial_source_sha256"]==bytes2hex(sha256(read(input)))
            for (a,b) in zip(initial.cells,parent.cells), name in fieldnames(ServerBase.KagomeSUTensors)
                x,y=getfield(a,name),getfield(b,name)
                @test isnothing(x) ? isnothing(y) : norm(x-y)<1e-12
            end
            @test TOML.parsefile(joinpath(cfg.out,"D$D","lineage.toml"))["parent_D"]==D-1
        end
        files=[joinpath(cfg.out,"D$D",tail...) for D in cfg.dims
            for tail in (("final.jld2",),("ctm","observables.toml"))]
        hashes=[bytes2hex(sha256(read(p))) for p in files]
        run_u1_continuation(cfg)
        @test [bytes2hex(sha256(read(p))) for p in files]==hashes
        @test_throws ErrorException run_u1_continuation(merge(cfg,(chi=3,)))
        @test_throws ErrorException u1_continuation_plan(merge(cfg,(initial_sha256=repeat("0",64),)))
        @test_throws ErrorException u1_continuation_plan(merge(cfg,(dims=[6],)))
        @test_throws ErrorException u1_continuation_plan(merge(cfg,(out=tmp,)))
        # The same physical parent reserialized with different metadata is
        # still a changed source: a checkpoint must not silently mix it.
        ServerU1.save_u1_checkpoint(source,seed,ServerU1.u1_heisenberg_triangle();
            metadata=Dict("Dmax"=>4,"h"=>.38,"changed"=>true))
        @test_throws ErrorException u1_continuation_check_parent(joinpath(cfg.out,"D5"),source,
            bytes2hex(sha256(read(joinpath(cfg.out,"configuration.toml")))),5)
    end
end
println("U1 continuation checks passed; no D11-D15 numerical run started.")
