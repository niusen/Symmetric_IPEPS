# SMALL D4 fixture, chi=2,3. One Lmax=12 contraction extracts L=3,6.
using Test,TensorKit,LinearAlgebra,TOML,SHA
native=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..","..",
    "src","bosonic","kagome","native_measurement")))
include(get(ENV,"KAGOME_VARIANCE_MODULE",joinpath(native,"energy_variance","KagomeEnergyVariance.jl")))
const V=KagomeEnergyVariance;const N=V.N
const ServerU1=N.KagomeU1Cells;const ServerBase=N.BS
include(joinpath(native,"u1_seed_io.jl"))
root=normpath(joinpath(native,"..","..","..",".."))
seed=load_u1_seed(joinpath(root,"Kagome","simple_update","multicell","data","u1_D4_optimized_seed.toml"))
BLAS.set_num_threads(1)
@testset "small batch: shared CTM, all outputs, restart and changed-settings protection" begin
    mktempdir() do tmp
        input=joinpath(tmp,"D4.jld2")
        N.KagomeU1Cells.save_u1_checkpoint(input,seed,N.KagomeU1Cells.u1_heisenberg_triangle())
        hash=V.Plan.file_sha256(input)
        cfg=(dims=[4],checkpoints=Dict(4=>input),chis=[2,3],sizes=[3,6],contraction_size=12,
            out=joinpath(tmp,"batch"),threads=1,ctm_iterations=2,ctm_tolerance=1.,sweeps=1,source_shift=1.)
        rows=V.run_variance_batch(cfg;startup_seconds=.25)
        @test length(rows)==2
        @test V.Plan.file_sha256(input)==hash
        @test all(r["result"]["anchor_count"]==18 && r["result"]["full_anchor_sum"] for r in rows)
        @test all(r["result"]["timing"]["norm_environment_reused"] for r in rows)
        @test all(r["run_timing"]["case_wall_seconds"]>0 for r in rows)
        @test TOML.parsefile(joinpath(cfg.out,"batch_completed.toml"))["completed_cases"]==2
        @test countlines(joinpath(cfg.out,"summary.csv"))==5
        @test countlines(joinpath(cfg.out,"variance_by_L_all.csv"))==5
        @test countlines(joinpath(cfg.out,"timing_by_anchor_all.csv"))==37
        for chi in cfg.chis
            matched=filter(r->r["settings"]["chi"]==chi,rows)
            @test length(matched)==1 && length(matched[1]["result"]["cumulative"])==2
            @test isfile(joinpath(cfg.out,"D4","chi$chi","norm_ctm_timing.toml"))
        end
        # Cached and fresh norm CTMRG must give the same connected correlators.
        f=N.build_factors(seed)
        fresh=V.exchange_variance(f;L=12,radii=[1],chi=2,ctm_iterations=2,
            ctm_tolerance=1.,sweeps=1,verbose=false)
        stored=TOML.parsefile(joinpath(cfg.out,"D4","chi2","variance.toml"))
        @test fresh["exchange_energy_per_spin"]≈stored["exchange_energy_per_spin"] atol=1e-12
        @test length(fresh["connected_terms"])==972
        @test fresh["cumulative"][1]["variance_per_spin"]≈stored["cumulative"][1]["variance_per_spin"] atol=1e-12
        @test [c["window_size"] for c in stored["cumulative"]]==[3,6]
        @test length(stored["connected_terms"])==18*6^2*6
        for c in stored["cumulative"]
            w=c["window_size"]
            selected=filter(r->V.variance_window_contains(r["dx"],r["dy"],w),stored["connected_terms"])
            @test length(selected)==18*w^2*6
            @test sum(r["connected_real"] for r in selected)/9≈c["variance_per_spin"] atol=1e-12
        end
        wrong=(f=f,environment=nothing,ctm=nothing,chi=99,iterations=2,tolerance=1.)
        @test_throws ErrorException V.exchange_variance(f;L=6,radii=[0],chi=2,
            ctm_iterations=2,ctm_tolerance=1.,norm_environment=wrong,verbose=false)
        # Finished cases must not be measured again or rewritten on restart.
        jobs=V.Plan.variance_jobs(cfg)
        files=[joinpath(j.out,"variance.json") for j in jobs]
        hashes=V.Plan.file_sha256.(files)
        again=V.run_variance_batch(cfg)
        @test length(again)==2
        @test V.Plan.file_sha256.(files)==hashes
        @test countlines(joinpath(cfg.out,"summary.csv"))==5
        @test_throws ErrorException V.run_variance_batch(merge(cfg,(sweeps=2,)))
        # An unfinished case is rerun; the other completed case survives.
        rm(joinpath(first(jobs).out,"completed.toml"))
        partial=V.run_variance_batch(cfg)
        @test length(partial)==2
        @test V.Plan.file_sha256.(files[2:end])==hashes[2:end]
        @test countlines(joinpath(cfg.out,"summary.csv"))==5
        @test V.Plan.file_sha256(input)==hash
    end
end
println("SMALL VARIANCE BATCH PASSED; no D>4, no chi40/60/80 calculation was run")
