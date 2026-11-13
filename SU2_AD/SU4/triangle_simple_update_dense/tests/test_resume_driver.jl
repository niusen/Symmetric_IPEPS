# Test restart bookkeeping with small MAT fixtures; never run a D=9 contraction.
using Test, MAT
include(joinpath(@__DIR__, "../../../src/bosonic/triangle/su4_dense/Resume.jl"))
const LR = DenseSU4LongRun
const RS = DenseSU4Resume

function fixture_result(cfg, D, seed; converged=true)
    path = joinpath(LR.run_directory(cfg,D,seed), "result.mat")
    mkpath(dirname(path))
    data = Dict{String,Any}("complete"=>true,"Dmax"=>D,"seed"=>seed,"J"=>cfg.J,
        "convention"=>string(cfg.convention),"Lx"=>cfg.Lx,"Ly"=>cfg.Ly,
        "tensor_format"=>"TensorMap_ComplexSpace","environment"=>"ctmrg",
        "energy_per_site"=>-D/10,"converged"=>converged,"total_sweeps"=>2300,
        "history"=>[1.0 20.0; 2.0 40.0],"history_columns"=>["stage","elapsed_seconds"],
        "initialization_mode"=>string(cfg.initialization),"source_D"=>D-1,
        "expansion_noise"=>cfg.expansion_noise,"ctm_chi"=>last(cfg.ctm_chis),
        "ctm_max_iterations"=>cfg.ctm_max_iterations,"ctm_corner_tol"=>cfg.ctm_corner_tol,
        "ctm_chi_tol"=>cfg.ctm_chi_tol)
    matwrite(path,data)
    return RS.saved_result_row(cfg,D,seed)
end

@testset "D9 restart preserves continuation and prior results" begin
    mktempdir() do root
        cfg=LR.LongRunConfig(output_dir=root,Ds=collect(4:10),seeds=[1234,5678])
        checkpoint=joinpath(LR.run_directory(cfg,9,1234),"checkpoint.jls")
        mkpath(dirname(checkpoint)); write(checkpoint,"fixture")
        for seed in cfg.seeds
            fixture_result(cfg,8,seed)
        end
        # Complete-but-unconverged is a completed measurement and must be kept.
        fixture_result(cfg,9,5678;converged=false)
        calls=Tuple{Int,Int}[]
        runner=function(c,D,seed)
            @test c === cfg
            @test LR.signature(c,9).source_D==8
            push!(calls,(D,seed))
            fixture_result(c,D,seed)
        end
        rows=RS.resume_batch(cfg;runner)
        @test calls==[(9,1234),(10,1234),(10,5678)]
        @test length(rows)==6
        @test only(filter(r->r.D==9 && r.seed==5678,rows)).status==2
        @test all(r->r.elapsed_seconds==40.0,rows)
        summary=matread(joinpath(LR.batch_directory(cfg),"scan_summary.mat"))
        @test summary["energy_per_site"][5,1]≈-0.8
        @test summary["status"][6,2]==2
        @test summary["status"][7,1]==1
        @test length(readlines(joinpath(LR.batch_directory(cfg),"scan_summary.csv")))==7
        # Restarting the restart must not call any numerical work again.
        again=RS.resume_batch(cfg;runner=(args...)->error("must skip completed results"))
        @test length(again)==6
        # A mismatched saved measurement is rejected before work or summary mutation.
        previous=read(joinpath(LR.batch_directory(cfg),"scan_summary.csv"),String)
        changed=LR.LongRunConfig(output_dir=root,Ds=collect(4:10),seeds=[1234,5678],J=2.0)
        # Match the alternate model's directory to exercise metadata validation.
        mkpath(LR.run_directory(changed,9,1234))
        cp(checkpoint,joinpath(LR.run_directory(changed,9,1234),"checkpoint.jls"))
        cp(joinpath(LR.run_directory(cfg,8,1234),"result.mat"),
            joinpath(LR.run_directory(changed,9,1234),"result.mat"))
        @test_throws ErrorException RS.resume_batch(changed)
        @test read(joinpath(LR.batch_directory(cfg),"scan_summary.csv"),String)==previous
    end
end

@testset "Restart rejects wrong roots and stops on failure" begin
    mktempdir() do root
        cfg=LR.LongRunConfig(output_dir=root,Ds=collect(4:10),seeds=[1234])
        @test_throws ErrorException RS.resume_batch(cfg)
        @test !isfile(joinpath(LR.batch_directory(cfg),"scan_summary.csv"))
        checkpoint=joinpath(LR.run_directory(cfg,9,1234),"checkpoint.jls")
        mkpath(dirname(checkpoint)); write(checkpoint,"fixture")
        fixture_result(cfg,8,1234)
        calls=Int[]
        runner=(c,D,seed)->begin push!(calls,D); error("simulated measurement failure") end
        @test_throws ErrorException RS.resume_batch(cfg;runner)
        @test calls==[9]
        summary=matread(joinpath(LR.batch_directory(cfg),"scan_summary.mat"))
        @test summary["status"][5,1]==1
        @test summary["status"][6,1]==-1
        @test summary["status"][7,1]==0
        @test occursin("simulated measurement failure",read(joinpath(LR.batch_directory(cfg),"failures.log"),String))
    end
end
