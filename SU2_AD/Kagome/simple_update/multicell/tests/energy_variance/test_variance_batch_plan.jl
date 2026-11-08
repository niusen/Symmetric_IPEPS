using Test
include(get(ENV,"KAGOME_VARIANCE_PLAN",normpath(joinpath(@__DIR__,"..","..","..","..","..",
    "src","bosonic","kagome","native_measurement","energy_variance","variance_batch_plan.jl"))))
const P=KagomeVariancePlan
@testset "variance batch plan, no numerical calculations" begin
    mktempdir() do root
        file=joinpath(root,"input.jld2");write(file,"preflight fixture only")
        cfg=(dims=collect(4:15),checkpoints=Dict(D=>file for D in 4:15),chis=[40,60,80],
            sizes=[15,18,21],contraction_size=27,out=joinpath(root,"results"),threads=10,
            ctm_iterations=50,ctm_tolerance=1e-8,sweeps=2,source_shift=1.)
        jobs=P.variance_jobs(cfg)
        @test length(jobs)==36
        @test length(unique(j.out for j in jobs))==36
        @test all(count(j->j.D==D,jobs)==3 for D in 4:15)
        @test all(count(j->j.D==D && j.chi==chi,jobs)==1 for D in 4:15 for chi in (40,60,80))
        @test length(jobs)*length(cfg.sizes)==108
        hash=P.file_sha256(file)
        @test length(hash)==64
        settings=P.case_settings(cfg,first(jobs),hash,"fixture code")
        @test settings["D"]==4 && settings["chi"]==40 && settings["L"]==27
        @test settings["window_sizes"]==[15,18,21]
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[24],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[16],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(chis=[40,40],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(dims=[3],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(checkpoints=Dict(4=>joinpath(root,"missing")),)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[21,15,18],)))
    end
end
