using Test
plan_file=get(ENV,"KAGOME_VARIANCE_PLAN",normpath(joinpath(@__DIR__,"..","..","..","..","..",
    "src","bosonic","kagome","native_measurement","energy_variance","variance_batch_plan.jl")))
include(plan_file)
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
        @test settings["extra_bottom_warmup"]===false
        legacy=P.case_settings(merge(cfg,(extra_bottom_warmup=true,)),first(jobs),hash,"fixture code")
        @test legacy["extra_bottom_warmup"]===true
        @test_throws ErrorException P.variance_jobs(merge(cfg,(extra_bottom_warmup="false",)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[24],)))
        @test length(P.variance_jobs(merge(cfg,(sizes=[16],))))==36
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[3.5],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(chis=[40,40],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(dims=[3],)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(checkpoints=Dict(4=>joinpath(root,"missing")),)))
        @test_throws ErrorException P.variance_jobs(merge(cfg,(sizes=[21,15,18],)))
        # Current production task: one fixed chi, ascending D, five nested sums.
        current=merge(cfg,(chis=[40],sizes=[3,6,8,12,15],contraction_size=21,
            sweeps=2,extra_bottom_warmup=false))
        planned=P.variance_jobs(current)
        @test length(planned)==12
        @test [j.D for j in planned]==collect(4:15)
        @test all(j.chi==40 && j.L==21 for j in planned)
        @test length(planned)*length(current.sizes)==60
        @test P.case_settings(current,first(planned),hash,"fixture")["window_sizes"]==[3,6,8,12,15]
        @test_throws ErrorException P.variance_jobs(merge(current,(contraction_size=20,)))
        @test_throws ErrorException P.variance_jobs(merge(current,(contraction_size=18,)))
    end
end

# Test the actual pure window predicate without loading TensorKit or contracting
# any tensors. This catches the asymmetric endpoints of an even-width window.
workflow=joinpath(dirname(plan_file),"variance_workflow.jl")
predicate=only(filter(line->startswith(line,"variance_window_contains("),readlines(workflow)))
Base.include_string(@__MODULE__,predicate,workflow)
@testset "reported windows, including L8, use exactly L squared cell origins" begin
    for w in (3,6,8,12,15)
        selected=[(x,y) for x in -10:10 for y in -10:10 if variance_window_contains(x,y,w)]
        @test length(selected)==w^2
        @test minimum(first.(selected))==-div(w-1,2)
        @test maximum(first.(selected))==div(w,2)
    end
    @test variance_window_contains(-3,4,8)
    @test !variance_window_contains(-4,0,8)
    @test !variance_window_contains(0,5,8)
end

module ServerTaskSettings
const KAGOME_VARIANCE_SETTINGS_ONLY=true
include(joinpath(@__DIR__,"..","..","run_energy_variance_server.jl"))
end
@testset "actual server settings: fixed chi40, ascending D, five windows" begin
    c=ServerTaskSettings.variance_settings
    jobs=P.variance_jobs(c;check_inputs=false)
    @test c.dims==collect(4:15)
    @test c.chis==[40]
    @test c.sizes==[3,6,8,12,15]
    @test c.sweeps==2 && c.extra_bottom_warmup===false
    @test c.threads==10 && c.ctm_iterations==50
    @test c.contraction_size==21
    @test length(jobs)==12 && length(jobs)*length(c.sizes)==60
    @test [j.D for j in jobs]==collect(4:15)
end
