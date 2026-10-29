using Test,TOML
include("../../../src/bosonic/honeycomb/star/StarRunConfig.jl")
include("../server/run_star_simple_update.jl")

@testset "Star server configuration" begin
    template=joinpath(@__DIR__,"../config/StarSimpleUpdate.toml");
    c=read_star_config(template);
    @test (c.Lx,c.Ly)==(4,4)
    @test all(s.Dmax==24 for s in c.stages)
    @test sum(s.steps*s.dt for s in c.stages)≈11.
    @test c.ctm_enabled
    @test c.output_dir==normpath(joinpath(@__DIR__,"../server_runs/star_D24"))
    pilot=read_star_config(joinpath(@__DIR__,"../config/StarSimpleUpdate_pilot.toml"));
    @test !pilot.ctm_enabled # Only the explicit lightweight pilot skips observables.
    @test pilot.output_dir==normpath(joinpath(@__DIR__,"../server_runs/star_D24_pilot"))
    mktempdir() do dir
        raw=TOML.parsefile(template);
        path=joinpath(dir,"config.toml");
        raw["run"]["output_dir"]="relative_output";
        open(io->TOML.print(io,raw),path,"w");
        @test read_star_config(path).output_dir==joinpath(dir,"relative_output")
        @test read_star_config(path;output_dir="override").output_dir==joinpath(dir,"override")
        raw["schedule"][1]["Dmax"]=32;
        open(io->TOML.print(io,raw),path,"w");
        @test first(read_star_config(path).stages).Dmax==32
        for (section,key,value) in (("run","Dmax",8),("run","checkpoint_every",0),
                ("run","max_seconds",-1.),("run","unknown_key",1),("ctm","chis",[96,48]))
            bad=deepcopy(raw);bad[section][key]=value;
            open(io->TOML.print(io,bad),path,"w");
            @test_throws ArgumentError read_star_config(path)
        end
        bad=deepcopy(raw);bad["schedule"][1]["dt"]=0.;
        open(io->TOML.print(io,bad),path,"w");
        @test_throws ArgumentError read_star_config(path)
    end
    mktempdir() do dir
        c=(output_dir=dir,);
        options=star_cli(["--auto-resume"]);
        @test !star_should_resume(options,c)
        touch(joinpath(dir,"state_latest.jld2"));
        @test star_should_resume(options,c)
        @test !star_should_resume(star_cli(String[]),c)
        @test star_should_resume(star_cli(["--resume"]),c)
        @test !star_should_resume(star_cli(["--auto-resume","--init-checkpoint","parent.jld2"]),c)
        @test_throws ErrorException star_cli(["--resume","--init-checkpoint","parent.jld2"])
    end
    mktempdir() do dir
        # No SU2_AD/SU4/honeycomb names: local src, a sibling library, and an
        # explicitly selected external source tree are all valid layouts.
        source=joinpath(dir,"my_run","src","bosonic","honeycomb","star");mkpath(source);
        for file in ("StarRunConfig.jl","StarSimpleUpdateRun.jl")
            touch(joinpath(source,file));
        end
        touch(joinpath(dirname(source),"Honeycomb_SU4.jl"));
        entry=joinpath(dir,"my_run","server");mkpath(entry);
        @test star_source_dir(entry_dir=entry,work_dir=dir,env=Dict())==source
        @test star_source_dir(entry_dir=dir,work_dir=dir,
            env=Dict("SU4_SRC_DIR"=>joinpath(dir,"my_run","src")))==source
        @test star_source_dir(entry_dir=dir,work_dir=dir,
            env=Dict("SU2_AD_ROOT"=>joinpath(dir,"my_run")))==source
        @test_throws ErrorException star_source_dir(entry_dir=entry,work_dir=dir,
            env=Dict("SU4_SRC_DIR"=>joinpath(dir,"missing")))
        config=joinpath(dir,"my_run","config");mkpath(config);
        touch(joinpath(config,"StarSimpleUpdate.toml"));
        @test star_default_config(entry_dir=entry,work_dir=dir)==joinpath(config,"StarSimpleUpdate.toml")
        flat=joinpath(dir,"flat");mkpath(flat);touch(joinpath(flat,"StarSimpleUpdate.toml"));
        @test star_default_config(entry_dir=flat,work_dir=dir)==joinpath(flat,"StarSimpleUpdate.toml")
    end
end
