# Small CPU check only: D=2, two tiny chi values, no production SU run.
using Test, TOML, SHA, JLD2
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","charged_su2_convergence_driver.jl"))
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","native_measurement","chi_scan_workflow.jl"))
const Scan=KagomeChiScan
@testset "fixed-state chi scan: measurements, restart, immutable source" begin
    mktempdir() do root
        source=joinpath(root,"fixed_D2.jld2")
        s=KagomeU1Cells.to_u1(convergence_initial_state(2,:integer_singlet,1234))
        KagomeU1Cells.save_u1_checkpoint(source,s,KagomeU1Cells.u1_heisenberg_triangle())
        original=bytes2hex(sha256(read(source)))
        cfg=(sources=[(D=2,checkpoint=source)],chis=[2,4],h=.38,threads=2,
            device="cpu",ctm_iterations=4,ctm_tolerance=1e-8,out=joinpath(root,"scan"))
        @test_throws ErrorException Scan.chi_scan_plan(merge(cfg,(chis=[4,2],)))
        @test_throws ErrorException Scan.chi_scan_plan(merge(cfg,(sources=[(D=4,checkpoint=source)],)))
        @test_throws ErrorException Scan.chi_scan_plan(merge(cfg,(sources=[(D=2,checkpoint=source*".missing")],)))
        @test !ispath(cfg.out)
        @test Scan.run_chi_scan(cfg)==abspath(cfg.out)
        @test bytes2hex(sha256(read(source)))==original
        rows=TOML.parsefile(joinpath(cfg.out,"summary.toml"))["results"]
        @test length(rows)==2
        @test getindex.(rows,"chi")==[2,4]
        @test all(r->abs(r["energy_per_spin"]+.3544444444444445)<1e-10,rows)
        @test all(r->r["ctm_iterations"]<=4,rows)
        @test rows[1]["previous_chi"]==""
        @test rows[2]["previous_chi"]==2
        @test abs(rows[2]["max_bond_delta_vs_previous_chi"])<1e-10
        @test abs(rows[2]["delta_exchange_vs_previous_chi"])<1e-10
        @test length(readlines(joinpath(cfg.out,"bond_energies.csv")))==1+2*18
        @test length(readlines(joinpath(cfg.out,"site_magnetization.csv")))==1+2*9
        @test length(readlines(joinpath(cfg.out,"triangle_observables.csv")))==1+2*6
        point=joinpath(cfg.out,"D2","chi4","observables.toml")
        before=bytes2hex(sha256(read(point)));modified=mtime(point)
        Scan.run_chi_scan(cfg)
        @test bytes2hex(sha256(read(point)))==before
        @test mtime(point)==modified
        @test_throws ErrorException Scan.run_chi_scan(merge(cfg,(ctm_tolerance=1e-7,)))
        @test_throws ErrorException Scan.run_chi_scan(merge(cfg,(out=root,)))
        @test bytes2hex(sha256(read(source)))==original
    end
end
