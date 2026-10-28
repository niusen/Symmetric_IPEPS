using Test, TensorKit, LinearAlgebra, Random, JLD2
BLAS.set_num_threads(10)
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","workflows","run.jl"))
const InitSU=KagomeU1Cells.SU2
const InitBS=KagomeU1Cells.BS

@testset "Charged initialization and sqrt3-to-rectangular evolution" begin
    mktempdir() do folder
        source=InitSU.initialize_charged(;rng=MersenneTwister(108))
        file=joinpath(folder,"source.jld2")
        InitSU.save_charged_checkpoint(file,source,InitBS.heisenberg_triangle())
        small=main(["--symmetry","u1","--cell","sqrt3xsqrt3","--Dmax","3",
            "--init",file,"--smoke","--out",joinpath(folder,"sqrt3")])
        @test abs(u1_torus_observables(small).magnetization_normalized-1/9)<3e-12
        rect=main(["--symmetry","u1","--cell","3x3","--Dmax","3",
            "--init",file,"--smoke","--out",joinpath(folder,"rect")])
        @test length(rect.cells)==9
        @test length(rect.auxiliary_charges)==3
        @test sum(values(rect.auxiliary_charges))/27==1/9
        # A copied initial state must evolve identically when representatives
        # are independent, without perturbations or representative averaging.
        for (r,(x,y)) in enumerate(rect.geometry.representatives)
            owner=InitBS.representative(small.geometry,x,y)
            for name in (:B_a,:B_b,:B_c,:T_u,:T_d)
                a,b=getfield(rect.cells[r],name),getfield(small.cells[owner],name)
                @test norm(a-b)<3e-11
            end
        end
        # Exercise the U1 --init branch separately from SU2 projection.
        file=joinpath(folder,"sqrt3","final.jld2")
        u1=main(["--symmetry","u1","--cell","sqrt3xsqrt3","--Dmax","3",
            "--init",file,"--smoke","--out",joinpath(folder,"from_u1")])
        @test abs(u1_torus_observables(u1).magnetization_normalized-1/9)<3e-12
        @test isfile(joinpath(folder,"from_u1","u1_state.toml"))
    end
end
println("U1 initialization checks completed / TensorKit ",pkgversion(TensorKit))
