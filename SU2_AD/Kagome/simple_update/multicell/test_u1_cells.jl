using Test, TensorKit, LinearAlgebra, Random, TOML
BLAS.set_num_threads(10)
include(joinpath(@__DIR__,"verify_charged_dense.jl"))
include(joinpath(@__DIR__,"run_charged_trial.jl"))
include(joinpath(@__DIR__,"run.jl"))
const US=KagomeU1Cells
const CS1=KagomeChargedSU2

function distance(a,b)
    a,b=a/norm(a),b/norm(b);z=dot(vec(a),vec(b))
    norm(a-b*conj(z)/abs(z))
end

@testset "Exact restriction, multiplicities, duals and auxiliary components" begin
    for V in (SU2Space(0=>1,1/2=>1),SU2Space(0=>2,1/2=>2,1=>1))
        for complex in (false,true)
            source=CS1.initialize_charged(;Vv=V,complex,rng=MersenneTwister(21))
            converted=to_u1(source)
            @test check_u1(converted)===converted
            @test dim(space(converted.cells[1].T_u,4))==1
            for (old,new) in zip(source.raw.cells,converted.cells)
                for name in (:B_a,:B_b,:B_c,:T_d)
                    t=getfield(old,name);u=getfield(new,name)
                    maps=[US._axis_restriction(space(t,i))[2] for i in 1:numind(t)]
                    @test norm(convert(Array,u)-convert(Array,t)[maps...])<1e-11
                    @test all(sectortype(space(u,i))==U1Irrep for i in 1:numind(u))
                end
            end
            @test US._axis_restriction(V)[2]==US._axis_restriction(V')[2]
        end
    end
    for j in (1//2,3//2), component in 1:Int(2j+1)
        source=CS1.initialize_charged(;auxiliary_spin=j,rng=MersenneTwister(41))
        converted=to_u1(source;component)
        ref=CS1.closed_sqrt3_wavefunction(source)
        @test distance(u1_torus_wavefunction(converted),copy(selectdim(ref,10,component)))<3e-12
        old=CS1.exact_torus_observables(source;component,h=.38)
        new=u1_torus_observables(converted;h=.38)
        @test abs(old.field_energy_per_site-new.field_energy_per_site)<3e-12
        @test abs(old.magnetization_normalized-new.magnetization_normalized)<3e-12
        @test abs(old.spin_squared-new.spin_squared)<3e-12
    end
end

@testset "U1 SU frees multiplets, preserves charge and checkpoint continuation" begin
    source=CS1.initialize_charged(;rng=MersenneTwister(1234))
    initial=to_u1(source);H=u1_heisenberg_triangle()
    before=deepcopy(initial)
    for Dmax in (2,3)
        updated=itebd_u1(initial,H,.002,.001;Dmax)
        @test check_u1(updated;Dmax)===updated
        @test norm(initial.cells[1].T_u-before.cells[1].T_u)==0
        obs=u1_torus_observables(updated;h=.38)
        @test abs(obs.magnetization_normalized-1/9)<3e-12
        @test obs.variance_Sz<3e-12
        @test isfinite(obs.field_energy_per_site)
        @test all(dim(space(c.T_u,i))<=Dmax for c in updated.cells for i in 1:3)
        mktempdir() do folder
            file=joinpath(folder,"u1.jld2")
            save_u1_checkpoint(file,updated,H;metadata=Dict("test"=>true))
            restored,data=load_u1_checkpoint(file)
            @test data["metadata"]["test"]
            @test distance(u1_torus_wavefunction(restored),u1_torus_wavefunction(updated))<3e-12
            direct=itebd_u1(updated,H,.001,.0005;Dmax)
            resumed=itebd_u1(restored,data["H"],.001,.0005;Dmax)
            @test distance(u1_torus_wavefunction(direct),u1_torus_wavefunction(resumed))<3e-12
        end
    end
    # Equal singular values in distinct U1 sectors need not be kept together.
    # The charge origin of an SVD bond is gauge-dependent, so do not assert
    # that a particular run must produce an asymmetric set of charge labels.
    V=U1Space(1=>1,-1=>1)
    matrix=TensorMap(Matrix{Float64}(I,2,2),V,V)
    a,b,c=US.Kernels.tsvd(matrix;trunc=US.Kernels.truncdim(1))
    a,b,c=US.Kernels.Truncations(a,b,c,1,1e-6)
    @test dim(space(b,1))==1
end

@testset "Default U1 export, explicit dense fallback and U1-state re-export" begin
    mktempdir() do folder
        source=CS1.initialize_charged(;rng=MersenneTwister(67))
        input=joinpath(folder,"source.jld2")
        CS1.save_charged_checkpoint(input,source,CS1.BaseSU.heisenberg_triangle())
        out=joinpath(folder,"default")
        result=export_and_verify(input,out)
        @test result["representation"]=="u1"
        @test isfile(joinpath(out,"u1_state.jld2"))
        @test !isfile(joinpath(out,"dense_state.jld2"))
        @test TOML.parsefile(joinpath(out,"u1_state.toml"))["symmetry"]=="U1"
        again=export_and_verify(joinpath(out,"u1_state.jld2"),joinpath(folder,"again"))
        @test again["absolute_difference_from_SU2_reference"]["magnetization"]<3e-12
        dense=export_and_verify(input,joinpath(folder,"dense");representation=:dense)
        @test dense["representation"]=="dense"
        @test isfile(joinpath(folder,"dense","dense_state.toml"))
    end
end

@testset "Both cell geometries, U1 runner and SU2-projected charged initializer" begin
    mktempdir() do folder
        for cell in ("sqrt3xsqrt3","3x3")
            state=main(["--symmetry","u1","--cell",cell,"--Dmax","3","--smoke",
                        "--out",joinpath(folder,cell)])
            @test check_u1(state)===state
            @test length(state.cells)==(cell=="3x3" ? 9 : 3)
        end
        s,report=charged_trial(joinpath(folder,"charged_u1");Dmax=3,symmetry=:u1)
        @test report["symmetry"]=="u1"
        @test abs(report["final"]["magnetization_normalized"]-1/9)<3e-12
        @test check_u1(s)===s
        original,legacy=charged_trial(joinpath(folder,"charged_su2");Dmax=3,symmetry=:su2)
        @test legacy["symmetry"]=="su2"
        @test abs(legacy["final"]["magnetization_normalized"]-1/9)<3e-12
    end
end

println("U1 tests completed: Julia ",VERSION," / TensorKit ",pkgversion(TensorKit))
