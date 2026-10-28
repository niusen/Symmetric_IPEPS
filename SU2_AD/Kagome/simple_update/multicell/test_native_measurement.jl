# CPU / small D only. No production scan and no Python process.
using Test
include(joinpath(@__DIR__,"charged_su2_convergence_driver.jl"))
include(joinpath(@__DIR__,"..","..","..","src","bosonic","kagome","native_measurement","KagomeNativeMeasurement.jl"))
const Native=KagomeNativeMeasurement
BLAS.set_num_threads(2)
@testset "pure TensorKit D2 conversion, CTMRG and observables" begin
    s=convergence_initial_state(2,:integer_singlet,1234)
    H=ConvBase.heisenberg_triangle()
    s=ConvSU.itebd_charged(s,H,.02,.01;Dmax=2)
    u=KagomeU1Cells.to_u1(s);dense=Native.to_dense(u)
    for (a,b) in zip(u.cells,dense.cells),name in fieldnames(ConvBase.KagomeSUTensors)
        ta,tb=getfield(a,name),getfield(b,name)
        @test isnothing(ta)==isnothing(tb)
        isnothing(ta) && continue
        @test norm(convert(Array,ta)-convert(Array,tb))<1e-12
    end
    for V in (ComplexSpace(3),U1Space(0=>1,1=>2))
        M=complex(unitary(V,V))
        uu,ss,vv=Native.CTM.native_svd(M,2,1e-12)
        si=complex(Native.CTM.safe_inv_sqrt(ss,1e-12))
        @test dim(space(ss,1))==2
        @test norm(si*ss*si-complex(unitary(space(ss,1),space(ss,1))))<1e-12
        @test isfinite(norm(uu*ss*vv))
    end
    if !isdefined(@__MODULE__,:NATIVE_CONVERSION_ONLY)
    for representation in (:u1,:dense)
        println("TEST D2 ",representation);flush(stdout)
        result,state=Native.measure_state(s;representation,device="cpu",chi=4,iterations=10,tolerance=1e-8)
        println("D2 ",representation," E=",result["energy_per_spin"]," m=",result["m_normalized"]);flush(stdout)
        @test result["energy_per_spin"]≈-0.3544444444444445 atol=1e-10
        @test result["m_normalized"]≈1/9 atol=1e-10
        @test result["field_energy_per_spin"]≈-.38/18 atol=1e-10
        rows=Native.grouped_observables(result,2)
        @test length(rows.sites)==9
        @test length(rows.triangles)==6
        @test length(rows.bonds)==18
        @test sum(r["energy"] for r in rows.bonds)/9≈result["exchange_energy_per_spin"] atol=1e-11
        @test all(abs(r["C_SzSzSz"])<=1/8+1e-12 for r in rows.triangles)
        @test result["ctm"]["iterations"]<=10
        @test all(d<=4 for d in result["actual_corner_dimensions"])
    end
    end
end
