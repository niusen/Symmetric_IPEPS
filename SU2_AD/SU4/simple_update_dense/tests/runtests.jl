using Test, LinearAlgebra, Random, MAT
if !isdefined(@__MODULE__,:TensorMapSU4)
    include(joinpath(@__DIR__,"../../../src/bosonic/triangle/su4_dense/TensorMapSU4.jl"))
end
using .TensorMapSU4
using .TensorMapSU4: @tensor, TensorMap, codomain, domain
if !isdefined(@__MODULE__,:DenseSU4LongRun)
    include(joinpath(@__DIR__,"../LongRun_D4_D16.jl"))
end
const SU4TM=TensorMapSU4

function exact_triangle_h()
    h=zeros(64,64)
    idx(a,b,c)=a+4*(b-1)+16*(c-1)
    for a=1:4,b=1:4,c=1:4
        k=idx(a,b,c)
        h[idx(b,a,c),k]+=1
        h[idx(c,b,a),k]+=1
        h[idx(a,c,b),k]+=1
        h[k,k]-=3/4
    end
    h
end
function triangle_wavefunction(t1,t2,t3,b)
    @tensor psi[-1,-2,-3,-4,-5,-6,-7,-8,-9] := t1[-1,-2,-3,a]*b[z,a,c]*t2[-4,-5,z,-6]*t3[c,-7,-8,-9]
    a=permutedims(convert(Array,psi),(2,5,7,1,3,4,6,8,9))
    reshape(a,64,:)
end
function test_gate_update()
    @testset "Original iPESS gate: exact untruncated evolution" begin
        s=init_state(2,2;D=2,seed=41)
        p=Dict("J"=>1.0,"K"=>0.0,"Φ"=>0.0)
        h=exact_triangle_h()
        for dt in (0.0,0.07)
            g=SU4TM.gate_RU_LD_RD(p,model_settings(2,2),dt,SU4TM.ComplexSpace)[1]
            @test reshape(convert(Array,g),64,64) ≈ exp(-dt*h) atol=2e-12
            ts=(s.T[1,2],s.T[2,1],s.T[1,1],s.B[1,1])
            expected=exp(-dt*h)*triangle_wavefunction(ts...)
            out=SU4TM.triangle_gate_iPESS_simplified(16,g,ts...,1e-14)
            actual=triangle_wavefunction(out[1:4]...)
            @test abs(dot(vec(actual),vec(expected)))/(norm(actual)*norm(expected)) ≈ 1 atol=2e-12
        end
    end
end
function test_dense_projector()
    @testset "Dense CTM truncation: degeneracy and zero values" begin
        Core.eval(SU4TM, :(chi=1))
        Core.eval(SU4TM, :(multiplet_tol=0.0))
        Core.eval(SU4TM, :(projector_trun_tol=1e-12))
        S=SU4TM.DiagonalTensorMap([1.0,1.0,0.0],SU4TM.ComplexSpace(3))
        inverse=convert(Array,SU4TM.sdiag_inv_sqrt(S))
        @test diag(inverse)==[1.0,0.0,0.0]
        @test all(isfinite,inverse)
    end
end
function product_state(vectors)
    s=init_state(size(vectors)...;D=1,seed=17)
    for k in eachindex(s.T)
        t=s.T[k]; b=s.B[k]
        s.T[k]=TensorMap(reshape(vectors[k],1,4,1,1),codomain(t),domain(t))
        s.B[k]=TensorMap(ones(1,1,1),codomain(b),domain(b))
    end
    s
end
function test_product_measurements()
    @testset "CTMRG exact products and every bond coordinate" begin
        Random.seed!(71)
        sets=([ComplexF64.(Matrix{Float64}(I,4,4)[:,x+2(y-1)]) for x=1:2,y=1:2],
              [ComplexF64[1,0,0,0] for x=1:2,y=1:2],
              [normalize(randn(ComplexF64,4)) for x=1:2,y=1:2])
        for vectors in sets
            o=measure_state(product_state(vectors);chis=[1,2],max_iterations=3,verbose=false)
            @test length(nearest_neighbor_bonds(o))==12
            expected=0.0
            for b in nearest_neighbor_bonds(o)
                p=abs2(dot(vectors[b.source...],vectors[b.target...]))
                @test b.exchange ≈ p atol=1e-10
                @test b.connected_generator_dot ≈ 0 atol=1e-10
                expected+=p-0.25
            end
            @test o.energy_per_site ≈ expected/4 atol=1e-10
        end
    end
end
function test_d4_smoke()
    @testset "D=4 original SU -> CTMRG" begin
        s=init_state(2,2;D=4,seed=1234)
        g=SU4TM.gate_RU_LD_RD(Dict("J"=>1.0,"K"=>0.0,"Φ"=>0.0),model_settings(2,2),0.01,SU4TM.ComplexSpace)
        SU4TM.triangle_update_iPESS(4,1,s.B,s.T,s.lambda1,s.lambda2,s.lambda3,g,1e-12)
        o=measure_state(s;chis=[4],max_iterations=2,verbose=false)
        @test isfinite(o.energy_per_site)
        @test all(b->isfinite(b.coupling),nearest_neighbor_bonds(o))
        @test all(t->t isa TensorMap,s.T)
        println("D4 SMOKE (one sweep, chi=4, two CTMRG iterations; NOT converged): E/site=",o.energy_per_site)
    end
end
function test_batch_resume()
    @testset "Server checkpoint, resume and MAT" begin
        LR=DenseSU4LongRun
        mktempdir() do dir
            cfg=LR.LongRunConfig(Ds=[1],seeds=[77],J=0.0,schedule=[(0.1,2),(0.01,2)],
                min_sweeps=1,measure_every=1,checkpoint_every=1,stable_checks=1,
                blas_threads=1,ctm_chis=[1,2],ctm_max_iterations=3,output_dir=dir)
            r=LR.run_batch(cfg)[1]
            @test r.complete
            r.complete || error("batch failed")
            m=matread(r.matfile)
            @test m["environment"]=="ctmrg"
            @test size(m["bond_table"])==(12,17)
            @test abs(m["energy_per_site"])<1e-12
            @test m["tensor_format"]=="TensorMap_ComplexSpace"
            r2=LR.run_one(cfg,1,77)
            @test r2.total_sweeps==r.total_sweeps
            @test r2.energy ≈ r.energy atol=1e-12
            @test load_state(joinpath(dirname(r.matfile),"checkpoint.jls")).state.T[1] isa TensorMap
        end
        @test_throws ArgumentError LR.validate(LR.LongRunConfig(ctm_max_iterations=51))
    end
end
function run_su4_tests()
    BLAS.set_num_threads(1)
    test_dense_projector()
    test_gate_update()
    test_product_measurements()
    test_d4_smoke()
    test_batch_resume()
end
if abspath(PROGRAM_FILE)==@__FILE__
    run_su4_tests()
end
