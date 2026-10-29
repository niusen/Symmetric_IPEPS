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
            checkpoint=joinpath(dirname(r.matfile),"checkpoint.jls")
            saved=load_state(checkpoint)
            @test saved.state.T[1] isa TensorMap
            # The old, ten-times-stricter checkpoint must resume without
            # repeating SU; convergence is rechecked from its saved history.
            legacy=merge(saved.metadata.longrun,
                (signature=merge(saved.metadata.longrun.signature,(lambda_tol_scale=1e-4,)),
                 stages=NamedTuple[merge(stage,(converged=false,)) for stage in saved.metadata.longrun.stages]))
            save_state(checkpoint,saved.state;metadata=merge(saved.metadata,(longrun=legacy,)))
            r3=LR.run_one(cfg,1,77)
            @test r3.total_sweeps==r.total_sweeps
            @test r3.converged
            migrated=load_state(checkpoint)
            @test migrated.metadata.longrun.signature.lambda_tol_scale==1e-3
            @test all(stage->stage.converged,migrated.metadata.longrun.stages)
        end
        @test LR.LongRunConfig().lambda_tol_scale==1e-3
        @test_throws ArgumentError LR.validate(LR.LongRunConfig(ctm_max_iterations=51))
    end
end
function test_bond_expansion()
    @testset "Dense iPESS virtual-space expansion" begin
        s=init_state(2,2;D=1,seed=31)
        original=deepcopy(s)
        padded=expand_state(s,2;noise=0.0,seed=99)
        oldpsi=triangle_wavefunction(s.T[1,2],s.T[2,1],s.T[1,1],s.B[1,1])
        newpsi=triangle_wavefunction(padded.T[1,2],padded.T[2,1],padded.T[1,1],padded.B[1,1])
        @test newpsi[:,1] ≈ oldpsi[:,1] atol=1e-12
        @test norm(newpsi[:,2:end])==0
        perturbed=expand_state(s,2;noise=1e-3,seed=99)
        repeated=expand_state(s,2;noise=1e-3,seed=99)
        @test perturbed.T==repeated.T
        @test perturbed.B==repeated.B
        @test s.B==original.B && s.T==original.T
        @test all(l->diag(convert(Array,l))[end]>0,perturbed.lambda1)
        @test all(t->all(k->SU4TM.dim(SU4TM.space(t,k))==2,1:3),perturbed.B)
        gates=SU4TM.gate_RU_LD_RD(Dict("J"=>1.0,"K"=>0.0,"Φ"=>0.0),
            model_settings(2,2),0.005,SU4TM.ComplexSpace)
        SU4TM.triangle_update_iPESS(2,1,perturbed.B,perturbed.T,
            perturbed.lambda1,perturbed.lambda2,perturbed.lambda3,gates,1e-12)
        @test all(t->isfinite(norm(t)),perturbed.B)
        @test all(t->isfinite(norm(t)),perturbed.T)
        @test maximum(SU4TM.dim(SU4TM.space(b,k)) for b in perturbed.B for k=1:3)==2
        @test_throws ArgumentError expand_state(s,1)
        @test_throws ArgumentError expand_state(s,2;noise=-1.0)
    end
end
function test_continuation_batch()
    @testset "Previous-D chain, existing state reuse and restart" begin
        LR=DenseSU4LongRun
        mktempdir() do dir
            options=(Ds=[1,2],seeds=[77],J=0.0,schedule=[(0.1,2)],
                continuation_schedule=[(0.005,2)],min_sweeps=1,measure_every=1,
                checkpoint_every=1,stable_checks=1,blas_threads=1,
                ctm_chis=[1,2],ctm_max_iterations=3,output_dir=dir)
            independent=LR.LongRunConfig(;options...,initialization=:random)
            base=LR.run_one(independent,1,77)
            original_checkpoint=joinpath(dirname(base.matfile),"checkpoint.jls")
            original_bytes=read(original_checkpoint)
            ignored_checkpoint=joinpath(LR.run_directory(independent,2,77),"checkpoint.jls")
            mkpath(dirname(ignored_checkpoint))
            write(ignored_checkpoint,"independent D=2 result must not be overwritten")
            cfg=LR.LongRunConfig(;options...,initialization=:previous_D)
            rows=LR.run_batch(cfg)
            @test all(r->r.complete,rows)
            all(r->r.complete,rows) || error("continuation batch failed")
            first_data=matread(rows[1].matfile)
            next_data=matread(rows[2].matfile)
            @test first_data["initialization"]=="existing_checkpoint"
            @test rows[1].total_sweeps==base.total_sweeps
            @test next_data["initialization"]=="previous_D"
            @test next_data["source_D"]==1
            @test next_data["initial_D"]==1
            @test next_data["schedule"]≈[0.005 2.0]
            @test next_data["expansion_noise"]==cfg.expansion_noise
            @test read(original_checkpoint)==original_bytes
            @test read(ignored_checkpoint,String)=="independent D=2 result must not be overwritten"
            @test isfile(joinpath(dir,"continuation","scan_summary.mat"))
            resumed=LR.run_one(cfg,2,77)
            @test resumed.total_sweeps==rows[2].total_sweeps
            @test resumed.energy≈rows[2].energy atol=1e-10
            saved=load_state(joinpath(dirname(resumed.matfile),"checkpoint.jls"))
            @test saved.metadata.initialization.source_D==1
            @test saved.metadata.longrun.signature.schedule==cfg.continuation_schedule
        end
        @test_throws ArgumentError LR.validate(LR.LongRunConfig(Ds=[5,4]))
        @test_throws ArgumentError LR.validate(LR.LongRunConfig(expansion_noise=0.0))
    end
end
function test_schedule_cutoff()
    @testset "Stop at dt=0.001 and preserve older checkpoint history" begin
        LR=DenseSU4LongRun
        @test last(LR.LongRunConfig().schedule)[1]==0.001
        @test last(LR.LongRunConfig().continuation_schedule)[1]==0.001
        mktempdir() do dir
            options=(Ds=[1],seeds=[77],J=0.0,initialization=:random,
                min_sweeps=1,measure_every=1,checkpoint_every=1,stable_checks=1,
                blas_threads=1,ctm_chis=[1,2],ctm_max_iterations=3,output_dir=dir)
            oldcfg=LR.LongRunConfig(;options...,schedule=[(0.001,1),(0.0001,1)])
            old=LR.run_one(oldcfg,1,77)
            checkpoint=joinpath(dirname(old.matfile),"checkpoint.jls")
            saved=load_state(checkpoint)
            legacy=merge(saved.metadata.longrun,
                (signature=merge(saved.metadata.longrun.signature,(lambda_tol_scale=1e-4,)),))
            save_state(checkpoint,saved.state;metadata=merge(saved.metadata,(longrun=legacy,)))
            cfg=LR.LongRunConfig(;options...,schedule=[(0.001,1)])
            restored=LR.run_one(cfg,1,77)
            data=matread(restored.matfile)
            @test restored.total_sweeps==old.total_sweeps
            @test data["historical_finer_steps_retained"]==true
            @test data["su_convergence_dt"]==0.001
            @test size(data["stage_summary"],1)==2
            @test data["su_converged"]==true
            current=load_state(checkpoint)
            @test current.state.B==saved.state.B && current.state.T==saved.state.T
            @test current.metadata.longrun.progress.complete
            @test current.metadata.longrun.signature.schedule==cfg.schedule
            # A checkpoint still at the first stage must finish that stage,
            # then stop without entering the removed finer stage.
            fresh=init_state(2,2;D=1,seed=77)
            progress=(stage=1,sweep=0,total_sweeps=0,stable=0,block_lambda=0.0,
                elapsed_seconds=0.0,complete=false)
            LR.write_checkpoint(checkpoint,fresh,oldcfg,1,77,progress,NamedTuple[],NamedTuple[])
            resumed=LR.run_one(cfg,1,77)
            @test resumed.total_sweeps==1
            @test matread(resumed.matfile)["historical_finer_steps_retained"]==false
            @test_throws ErrorException LR.run_one(LR.LongRunConfig(;options...,schedule=[(0.001,2)]),1,77)
        end
    end
end
function run_su4_tests()
    BLAS.set_num_threads(1)
    test_dense_projector()
    test_gate_update()
    test_product_measurements()
    test_d4_smoke()
    test_batch_resume()
    test_bond_expansion()
    test_continuation_batch()
    test_schedule_cutoff()
end
if abspath(PROGRAM_FILE)==@__FILE__
    run_su4_tests()
end
