using Test, LinearAlgebra, TensorKit, SUNRepresentations, MAT
include(joinpath(@__DIR__,"../../../src/bosonic/triangle/su4_six/TriangleSU4Six.jl"))
using .TriangleSU4Six
const SIX=TriangleSU4Six
BLAS.set_num_threads(1)

function test_model()
    @testset "Two-particle SU4 six-irrep normalization and symmetry" begin
        dense=six_model(); symmetric=six_model(backend=:su4)
        H=reshape(convert(Array,dense.h),36,36)
        P=reshape(convert(Array,dense.swap),36,36)
        Q=reshape(convert(Array,dense.singlet),36,36)
        @test eigvals(Hermitian(H))≈vcat(-5.0,fill(-1.,15),fill(1.,20)) atol=1e-12
        @test H≈H' atol=1e-13
        @test tr(H)≈0 atol=1e-12
        @test Q*Q≈Q atol=1e-12
        @test tr(Q)≈1 atol=1e-12
        @test H*Q≈-5Q atol=1e-12
        @test P*P≈I atol=1e-12
        C=dense.C
        @test sum(C[a,b]*C[b,a] for a=1:4,b=1:4)≈5I atol=1e-12
        for a=1:4,b=1:4
            G=kron(Matrix{ComplexF64}(I,6,6),C[a,b])+kron(C[a,b],Matrix{ComplexF64}(I,6,6))
            @test norm(H*G-G*H)<1e-12
        end
        R=symmetric.rotation; U=kron(R,R)
        @test U*reshape(convert(Array,symmetric.h),36,36)*U'≈H atol=1e-11
        @test U*reshape(convert(Array,symmetric.singlet),36,36)*U'≈Q atol=1e-11
        @test sort([dim(s) for (s,_) in blocks(symmetric.h)])==[1,15,20]
        for model in (dense,symmetric)
            SIX.bond_factors(model.h); SIX.bond_factors(model.swap)
            @test true # reconstruction is checked inside bond_factors
        end
        gd=triangle_gates(dense,0.01)[1]
        gs=triangle_gates(symmetric,0.01)[1]
        U3=kron(R,kron(R,R))
        @test U3*reshape(convert(Array,gs),216,216)*U3'≈reshape(convert(Array,gd),216,216) atol=1e-11
        @test reshape(convert(Array,triangle_gates(dense,0.0)[1]),216,216)≈I atol=1e-12
    end
end

function test_dense_product()
    @testset "Dense product: every NN coordinate, energy, singlet and connected correlator" begin
        s=initial_state(D=1)
        vectors=[normalize(ComplexF64[cos(k+j),sin(2k-j),k/5,j/3,im*k/4,im*j/7]) for k=1:2,j=1:2]
        for k in eachindex(s.T)
            t=s.T[k]
            s.T[k]=TensorMap(reshape(vectors[k],1,6,1,1),codomain(t),domain(t))
            s.B[k]=s.B[k]/norm(s.B[k])
        end
        obs=measure_state(s;chis=[1,2],max_iterations=3,verbose=false)
        m=six_model(); H=reshape(convert(Array,m.h),36,36); P1=reshape(convert(Array,m.singlet),36,36)
        energy=0.0
        for b in nearest_neighbor_bonds(obs)
            v=kron(vectors[b.target...],vectors[b.source...])
            h=real(dot(v,H*v)); p=real(dot(v,P1*v)); energy+=h
            @test b.coupling≈h atol=1e-9
            @test b.exchange≈abs2(dot(vectors[b.source...],vectors[b.target...])) atol=1e-9
            @test getproperty(obs.singlet_weight,b.direction)[b.simplex...]≈p atol=1e-9
            @test b.connected_generator_dot≈0 atol=1e-9
        end
        @test obs.energy_per_site≈energy/4 atol=1e-9
        @test length(nearest_neighbor_bonds(obs))==12
        data=observable_mat_data(obs)
        @test data["particles_per_site"]==2
        @test size(data["bond_table"],1)==12
        @test "singlet_weight" in data["bond_table_columns"]
    end
end

function local_triangle_wavefunction(t1,t2,t3,b)
    SIX.@tensor psi[-1,-2,-3,-4,-5,-6,-7,-8,-9] := t1[-1,-2,-3,a]*b[z,a,c]*t2[-4,-5,z,-6]*t3[c,-7,-8,-9]
    values=permutedims(convert(Array,psi),(2,5,7,1,3,4,6,8,9))
    return reshape(values,216,:)
end

function test_exact_local_update()
    @testset "Old triangle update equals exact untruncated six-state evolution" begin
        state=initial_state(D=1,seed=52)
        gate=triangle_gates(six_model(),0.03)[1]
        tensors=(state.T[1,2],state.T[2,1],state.T[1,1],state.B[1,1])
        expected=reshape(convert(Array,gate),216,216)*local_triangle_wavefunction(tensors...)
        out=SIX.triangle_gate_iPESS_simplified(6,gate,tensors...,1e-14)
        actual=local_triangle_wavefunction(out[1:4]...)
        @test abs(dot(vec(actual),vec(expected)))/(norm(actual)*norm(expected))≈1 atol=1e-12
    end
end

function test_network_mapping()
    @testset "CTMRG square cell equals the original periodic iPESS network" begin
        s=initial_state(D=2,seed=71)
        ta,tb,tc,td=map(t->convert(Array,t),(s.T[1,1],s.T[2,1],s.T[1,2],s.T[2,2]))
        ba,bb,bc,bd=map(t->convert(Array,t),(s.B[1,1],s.B[2,1],s.B[1,2],s.B[2,2]))
        SIX.@tensor direct[-1,-2,-3,-4] := ta[1,-1,2,3]*tb[4,-2,5,6]*tc[7,-3,8,9]*td[10,-4,11,12]*ba[5,9,1]*bb[2,12,4]*bc[11,3,7]*bd[8,6,10]
        A=SIX.measurement_cell(s)
        a,b,c,d=map(t->convert(Array,t),(A[1][1],A[2][1],A[1][2],A[2][2]))
        SIX.@tensor square[-1,-2,-3,-4] := a[1,2,3,4,-1]*b[3,5,1,6,-2]*c[7,4,8,2,-3]*d[8,6,7,5,-4]
        @test abs(dot(vec(direct),vec(square)))/(norm(direct)*norm(square))≈1 atol=1e-12
    end
end

function test_symmetric_dimers()
    @testset "Exact singlet dimers: SU4 blocks and dense contractions agree" begin
        observations=Any[]
        for backend in (:su4,:dense)
            s=initial_dimer_state(;backend)
            obs=measure_state(s;chis=[6,7],max_iterations=3,verbose=false,onsite=false)
            push!(observations,obs)
            @test obs.energy_per_site≈-2.5 atol=1e-8
            values=sort([b.coupling for b in nearest_neighbor_bonds(obs)])
            @test values≈vcat([-5.,-5.],zeros(10)) atol=1e-8
            @test sum(sum(v) for v in obs.singlet_weight)≈2+10/36 atol=1e-8
            for d in (:x,:y,:diagonal)
                @test getproperty(obs.singlet_weight,d)+getproperty(obs.adjoint_weight,d)+getproperty(obs.symmetric20_weight,d)≈ones(2,2) atol=1e-9
            end
        end
        for field in (:coupling,:exchange,:singlet_weight),d in (:x,:y,:diagonal)
            @test getproperty(getproperty(observations[1],field),d)≈getproperty(getproperty(observations[2],field),d) atol=1e-8
        end
    end
end

function test_update_resume()
    @testset "Original simple update, physical dimension and six-irrep restart" begin
        for backend in (:dense,:su4)
            D=backend==:dense ? 2 : 7
            s=initial_state(;backend,D)
            model=six_model(;backend)
            simple_update!(s;model,Dmax=D,dt=0.001,sweeps=1)
            @test all(t->isfinite(norm(t)) && norm(t)>0,s.T)
            @test all(t->dim(space(t,2))==6,s.T)
            @test maximum(dim(space(b,k)) for b in s.B for k=1:3)<=D
            @test all(t->sectortype(t)==(backend==:dense ? Trivial : SU4Irrep),s.T)
        end
        for backend in (:dense,:su4)
            mktempdir() do root
                D=backend==:dense ? 1 : 7
                cfg=SixRunConfig(;backend,Dmax=D,output_dir=root,schedule=[(0.001,1)],min_sweeps=1,
                    check_every=1,checkpoint_every=1,stable_checks=1,chis=[backend==:dense ? 1 : 16],
                    ctm_max_iterations=2,blas_threads=1)
                first_run=run_simulation(cfg)
                saved=SIX.load_checkpoint(first_run.checkpoint)
                @test saved.progress.complete
                @test first_run.total_sweeps==1
                second_run=run_simulation(cfg)
                @test second_run.total_sweeps==1
                @test second_run.obs.energy_per_site≈first_run.obs.energy_per_site atol=1e-10
                @test matread(second_run.matfile)["physical_dimension"]==6
                bad=SixRunConfig(;backend,Dmax=D,output_dir=root,schedule=[(0.002,1)])
                @test_throws ErrorException run_simulation(bad)
            end
        end
    end
end

function test_graded_ctm_inverse()
    @testset "Six-model-only CTM inverse and unchanged old dispatch" begin
        V=Rep[SU{4}]((0,0,0,0)=>2,(1,1,0,0)=>1)
        S=DiagonalTensorMap([1.,0.,0.5],V)
        for ch in (7,6,1)
            Core.eval(SIX,:(chi=$ch))
            Core.eval(SIX,:(multiplet_tol=0.0))
            Core.eval(SIX,:(projector_trun_tol=1e-12))
            actual=diag(convert(Array,SIX.sdiag_inv_sqrt(S)))
            original=diag(convert(Array,S))
            expected=[v>0 && (ch==7 || v>0.75) ? inv(sqrt(v)) : 0.0 for v in original]
            @test all(isfinite,actual)
            @test actual≈expected atol=1e-12
        end
        su2=DiagonalTensorMap([1.,0.,0.5],Rep[SU₂](0=>2,1/2=>1))
        @test basename(string(which(SIX.sdiag_inv_sqrt,(typeof(su2),)).file))=="CTMRG.jl"
        @test basename(string(which(SIX.sdiag_inv_sqrt,(typeof(S),)).file))=="CTMRGCompat.jl"
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    test_model()
    test_exact_local_update()
    test_dense_product()
    test_network_mapping()
    test_symmetric_dimers()
    test_update_resume()
    test_graded_ctm_inverse()
end
