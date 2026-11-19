using Test,LinearAlgebra,TensorKit,SUNRepresentations
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
if !isdefined(@__MODULE__,:HC);const HC=HoneycombSU4;end
using .HoneycombSU4
BLAS.set_num_threads(1);

@testset "SU4 original D9: CTM and two-site full update" begin
    T,_,_,_=initial_honeycomb_star_iPEPS();
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
        CTM_ite_info=true,CTM_conv_info=true,grad_checkpoint=false);
    env=honeycomb_fu_environment(A,16,setting);
    println("SU4 FU environment ready");flush(stdout);
    ob=honeycomb_energy_FU(A,env);
    @test ob.E_P≈-9/16 atol=1e-8
    @test length(ob.bonds)==24
    reference=honeycomb_energy_CTM(Dict("J"=>1.),T,16,setting;convention=:permutation,CTM0=env.CTM);
    @test ob.E_P≈reference.energy_per_site atol=1e-8
    @test sort([r.P for r in ob.bonds])≈sort(reference.energies) atol=1e-8
    reference_by_bond=Dict((b.axis,b.site1,b.site2)=>e for (b,e) in zip(reference.bonds,reference.energies));
    for r in ob.bonds
        su1=(r.site1[1],mod1(2-r.site1[2],4));su2=(r.site2[1],mod1(2-r.site2[2],4));
        axis=r.direction==:x ? :horizontal : :vertical;
        @test r.P≈reference_by_bond[(axis,su1,su2)] atol=1e-8
    end
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.001);
    settings=SquareJ1FullUpdateSettings(Dmax=9,multiplet_tol=0.,maxiter=1,verbose=false);
    b0=honeycomb_fu_bond_groups(A)[3][1];
    id1,id2,_=honeycomb_full_update_bond(A,env,unitary(codomain(H),domain(H)),b0;
        settings,space_policy=:fixed_space);
    @test norm(id1-A[b0.site1])<1e-10
    @test norm(id2-A[b0.site2])<1e-10
    for b in (honeycomb_fu_bond_groups(A)[1][1],honeycomb_fu_bond_groups(A)[3][1]),policy in (:fixed_space,:tensorkit)
        println("SU4 FU START ",(direction=b.direction,policy=policy));flush(stdout);
        left,right,r=honeycomb_full_update_bond(A,env,gate,b;settings,space_policy=policy);
        @test r.loss_final<=r.loss_initial+1e-10
        @test 0<=r.fidelity<=1+1e-8
        @test norm(left)≈1 && norm(right)≈1
        leg=b.direction==:x ? 3 : 2;
        @test dim(space(left,leg))<=9
        if policy==:fixed_space;@test space(left,leg)==space(A[b.site1],leg);end
        println("SU4 FU RESULT ",(direction=b.direction,policy=policy,initial=r.loss_initial,
            final=r.loss_final,accepted=r.accepted_steps,old=string(r.old_bond_space),new=string(r.new_bond_space)));flush(stdout);
    end
end
