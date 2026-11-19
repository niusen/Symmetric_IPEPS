using Test,LinearAlgebra,TensorKit,SUNRepresentations
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
if !isdefined(@__MODULE__,:HC);const HC=HoneycombSU4;end
using .HoneycombSU4
BLAS.set_num_threads(1);

@testset "Nonproduct dense environment improves the variational fit" begin
    T,_,_=initial_honeycomb_iPEPS(2,2;symmetry=:dense,D=2,seed=92);
    # Keep this optimizer regression away from a random near-critical norm
    # network: a product component plus nonzero random entanglement.
    for i in eachindex(T)
        data=convert(Array,T[i]);data[1,1,1,1,1]+=5;
        T[i]=TensorMap(data,codomain(T[i]),domain(T[i]));T[i]/=norm(T[i]);
    end
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));backup=deepcopy(A);
    setting=LS_CTMRG_settings(CTM_ite_nums=100,CTM_conv_tol=1e-7,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    env=honeycomb_fu_environment(A,8,setting);
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.05);
    settings=SquareJ1FullUpdateSettings(Dmax=2,multiplet_tol=0.,maxiter=4,verbose=false);
    b=honeycomb_fu_bond_groups(A)[1][1];
    _,_,r=honeycomb_full_update_bond(A,env,gate,b;settings,space_policy=:fixed_space);
    @test isfinite(r.fidelity)
    @test r.accepted_steps>0
    @test r.loss_final<r.loss_initial-1e-12
    @test all(norm(x-y)==0 for (x,y) in zip(A,backup))
    println("NONPRODUCT FU FIT ",(initial=r.loss_initial,final=r.loss_final,accepted=r.accepted_steps));flush(stdout);
end
@testset "Honeycomb FU real-edge geometry" begin
    T,_,_,_=initial_honeycomb_star_iPEPS();
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));
    groups=honeycomb_fu_bond_groups(A);
    @test length.(groups)==[8,8,8]
    @test length(Set((b.direction,b.site1,b.site2) for b in reduce(vcat,groups)))==24
    @test all(dim(space(A[b.site1],b.direction==:x ? 3 : 2))==9 for b in reduce(vcat,groups))
end

@testset "Fixed quantum numbers versus native dimension SVD" begin
    T,_,_,_=initial_honeycomb_star_iPEPS();
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.01);
    for b in (honeycomb_fu_bond_groups(A)[1][1],honeycomb_fu_bond_groups(A)[3][1])
        q1,k1,k2,q2=HC._square_fu_split_reduced(A[b.site1],A[b.site2],b.direction);
        old=HC._square_fu_gated_bond(k1,k2,unitary(space(H,1)*space(H,2),space(H,1)*space(H,2)));
        f1,f2,_=HC._honeycomb_fu_initializer(:fixed_space,9)(old,k1,k2);
        @test space(f1)==space(k1) && space(f2)==space(k2)
        @test norm(HC._square_fu_gated_bond(f1,f2,unitary(codomain(H),domain(H)))-old)/norm(old)<1e-10
        M=HC._square_fu_gated_bond(k1,k2,gate);
        n1,n2,S=HC._honeycomb_fu_initializer(:tensorkit,24)(M,k1,k2);
        @test dim(space(S,1))<=24
        @test norm(HC._square_fu_gated_bond(n1,n2,unitary(codomain(H),domain(H)))-M)/norm(M)<1e-10
        f1,f2,_=HC._honeycomb_fu_initializer(:fixed_space,9)(M,k1,k2);
        @test space(f1)==space(k1) && space(f2)==space(k2)
    end
    @test_throws ArgumentError HC._honeycomb_fu_initializer(:unknown,9)
    @test_throws ArgumentError HC._honeycomb_fu_initializer(:fixed_space,4)(
        HC._square_fu_gated_bond(HC._square_fu_split_reduced(A[1,1],A[2,1],:x)[2:3]...,gate),
        HC._square_fu_split_reduced(A[1,1],A[2,1],:x)[2:3]...)
end

@testset "Mixed SU4 bra/ket with a non-self-dual internal bond" begin
    T,_,_,_=initial_honeycomb_star_iPEPS();
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));V4=honeycomb_virtual_space(["4"=>1]);
    function select_leg(A,leg,W)
        P=TensorMap{eltype(A)}(undef,W,space(A,leg));fill!(P,0);
        for (c,b) in blocks(P),k=1:min(size(b)...);b[k,k]=1;end
        return HC._star_mode_product(A,P,leg)
    end
    new1=select_leg(A[1,1],3,V4);new2=select_leg(A[2,1],1,V4');
    aa1,_=HC.build_square_cross_double_layer_open(new1,A[1,1]);
    aa2,_=HC.build_square_cross_double_layer_open(new2,A[2,1]);
    @test norm(new1)>0 && norm(new2)>0
    @test space(aa1,3)'==space(aa2,1)
    @test norm(aa1)>0 && norm(aa2)>0
end

@testset "Dense product CTM measurements and actual FU optimizer" begin
    T,_,_=initial_honeycomb_iPEPS(2,2;symmetry=:dense,D=1,seed=71);
    A=HC.square_fu_cell_to_matrix(convert_honeycomb_to_iPEPS(T));
    setting=LS_CTMRG_settings(CTM_ite_nums=20,CTM_conv_tol=1e-9,CTM_ite_info=false,CTM_conv_info=false,grad_checkpoint=false);
    env=honeycomb_fu_environment(A,8,setting);
    ob=honeycomb_energy_FU(A,env);
    @test length(ob.bonds)==6
    for r in ob.bonds
        a=vec(convert(Array,A[r.site1...]));b=vec(convert(Array,A[r.site2...]));
        @test r.P≈abs2(dot(a,b))/(norm(a)^2*norm(b)^2) atol=1e-9
    end
    H=Hamiltonian_honeycomb_SU4(space(A[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.02);
    settings=SquareJ1FullUpdateSettings(Dmax=2,multiplet_tol=0.,maxiter=2,verbose=false);
    for b in (honeycomb_fu_bond_groups(A)[1][1],honeycomb_fu_bond_groups(A)[3][1]),policy in (:fixed_space,:tensorkit)
        left,right,r=honeycomb_full_update_bond(A,env,gate,b;settings,space_policy=policy);
        @test r.loss_final<=r.loss_initial+1e-10
        @test 0<=r.fidelity<=1+1e-8
        @test norm(left)≈1 && norm(right)≈1
        @test dim(space(left,b.direction==:x ? 3 : 2))==(policy==:fixed_space ? 1 : 2)
    end
end
