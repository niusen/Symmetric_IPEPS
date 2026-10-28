using Test, LinearAlgebra, TensorKit, SUNRepresentations
BLAS.set_num_threads(1);
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
using .HoneycombSU4
const HC=HoneycombSU4;

function pair_matrix(psi)
    # Physical indices first, external indices afterwards.
    data=permutedims(convert(Array,psi),(7,8,1,2,3,4,5,6));
    return reshape(data,16,:)
end

function normalized_overlap(a,b)
    return abs(dot(a,b))/(norm(a)*norm(b))
end

@testset "Honeycomb topology and product-state energy" begin
    bonds=honeycomb_bonds(4,2);
    @test length(bonds)==12;
    for d in (:x,:y,:z)
        layer=filter(b->b.direction==d,bonds);
        @test length(layer)==4;
        @test length(unique(vcat([b.site1 for b in layer],[b.site2 for b in layer])))==8;
    end
    @test_throws ArgumentError honeycomb_bonds(3,2);
    T,lx,ly=initial_honeycomb_iPEPS(2,2;D=1);
    for k in eachindex(T)
        t=T[k];
        data=convert(Array,t); fill!(data,0); data[1,1,1,1,1]=1;
        T[k]=TensorMap(data,codomain(t),domain(t));
    end
    p=Dict("J"=>1.0);
    ob=honeycomb_energy_simple(p,T,lx,ly);
    @test ob.energy_per_site≈9/8;
    @test ob.bond_energy.x≈3/4;
    @test honeycomb_energy_simple(p,T,lx,ly;convention=:permutation).energy_per_site≈3/2;
    @test honeycomb_energy_simple(Dict("J"=>0.0),T,lx,ly).energy_per_site==0;
end

@testset "Reused two-site SU equals exact evolution without truncation" begin
    for direction in (:x,:y,:z)
        T,lx,ly=initial_honeycomb_iPEPS(2,2;D=2,seed=17);
        bond=first(filter(b->b.direction==direction,honeycomb_bonds(2,2)));
        before=pair_matrix(HC.honeycomb_bond_wavefunction(T,lx,ly,bond));
        H=Hamiltonian_honeycomb_SU4(space(T[1],5));
        g,_=HC.trotter_gate(H,0.03);
        x,y=bond.site1;
        if bond.axis==:horizontal
            HC.tebd_xbond(1,T,lx,ly,g,x+0.5,y,64);
        else
            HC.tebd_ybond(1,T,lx,ly,g,x,y-0.5,64);
        end
        after=pair_matrix(HC.honeycomb_bond_wavefunction(T,lx,ly,bond));
        expected=reshape(convert(Array,g),16,16)*before;
        @test normalized_overlap(after,expected)>1-1e-10;
    end
end

@testset "SU(4) invariant spaces, complete multiplets, and operators" begin
    V=honeycomb_virtual_space(["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1]);
    @test dim(V)==15;
    T,lx,ly=initial_honeycomb_iPEPS(2,2;symmetry=:SU4,V=V,seed=8);
    @test all(t->norm(t)≈1,T);
    @test all(t->space(t,5)==Rep[SU₄]("4"=>1),T);
    @test all(t->minimum(dim(space(t,k)) for k=1:4)==1,T);
    @test_throws ArgumentError initial_honeycomb_iPEPS(2,2;symmetry=:SU4);
    @test_throws ArgumentError initial_honeycomb_iPEPS(2,2;symmetry=:SU4,V=Rep[SU₄]("1"=>1));
    Sa,Sb=HC.SUN_spin(4)[1:2];
    @tensor Hold[:]:=Sa[1,2,-1,-3]*Sb[2,1,-2,-4];
    H=Hamiltonian_honeycomb_SU4(space(T[1],5));
    @test norm(HC.permute(Hold,(1,2,),(3,4,))-H)<1e-11;
    @test eigvals(Hermitian(reshape(convert(Array,H),16,16)))≈vcat(fill(-1.25,6),fill(0.75,10));
    # A small, nonzero four-fundamental singlet per horizontal chain.
    # Bond N-alities advance 0 -> 1 -> 2 -> 3 -> 0 along x.
    irreps=("1","4","6","4⁺");
    Vh=[Rep[SU₄](irreps[x]=>1) for x=1:4,y=1:2];
    Vv=fill(Rep[SU₄]("1"=>1),4,2);
    Ts,xs,ys=initial_honeycomb_iPEPS(4,2;symmetry=:SU4,V=(horizontal=Vh,vertical=Vv));
    ob=honeycomb_energy_simple(Dict("J"=>1.0),Ts,xs,ys);
    @test isfinite(ob.energy_per_site);
    # Update a horizontal edge; comparison is made on the same external spaces.
    bond=first(honeycomb_bonds(4,2));
    before=pair_matrix(HC.honeycomb_bond_wavefunction(Ts,xs,ys,bond));
    g,_=HC.trotter_gate(H,0.02);
    HC.tebd_xbond(1,Ts,xs,ys,g,1.5,1,32);
    after=pair_matrix(HC.honeycomb_bond_wavefunction(Ts,xs,ys,bond));
    @test normalized_overlap(after,reshape(convert(Array,g),16,16)*before)>1-1e-10;
    @test sectortype(xs[2,1])==sectortype(V);
    itebd_honeycomb(Dict("J"=>1.0),Ts,xs,ys,0.01,0.01,16;verbose=false);
    @test all(t->isfinite(norm(t)),Ts);
    @test all(dim(space(s,1))<=16 for s in vcat(vec(xs),vec(ys)));
end

@testset "Dense SU sweep and CTMRG product-state normalization" begin
    T,lx,ly=initial_honeycomb_iPEPS(2,2;D=1,seed=22);
    setting=LS_CTMRG_settings(CTM_ite_nums=6,CTM_conv_tol=1e-9,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    simple=honeycomb_energy_simple(Dict("J"=>1.0),T,lx,ly);
    ctm=honeycomb_energy_CTM(Dict("J"=>1.0),T,4,setting);
    @test ctm.energy_per_site≈simple.energy_per_site atol=1e-10;
    @test collect(values(ctm.bond_energy))≈collect(values(simple.bond_energy)) atol=1e-10;
    # A 4x4 product state also checks the SU -> CTMRG y reflection.
    T4,lx4,ly4=initial_honeycomb_iPEPS(4,4;D=1,seed=24);
    local4=honeycomb_energy_simple(Dict("J"=>1.0),T4,lx4,ly4);
    ctm4=honeycomb_energy_CTM(Dict("J"=>1.0),T4,4,setting);
    @test ctm4.energies≈local4.energies atol=1e-10;
    itebd_honeycomb(Dict("J"=>1.0),T,lx,ly,0.02,0.01,2;verbose=false);
    @test all(isfinite(norm(t)) for t in T);
    for x=1:2,y=1:2
        @test dim(space(T[x,y],iseven(x+y) ? 4 : 2))==1;
    end
    @test isfinite(honeycomb_energy_simple(Dict("J"=>1.0),T,lx,ly).energy_per_site);
    itebd_honeycomb(Dict("J"=>0.0),T,lx,ly,0.01,0.01,2;verbose=false);
end

@testset "SU(4) CTMRG: exact four-site singlet chains" begin
    irreps=("1","4","6","4⁺");
    Vh=[Rep[SU₄](irreps[x]=>1) for x=1:4,y=1:4];
    Vv=fill(Rep[SU₄]("1"=>1),4,4);
    T,lx,ly=initial_honeycomb_iPEPS(4,4;symmetry=:SU4,V=(horizontal=Vh,vertical=Vv));
    setting=LS_CTMRG_settings(CTM_ite_nums=12,CTM_conv_tol=1e-9,
        CTM_ite_info=false,CTM_conv_info=true,grad_checkpoint=false);
    ob=honeycomb_energy_CTM(Dict("J"=>1.0),T,36,setting);
    # Each disconnected row is a four-fundamental singlet, with three internal
    # bonds at -5/4 and one product boundary bond at zero. All z bonds are zero.
    @test ob.energy_per_site≈-15/16 atol=1e-9;
    @test ob.bond_energy.z≈0 atol=1e-9;
    @test all(isfinite,ob.energies);
end
