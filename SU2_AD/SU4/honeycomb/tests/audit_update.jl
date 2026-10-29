using Test, LinearAlgebra, TensorKit, SUNRepresentations
BLAS.set_num_threads(1);
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
using .HoneycombSU4
const HC=HoneycombSU4;

function pair_array(psi)
    return reshape(permutedims(convert(Array,psi),(7,8,1,2,3,4,5,6)),16,:)
end

@testset "SU4: nontrivial inter-singlet gates in both directions" begin
    for axis in (:horizontal,:vertical),dt in (0.0,0.03)
        irreps=("1","4","6","4⁺");
        vh=[Rep[SU₄](irreps[x]=>1) for x=1:4,y=1:2];
        vv=fill(Rep[SU₄]("1"=>1),4,2);
        T,lx,ly=initial_honeycomb_iPEPS(4,2;symmetry=:SU4,V=(horizontal=vh,vertical=vv));
        bonds=honeycomb_bonds(4,2);
        bond=first(filter(b->b.axis==axis && b.site1==(axis==:horizontal ? (4,1) : (1,1)),bonds));
        before=pair_array(HC.honeycomb_bond_wavefunction(T,lx,ly,bond));
        H=Hamiltonian_honeycomb_SU4(space(T[1],5));
        gate,_=HC.trotter_gate(H,dt);
        x,y=bond.site1;
        if axis==:horizontal
            HC.tebd_xbond(1,T,lx,ly,gate,x+0.5,y,64);
        else
            HC.tebd_ybond(1,T,lx,ly,gate,x,y-0.5,64);
        end
        after=pair_array(HC.honeycomb_bond_wavefunction(T,lx,ly,bond));
        expected=reshape(convert(Array,gate),16,16)*before;
        phase=dot(after,expected)/dot(after,after);
        error=norm(phase*after-expected)/norm(expected);
        println("EXACT ",(axis=axis,dt=dt,relative_error=error));flush(stdout);
        @test error<1e-10;
    end
end

@testset "Backend total-dimension truncation diagnostics" begin
    V=Rep[SU₄]("1"=>1,"4"=>1,"6"=>1);
    S=DiagonalTensorMap(unitary(V,V));
    for (c,b) in blocks(S)
        b[1,1]=dim(c)==4 ? 1.0 : dim(c)==6 ? 0.9 : 0.8;
    end
    _,s,_=HC.tsvd(S;trunc=HC.truncdim(5));
    println("TRUNCATION ",(cap=5,kept_D=dim(space(s,1)),space=space(s,1),
        kept_squared_norm=norm(s)^2,available_four_plus_one=4+0.8^2));flush(stdout);
    @test dim(space(s,1))<=5;
    # This backend currently stops at the first over-budget multiplet.
    # Record the behavior; retaining 4+1 is a strictly better feasible choice.
    @test dim(space(s,1))==4;
    @test norm(s)^2<4+0.8^2;

    Vbig=Rep[SU₄]("1"=>1,"4"=>1,"10"=>1);
    Sbig=DiagonalTensorMap(unitary(Vbig,Vbig));
    for (c,b) in blocks(Sbig)
        b[1,1]=dim(c)==10 ? 1.2 : dim(c)==4 ? 1.0 : 0.8;
    end
    _,sbig,_=HC.tsvd(Sbig;trunc=HC.truncdim(9));
    println("EMPTY ",(cap=9,kept_D=dim(space(sbig,1)),space=space(sbig,1)));flush(stdout);
    @test dim(space(sbig,1))==0;
end
