using Test,LinearAlgebra,TensorKit,SUNRepresentations,Random
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
if !isdefined(@__MODULE__,:HC);const HC=HoneycombSU4;end
BLAS.set_num_threads(1);

function star_relative_error(a,b)
    scale=dot(a,b)/dot(a,a);
    return norm(scale*a-b)/norm(b)
end

# Independent contraction into the OLD external residual basis. This checks
# the lambda/gauge conversion, not a newly computed basis after the update.
function star_in_reference_basis(T,g,prepared)
    C=HC._star_strip(T[g.center...],(g.legs...,5));K=Any[];
    for i=1:3
        B=HC._star_strip(T[g.leaves[i]...],prepared.orders[i]);
        W=prepared.weights[i];
        B=HC._star_mode_product(HC._star_mode_product(B,W[1],1),W[2],2);
        Q=permute(prepared.residuals[i],((1,2),(3,)))';
        @tensor reduced[:]:=Q[-1,a,b]*B[a,b,-2,-3];
        push!(K,reduced);
    end
    K1,K2,K3=K;
    @tensor theta[:]:=C[a,b,c,-1]*K1[-2,-3,a]*K2[-4,-5,b]*K3[-6,-7,c];
    return theta
end

function star_reconstruct(core,U)
    U1,U2,U3=U;
    @tensor theta[:]:=core[-1,a,b,c]*U1[-2,-3,a]*U2[-4,-5,b]*U3[-6,-7,c];
    return theta
end

function star_connections_match(T,lx,ly)
    for b in HC.honeycomb_bonds(size(T)...)
        i,j=b.axis==:horizontal ? (3,1) : (2,4);
        space(T[b.site1...],i)'==space(T[b.site2...],j) || return false;
    end
    for x=1:size(T,1),y=1:size(T,2),leg in (1,3,iseven(x+y) ? 2 : 4)
        W=HC._star_weight(lx,ly,(x,y),leg);
        space(W,1)==space(T[x,y],leg) || return false;
        space(W,2)'==space(T[x,y],leg) || return false;
    end
    return true
end

@testset "Star coverings and bond multiplicity" begin
    groups=HC.honeycomb_star_groups(4,4);
    @test length(groups)==4 && all(length(g)==4 for g in groups)
    counts=Dict{Any,Int}();
    for centers in groups
        sites=Tuple{Int,Int}[];
        for c in centers
            g=HC.honeycomb_star_geometry(4,4,c);append!(sites,(c,g.leaves...));
            for leaf in g.leaves
                edge=Set((c,leaf));counts[edge]=get(counts,edge,0)+1;
            end
        end
        @test length(unique(sites))==16
    end
    @test length(counts)==24 && all(==(2),values(counts))
    @test_throws ArgumentError HC.honeycomb_star_groups(2,2)
end

@testset "Kagome-style simultaneous projectors and C3 covariance" begin
    rng=MersenneTwister(81);p=ComplexSpace(4);r=ComplexSpace(2);
    V=p*r*p*r*p*r*p;
    A=TensorMap(randn(rng,ComplexF64,dim(V)),V,one(V));
    rotate(A)=permute(A,(1,4,5,6,7,2,3));
    theta=A+rotate(A)+rotate(rotate(A));theta/=norm(theta);
    C,U,S,retained=HC.honeycomb_star_hosvd(theta,3;trun_tol=0.);
    @test norm(C)≈1
    @test all(norm(s)≈1 for s in S)
    @test 0<retained<=1+1e-12
    projected=star_reconstruct(C,U);
    @test norm(projected-rotate(projected))<1e-10
    reference=Any[];
    for left in ((2,3),(4,5),(6,7))
        right=Tuple(k for k=1:7 if !(k in left));
        u,s,_=HC.tsvd(permute(theta,(left,right));trunc=HC.truncdim(3));
        push!(reference,u);
    end
    U1,U2,U3=reference;
    @tensor G[:]:=theta[-1,a,b,c,d,e,f]*U1'[-2,a,b]*U2'[-3,c,d]*U3'[-4,e,f];
    @test star_relative_error(star_reconstruct(G,reference),projected)<1e-12
end

@testset "Dense exact gate and nonuniform lambda bookkeeping" begin
    T,lx,ly=HC.initial_honeycomb_iPEPS(4,4;D=2,seed=18);
    for (j,L) in enumerate(vcat(vec(lx),vec(ly)))
        for (_,b) in blocks(L),k=1:size(b,1);b[k,k]=.3+.07*k+.01*j;end
    end
    g=HC.honeycomb_star_geometry(4,4,(2,1));
    theta,prepared=HC._star_prepare(T,lx,ly,g);
    H=HC.honeycomb_star_hamiltonian(space(T[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.01);expected=HC._star_apply_gate(theta,gate);
    report=HC.honeycomb_star_update!(T,lx,ly,gate,g.center,nothing;trun_tol=1e-12);
    actual=star_in_reference_basis(T,g,prepared);
    @test star_relative_error(actual,expected)<1e-10
    @test report.retained_norm_squared≈1 atol=1e-12
    @test all(norm(T[s...])≈1 for s in (g.center,g.leaves...))
    @test isnothing(HC._check_honeycomb_state(T,lx,ly))
    @test star_connections_match(T,lx,ly)
end

@testset "SU4 D9 identity: both brick-wall parities and unequal conjugate weights" begin
    for center in ((1,1),(2,1))
        T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
        @test all(dim(space(L,1))==9 for L in lx)
        for L in vcat(vec(lx),vec(ly)),(c,b) in blocks(L)
            q=mod(sum(k*a for (k,a) in enumerate(SUNRepresentations.dynkin_label(c))),4);
            for k=1:size(b,1);b[k,k]=q==0 ? .9 : q==1 ? .6 : .3;end
        end
        g=HC.honeycomb_star_geometry(4,4,center);
        expected,prepared=HC._star_prepare(T,lx,ly,g);
        report=HC.honeycomb_star_update!(T,lx,ly,nothing,center,9;trun_tol=1e-10);
        @test star_relative_error(star_in_reference_basis(T,g,prepared),expected)<1e-9
        @test report.retained_norm_squared≈1 atol=1e-10
        @test all(d<=9 for d in report.dimensions)
        @test isnothing(HC._check_honeycomb_state(T,lx,ly;maxdim=9))
        @test star_connections_match(T,lx,ly)
    end
end

@testset "SU4 physical gate from D9 uses joint HOSVD projection" begin
    T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    g=HC.honeycomb_star_geometry(4,4,(1,1));theta,prepared=HC._star_prepare(T,lx,ly,g);
    H=HC.honeycomb_star_hamiltonian(space(T[1],5);convention=:permutation);
    gate,_=HC.trotter_gate(H,.001);
    target=HC._star_apply_gate(theta,gate);
    C,U,S,_=HC.honeycomb_star_hosvd(target,9);
    projected=star_reconstruct(C,U);
    HC.honeycomb_star_update!(T,lx,ly,gate,g.center,9);
    @test star_relative_error(star_in_reference_basis(T,g,prepared),projected)<1e-9
    @test isnothing(HC._check_honeycomb_state(T,lx,ly;maxdim=9))
    @test star_connections_match(T,lx,ly)
end
