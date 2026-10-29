# Tensor order: (L,D,R,U,physical), as in triangle_SimpleUpdate_iPEPS.jl.
# y increases UP in SU coordinates. CTMRG coordinates are reflected on export.
# All horizontal bonds exist. D(x,y)--U(x,y-1) exists iff iseven(x+y).
# The other vertical bonds are trivial representations with dimension one.

"""Initial virtual irreps and multiplicities, e.g. ["1"=>1,"4"=>1,"6"=>1,"4⁺"=>1]."""
function honeycomb_virtual_space(sectors)
    return Rep[SU₄](sectors...)
end

function honeycomb_bonds(Lx,Ly)
    Lx>=2 && Ly>=2 && iseven(Lx) && iseven(Ly) ||
        throw(ArgumentError("brick-wall periodic cells require even Lx,Ly >= 2"));
    bonds=NamedTuple[];
    for cx=1:Lx,cy=1:Ly
        push!(bonds,(direction=iseven(cx+cy) ? :x : :y,axis=:horizontal,
            site1=(cx,cy),site2=(mod1(cx+1,Lx),cy)));
        if iseven(cx+cy)
            push!(bonds,(direction=:z,axis=:vertical,
                site1=(cx,cy),site2=(cx,mod1(cy-1,Ly))));
        end
    end
    return bonds
end

"""
initial_honeycomb_iPEPS(Lx,Ly; symmetry=:dense,D=2,V=nothing,seed=1234,uniform=false)

SU(4): pass V=Rep[SU₄](...) or V=(x=Vx,y=Vy,z=Vz).
For site-dependent sectors, V=(horizontal=Vh,vertical=Vv) contains Lx×Ly
matrices: Vh[x,y] is the LEFT bond of (x,y); Vv[x,y] is its DOWN bond.
Missing vertical bonds are set to the trivial irrep regardless of Vv.
All physical sites carry the fundamental 4, never the conjugate 4⁺.
uniform=true copies the two parity tensors and requires direction-only V.
"""
function initial_honeycomb_iPEPS(Lx,Ly; symmetry=:dense,D=2,V=nothing,seed=1234,uniform=false)
    honeycomb_bonds(Lx,Ly);
    symmetry in (:dense,:SU4) || throw(ArgumentError("symmetry must be :dense or :SU4"));
    D>0 || throw(ArgumentError("D must be positive"));
    if symmetry==:dense
        isnothing(V) || throw(ArgumentError("use D for dense initialization"));
        Vp=ComplexSpace(4); V0=ComplexSpace(1); V=ComplexSpace(D);
    else
        isnothing(V) && throw(ArgumentError("specify initial SU(4) virtual irreps in V"));
        Vp=Rep[SU₄]("4"=>1); V0=Rep[SU₄]("1"=>1);
    end
    Vh=Matrix{Any}(undef,Lx,Ly); Vd=similar(Vh);
    cell_spaces=V isa NamedTuple && haskey(V,:horizontal);
    if cell_spaces
        size(V.horizontal)==(Lx,Ly) && size(V.vertical)==(Lx,Ly) ||
            throw(ArgumentError("virtual-space matrices must have size (Lx,Ly)"));
        uniform && throw(ArgumentError("uniform initialization requires direction-only virtual spaces"));
    end
    for cx=1:Lx,cy=1:Ly
        if cell_spaces
            Vh[cx,cy]=V.horizontal[cx,cy];
            Vd[cx,cy]=iseven(cx+cy) ? V.vertical[cx,cy] : V0;
        elseif V isa NamedTuple
            Vh[cx,cy]=iseven(cx-1+cy) ? V.x : V.y;
            Vd[cx,cy]=iseven(cx+cy) ? V.z : V0;
        else
            Vh[cx,cy]=V;
            Vd[cx,cy]=iseven(cx+cy) ? V : V0;
        end
    end
    rng=MersenneTwister(seed);
    T_set=Matrix{Any}(undef,Lx,Ly);
    lambdax_set=similar(T_set); lambday_set=similar(T_set);
    templates=Dict{Bool,Any}();
    for cx=1:Lx,cy=1:Ly
        Vl=Vh[cx,cy]; Vdown=Vd[cx,cy];
        Vr=Vh[mod1(cx+1,Lx),cy]; Vu=Vd[cx,mod1(cy+1,Ly)];
        all(W->sectortype(W)==sectortype(Vp),(Vl,Vdown,Vr,Vu)) ||
            throw(ArgumentError("virtual spaces must have the same symmetry as the physical space"));
        T=TensorMap{ComplexF64}(undef,Vl*Vdown*Vr'*Vu',Vp');
        # Fill the allowed blocks with a local RNG (no global Random.seed!).
        for (_,b) in blocks(T)
            randn!(rng,b);
        end
        norm(T)>0 || throw(ArgumentError("no invariant SU(4) tensor at ($cx,$cy): incompatible virtual irreps"));
        T=permute(T,(1,2,3,4,5,)); T=T/norm(T);
        if uniform
            parity=iseven(cx+cy);
            T=copy(get!(templates,parity,T));
        end
        T_set[cx,cy]=T;
        lambdax_set[cx,cy]=DiagonalTensorMap(unitary(Vl,Vl));
        lambday_set[cx,cy]=DiagonalTensorMap(unitary(Vdown',Vdown'));
    end
    return T_set,lambdax_set,lambday_set
end

"""Nearest-neighbor H=J(P_ij-I/4), or J*P_ij with convention=:permutation."""
function Hamiltonian_honeycomb_SU4(Vp;J=1.0,convention=:traceless)
    convention in (:traceless,:permutation) || throw(ArgumentError("unknown energy convention"));
    dim(Vp)==4 || throw(ArgumentError("physical dimension must be 4"));
    I2=unitary(Vp*Vp,Vp*Vp);
    Pij=permute(I2,(2,1,),(3,4,));
    return J*(Pij-(convention==:traceless ? I2/4 : 0*I2))
end

"""
Second-order two-site simple update. T contains sqrt(s) on each end of a bond,
and lambdax/y store sqrt(s), exactly as in the reused tebd_xbond/tebd_ybond.
Dmax counts total states, including SU(4) irrep dimensions. Initial sectors
are user supplied; the SVD may change them while preserving whole irreps.
Three bond directions remain independent; no C3 projection is imposed.
step_offset continues the direction-order cycle across checkpoint chunks.
truncation=:optimal maximizes the retained norm within Dmax using whole irreps.
truncation=:legacy reproduces the previous TensorKit rank-budget policy.
"""
function itebd_honeycomb(parameters,T_set,lambdax_set,lambday_set,tau,dt,Dmax;
                          convention=:traceless,rotate_order=true,verbose=true,
                          truncation=:optimal,step_offset=0)
    dt>0 && tau>=0 && Dmax>=1 || throw(ArgumentError("require dt>0, tau>=0, Dmax>=1"));
    truncation in (:optimal,:legacy) || throw(ArgumentError("use truncation=:optimal or :legacy"));
    budget=truncation==:optimal ? WholeMultipletBudget(Dmax) : Dmax;
    steps=round(Int,tau/dt);
    isapprox(steps*dt,tau;atol=1e-12,rtol=1e-10) || throw(ArgumentError("tau must be an integer multiple of dt"));
    Lx,Ly=size(T_set); bonds=honeycomb_bonds(Lx,Ly);
    size(lambdax_set)==size(lambday_set)==size(T_set) || throw(DimensionMismatch("inconsistent cells"));
    H=Hamiltonian_honeycomb_SU4(space(T_set[1],5);J=parameters["J"],convention=convention);
    if iszero(norm(H))
        gate=unitary(codomain(H),domain(H)); gate_half=gate;
    else
        gate,gate_half=trotter_gate(H,dt);
    end
    directions=(:x,:y,:z);
    for ct=1:steps
        offset=rotate_order ? mod(ct+step_offset-1,3) : 0;
        a,b,c=ntuple(k->directions[mod1(k+offset,3)],3);
        for (direction,g) in ((a,gate_half),(b,gate_half),(c,gate),(b,gate_half),(a,gate_half))
            for bond in bonds
                bond.direction==direction || continue;
                cx,cy=bond.site1;
                if bond.axis==:horizontal
                    tebd_xbond(1,T_set,lambdax_set,lambday_set,g,cx+0.5,cy,budget);
                else
                    tebd_ybond(1,T_set,lambdax_set,lambday_set,g,cx,cy-0.5,budget);
                end
            end
        end
        if verbose && (ct==1 || mod(ct,20)==0 || ct==steps)
            println("honeycomb SU: iteration ",ct,"/",steps,", dt=",dt,", Dmax=",Dmax);flush(stdout);
        end
    end
    return T_set,lambdax_set,lambday_set
end

function convert_honeycomb_to_iPEPS(T_set)
    nx,ny=size(T_set);
    # SU D points toward y-1; the old CTMRG D points toward cy+1.
    return ntuple(cx->ntuple(cy->T_set[cx,mod1(2-cy,ny)],ny),nx)
end
