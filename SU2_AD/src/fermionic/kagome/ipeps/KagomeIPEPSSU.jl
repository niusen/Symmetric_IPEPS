module KagomeIPEPSSU
using TensorKit, LinearAlgebra, JLD2
using ..KagomeFermionicZ2
using ..KagomeGFPEPSValidation
using ..KagomeSpinOptimization: HeisenbergField
const KF=KagomeFermionicZ2
export run_native_su, SquareState, ipeps_su!, ctm_cell, run_ipeps_su, load_square, save_square

"""CTM-compatible A[L,D,R,U,p] carries half of each link weight.
weight[x,y,i] is the other half used as the SU metric on that external leg.
"""
struct SquareState
    A::Matrix{TensorMap}
    weight::Array{TensorMap,3}
    fusion::TensorMap
end
"""Tile a native state periodically, with independent tensors and SU weights in each copy."""
function tile_su_cell(s,unitcell::Tuple{Int,Int})
    nx,ny=s isa SquareState ? size(s.A) : size(s.T_u)
    lx,ly=unitcell
    lx>0 && ly>0 && lx%nx==0 && ly%ny==0 ||
        throw(ArgumentError("requested cell $unitcell must be an integer tiling of $((nx,ny))"))
    # A square down-triangle gate writes to three distinct square tensors.
    s isa SquareState && min(lx,ly)<2 && throw(ArgumentError("square SU needs at least two rows and columns"))
    unitcell==(nx,ny) && return s
    rx,ry=lx÷nx,ly÷ny
    tile(A::AbstractMatrix)=copy.(repeat(A,rx,ry))
    tile(A::AbstractArray{<:Any,3})=copy.(repeat(A,rx,ry,1))
    if s isa SquareState
        return SquareState(tile(s.A),tile(s.weight),s.fusion)
    end
    typeof(s)(tile(s.T_u),tile(s.T_d),tile(s.B),tile(s.lambda_u),tile(s.lambda_d))
end

function SquareState(cell,U;unitcell=nothing)
    nx,ny=length(cell),length(first(cell))
    all(length(col)==ny for col in cell) || throw(ArgumentError("ragged cell"))
    A=Matrix{TensorMap}(undef,nx,ny); w=Array{TensorMap}(undef,nx,ny,4)
    for x in 1:nx,y in 1:ny
        a=A[x,y]=cell[x][y]
        space(a,5)==space(U,1) || throw(SpaceMismatch("physical fusion"))
        for j in 1:4
            v=space(a,j); w[x,y,j]=unitary(v,v)
        end
    end
    tile_su_cell(SquareState(A,w,U),something(unitcell,(max(2,nx),max(2,ny))))
end
ctm_cell(s::SquareState)=ntuple(x->Tuple(s.A[x,:]),size(s.A,1))
function save_square(path,s::SquareState;metadata=(;))
    # Save fields rather than Julia module-qualified state types.
    jldsave(path;format="kagome_native_ipeps_su_v1",A=s.A,weight=s.weight,fusion=s.fusion,metadata)
end
function load_square(path)
    r=load(path); r["format"]=="kagome_native_ipeps_su_v1" || error("not a native iPEPS SU checkpoint")
    SquareState(r["A"],r["weight"],r["fusion"])
end

plain(a,perm)=TensorKit.permute(a,(perm,()))
matrix(a,l,r)=TensorKit.permute(a,(l,r))
# The old square CTM stores (L,D,R,U,p) in memory, but its fermionic order
# is (L,U,p,R,D). The first index relabeling is therefore ungraded.
function unpack(a,U)
    b=plain(a,(1,4,5,3,2))
    @tensor c[-1,-2,-3,-4,-5,-6,-7] := b[-1,-2,1,-6,-7]*conj(U[1,-3,-4,-5])
    KF._allout(c)
end
function pack(c,U)
    @tensor b[-1,-2,-3,-4,-5] := c[-1,-2,1,2,3,-4,-5]*U[-3,1,2,3]
    plain(b,(1,5,4,2,3))
end

# Path LD.c -> LU.b -> RU.a, as in evo_hopping_LU_RU_LD. Expose only
# one microscopic spin/orbital per block; the two spectators are QR-compressed
# with all external legs. No gate on three 64-dimensional blocks is formed.
const ROUTE=((1,3,4,6,7,5,2), (7,4,1,2,3,5,6), (1,3,2,4,5,6,7))
const EXT=((1,2,3,4,5),(3,4,5,6),(3,4,5,6,7))
const KEEP=((6,7),(1,2,7),(1,2))
const CORE_ORDER=((1,2,3),(2,3,1,4),(2,3,1))
function reduce_block(a,U,k)
    b=KF.fpermute(unpack(a,U),ROUTE[k])
    block=matrix(b,EXT[k],KEEP[k])
    q,r = if isdefined(TensorKit,:left_orth)
        TensorKit.left_orth(block)
    else
        TensorKit.leftorth(block)
    end
    q,plain(r,CORE_ORDER[k])
end
function restore(q,c,U,k)
    t=plain(c,Tuple(invperm(collect(CORE_ORDER[k]))))
    b=q*matrix(t,(1,),Tuple(2:KF._rank(t)))
    b=plain(b,Tuple(invperm(collect((EXT[k]...,KEEP[k]...)))))
    pack(KF.fpermute(b,Tuple(invperm(collect(ROUTE[k])))),U)
end
function parity(a,i)
    V=space(a,i)
    KF._leg_map(a,unitary(V,V)-2KF._odd_projector(V),i)
end
function metric(s,xy,axes,a;inverse=false)
    for j in axes
        w=s.weight[xy...,j]
        a=KF._leg_map(a,inverse ? KF._pinv_weight(w) : w,j)
    end
    a
end
function dualweight(S,V)
    w=TensorMap(sqrt(S/norm(S)))
    space(w,1)==V && return w
    space(w,1)'==V || throw(SpaceMismatch("bond weight"))
    TensorKit.permute(w,((2,),(1,)))
end
# Reverse (a,b,c) -> (c,b,a) on both sides as a graded many-body basis
# transformation. Permuting bra and ket independently avoids bending signs.
function reverse_gate(g,P)
    I3=unitary(P*P*P,P*P*P)
    sw=KF.fpermute(I3,(3,2,1,4,5,6))
    F=matrix(sw,(1,2,3),(4,5,6))
    F*g*F'
end
down_blocks(x,y,lx,ly)=((x,mod1(y+1,ly)),(x,y),(mod1(x+1,lx),y))

"""Group down triangles with disjoint square tensors, including across periodic boundaries."""
function down_layers(lx,ly)
    layers=Vector{CartesianIndex{2}}[]
    occupied=BitMatrix[]
    for site in CartesianIndices((lx,ly))
        vertices=down_blocks(Tuple(site)...,lx,ly)
        layer=findfirst(mask->all(p->!mask[p...],vertices),occupied)
        if isnothing(layer)
            push!(layers,CartesianIndex{2}[])
            push!(occupied,falses(lx,ly))
            layer=length(layers)
        end
        push!(layers[layer],site)
        for p in vertices
            occupied[layer][p...]=true
        end
    end
    layers
end

function down!(s,x,y,g,Dmax,tol)
    pos=down_blocks(x,y,size(s.A)...)
    external=((1,2,3),(1,4),(2,3,4))
    qr=ntuple(k->reduce_block(metric(s,pos[k],external[k],s.A[pos[k]...]),s.fusion,k),3)
    l,m,r=last.(qr)
    # Legacy vertical incoming-edge parity. It is removed again on reconstruction.
    l=parity(l,3)
    @tensor z[-1,-2,-3,-4,-5,-6] := l[-1,-2,1]*m[1,-3,-4,2]*r[2,-5,-6]
    z=KF.fpermute(z,(1,2,3,5,4,6)) # eC,c,b,a,eB,eA
    @tensor t[-1,-2,-3,-4,-5,-6] := z[-1,1,2,3,-5,-6]*g[-2,-3,-4,1,2,3]
    t=KF.fpermute(t,(1,2,3,5,4,6)) # eC,c,b,eB,a,eA
    M=matrix(t,(1,2),(3,4,5,6))
    u,d,v=KF._svd(M,Dmax,tol)
    M2=matrix(d*v,(1,2,3),(4,5))
    u2,d2,v2=KF._svd(M2,Dmax,tol)
    # Sequential SVD keeps the first spectrum during the second truncation.
    # Move half of each spectrum to each endpoint, also undoing the old metric.
    l=parity(KF._allout(u*sqrt(d)),3)
    m=KF._leg_map(KF._allout(u2*sqrt(d2)),KF._pinv_weight(TensorMap(sqrt(d))),1)
    r=KF._allout(sqrt(d2)*v2)
    raw=ntuple(k->metric(s,pos[k],external[k],restore(first(qr[k]),(l,m,r)[k],s.fusion,k);inverse=true),3)
    for k in 1:3
        s.A[pos[k]...]=KF._normalized(raw[k])
    end
    for (k,j,S) in ((1,4,d),(2,2,d),(2,3,d2),(3,1,d2))
        s.weight[pos[k]...,j]=dualweight(S,space(raw[k],j))
    end
    clamp(1-(norm(d2)/norm(M))^2,0.,1.)
end

function ipeps_su!(s::SquareState,model,tau,dt,Dmax;trun_tol=1e-10,callback=nothing)
    isfinite(dt) && dt>0 && isfinite(tau) && tau>=0 && Dmax>0 && 0<=trun_tol<1 || throw(ArgumentError("invalid SU settings"))
    n=round(Int,tau/dt); isapprox(n*dt,tau;atol=1e-12,rtol=1e-10) || throw(ArgumentError("tau must be a multiple of dt"))
    nx,ny=size(s.A); P=domain(s.fusion)[1]
    ng=KF.magnetic_period(model)
    nx%ng!=0 && throw(ArgumentError("parton gauge requires three columns"))
    up=s.fusion*first(trotter_gate(KF.triangle_at(model,:up,1),dt))*s.fusion'
    down=[reverse_gate(last(trotter_gate(KF.triangle_at(model,:dn,x),dt)),P) for x in 1:ng]
    layers=down_layers(nx,ny); history=NamedTuple[]
    for step in 1:n
        e=0.0
        for pass in 1:2
            for layer in (pass==1 ? layers : Iterators.reverse(layers)), site in layer
                x,y=Tuple(site); e=max(e,down!(s,x,y,down[mod1(x,ng)],Dmax,trun_tol))
            end
            if pass==1
                for site in CartesianIndices(s.A)
                    s.A[site]=KF._normalized(KF._leg_map(s.A[site],up,5))
                end
            end
        end
        r=(step=step,tau=step*dt,discarded_weight=e);push!(history,r)
        isnothing(callback) || callback(s,r)
    end
    history
end

function square_measure(s,model;chi,maxiter,tol,verbose)
    A,U=s isa SquareState ? (ctm_cell(s),s.fusion) :
        (KF.convert_iPESS_to_iPEPS(s),last(KF.build_A_kagome(s,1,1)))
    nx,ny=length(A),length(first(A))
    if model isa HeisenbergField
        r=measure_heisenberg(A,U;J=model.J,h=model.h,chi,maxiter,tol,verbose)
        return (;r.energy_per_site,r.exchange_per_site,r.magnetization_per_site,r.mz_over_saturation,
                cell=(nx,ny),r.magnetization,r.up_bonds,r.down_bonds,r.iterations,r.ctm_error,r.converged)
    else
        r=measure_gaussian_cell(A,U,model.mu;model,chi,maxiter,tol,verbose)
        max_virtual_D=s isa SquareState ? maximum(dim(space(a,j)) for a in s.A for j in 1:4) :
                                          maximum(dim(space(b,j)) for b in s.B for j in 1:2)
        # Existing hopping arrays use b=(f_up,f_down†); keep them for compatibility.
        # Density and bond kinetic energies below are for physical electrons.
        function bond_energy(hop,orientation)
            [2real(parton_hoppings(model,orientation,mod1(i,nx))[e][σ]*hop[i,e,σ])
             for i in axes(hop,1), e in 1:3, σ in 1:2]
        end
        return (;energy_per_site=r.grand_per_site,r.energy_without_field_per_site,r.kinetic_per_site,
                r.n_up_per_site,r.n_down_per_site,r.up_hoppings,r.hoppings,
                n_up_sites=real.(r.density[:,:,1]),n_down_sites=1 .- real.(r.density[:,:,2]),
                up_bond_energy=bond_energy(r.up_hoppings,:up),down_bond_energy=bond_energy(r.hoppings,:dn),
                cell=(nx,ny),max_virtual_D,chi,r.iterations,r.ctm_error,r.converged)
    end
end

"""Optimize a native iPEPS or iPESS MAT seed, or resume its JLD2 bond weights.
A model is chosen explicitly even on restart (changing the field is allowed).
The optional `unitcell=(Lx,Ly)` periodically tiles the seed before the initial measurement.
"""
function run_native_su(family,input,output,kind;schedule,Dmax,J=1.,h=0.,mu=nothing,t=(1.,1.),
                       onsite_U=0.,chi=16,maxiter=100,tol=1e-7,trun_tol=1e-10,verbose=true,provenance=(;),unitcell=nothing)
    family in (:ipeps,:ipess) || throw(ArgumentError("invalid tensor family"))
    square=family===:ipeps
    kind=Symbol(kind); spin=kind===:spin
    kind in (:spin,:orig,:mild,:zero,:hubbard) || throw(ArgumentError("unknown model: $kind"))
    !iszero(onsite_U) && kind!==:hubbard && throw(ArgumentError("onsite_U requires :hubbard"))
    isfile(input) || throw(ArgumentError("missing input: $input"))
    if endswith(lowercase(input),".mat")
        if square
            cell,U,seed_mu=import_gaussian_ipeps(input;gutzwiller=spin)
            s=SquareState(cell,U;unitcell)
        else
            s,seed_mu=import_gaussian_ipess(input;gutzwiller=spin)
        end
    elseif endswith(lowercase(input),".jld2")
        if square
            s=load_square(input); meta=load(input,"metadata")
        else
            s,_,meta=(spin ? load_projected_checkpoint : KF.load_checkpoint)(input)
        end
        seed_mu=hasproperty(meta,:mu) ? meta.mu : get(meta,"mu",nothing)
        isnothing(seed_mu) && isnothing(mu) && !spin && throw(ArgumentError("set mu for this checkpoint"))
    else
        throw(ArgumentError("expected a native MAT seed or an SU checkpoint for the chosen family"))
    end
    if !isnothing(unitcell)
        s=tile_su_cell(s,unitcell)
    end
    cell_size=square ? size(s.A) : size(s.T_u)
    if square && dim(domain(s.fusion)[1]) != (spin ? 2 : 4)
        throw(ArgumentError("checkpoint physical space does not match model"))
    end
    fields=Tuple(isnothing(mu) ? (isnothing(seed_mu) ? (0.,0.) : seed_mu) : mu)
    model=spin ? HeisenbergField(J,h) : kind===:hubbard ? hole_hubbard(;t,U=onsite_U,mu=fields) :
          parton_model(kind;mu=fields,amplitude=t)
    Dmax>0 && chi>0 && maxiter>0 && tol>0 || throw(ArgumentError("invalid truncation or CTM settings"))
    ispath(output) && throw(ArgumentError("choose a fresh output directory: $output"))
    mkpath(dirname(abspath(output))); mkdir(output)
    meta=(;family,model=kind,mu=fields,t=Tuple(t),J,h,onsite_U,Dmax,chi,maxiter,tol,trun_tol,schedule,input=abspath(input),provenance,unitcell=cell_size)
    save(step)=begin
        path=joinpath(output,"state.jld2"); tmp=path*".tmp"
        md=merge(meta,(;step))
        square ? save_square(tmp,s;metadata=md) : KF.save_checkpoint(tmp,s;metadata=md)
        Base.Filesystem.rename(tmp,path)
    end
    step=0;save(step)
    function observe(name)
        r=square_measure(s,model;chi,maxiter,tol,verbose)
        path=joinpath(output,name*".jld2")
        jldsave(path*".tmp";measurement=r,parameters=meta)
        Base.Filesystem.rename(path*".tmp",path)
        println(name," E/site=",r.energy_per_site," CTM error=",r.ctm_error," converged=",r.converged)
        r
    end
    initial=observe("initial")
    open(joinpath(output,"history.tsv"),"w") do io
        println(io,"step\tdt\tmax_discarded_weight")
        for (tau,dt) in schedule
            cb=function (_,r)
                step+=1
                println(io,step,'\t',dt,'\t',r.discarded_weight);flush(io);save(step)
            end
            (square ? ipeps_su! : KF.itebd!)(s,model,tau,dt,Dmax;trun_tol,callback=cb)
        end
    end
    final=observe("final")
    (;state=s,initial,final,steps=step)
end

run_ipeps_su(args...;kwargs...)=run_native_su(:ipeps,args...;kwargs...)

end
