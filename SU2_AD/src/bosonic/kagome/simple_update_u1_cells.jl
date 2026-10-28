"""Exact SU(2)->U(1) restriction and genuine block-sparse kagome Simple Update."""
module KagomeU1Cells
using TensorKit, LinearAlgebra, JLD2
if !isdefined(parentmodule(@__MODULE__),:KagomeChargedSU2)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"simple_update_charged_cell.jl"))
end
import ..KagomeChargedSU2
const SU2 = KagomeChargedSU2
const BS = SU2.BaseSU

export U1KagomeState, su2_axis_charges, u1_axis_charges, restrict_tensor,
       to_u1, check_u1, itebd_u1, u1_dense_cells, u1_torus_wavefunction,
       save_u1_checkpoint, load_u1_checkpoint, export_u1_portable, u1_heisenberg_triangle,
       u1_torus_observables, expand_u1_rectangular

module Kernels
include(joinpath(@__DIR__,"simple_update_compat.jl"))
function Truncations(U,S,V,Dmax,tolerance)
    # U(1) irreps have dimension one. Do not impose SU(2) multiplet ties.
    data = convert(Dict,S)
    largest = maximum(abs,diag(convert(Array,S)))
    for sector in sectors(space(S,1))
        values = diag(data[:data][string(sector)])
        values[abs.(values) .< tolerance*largest] .= 0
        data[:data][string(sector)] = diagm(values)
    end
    filtered = convert(TensorMap,data)
    a,s,b = tsvd(filtered;trunc=truncerr(1e-12))
    norm(s)>0 || error("U(1) truncation removed the whole bond")
    return U*a,s/norm(s),b*V
end
include(joinpath(@__DIR__,"funs_1up_1down.jl"))
end

struct U1KagomeState
    geometry::BS.KagomeSUCellGeometry
    cells::Vector{BS.KagomeSUTensors}
    auxiliary_charges::Dict{Int,Int}
end

"""Dense-axis q=2Sz labels, including dual signs and irrep multiplicity."""
function su2_axis_charges(V)
    sectortype(V)==SU2Irrep || throw(ArgumentError("expected an SU(2) space"))
    q = zeros(Int,dim(V))
    sign = isdual(V) ? -1 : 1
    for sector in sectors(V)
        n = Int(dim(sector)); start = first(axes(V,sector))
        # TensorKit dense basis: m is fastest, then the degeneracy index.
        for multiplicity in 1:dim(V,sector), k in 1:n
            q[start+(multiplicity-1)*n+k-1] = sign*(n+1-2k)
        end
    end
    return q
end

function u1_axis_charges(V)
    sectortype(V)==U1Irrep || throw(ArgumentError("expected a U(1) space"))
    q = zeros(Int,dim(V))
    for sector in sectors(V)
        q[axes(V,sector)] .= Int(sector.charge)
    end
    return q
end

function _axis_restriction(V;component=nothing)
    q = su2_axis_charges(V)
    if isnothing(component)
        labels = isdual(V) ? -q : q
        counts = Dict(value=>count(==(value),labels) for value in unique(labels))
        W = U1Space((value=>n for (value,n) in counts)...)
        isdual(V) && (W=W')
        order = Int[]
        for sector in sectors(W)
            append!(order,findall(==(Int(sector.charge)),q))
        end
    else
        1<=component<=length(q) || throw(BoundsError())
        label = isdual(V) ? -q[component] : q[component]
        W = U1Space(label=>1)
        isdual(V) && (W=W')
        order = [Int(component)]
    end
    return W,order
end

"""Restrict an SU(2) TensorMap exactly; optionally select one auxiliary component.
No coefficient is truncated. U(1) sector ordering is obtained from the actual
spaces, not assumed numerically sorted. Incoming/dual legs retain their arrows.
"""
function restrict_tensor(t;project_axis=0,component=1)
    n = numind(t)
    axes_data = [_axis_restriction(space(t,i);component=i==project_axis ? component : nothing)
                 for i in 1:n]
    spaces = first.(axes_data)
    value = convert(Array,t)[last.(axes_data)...]
    out = foldl(⊗,spaces[1:numout(t)];init=one(spaces[1]))
    incoming = foldl(⊗,dual.(spaces[numout(t)+1:end]);init=one(spaces[1]))
    result = TensorMap(value,out,incoming;tol=1e-12)
    norm(convert(Array,result)-value) <= 1e-12*max(norm(value),1.) ||
        error("restriction changed tensor coefficients")
    return result
end

"""Convert a charged SU(2) state (with projection), or an ordinary SU(2) cell.
The selected auxiliary representation becomes a ONE-dimensional charged leg.
It is retained so all TensorKit tensors remain invariant intertwiners.
"""
function to_u1(source;component=1)
    charged = hasproperty(source,:raw) && hasproperty(source,:auxiliary_spin)
    original = charged ? source.raw : source
    g = original.geometry
    geometry = BS.KagomeSUCellGeometry(g.Lx,g.Ly;pattern=g.pattern)
    charges = Dict{Int,Int}()
    cells = BS.KagomeSUTensors[]
    for (r,c) in enumerate(original.cells)
        values = Any[]
        for name in fieldnames(BS.KagomeSUTensors)
            t = getfield(c,name)
            if isnothing(t)
                push!(values,nothing)
            elseif charged && r==source.anchor && name==:T_u
                push!(values,restrict_tensor(t;project_axis=4,component))
                charges[r] = su2_axis_charges(space(t,4))[component]
            else
                push!(values,restrict_tensor(t))
            end
        end
        push!(cells,BS.KagomeSUTensors(values...))
    end
    return check_u1(U1KagomeState(geometry,cells,charges))
end

"""Expand a tied sqrt3 cell to nine independent PESS without changing the state."""
function expand_u1_rectangular(s::U1KagomeState)
    s.geometry.pattern==:sqrt3xsqrt3 || error("expansion expects sqrt3xsqrt3")
    g=BS.KagomeSUCellGeometry(3,3)
    owners=[BS.representative(s.geometry,x,y) for (x,y) in g.representatives]
    cells=[deepcopy(s.cells[r]) for r in owners]
    charges=Dict(k=>s.auxiliary_charges[r] for (k,r) in enumerate(owners) if haskey(s.auxiliary_charges,r))
    return check_u1(U1KagomeState(g,cells,charges))
end

function check_u1(s::U1KagomeState;Dmax=nothing)
    length(s.cells)==length(s.geometry.representatives) || error("wrong representative count")
    for (r,c) in enumerate(s.cells)
        for name in (:T_u,:T_d,BS.B_FIELDS...)
            t=getfield(c,name)
            sectortype(space(t,1))==U1Irrep || error("expected U(1) TensorMaps")
            isfinite(norm(t)) && norm(t)>0 || error("nonfinite or zero U(1) tensor")
        end
        for name in BS.B_FIELDS
            u1_axis_charges(space(getfield(c,name),3))==[1,-1] || error("physical basis must be up,down with q=2Sz")
        end
        haskey(s.auxiliary_charges,r) == (numind(c.T_u)==4) || error("wrong auxiliary placement")
        if haskey(s.auxiliary_charges,r)
            dim(space(c.T_u,4))==1 || error("the auxiliary leg must have dimension one")
            u1_axis_charges(space(c.T_u,4))==[s.auxiliary_charges[r]] || error("wrong charge")
        end
        for orientation in (:up,:dn)
            core=orientation==:up ? c.T_u : c.T_d
            for (axis,(owner,b)) in enumerate(BS.triangle_sites(s.geometry,r,orientation))
                bond=getfield(s.cells[owner],BS.B_FIELDS[b])
                leg=orientation==:up ? 2 : 1
                space(core,axis)==space(bond,leg)' || throw(SpaceMismatch("U(1) simplex/B mismatch"))
                isnothing(Dmax) || dim(space(core,axis))<=Dmax || error("Dmax exceeded")
            end
        end
        for b in 1:3, (fields,leg) in ((BS.U_FIELDS,2),(BS.D_FIELDS,1))
            w=getfield(c,fields[b]);isnothing(w) && continue
            V=space(getfield(c,BS.B_FIELDS[b]),leg)'
            space(w,1)==V && space(w,2)==V' || throw(SpaceMismatch("U(1) lambda/B mismatch"))
            isfinite(norm(w)) && norm(w)>0 || error("nonfinite or zero lambda")
        end
    end
    return s
end

function _charged_up(c,gate,tolerance,Dmax)
    wa,wb,wc=c.lambda_d_a,c.lambda_d_b,c.lambda_d_c
    @tensor a[:] := c.B_a[1,-2,-3]*wa[1,-1]
    @tensor b[:] := c.B_b[1,-2,-3]*wb[1,-1]
    @tensor d[:] := c.B_c[1,-2,-3]*wc[1,-1]
    @tensor A[:] := c.T_u[1,2,3,-7]*a[-1,1,-2]*b[-3,2,-4]*d[-5,3,-6]
    @tensor A[:] := gate[-2,-4,-6,1,2,3]*A[-1,1,-3,2,-5,3,-7]
    A=A/norm(A)
    Ua,la,Va=Kernels.tsvd(A,(1,2),(3,4,5,6,7);trunc=Kernels.truncdim(Dmax))
    Ua,la,Va=Kernels.Truncations(Ua,la,Va,Dmax,tolerance)
    Ub,lb,Vb=Kernels.tsvd(A,(3,4),(1,2,5,6,7);trunc=Kernels.truncdim(Dmax))
    Ub,lb,Vb=Kernels.Truncations(Ub,lb,Vb,Dmax,tolerance)
    Uc,lc,Vc=Kernels.tsvd(A,(5,6),(1,2,3,4,7);trunc=Kernels.truncdim(Dmax))
    Uc,lc,Vc=Kernels.Truncations(Uc,lc,Vc,Dmax,tolerance)
    @tensor core[:] := A[1,2,3,4,5,6,-4]*Ua'[-1,1,2]*Ub'[-2,3,4]*Uc'[-3,5,6]
    Ba=permute(Ua,((1,3,2),()));Bb=permute(Ub,((1,3,2),()));Bc=permute(Uc,((1,3,2),()))
    ia,ib,ic=pinv(wa),pinv(wb),pinv(wc)
    @tensor Ba[:] := Ba[1,-2,-3]*ia[1,-1]
    @tensor Bb[:] := Bb[1,-2,-3]*ib[1,-1]
    @tensor Bc[:] := Bc[1,-2,-3]*ic[1,-1]
    return Ba,Bb,Bc,la/norm(la),lb/norm(lb),lc/norm(lc),core/norm(core)
end

function _sweep!(s,gate,orientation;Dmax,trun_tol)
    opposite=orientation==:dn ? BS.U_FIELDS : BS.D_FIELDS
    current=orientation==:dn ? BS.D_FIELDS : BS.U_FIELDS
    simplex=orientation==:dn ? :T_d : :T_u
    for r in eachindex(s.cells)
        c=s.cells[r]
        sites=BS.triangle_sites(s.geometry,r,orientation)
        if orientation==:up && haskey(s.auxiliary_charges,r)
            result=_charged_up(c,gate,trun_tol,Dmax)
        else
            bonds=[getfield(s.cells[owner],BS.B_FIELDS[b]) for (owner,b) in sites]
            weights=[getfield(s.cells[owner],opposite[b]) for (owner,b) in sites]
            any(isnothing,weights) && error("perform the initial down half-step first")
            fn=orientation==:dn ? Kernels.Tri_T_dn : Kernels.Tri_T_up
            result=fn(getfield(c,simplex),bonds...,weights...,gate,trun_tol,Dmax,false)
        end
        for (i,(owner,b)) in enumerate(sites)
            setfield!(s.cells[owner],BS.B_FIELDS[b],result[i])
            setfield!(s.cells[owner],current[b],result[3+i])
        end
        setfield!(c,simplex,result[7])
    end
    return s
end

function itebd_u1(s::U1KagomeState,H,tau::Real,dt::Real;Dmax::Integer,trun_tol=1e-6,callback=nothing)
    isfinite(dt) && dt>0 && isfinite(tau) && tau>=0 || throw(ArgumentError("invalid dt/tau"))
    Dmax>0 && 0<=trun_tol<1 || throw(ArgumentError("invalid truncation"))
    check_u1(s)
    result=deepcopy(s)
    gate,half=Kernels.trotter_gate(H,dt)
    _sweep!(result,half,:dn;Dmax,trun_tol)
    steps=round(Int,tau/dt)
    for step in 1:steps
        _sweep!(result,gate,:up;Dmax,trun_tol)
        _sweep!(result,gate,:dn;Dmax,trun_tol)
        isnothing(callback) || callback(result,step,steps)
    end
    _sweep!(result,half,:up;Dmax,trun_tol)
    return check_u1(result;Dmax)
end

u1_heisenberg_triangle(;J1=1.)=restrict_tensor(BS.heisenberg_triangle(;J1))

function u1_dense_cells(s::U1KagomeState)
    [begin
        arrays=Any[]
        for name in (:B_a,:B_b,:B_c,:T_u,:T_d)
            t=getfield(c,name);value=convert(Array,t)
            if name in (:B_a,:B_b,:B_c)
                q=u1_axis_charges(space(t,3))
                value=value[:,:,[only(findall(==(charge),q)) for charge in (1,-1)]]
            elseif name==:T_u && haskey(s.auxiliary_charges,r)
                value=copy(selectdim(value,4,1))
            end
            push!(arrays,value)
        end
        (B1=arrays[1],B2=arrays[2],B3=arrays[3],Tup=arrays[4],Tdn=arrays[5])
    end for (r,c) in enumerate(s.cells)]
end

function u1_torus_wavefunction(s::U1KagomeState)
    s.geometry.pattern==:sqrt3xsqrt3 && length(s.auxiliary_charges)==1 ||
        error("finite torus helper requires a single charged sqrt3 cell")
    anchor=only(keys(s.auxiliary_charges))
    c1,c2,c3=s.cells[[mod1(anchor+k,3) for k in 0:2]]
    a1,b1,d1,t1=c1.B_a,c1.B_b,c1.B_c,c1.T_u
    a2,b2,d2,t2=c2.B_a,c2.B_b,c2.B_c,c2.T_u
    a3,b3,d3,t3=c3.B_a,c3.B_b,c3.B_c,c3.T_u
    @tensor f1[:] := t1[1,2,3,-7]*a1[-1,1,-4]*b1[-2,2,-5]*d1[-3,3,-6]
    @tensor f2[:] := t2[1,2,3]*a2[-1,1,-4]*b2[-2,2,-5]*d2[-3,3,-6]
    @tensor f3[:] := t3[1,2,3]*a3[-1,1,-4]*b3[-2,2,-5]*d3[-3,3,-6]
    td1,td2,td3=c1.T_d,c2.T_d,c3.T_d
    @tensoropt psi[:] := f1[1,2,3,-1,-2,-3,-10]*f2[4,5,6,-4,-5,-6]*
        f3[7,8,9,-7,-8,-9]*td1[4,2,9]*td2[7,5,3]*td3[1,8,6]
    return dropdims(convert(Array,psi);dims=10)
end

function u1_torus_observables(s::U1KagomeState;h=0.,J1=1.)
    psi=vec(u1_torus_wavefunction(s)); norm(psi)>0 || error("zero finite-torus state")
    psi=psi/norm(psi)
    sites=zeros(9); second=0.;acted=zeros(ComplexF64,512)
    plus=zeros(ComplexF64,512);minus=zeros(ComplexF64,512)
    anchor=only(keys(s.auxiliary_charges))
    order=[mod1(anchor+k,3) for k in 0:2]
    indices=Dict((r,b)=>3(k-1)+b for (k,r) in enumerate(order) for b in 1:3)
    bonds=Tuple{Int,Int}[]
    for orientation in (:up,:dn),r in 1:3
        t=BS.triangle_sites(s.geometry,r,orientation)
        append!(bonds,[(indices[t[k]],indices[t[mod1(k+1,3)]]) for k in 1:3])
    end
    for index in 0:511
        second+=abs2(psi[index+1])*(4.5-count_ones(index))^2
        for i in 1:9
            b=(index>>(i-1))&1
            sites[i]+=abs2(psi[index+1])*(.5-b)
            (b==1 ? plus : minus)[xor(index,1<<(i-1))+1]+=psi[index+1]
        end
        for (i,j) in bonds
            a,b=(index>>(i-1))&1,(index>>(j-1))&1
            acted[index+1]+=J1*(.5-a)*(.5-b)*psi[index+1]
            a==b || (acted[xor(index,(1<<(i-1))|(1<<(j-1)))+1]+=.5J1*psi[index+1])
        end
    end
    exchange=real(dot(psi,acted))/9; total=sum(sites)
    return (exchange_energy_per_site=exchange,field_energy_per_site=exchange+h*total/9,
            magnetization_normalized=-2total/9,total_Sz=total,site_Sz=sites,
            spin_squared=second+.5*(norm(plus)^2+norm(minus)^2),
            variance_Sz=max(0.,second-total^2),measurement="exact nine-site U1 torus; not CTMRG")
end

function save_u1_checkpoint(path,s::U1KagomeState,H;metadata=Dict{String,Any}())
    check_u1(s);target=abspath(path);mkpath(dirname(target));temporary=tempname(dirname(target))
    try
        jldsave(temporary;format="kagome_u1_cells_v1",state=s,H,metadata)
        mv(temporary,target;force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return target
end
function load_u1_checkpoint(path)
    data=JLD2.load(path)
    data["format"]=="kagome_u1_cells_v1" || error("not a U(1) checkpoint")
    return check_u1(data["state"]),data
end

function export_u1_portable(s::U1KagomeState)
    result=Dict{String,Any}()
    for (r,c) in enumerate(s.cells)
        tensors=Dict{String,Any}()
        for (name,label) in zip((:B_a,:B_b,:B_c,:T_u,:T_d),("B1","B2","B3","Tup","Tdn"))
            t=getfield(c,name);a=convert(Array,t)
            n=numind(t);q=[u1_axis_charges(space(t,i)) for i in 1:n]
            signs=[isdual(space(t,i)) ? 1 : -1 for i in 1:n]
            total=0
            if name==:T_u && haskey(s.auxiliary_charges,r)
                a=copy(selectdim(a,4,1));q=q[1:3];signs=signs[1:3]
                total=s.auxiliary_charges[r]
            end
            # YASTN arrows are opposite TensorKit all-out generators, matching
            # the existing kagome Q(+,+,+,-,-,-), R(-,-,-) factor conventions.
            labels=[-signs[i].*q[i] for i in eachindex(q)]
            tensors[label]=Dict("dims"=>collect(size(a)),"real"=>real.(vec(a)),"imag"=>imag.(vec(a)),
                                "signatures"=>signs,"axis_charges"=>labels,"total_charge"=>total)
        end
        result["$r"]=tensors
    end
    g=s.geometry
    return Dict("format"=>"kagome_u1_portable_v1","symmetry"=>"U1","charge_unit"=>"2Sz",
                "Lx"=>g.Lx,"Ly"=>g.Ly,"cell_pattern"=>string(g.pattern),
                "site_map"=>Dict("$x,$y"=>string(BS.representative(g,x,y)) for x in 1:g.Lx for y in 1:g.Ly),
                "cells"=>result,"order"=>"F")
end

end # module
