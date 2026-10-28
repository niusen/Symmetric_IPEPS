# Exact U1 export by default, explicit dense fallback, independent finite checks,
# and optional block U1 / dense factorized CTMRG measurement.
using TensorKit, LinearAlgebra, JLD2, TOML, SHA
if !isdefined(@__MODULE__,:KagomeChargedSU2)
    include(joinpath(@__DIR__,"..","simple_update_charged_cell.jl"))
end
using .KagomeChargedSU2
if !isdefined(@__MODULE__,:KagomeU1Cells)
    include(joinpath(@__DIR__,"..","simple_update_u1_cells.jl"))
end
using .KagomeU1Cells

const DENSE_NAMES = (:B1,:B2,:B3,:Tup,:Tdn)

function padded_dense_cells(cells)
    D = maximum(size(getfield(c,name),axis) for c in cells for name in DENSE_NAMES
                for axis in 1:(name in (:B1,:B2,:B3) ? 2 : 3))
    padded = [NamedTuple{DENSE_NAMES}(Tuple(begin
        t = getfield(c,name)
        shape = name in (:B1,:B2,:B3) ? (D,D,2) : (D,D,D)
        value = zeros(ComplexF64,shape)
        value[axes(t)...] .= t
        value
    end for name in DENSE_NAMES)) for c in cells]
    return padded,D
end

function dense_torus_wavefunction(cells,anchor)
    c1,c2,c3 = cells[[mod1(anchor+k,3) for k in 0:2]]
    a1,b1,d1,t1 = c1.B1,c1.B2,c1.B3,c1.Tup
    a2,b2,d2,t2 = c2.B1,c2.B2,c2.B3,c2.Tup
    a3,b3,d3,t3 = c3.B1,c3.B2,c3.B3,c3.Tup
    @tensor f1[:] := t1[1,2,3]*a1[-1,1,-4]*b1[-2,2,-5]*d1[-3,3,-6]
    @tensor f2[:] := t2[1,2,3]*a2[-1,1,-4]*b2[-2,2,-5]*d2[-3,3,-6]
    @tensor f3[:] := t3[1,2,3]*a3[-1,1,-4]*b3[-2,2,-5]*d3[-3,3,-6]
    td1,td2,td3 = c1.Tdn,c2.Tdn,c3.Tdn
    @tensoropt psi[:] := f1[1,2,3,-1,-2,-3]*f2[4,5,6,-4,-5,-6]*
        f3[7,8,9,-7,-8,-9]*td1[4,2,9]*td2[7,5,3]*td3[1,8,6]
    norm(psi) > 0 || error("zero projected dense state")
    return psi/norm(psi)
end

function apply_local(psi,operator,site)
    order = [site;[i for i in 1:9 if i != site]]
    acted = operator*reshape(permutedims(psi,order),2,:)
    return permutedims(reshape(acted,ntuple(_->2,9)),invperm(order))
end

function dense_finite_measure(psi,state,h)
    sx = ComplexF64[0 1;1 0]/2
    sy = ComplexF64[0 -im;im 0]/2
    sz = ComplexF64[1 0;0 -1]/2
    sites = [real(dot(psi,apply_local(psi,sz,i))) for i in 1:9]
    # Build bond ownership from geometry rather than the reference's bit code.
    order = [mod1(state.anchor+k,3) for k in 0:2]
    index = Dict((r,b)=>3(k-1)+b for (k,r) in enumerate(order) for b in 1:3)
    acted = zeros(ComplexF64,size(psi))
    geometry = state.raw.geometry
    for orientation in (:up,:dn), r in 1:3
        triangle = KagomeChargedSU2.BaseSU.triangle_sites(geometry,r,orientation)
        for edge in 1:3
            i,j = index[triangle[edge]],index[triangle[mod1(edge+1,3)]]
            for op in (sx,sy,sz)
                acted .+= apply_local(apply_local(psi,op,j),op,i)
            end
        end
    end
    exchange = real(dot(psi,acted))/9
    spin_squared = sum(norm(sum(apply_local(psi,op,i) for i in 1:9))^2
                       for op in (sx,sy,sz))
    return Dict("exchange_energy_per_site"=>exchange,
                "field_energy_per_site"=>exchange+h*sum(sites)/9,
                "site_Sz"=>sites,"total_Sz"=>sum(sites),
                "m_normalized"=>-2sum(sites)/9,"spin_squared"=>spin_squared,
                "method"=>"local dense tensors, exact finite nine-site periodic torus")
end

function export_and_verify(input,out;h=.38,component=1,threads=10,representation=:u1)
    representation in (:u1,:dense) || error("representation must be u1 or dense")
    ispath(out) && error("use a NEW output directory")
    BLAS.set_num_threads(threads)
    data=JLD2.load(input)
    is_u1=get(data,"format","")=="kagome_u1_cells_v1"
    if is_u1
        component==1 || error("a U1 checkpoint already has its auxiliary component selected")
        u1state=check_u1(data["state"])
        anchor=only(keys(u1state.auxiliary_charges))
        state=(raw=(geometry=u1state.geometry,),anchor=anchor,auxiliary_spin=u1state.auxiliary_charges[anchor]//2)
        raw=u1_dense_cells(u1state)
        reference=u1_torus_observables(u1state;h)
    else
        state,data=load_charged_checkpoint(input)
        reference=exact_torus_observables(state;component,h)
        u1state=to_u1(state;component)
        raw=representation==:u1 ? u1_dense_cells(u1state) : polarized_dense_cells(state;component)
    end
    cells,D = padded_dense_cells(raw)
    D <= 6 || error("finite verification is restricted to actual D<=6")
    psi_raw = dense_torus_wavefunction(raw,state.anchor)
    psi = dense_torus_wavefunction(cells,state.anchor)
    overlap = dot(psi_raw,psi)
    padding_error = norm(psi_raw-psi*conj(overlap)/abs(overlap))
    padding_error < 1e-12 || error("zero padding changed the physical state")
    measured = dense_finite_measure(psi,state,h)
    differences = Dict(
        "exchange_energy"=>abs(measured["exchange_energy_per_site"]-reference.exchange_energy_per_site),
        "field_energy"=>abs(measured["field_energy_per_site"]-reference.field_energy_per_site),
        "magnetization"=>abs(measured["m_normalized"]-reference.magnetization_normalized),
        "site_Sz"=>maximum(abs.(measured["site_Sz"]-reference.site_Sz)),
        "spin_squared"=>abs(measured["spin_squared"]-reference.spin_squared))
    maximum(values(differences)) < 1e-11 || error("independent dense/reference disagreement")
    mkpath(out)
    source_sha256 = bytes2hex(sha256(read(input)))
    encode(t) = Dict("dims"=>collect(size(t)),"real"=>real.(vec(t)),"imag"=>imag.(vec(t)))
    map = Dict("$x,$y"=>"$(1+mod(x-y,3)),1" for x in 1:3 for y in 1:3)
    portable = Dict("format"=>"kagome_julia_dense_v2",
        "convention"=>"Symmetric_IPEPS/SU2_AD/src/bosonic/kagome_AD_SU2_cell.jl:build_A_kagome",
        "Lx"=>3,"Ly"=>3,"D"=>D,"dtype"=>"complex128","order"=>"F",
        "cell_pattern"=>"sqrt3xsqrt3","site_map"=>map,
        "cells"=>Dict("$r,1"=>Dict(string(name)=>encode(getfield(c,name))
                                    for name in DENSE_NAMES) for (r,c) in enumerate(cells)),
        "conversion"=>Dict("source_sha256"=>source_sha256,"source"=>abspath(input),
            "auxiliary_spin"=>string(state.auxiliary_spin),"component"=>component,
            "padding"=>"leading-index isometric zero embedding; independently verified",
            "padding_wavefunction_error"=>padding_error,
            "original_shapes"=>Dict("$r,1"=>Dict(string(n)=>collect(size(getfield(c,n)))
                for n in DENSE_NAMES) for (r,c) in enumerate(raw))))
    prefix=representation==:u1 ? "u1" : "dense"
    if representation==:u1
        portable=export_u1_portable(u1state)
        portable["conversion"]=Dict("source_sha256"=>source_sha256,"source"=>abspath(input),
                                     "component"=>component,"auxiliary_charge_unit"=>"2Sz")
        H=is_u1 ? data["H"] : restrict_tensor(data["H"])
        save_u1_checkpoint(joinpath(out,"u1_state.jld2"),u1state,H;
                           metadata=Dict("source_sha256"=>source_sha256,"h"=>h,"symmetry"=>"u1"))
    else
        jldsave(joinpath(out,"dense_state.jld2");cells,site_map=map,source_sha256)
    end
    open(joinpath(out,"$(prefix)_state.toml"),"w") do io
        TOML.print(io,portable;sorted=true)
    end
    report = Dict("h"=>h,"representation"=>string(representation),"padded_D"=>D,"TensorKit"=>string(pkgversion(TensorKit)),
                  "source_sha256"=>source_sha256,"finite_dense"=>measured,
                  "padding_wavefunction_error"=>padding_error,
                  "absolute_difference_from_SU2_reference"=>differences)
    open(joinpath(out,"$(prefix)_finite_report.toml"),"w") do io
        TOML.print(io,report;sorted=true)
    end
    println(uppercase(string(representation)),"_FINITE ",measured)
    println("maximum reference difference: ",maximum(values(differences)))
    return report
end

function main_dense(args=ARGS)
    length(args)>=2 || error("usage: verify_charged_dense.jl CHECKPOINT NEW_OUT [--representation u1|dense] [--ctm --device cpu --chi 8]")
    input,out=args[1],abspath(args[2]);measure=false
    opts=Dict("h"=>"0.38","component"=>"1","threads"=>"10","representation"=>"u1",
        "device"=>"cpu","chi"=>"8","ctm-iterations"=>"50","ctm-tol"=>"1e-6")
    i=3
    while i<=length(args)
        if args[i]=="--ctm"
            measure=true
        else
            key=replace(args[i],"--"=>"")
            haskey(opts,key) && i<length(args) || error("invalid native Julia option: $(args[i])")
            i+=1;opts[key]=args[i]
        end
        i+=1
    end
    if !isdefined(@__MODULE__,:KagomeNativeMeasurement)
        include(joinpath(@__DIR__,"..","native_measurement","KagomeNativeMeasurement.jl"))
    end
    cfg=(input=input,out=out,representation=Symbol(opts["representation"]),
        component=parse(Int,opts["component"]),h=parse(Float64,opts["h"]),threads=parse(Int,opts["threads"]),
        device=opts["device"],chi=parse(Int,opts["chi"]),ctm_iterations=parse(Int,opts["ctm-iterations"]),
        ctm_tolerance=parse(Float64,opts["ctm-tol"]),measure=measure)
    Base.invokelatest(KagomeNativeMeasurement.run_native_conversion,cfg)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    main_dense()
end
