# Convergence driver for the charged SU2 prototype; original kernels are unchanged.
using TensorKit, LinearAlgebra, Random, TOML, JLD2, Dates, Printf
if !isdefined(@__MODULE__, :export_and_verify)
    include(joinpath(@__DIR__, "verify_charged_dense.jl"))
end
const ConvSU = KagomeChargedSU2
const ConvBase = ConvSU.BaseSU

function lambda_signature(s)
    result=Dict{String,Vector{Float64}}()
    for (r,c) in enumerate(s.raw.cells), name in (ConvBase.U_FIELDS...,ConvBase.D_FIELDS...)
        w=getfield(c,name);isnothing(w) && continue
        # Compare each irrep separately; multiplicity basis phases do not matter.
        for q in sectors(space(w,1))
            result["$r/$name/$(q)"]=sort(Float64.(abs.(diag(block(w,q))));rev=true)
        end
    end
    return result
end
function spectrum_change(a,b)
    keys(a)==keys(b) || return Inf
    maximum((length(a[k])==length(b[k]) ? norm(a[k]-b[k]) : Inf) for k in keys(a);init=0.)
end
function finite_vector(s)
    # Local expansion is exact and avoids compiling a rank-10 SU2 TensorMap.
    psi=dense_torus_wavefunction(ConvSU.polarized_dense_cells(s),s.anchor)
    return vec(psi)
end
function vector_change(a,b)
    (isnothing(a)||isnothing(b)) && return Inf
    z=dot(a,b)
    return norm(a-b*(abs(z)>0 ? conj(z)/abs(z) : 1))
end

"""Exact original step sequence, with optional stopping after stable blocks."""
function converged_stage(s,H,dt,tau,D,cfg;early_stop=true,progress=(args...)->nothing)
    isfinite(dt) && dt>0 && isfinite(tau) && tau>=0 && D>0 || error("invalid dt, tau or D")
    cfg.check_every>0 && cfg.minimum_steps>=0 && cfg.stable_checks>0 || error("invalid stopping counts")
    isfinite(cfg.lambda_tolerance) && cfg.lambda_tolerance>0 &&
        isfinite(cfg.wavefunction_tolerance) && cfg.wavefunction_tolerance>0 || error("invalid stopping tolerances")
    isfinite(cfg.truncation_tolerance) && 0<=cfg.truncation_tolerance<1 || error("invalid truncation tolerance")
    ConvSU.check_charged(s)
    result=deepcopy(s)
    gate,half=ConvSU.Kernel.trotter_gate(H,dt)
    ConvBase.sweep!(result.raw,half,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
    previous=nothing;previous_vector=nothing;stable=0;drift=Inf;wave_drift=Inf;actual=0
    target=round(Int,tau/dt)
    for step in 1:target
        ConvSU._up_sweep!(result,gate;Dmax=D,trun_tol=cfg.truncation_tolerance)
        ConvBase.sweep!(result.raw,gate,:dn;Dmax=D,trun_tol=cfg.truncation_tolerance)
        actual=step
        if step%cfg.check_every==0 || step==target
            signature=lambda_signature(result);wave=finite_vector(result)
            if !isnothing(previous)
                drift=spectrum_change(signature,previous)
                wave_drift=vector_change(wave,previous_vector)
                stable=(drift<cfg.lambda_tolerance && wave_drift<cfg.wavefunction_tolerance) ? stable+1 : 0
            end
            progress(step,target,drift,wave_drift)
            previous=signature;previous_vector=wave
            early_stop && step>=cfg.minimum_steps && stable>=cfg.stable_checks && break
        end
    end
    ConvSU._up_sweep!(result,half;Dmax=D,trun_tol=cfg.truncation_tolerance)
    ConvSU.check_charged(result;Dmax=D)
    return result,Dict("requested_steps"=>target,"actual_steps"=>actual,"dt"=>dt,
        "tau_actual"=>actual*dt,"lambda_drift"=>drift,"finite_wavefunction_drift"=>wave_drift,
        "stable_checks"=>stable,"stopped_early"=>actual<target)
end

function parity_initializer(seed_state,D,kind,rng;parity_pattern=Bool[0 0 1;0 1 1;1 0 1])
    # A fixed, globally compatible parity pattern is safer than independent
    # Dmax=2 HOSVD projections of a random D3 state: those can make Tu zero.
    pattern=seed_state
    size(parity_pattern)==(3,3) || error("expected (representative, B-index) parity matrix")
    integer=kind==:integer_two_singlets ? SU2Space(0=>2) :
        kind in (:truncated,:integer_singlet) ? SU2Space(0=>1) :
        kind==:integer_triplet ? SU2Space(1=>1) : SU2Space(0=>1,1=>1)
    Vp=SU2Space(1/2=>1)
    down=Dict();up=Dict()
    for (r,c) in enumerate(pattern.raw.cells),b in 1:3
        up[(r,b)]=parity_pattern[r,b] ? SU2Space(1/2=>1) : integer
        down[(r,b)]=parity_pattern[r,b] ? integer : SU2Space(1/2=>1)
    end
    g=pattern.raw.geometry;cells=ConvBase.KagomeSUTensors[]
    for r in eachindex(g.representatives)
        bs=[permute(randn(rng,ComplexF64,down[(r,b)]'*up[(r,b)]',Vp'),((1,2,3),())) for b in 1:3]
        vu=[up[(r,b)] for b in 1:3]
        vd=[down[site] for site in ConvBase.triangle_sites(g,r,:dn)]
        Tu=r==pattern.anchor ? randn(rng,ComplexF64,vu[1],vu[2]'*vu[3]'*SU2Space(1/2=>1)') :
            randn(rng,ComplexF64,vu[1],vu[2]'*vu[3]')
        Tu=permute(Tu,(Tuple(1:numind(Tu)),()))
        Td=permute(randn(rng,ComplexF64,vd[1],vd[2]'*vd[3]'),((1,2,3),()))
        all(t->norm(t)>0,[bs;[Tu,Td]]) || error("zero tensor in parity initialization")
        weights=[unitary(up[(r,b)],up[(r,b)]) for b in 1:3]
        push!(cells,ConvBase.KagomeSUTensors(Tu/norm(Tu),Td/norm(Td),(B/norm(B) for B in bs)...,
            weights...,nothing,nothing,nothing))
    end
    return ConvSU.check_charged(ConvSU.ChargedKagomeState(ConvBase.KagomeSUCell(g,cells),pattern.anchor,1//2);Dmax=D)
end

function rvb_spinon_initializer(;down_sign=1.,charged_pair=0.)
    s=ConvSU.initialize_charged(;rng=MersenneTwister(0))
    epsilon(a,b)=a==1 && b== -1 ? 1. : a== -1 && b==1 ? -1. : 0.
    for (r,c) in enumerate(s.raw.cells),name in (:T_u,:T_d,ConvBase.B_FIELDS...)
        t=getfield(c,name);A=zeros(ComplexF64,size(convert(Array,t)))
        q=[KagomeU1Cells.su2_axis_charges(space(t,i)) for i in 1:numind(t)]
        for index in CartesianIndices(A)
            labels=[q[i][index[i]] for i in 1:numind(t)]
            if name in ConvBase.B_FIELDS
                d,u,p=labels
                A[index]=(d==0 && u== -p || u==0 && d== -p) ? 1. : 0.
            elseif name==:T_u && r==s.anchor
                for leg in 1:3
                    if all(labels[k]==0 for k in 1:3 if k!=leg)
                        A[index]+=epsilon(labels[leg],labels[4])
                    end
                end
                A[index]+=charged_pair*epsilon(labels[1],labels[2])*epsilon(labels[3],labels[4])
            else
                all(==(0),labels) && (A[index]=1.)
                sign=name==:T_d ? down_sign : 1.
                for (a,b,empty) in ((1,2,3),(2,3,1),(3,1,2))
                    labels[empty]==0 && (A[index]+=sign*epsilon(labels[a],labels[b]))
                end
            end
        end
        updated=TensorMap(A,codomain(t),domain(t);tol=1e-12)
        norm(convert(Array,updated)-A)<1e-12 || error("RVB coefficient arrows violate SU2")
        norm(updated)>0 || error("zero RVB initializer")
        setfield!(c,name,updated/norm(updated))
    end
    return ConvSU.check_charged(s)
end

function convergence_initial_state(D,kind,seed;parity_pattern=Bool[0 0 1;0 1 1;1 0 1])
    rng=MersenneTwister(seed)
    if kind in (:rvb_spinon,:rvb_opposite_down,:rvb_charged_pair)
        D==3 || error("RVB initializer here uses D=3")
        return rvb_spinon_initializer(;down_sign=kind==:rvb_opposite_down ? -1. : 1.,
            charged_pair=kind==:rvb_charged_pair ? 1. : 0.)
    end
    if kind in (:homogeneous,:uniform)
        V=D==4 ? SU2Space(0=>2,1/2=>1) : SU2Space(0=>1,1/2=>1)
        s=ConvSU.initialize_charged(;Vv=V,rng)
        if kind==:uniform
            # Deterministic real, equal reduced coefficients: a different
            # physical starting state, not noise added to the random branch.
            for c in s.raw.cells,name in (:T_u,:T_d,ConvBase.B_FIELDS...)
                t=getfield(c,name)
                for (_,b) in blocks(t)
                    fill!(b,1)
                end
                setfield!(c,name,t/norm(t))
            end
        end
        return s
    end
    seed_state=ConvSU.initialize_charged(;rng)
    return parity_initializer(seed_state,D,kind,rng;parity_pattern)
end

function physical_bond_ranks(s)
    result=Dict[]
    for (r,c) in enumerate(s.raw.cells),b in 1:3,leg in 1:2
        A=convert(Array,getfield(c,ConvBase.B_FIELDS[b]))
        order=(leg,(k for k in 1:3 if k!=leg)...)
        values=svdvals(reshape(permutedims(A,order),size(A,leg),:))
        push!(result,Dict("cell"=>r,"bond"=>b,"leg"=>leg,"dimension"=>size(A,leg),
            "matrix_rank"=>count(>(maximum(values)*1e-10),values),"singular_values"=>values))
    end
    return result
end

function save_convergence_state(folder,s,H,metadata,cfg)
    mkpath(folder)
    ConvSU.save_charged_checkpoint(joinpath(folder,"state.jld2"),s,H;metadata)
    u1=to_u1(s)
    save_u1_checkpoint(joinpath(folder,"u1_state.jld2"),u1,u1_heisenberg_triangle();metadata)
    open(joinpath(folder,"u1_state.toml"),"w") do io
        TOML.print(io,export_u1_portable(u1);sorted=true)
    end
    # Explicit dense measurement export for small-D CTM speed; SU remains SU2.
    raw=ConvSU.polarized_dense_cells(s)
    cells,D=padded_dense_cells(raw)
    encode(t)=Dict("dims"=>collect(size(t)),"real"=>real.(vec(t)),"imag"=>imag.(vec(t)))
    data=Dict("format"=>"kagome_julia_dense_v2",
        "convention"=>"Symmetric_IPEPS/SU2_AD/src/bosonic/kagome_AD_SU2_cell.jl:build_A_kagome",
        "Lx"=>3,"Ly"=>3,"D"=>D,
        "dtype"=>"complex128","order"=>"F","cell_pattern"=>"sqrt3xsqrt3",
        "site_map"=>Dict("$x,$y"=>"$(1+mod(x-y,3)),1" for x in 1:3 for y in 1:3),
        "cells"=>Dict("$r,1"=>Dict(string(n)=>encode(getfield(c,n)) for n in DENSE_NAMES) for (r,c) in enumerate(cells)))
    open(joinpath(folder,"dense_state.toml"),"w") do io
        TOML.print(io,data;sorted=true)
    end
    psi=dense_torus_wavefunction(raw,s.anchor)
    obs=dense_finite_measure(psi,s,cfg.h)
    report=Dict("finite"=>obs,"metadata"=>metadata,"B_ranks"=>physical_bond_ranks(s),
        "bonds"=>[Dict(string(k)=>v for (k,v) in pairs(row)) for row in ConvBase.bond_report(s.raw)])
    open(joinpath(folder,"report.toml"),"w") do io
        TOML.print(io,report;sorted=true)
    end
    return report
end

function run_su2_convergence(cfg)
    BLAS.set_num_threads(cfg.threads)
    ispath(cfg.out) && error("choose a NEW output folder")
    mkpath(cfg.out);H=ConvBase.heisenberg_triangle()
    for seed in cfg.seeds,case in cfg.cases
        D,kind=case.D,case.initialization
        label="D$(D)_$(kind)_seed$(seed)";folder=joinpath(cfg.out,label);mkpath(folder)
        println("START ",label);flush(stdout)
        state=convergence_initial_state(D,kind,seed;parity_pattern=cfg.parity_pattern)
        history=Dict[]
        metadata=Dict{String,Any}("Dmax"=>D,"initialization"=>string(kind),"seed"=>seed,
            "h"=>cfg.h,"symmetry"=>"su2","TensorKit"=>string(pkgversion(TensorKit)),
            "cell"=>"sqrt3xsqrt3","auxiliary_spin"=>"1/2","component"=>1)
        metadata["parity_pattern_up_half"]=collect.(eachrow(cfg.parity_pattern))
        metadata["initial_B_ranks"]=physical_bond_ranks(state)
        ConvSU.save_charged_checkpoint(joinpath(folder,"initial.jld2"),state,H;metadata)
        for (k,stage) in enumerate(cfg.schedule)
            stamp=time()
            state,info=converged_stage(state,H,stage.dt,stage.tau,D,cfg;
                progress=(step,target,dl,dp)->begin
                    println(label," stage=",k," step=",step,"/",target," dLambda=",dl," dPsi9=",dp);flush(stdout)
                end)
            info["seconds"]=time()-stamp;info["stage"]=k;push!(history,info)
            metadata["history"]=history;metadata["completed_stages"]=k
            ConvSU.save_charged_checkpoint(joinpath(folder,@sprintf("stage_%02d.jld2",k)),state,H;metadata)
            open(joinpath(folder,"progress.toml"),"w") do io
                TOML.print(io,metadata;sorted=true)
            end
        end
        save_convergence_state(joinpath(folder,"before_hold"),state,H,metadata,cfg)
        state,hold=converged_stage(state,H,cfg.hold_dt,cfg.hold_tau,D,cfg;early_stop=false,
            progress=(step,target,dl,dp)->begin
                println(label," HOLD step=",step,"/",target," dLambda=",dl," dPsi9=",dp);flush(stdout)
            end)
        metadata["hold"]=hold
        report=save_convergence_state(joinpath(folder,"after_hold"),state,H,metadata,cfg)
        println("READY ",label," finiteE=",report["finite"]["field_energy_per_site"],
            " hold_lambda_drift=",hold["lambda_drift"]);flush(stdout)
    end
    return cfg.out
end

function continue_su2_convergence(input,out,cfg;schedule=cfg.schedule)
    BLAS.set_num_threads(cfg.threads)
    ispath(out) && error("choose a NEW continuation output folder")
    state,data=ConvSU.load_charged_checkpoint(input)
    H=data["H"];metadata=deepcopy(data["metadata"])
    D=metadata["Dmax"];metadata["continued_from"]=abspath(input)
    metadata["h"]==cfg.h || error("field mismatch")
    history=metadata["history"];offset=length(history);mkpath(out)
    for (k,stage) in enumerate(schedule)
        stamp=time()
        state,info=converged_stage(state,H,stage.dt,stage.tau,D,cfg;
            progress=(step,target,dl,dp)->begin
                println(basename(out)," CONTINUE stage=",k," step=",step,"/",target,
                    " dLambda=",dl," dPsi9=",dp);flush(stdout)
            end)
        info["seconds"]=time()-stamp;info["stage"]=offset+k;push!(history,info)
        metadata["completed_stages"]=offset+k
        ConvSU.save_charged_checkpoint(joinpath(out,@sprintf("stage_%02d.jld2",k)),state,H;metadata)
    end
    save_convergence_state(joinpath(out,"before_hold"),state,H,metadata,cfg)
    state,hold=converged_stage(state,H,cfg.hold_dt,cfg.hold_tau,D,cfg;early_stop=false,
        progress=(step,target,dl,dp)->begin
            println(basename(out)," CONTINUE HOLD ",step,"/",target," dLambda=",dl," dPsi9=",dp);flush(stdout)
        end)
    metadata["hold"]=hold
    report=save_convergence_state(joinpath(out,"after_hold"),state,H,metadata,cfg)
    println("READY continued ",out," finiteE=",report["finite"]["field_energy_per_site"]);flush(stdout)
    return state
end
