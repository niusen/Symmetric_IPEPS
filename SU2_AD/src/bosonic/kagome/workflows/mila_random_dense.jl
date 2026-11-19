"""Independent dense 27-site random search, using the existing Julia SU/CTMRG.

No spatial symmetry or magnetization is imposed. B axes and up/down simplex
connectivity are exactly those of KagomeSU2Cells. All runtime settings belong
in the example script, not command-line options.
"""
module KagomeMilaRandom
using TensorKit, LinearAlgebra, Random, Statistics, JLD2, TOML, SHA, Printf
include(joinpath(@__DIR__,"..","native_measurement","KagomeNativeMeasurement.jl"))
const Native=KagomeNativeMeasurement
const BS=Native.BS
const SU=Native.KagomeU1Cells
export random_dense_state, triangle_gate_hamiltonian, evolve!, symmetry_rows, run_search, smoke_check

function random_dense_state(D,seed)
    D>0 || error("D must be positive")
    g=BS.KagomeSUCellGeometry(3,3;pattern=:rectangular)
    rng=MersenneTwister(seed);V=ComplexSpace(D);P=ComplexSpace(2)
    cells=BS.KagomeSUTensors[]
    for _ in g.representatives
        # Identical arrows, partitions and permutations to the legacy initializer.
        b=[permute(randn(rng,ComplexF64,V'⊗V',P'),((1,2,3),())) for _ in 1:3]
        t=[permute(randn(rng,ComplexF64,V,V'⊗V'),((1,2,3),())) for _ in 1:2]
        b=[x/norm(x) for x in b];t=[x/norm(x) for x in t]
        w=[unitary(space(x,2)',space(x,2)') for x in b]
        push!(cells,BS.KagomeSUTensors(t...,b...,w...,nothing,nothing,nothing))
    end
    check_dense(Native.DenseKagomeState(g,cells,Dict{Int,Int}()))
end

function check_dense(s;Dmax=typemax(Int))
    s.geometry.pattern==:rectangular && length(s.cells)==9 || error("expected nine independent PESS")
    isempty(s.auxiliary_charges) || error("dense random search must not fix total Sz")
    for (r,c) in enumerate(s.cells)
        for n in (:T_u,:T_d,BS.B_FIELDS...)
            t=getfield(c,n);isfinite(norm(t)) && norm(t)>0 || error("zero/nonfinite tensor")
        end
        for orientation in (:up,:dn)
            t=orientation==:up ? c.T_u : c.T_d;leg=orientation==:up ? 2 : 1
            for (axis,(owner,b)) in enumerate(BS.triangle_sites(s.geometry,r,orientation))
                space(t,axis)==space(getfield(s.cells[owner],BS.B_FIELDS[b]),leg)' || error("bond mismatch")
                dim(space(t,axis))<=Dmax || error("D exceeded")
            end
        end
    end
    s
end

function triangle_gate_hamiltonian(h;J=1.)
    H=Native.dense_tensor(BS.heisenberg_triangle(;J1=J))
    Z=sum(Native.embedded_operator(i,Native.SZ) for i in 1:3)
    Z=TensorMap(reshape(Z,ntuple(_->2,6)),codomain(H),domain(H))
    # Each site belongs to TWO updated triangles. The half field here makes
    # the sum of up/down Hamiltonians exactly H_exchange+h*sum(Sz).
    H+(h/2)*Z
end

function weight_values(s)
    [sort(abs.(diag(convert(Array,getfield(c,n))))) for c in s.cells for n in (BS.U_FIELDS...,BS.D_FIELDS...)]
end
function weight_drift(a,b)
    maximum(begin
        n=max(length(x),length(y));xx=zeros(n);yy=zeros(n)
        xx[1:length(x)]=x;yy[1:length(y)]=y;norm(xx-yy)
    end for (x,y) in zip(a,b))
end

function evolve!(s,D,h,schedule,folder;trun_tol=1e-8,progress_every=500,checkpoint_every=500)
    H=triangle_gate_hamiltonian(h);latest=joinpath(folder,"latest.jld2")
    completed=0;stage_step=0;phase="complete";drift=NaN
    if isfile(latest)
        d=JLD2.load(latest);s=d["state"];completed=d["completed"];stage_step=d["stage_step"];phase=d["phase"]
        println("RESUME D=$D completed=$completed phase=$phase step=$stage_step")
    end
    stamp=time()
    for k in completed+1:length(schedule)
        stage=schedule[k];steps=round(Int,stage.tau/stage.dt)
        gate,half=SU.Kernels.trotter_gate(H,stage.dt)
        continuing=phase=="body";start=continuing ? stage_step : 0
        if !continuing
            SU._sweep!(s,half,:dn;Dmax=D,trun_tol)
        end
        before=weight_values(s);last_sample=start
        for step in start+1:steps
            # The existing generic sweep selects the unchanged legacy up/down
            # HOSVD kernels; no charge-specific path is used for this dense state.
            SU._sweep!(s,gate,:up;Dmax=D,trun_tol)
            SU._sweep!(s,gate,:dn;Dmax=D,trun_tol)
            if step%100==0 || step==steps
                after=weight_values(s);drift=weight_drift(before,after)/max(step-last_sample,1)
                before=after;last_sample=step
            end
            if step%progress_every==0 || step==steps
                println("SU D=$D stage=$k dt=$(stage.dt) step=$step/$steps weight_drift_per_step=$drift elapsed=$(round(time()-stamp;digits=1))s");flush(stdout)
            end
            if step%checkpoint_every==0
                JLD2.jldsave(latest;state=s,completed=k-1,stage_step=step,phase="body")
            end
        end
        SU._sweep!(s,half,:up;Dmax=D,trun_tol)
        check_dense(s;Dmax=D)
        JLD2.jldsave(latest;state=s,completed=k,stage_step=0,phase="complete")
        phase="complete";stage_step=0
    end
    Native.save_converted_state(joinpath(folder,"final.jld2"),s;h)
    s,drift,time()-stamp
end

const A1=[1.,0.]
const A2=[.5,sqrt(3)/2]
const BASIS=(-A1/2,zeros(2),-A2/2)
siteid(x,y,b)=3*(mod(x,3)+3mod(y,3))+b
const POS=[x*A1+y*A2+BASIS[b] for y in 0:2 for x in 0:2 for b in 1:3]
const PERIOD=hcat(3A1,3A2)
edge(a,b)=minmax(a,b)
function identify(p)
    hits=findall(POS) do q
        v=PERIOD\(p-q);norm(v-round.(v))<1e-8
    end
    length(hits)==1 || error("point does not map to kagome sites")
    only(hits)
end

function observable_vectors(result)
    z=zeros(27);bonds=Dict{Tuple{Int,Int},Float64}()
    for c in result["cells"]
        x=c["x"]-1;y=c["y"]-1
        up=[siteid(x,y,b) for b in 1:3]
        dn=[siteid(x,y,2),siteid(x,y+1,3),siteid(x+1,y,1)]
        for b in 1:3;z[up[b]]=c["Sz"][b];end
        for (ids,values) in ((up,c["bonds_up"]),(dn,c["bonds_down"]))
            for (k,(a,b)) in enumerate(((1,2),(2,3),(3,1)))
                key=edge(ids[a],ids[b]);haskey(bonds,key) && error("duplicate physical bond")
                bonds[key]=values[k]
            end
        end
    end
    length(bonds)==54 || error("27-site cell must have 54 NN bonds")
    z,bonds
end
function symmetry_rows(result)
    z,bonds=observable_vectors(result);rows=Dict[]
    function add(kind,label,transform)
        p=[identify(transform(r)) for r in POS]
        sort(p)==collect(1:27) || error("invalid symmetry permutation")
        dz=z[p]-z;db=[bonds[edge(p[a],p[b])]-v for ((a,b),v) in bonds]
        push!(rows,Dict("kind"=>kind,"operation"=>label,"rms_Sz"=>sqrt(mean(abs2,dz)),
            "max_Sz"=>maximum(abs,dz),"rms_bond"=>sqrt(mean(abs2,db)),"max_bond"=>maximum(abs,db)))
    end
    for dx in 0:2,dy in 0:2
        dx==dy==0 && continue
        add("translation","($dx,$dy)",p->p+dx*A1+dy*A2)
    end
    theta=2pi/3;R=[cos(theta) -sin(theta);sin(theta) cos(theta)]
    for x in 0:2,y in 0:2
        c=(A1-A2)/2+x*A1+y*A2
        add("C3","hexagon($x,$y)",p->c+R*(p-c))
        for k in 0:5
            t=k*pi/6;M=[cos(2t) sin(2t);sin(2t) -cos(2t)]
            add("mirror","hexagon($x,$y),axis=$k*pi/6",p->c+M*(p-c))
        end
    end
    rows
end

function summary_row(result,D,seed,drift,seconds,folder)
    syms=symmetry_rows(result)
    best(kind)=argmin(r->max(r["rms_Sz"],r["rms_bond"]),filter(r->r["kind"]==kind,syms))
    c3=best("C3");mirror=best("mirror");trans=best("translation")
    z,b=observable_vectors(result)
    row=Dict{String,Any}("D"=>D,"seed"=>seed,"h"=>result["settings"]["h"],"chi"=>result["ctm"]["chi"],
        "energy"=>result["energy_per_spin"],"exchange_energy"=>result["exchange_energy_per_spin"],
        "m"=>result["m_normalized"],"ctm_ok"=>result["ctm"]["converged"],"ctm_error"=>result["ctm"]["error"],
        "ctm_iterations"=>result["ctm"]["iterations"],"lambda_drift_per_step"=>drift,"seconds_su"=>seconds,
        "site_contrast"=>maximum(z)-minimum(z),"C3_rms_Sz"=>c3["rms_Sz"],"C3_max_bond"=>c3["max_bond"],
        "C3_center"=>c3["operation"],"mirror_rms_Sz"=>mirror["rms_Sz"],"mirror_max_bond"=>mirror["max_bond"],
        "translation_rms_Sz"=>trans["rms_Sz"],"translation_max_bond"=>trans["max_bond"],
        "translation"=>trans["operation"],"folder"=>abspath(folder))
    Native.write_csv(joinpath(folder,"symmetry_checks.csv"),syms)
    Native.atomic_toml(joinpath(folder,"summary.toml"),row)
    row
end

function run_search(cfg)
    all(>(0),cfg.dims) && allunique(cfg.dims) && allunique(cfg.seeds) || error("invalid dims/seeds")
    cfg.chi>0 && 2<=cfg.ctm_iterations<=50 && cfg.threads>0 || error("invalid CTMRG/thread settings")
    BLAS.set_num_threads(cfg.threads);mkpath(cfg.out)
    config=Dict("format"=>"kagome_mila_dense_random_v1","dims"=>cfg.dims,"seeds"=>cfg.seeds,
        "h"=>cfg.h,"chi"=>cfg.chi,"ctm_iterations"=>cfg.ctm_iterations,"ctm_tolerance"=>cfg.ctm_tolerance,
        "truncation_tolerance"=>cfg.truncation_tolerance,"threads"=>cfg.threads,"cell"=>"rectangular3x3",
        "schedule"=>[Dict("dt"=>s.dt,"tau"=>s.tau) for s in cfg.schedule])
    cp=joinpath(cfg.out,"configuration.toml")
    isfile(cp) && TOML.parsefile(cp)!=config && error("settings changed: choose a new output folder")
    Native.atomic_toml(cp,config)
    summaries=Dict[]
    for seed in cfg.seeds,D in cfg.dims
        folder=joinpath(cfg.out,"D$(D)_seed$(seed)");mkpath(folder)
        done=joinpath(folder,"summary.toml")
        if isfile(done)
            push!(summaries,TOML.parsefile(done));continue
        end
        println("START independent random state: D=$D seed=$seed h=$(cfg.h) 27 sites");flush(stdout)
        s=random_dense_state(D,seed)
        if !isfile(joinpath(folder,"initial.jld2"))
            Native.save_converted_state(joinpath(folder,"initial.jld2"),s;h=cfg.h)
        end
        s,drift,seconds=evolve!(s,D,cfg.h,cfg.schedule,folder;trun_tol=cfg.truncation_tolerance,
            progress_every=cfg.progress_every,checkpoint_every=cfg.checkpoint_every)
        result=Native.measure_checkpoint(joinpath(folder,"final.jld2"),joinpath(folder,"ctm");representation=:dense,
            h=cfg.h,chi=cfg.chi,iterations=cfg.ctm_iterations,tolerance=cfg.ctm_tolerance,device="cpu",threads=cfg.threads,D)
        row=summary_row(result,D,seed,drift,seconds,folder);push!(summaries,row)
        Native.write_csv(joinpath(cfg.out,"summary.csv"),summaries)
        Native.write_json(joinpath(cfg.out,"summary.json"),summaries)
        println("POINT DONE ",row);flush(stdout);GC.gc()
    end
    Native.write_csv(joinpath(cfg.out,"summary.csv"),summaries)
    Native.write_json(joinpath(cfg.out,"summary.json"),summaries)
    summaries
end

function smoke_check(out)
    mkpath(out);BLAS.set_num_threads(2)
    s=random_dense_state(2,9871)
    s2=random_dense_state(2,9871)
    @assert norm(s.cells[1].B_a-s2.cells[1].B_a)==0
    @assert norm(s.cells[1].B_a-s.cells[2].B_a)>.01
    H0=triangle_gate_hamiltonian(0.);H1=triangle_gate_hamiltonian(.38)
    @assert norm(H1-H1')<1e-12
    @assert norm(convert(Array,H1-H0)-.19reshape(sum(Native.embedded_operator(i,Native.SZ) for i in 1:3),ntuple(_->2,6)))<1e-12
    s,drift,sec=evolve!(s,2,.38,[(dt=.03,tau=.06)],out;progress_every=1,checkpoint_every=100)
    result,_=Native.measure_state(s;representation=:dense,h=.38,chi=4,iterations=3,tolerance=1e-6,verbose=false)
    syms=symmetry_rows(result);@assert length(syms)==71
    z,bonds=observable_vectors(result)
    @assert abs(sum(values(bonds))/27-result["exchange_energy_per_spin"])<1e-12
    @assert abs(-2mean(z)-result["m_normalized"])<1e-12
    @assert abs(result["energy_per_spin"]-result["exchange_energy_per_spin"]-.38mean(z))<1e-12
    println("SMOKE PASSED dense SU, 27 sites, 54 bonds, half-field allocation, symmetry permutations")
    result
end
end
