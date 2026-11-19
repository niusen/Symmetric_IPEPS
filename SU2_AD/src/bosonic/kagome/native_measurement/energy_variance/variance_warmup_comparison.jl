"""Fixed-state comparison of the extra bottom warmup; production code untouched."""
module KagomeWarmupComparison
using TensorKit, LinearAlgebra, Random, TOML, SHA, Dates

native=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",normpath(joinpath(@__DIR__,"..")))
if !isdefined(parentmodule(@__MODULE__),:KagomeEnergyVariance)
    Base.include(parentmodule(@__MODULE__),joinpath(native,"energy_variance","KagomeEnergyVariance.jl"))
end
import ..KagomeEnergyVariance
const V=KagomeEnergyVariance
const N=V.N
const C=V.C

function dense_D2_fixture(;seed=701)
    g=N.BS.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    rng=MersenneTwister(seed)
    vv=ComplexSpace(2);vp=ComplexSpace(2);spectator=ComplexSpace(1)
    qs=Any[];rs=Any[];fusers=Any[]
    for _ in eachindex(g.representatives)
        P=vv⊗vv⊗vv⊗vp⊗vp⊗vp⊗spectator
        q=randn(rng,ComplexF64,P,one(P));push!(qs,q/norm(q))
        P=vv'⊗vv'⊗vv';r=randn(rng,ComplexF64,P,one(P));push!(rs,r/norm(r))
        push!(fusers,[unitary(fuse(vv'⊗vv),vv'⊗vv) for _ in 1:3])
    end
    N.NativeFactors((geometry=g,),qs,rs,fusers)
end

function factor_hash(f)
    io=IOBuffer()
    for t in [f.Q;f.R]
        # TensorKit 0.14 lacks Array(::Tensor); read its stored blocks instead.
        for (sector,b) in blocks(t)
            write(io,string(sector))
            write(io,reinterpret(UInt8,vec(Matrix(b))))
        end
    end
    bytes2hex(sha256(take!(io)))
end

function sweeps_without_bottom(ctx,positions;chi,sweeps)
    V.large_cell_sweeps(ctx,positions;chi,sweeps,extra_bottom_warmup=false)
end

function compare_case(f,env,ctm;chi,sweeps,grid=9,windows=[1,3],shift=1.,out)
    started=time();g=f.state.geometry
    grid>=maximum(windows)+6 || error("Insufficient padding for local patches")
    all(isodd,windows) || error("This diagnostic uses odd centered windows")
    catalogue=V.bond_catalogue(g)
    baseline=V.base_context(f,env,grid)
    means=Dict{Tuple{Int,Int,Symbol,Int},ComplexF64}()
    key(x,y,o,e)=(mod1(x,g.Lx),mod1(y,g.Ly),o,e)
    for x in 1:g.Lx,y in 1:g.Ly,o in (:up,:down),e in 1:3
        means[key(x,y,o,e)]=V.bond_expectation(baseline,x,y,o,e)
    end
    before=factor_hash(f)
    rows=Dict[];anchor_rows=Dict[];self_errors=[0.,0.]
    radius=div(maximum(windows),2)
    try
        for (index,term) in enumerate(catalogue)
            choices=[(x,y) for x in 2:grid-3 for y in 2:grid-3
                if N.owner(g,x,y)==term.representative && x-radius>=2 && y-radius>=2 &&
                   x+radius<=grid-2 && y+radius<=grid-2]
            x,y=argmin(p->abs(p[1]-(grid+1)/2)+abs(p[2]-(grid+1)/2),choices)
            ctx,cq,cr=V.central_context(f,env,grid,x,y,term.orientation,term.edge;shift)
            support=unique([cq;cr]);values=Vector{Vector{ComplexF64}}();times=Float64[]
            targets=[(dx,dy,o,e) for dx in -radius:radius for dy in -radius:radius
                     for o in (:up,:down) for e in 1:3]
            e0=means[key(x,y,term.orientation,term.edge)]
            for extra_bottom in (true,false)
                clock=time()
                mix=extra_bottom ? V.large_cell_sweeps(ctx,support;chi,sweeps,extra_bottom_warmup=true) :
                                   sweeps_without_bottom(ctx,support;chi,sweeps)
                covariance=ComplexF64[]
                for (dx,dy,o,e) in targets
                    eb=means[key(x+dx,y+dy,o,e)]
                    value=(shift+e0)*(V.bond_expectation(mix,x+dx,y+dy,o,e)-eb)
                    isfinite(real(value)) && isfinite(imag(value)) || error("Nonfinite correlation")
                    push!(covariance,value)
                end
                push!(times,time()-clock);push!(values,covariance)
            end
            for (i,(dx,dy,o,e)) in enumerate(targets)
                old,new=values[1][i],values[2][i]
                if dx==0 && dy==0 && o==term.orientation && e==term.edge
                    exact=3/16-e0/2-e0^2
                    self_errors[1]=max(self_errors[1],abs(old-exact))
                    self_errors[2]=max(self_errors[2],abs(new-exact))
                end
                push!(rows,Dict("source_representative"=>term.representative,
                    "source_orientation"=>string(term.orientation),"source_edge"=>term.edge,
                    "source_x"=>x,"source_y"=>y,"dx"=>dx,"dy"=>dy,
                    "target_orientation"=>string(o),"target_edge"=>e,
                    "old_real"=>real(old),"old_imag"=>imag(old),
                    "new_real"=>real(new),"new_imag"=>imag(new),"absolute_difference"=>abs(new-old)))
            end
            push!(anchor_rows,Dict("anchor"=>index,"old_seconds"=>times[1],"new_seconds"=>times[2]))
            println("chi=$chi sweeps=$sweeps anchor=$index/$(length(catalogue)); max pair delta=",
                    maximum(abs.(values[2].-values[1])));flush(stdout)
        end
    finally
        V.set_dimensions(g.Lx,g.Ly)
    end
    factor_hash(f)==before || error("Comparison changed the input factors")
    summary=Dict[];nspin=3length(g.representatives)
    for w in windows
        subset=filter(r->max(abs(r["dx"]),abs(r["dy"]))<=div(w,2),rows)
        old=sum(r["old_real"]+im*r["old_imag"] for r in subset)/nspin
        new=sum(r["new_real"]+im*r["new_imag"] for r in subset)/nspin
        push!(summary,Dict("D"=>2,"chi"=>chi,"sweeps"=>sweeps,"window_L"=>w,
            "contraction_grid"=>grid,"old_variance"=>real(old),"new_variance"=>real(new),
            "absolute_difference"=>abs(new-old),"relative_difference"=>abs(new-old)/max(abs(old),eps()),
            "old_imaginary"=>imag(old),"new_imaginary"=>imag(new),
            "max_pair_difference"=>maximum(r["absolute_difference"] for r in subset),
            "old_self_identity_error"=>self_errors[1],"new_self_identity_error"=>self_errors[2],
            "seconds"=>time()-started))
    end
    result=Dict("format"=>"kagome_bottom_warmup_comparison_v1","D"=>2,"chi"=>chi,
        "sweeps"=>sweeps,"grid"=>grid,"window_sizes"=>windows,"source_shift"=>shift,
        "state_sha256"=>before,"ctm"=>ctm,"summary"=>summary,
        "full_anchor_count"=>length(catalogue),"connected_pair_count"=>length(rows),
        "input_unchanged"=>true,"timing_note"=>"First calls include compilation; times are not a fair speed comparison",
        "interpretation"=>"Finite-window exchange variance density. Agreement of schedules is not an independent correctness proof.")
    mkpath(out)
    N.write_csv(joinpath(out,"summary.csv"),summary)
    N.write_csv(joinpath(out,"connected_comparison.csv"),rows)
    N.write_csv(joinpath(out,"anchor_timing.csv"),anchor_rows)
    N.atomic_toml(joinpath(out,"comparison.toml"),result)
    N.write_json(joinpath(out,"comparison.json"),result)
    println("CASE FINISHED: ",summary);flush(stdout)
    summary
end

function run_comparison(;out,chis=[4,8],sweep_counts=[1,2],seed=701,threads=10)
    ispath(out) && error("Choose a fresh output directory")
    mkpath(out);BLAS.set_num_threads(threads);N.prepare_device!("cpu")
    f=dense_D2_fixture(;seed);allrows=Dict[]
    meta=Dict("seed"=>seed,"D"=>2,"julia"=>string(VERSION),"TensorKit"=>string(pkgversion(TensorKit)),
        "threads"=>threads,"chi_values"=>chis,"sweep_counts"=>sweep_counts,"state_sha256"=>factor_hash(f))
    N.atomic_toml(joinpath(out,"settings.toml"),meta)
    for chi in chis
        println("NORM START: D2, chi=$chi");flush(stdout)
        env,ctm=N.native_ctmrg(f;chi,iterations=50,tolerance=1e-10,verbose=true)
        for sweeps in sweep_counts
            rows=compare_case(f,env,ctm;chi,sweeps,out=joinpath(out,"chi$(chi)_sweeps$(sweeps)"))
            append!(allrows,rows);N.write_csv(joinpath(out,"summary.csv"),allrows)
        end
    end
    println("ALL COMPARISONS FINISHED: ",out);flush(stdout)
    allrows
end
export run_comparison
end
