variance_window_contains(dx,dy,w)=(-div(w-1,2)<=dx<=div(w,2) && -div(w-1,2)<=dy<=div(w,2))

function exchange_variance(state;L=12,radii=[0,1,2],chi=8,ctm_iterations=30,
        ctm_tolerance=1e-8,sweeps=2,source_shift=1.,anchors=nothing,verbose=true,
        norm_environment=nothing,window_sizes=nothing)
    start=time()
    verbose && (println("Variance: building local factors (first call may compile Julia kernels)");flush(stdout))
    f=state isa N.NativeFactors ? state : N.build_factors(state);g=f.state.geometry
    factor_seconds=time()-start
    L%g.Lx==0 && L%g.Ly==0 && L>=6 || error("L must be >=6 and a multiple of the original display cell")
    windows=!isnothing(window_sizes)
    cutoffs=windows ? collect(window_sizes) : collect(radii)
    !isempty(cutoffs) && sort(unique(cutoffs))==cutoffs && first(cutoffs)>=(windows ? 1 : 0) || error("invalid summation windows")
    left=windows ? -div(last(cutoffs)-1,2) : -last(cutoffs)
    right=windows ? div(last(cutoffs),2) : last(cutoffs)
    contains(dx,dy,c)=windows ? variance_window_contains(dx,dy,c) : max(abs(dx),abs(dy))<=c
    source_shift>=1. || error("source_shift>=1 ensures a nonzero positive central overlap")
    chi>0 && sweeps>0 || error("invalid chi or sweeps")
    norm_start=time()
    if isnothing(norm_environment)
        verbose && (println("Variance: periodic norm CTMRG start");flush(stdout))
        original,ctm=N.native_ctmrg(f;chi,iterations=ctm_iterations,tolerance=ctm_tolerance,verbose)
    else
        # Reuse only the identical state's environment at identical CTM settings.
        # Mixed environments still have to be recomputed for EVERY L/source.
        norm_environment.f===f && norm_environment.chi==chi &&
            norm_environment.iterations==ctm_iterations && norm_environment.tolerance==ctm_tolerance ||
            error("Reused norm CTM state/settings do not match this calculation")
        original=norm_environment.environment;ctm=norm_environment.ctm
        set_dimensions(g.Lx,g.Ly)
        verbose && (println("Variance: reusing periodic norm CTM at chi=",chi);flush(stdout))
    end
    norm_seconds=time()-norm_start
    catalogue=bond_catalogue(g);selected=isnothing(anchors) ? catalogue : anchors
    full=length(selected)==length(catalogue) && Set(selected)==Set(catalogue)
    # Native cells can be rectangular. Their values are tiled into a square L.
    reference_start=time()
    verbose && (println("Variance: reference bond means start; norm CTMRG seconds=",round(norm_seconds;digits=3));flush(stdout))
    baseline=base_context(f,original,L)
    # Keep the original CTM coordinate phase in the disconnected subtraction.
    # Nominally equivalent positions can differ slightly at finite chi; using
    # a representative's mean everywhere would add an artificial long-range tail.
    means=Dict{Tuple{Int,Int,Symbol,Int},ComplexF64}()
    key(x,y,o,e)=(mod1(x,g.Lx),mod1(y,g.Ly),o,e)
    for x in 1:g.Lx,y in 1:g.Ly,o in (:up,:down),e in 1:3
        means[key(x,y,o,e)]=bond_expectation(baseline,x,y,o,e)
    end
    reference_seconds=time()-reference_start
    verbose && (println("Variance: reference means finished in ",round(reference_seconds;digits=3)," s; ",length(selected)," central bonds");flush(stdout))
    rows=Dict[];totals=zeros(ComplexF64,length(cutoffs));lc_start=time();anchor_timings=Dict[]
    try
        for (index,term) in enumerate(selected)
            choices=[(x,y) for x in 2:L-3 for y in 2:L-3 if N.owner(g,x,y)==term.representative &&
                x+left>=2 && y+left>=2 && x+right<=L-2 && y+right<=L-2]
            isempty(choices) && error("L is too small for the summation window and source representative")
            x,y=argmin(p->abs(p[1]-(L+1)/2)+abs(p[2]-(L+1)/2),choices)
            verbose && (println("Variance anchor $index/$(length(selected)): ",term," at ",(x,y));flush(stdout))
            center_start=time()
            ctx,cq,cr=central_context(f,original,L,x,y,term.orientation,term.edge;shift=source_shift)
            center_seconds=time()-center_start
            verbose && (println("  Center factors finished in ",round(center_seconds;digits=3)," s; mixed sweeps start");flush(stdout))
            support=unique([cq;cr])
            sweep_start=time();moves=Ref(0)
            progress=verbose ? info->begin
                moves[]+=1
                if moves[]%L==0
                    println("  Mixed sweep moves=",moves[],"; elapsed=",round(time()-sweep_start;digits=3)," s");flush(stdout)
                end
            end : nothing
            mixed=large_cell_sweeps(ctx,support;chi,sweeps,callback=progress)
            sweep_seconds=time()-sweep_start
            verbose && (println("  Mixed sweeps finished in ",round(sweep_seconds;digits=3)," s; target bonds start");flush(stdout))
            measurement_start=time()
            e0=means[key(x,y,term.orientation,term.edge)]+source_shift
            for dx in left:right,dy in left:right,orientation in (:up,:down),edge in 1:3
                xx,yy=x+dx,y+dy;r=N.owner(g,xx,yy)
                eb=means[key(xx,yy,orientation,edge)]
                ratio=bond_expectation(mixed,xx,yy,orientation,edge)
                covariance=e0*(ratio-eb)
                radius=max(abs(dx),abs(dy))
                for (j,c) in enumerate(cutoffs);contains(dx,dy,c) && (totals[j]+=covariance);end
                push!(rows,Dict("source_representative"=>term.representative,"source_orientation"=>string(term.orientation),
                    "source_x"=>x,"source_y"=>y,
                    "source_mean_real"=>real(e0-source_shift),"target_mean_real"=>real(eb),
                    "source_edge"=>term.edge,"dx"=>dx,"dy"=>dy,"target_representative"=>r,
                    "target_orientation"=>string(orientation),"target_edge"=>edge,"radius"=>radius,
                    "connected_real"=>real(covariance),"connected_imag"=>imag(covariance)))
            end
            measurement_seconds=time()-measurement_start
            push!(anchor_timings,Dict("representative"=>term.representative,
                "orientation"=>string(term.orientation),"edge"=>term.edge,
                "center_seconds"=>center_seconds,"sweep_seconds"=>sweep_seconds,
                "measurement_seconds"=>measurement_seconds))
            verbose && (println("  Anchor finished; target bonds seconds=",round(measurement_seconds;digits=3));flush(stdout))
        end
    finally
        set_dimensions(g.Lx,g.Ly)
    end
    nspin=3length(g.representatives)
    cumulative=[Dict((windows ? "window_size" : "radius")=>c,"variance_per_spin"=>real(totals[j])/nspin,
        "imaginary_per_spin"=>imag(totals[j])/nspin,
        "shell_increment_per_spin"=>real(totals[j]-(j==1 ? 0 : totals[j-1]))/nspin,
        "shell_absolute_sum_per_spin"=>sum(hypot(r["connected_real"],r["connected_imag"])
            for r in rows if contains(r["dx"],r["dy"],c) && (j==1 || !contains(r["dx"],r["dy"],cutoffs[j-1])))/nspin,
        "shell_max_abs_connected"=>maximum((hypot(r["connected_real"],r["connected_imag"])
            for r in rows if contains(r["dx"],r["dy"],c) && (j==1 || !contains(r["dx"],r["dy"],cutoffs[j-1])));init=0.))
        for (j,c) in enumerate(cutoffs)]
    Dict("format"=>"kagome_LC_exchange_variance_v1","full_anchor_sum"=>full,
         "n_physical_sites"=>nspin,"anchor_count"=>length(selected),"L"=>L,"chi"=>chi,
         "sweeps"=>sweeps,"source_shift"=>source_shift,"ctm"=>ctm,
         "summation_mode"=>windows ? "cell_windows" : "legacy_radius",
         "exchange_energy_per_spin"=>real(sum(values(means)))/(3g.Lx*g.Ly),
         "timing"=>Dict("factor_seconds"=>factor_seconds,"norm_ctm_seconds"=>norm_seconds,
             "norm_environment_reused"=>!isnothing(norm_environment),
             "reference_seconds"=>reference_seconds,"anchors"=>anchor_timings,
             "note"=>"Wall times include first-call Julia compilation; not peak-memory measurements"),
         "cumulative"=>cumulative,"connected_terms"=>rows,"seconds"=>time()-start,"lc_seconds"=>time()-lc_start,
         "interpretation"=>full ? "finite-window exchange variance density; check summation range, chi and sweeps" :
             "PARTIAL ANCHOR SUM: diagnostic only, not the variance")
end

function measure_variance_checkpoint(file,out;threads=10,kwargs...)
    ispath(out) && error("Choose a new output directory; existing results are not overwritten")
    BLAS.set_num_threads(threads);N.prepare_device!("cpu")
    state,_=N.KagomeU1Cells.load_u1_checkpoint(file)
    before=bytes2hex(sha256(read(file)))
    result=exchange_variance(state;kwargs...)
    result["source_checkpoint"]=abspath(file);result["source_sha256"]=before
    bytes2hex(sha256(read(file)))==before || error("source checkpoint changed during measurement")
    mkpath(out);N.write_json(joinpath(out,"variance.json"),result)
    N.write_csv(joinpath(out,"connected_terms.csv"),result["connected_terms"])
    table=result["summation_mode"]=="cell_windows" ? "variance_by_L.csv" : "variance_by_radius.csv"
    N.write_csv(joinpath(out,table),result["cumulative"])
    N.atomic_toml(joinpath(out,"variance.toml"),result)
    result
end
