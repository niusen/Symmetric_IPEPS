# Optional numerical audit of the original two-site SU routines. No tensor is
# rescaled or projected here; observations and snapshots are separate from SU.
mutable struct StarUpdateTrace
    folder::String
    streams::Dict{Symbol,IO}
    context::Any
    state::Any
    before::Any
    cache::Dict{Any,Any}
    first_loss::Bool
    sequence::Int
    last_stage::Symbol
    last_values::Any
end

_trace_cell(x)=x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x)
function _trace_line(io,values...)
    println(io,join(_trace_cell.(values),","));
end
_trace_prefix(t)=(t.sequence,t.context.sweep,t.context.layer,t.context.bond.direction,
    string(t.context.bond.site1),string(t.context.bond.site2));

function _trace_tensor_stats(A)
    lo=Inf;hi=0.;stored=0;zeros=0;finite=true;
    for (_,b) in blocks(A),v in (A isa DiagonalTensorMap ? diag(b) : b)
        a=abs(v);stored+=1;finite &= isfinite(a);hi=max(hi,a);
        if iszero(a);zeros+=1;else;lo=min(lo,a);end
    end
    return (norm=norm(A),min_nonzero=isinf(lo) ? 0. : lo,max_abs=hi,
        stored=stored,zeros=zeros,finite=finite,
        sectors=join(["$(c):$(size(b,1))x$(size(b,2))" for (c,b) in blocks(A)],";"));
end

function _trace_tensor(t,stage,name,A)
    n=_trace_tensor_stats(A);
    _trace_line(t.streams[:numbers],_trace_prefix(t)...,stage,name,n.norm,
        n.min_nonzero,n.max_abs,n.stored,n.zeros,n.finite,n.sectors);
    return n
end

function _trace_spectrum(t,stage,S)
    for (c,b) in blocks(S),k=1:size(b,1)
        _trace_line(t.streams[:spectrum],_trace_prefix(t)...,stage,string(c),Int(dim(c)),k,b[k,k]);
    end
end

function _trace_metric(t,stage,name,value)
    _trace_line(t.streams[:metrics],_trace_prefix(t)...,stage,name,value);
end

# With one virtual leg as codomain, each block is precisely that leg's sector.
# Frobenius norms include the irrep dimension, as does TensorKit.norm.
function _trace_leg_support(A,leg)
    M=TensorKit.permute(A,((leg,),Tuple(k for k=1:5 if k!=leg)));
    n=norm(A);
    return Dict(c=>sqrt(Float64(dim(c)))*norm(b)/n for (c,b) in blocks(M));
end

# Equivalent sector norms in the existing orthonormal fusion-tree basis.
# This computes all legs together without additional recoupling/permutation.
function _trace_fusion_support(A)
    TensorKit.numout(A)==5 && TensorKit.numin(A)==0 ||
        throw(ArgumentError("fusion support audit expects an all-outgoing five-leg site tensor"));
    result=[Dict{Any,Float64}() for _=1:5];n=norm(A);
    for (f,g) in fusiontrees(A)
        w=sqrt(Float64(dim(f.coupled)))*norm(A[f,g])/n;
        for leg=1:5
            c=f.uncoupled[leg];
            result[leg][c]=hypot(get(result[leg],c,0.),w);
        end
    end
    return result
end

function _trace_refresh!(t,sites)
    T=t.state[1];
    for site in sites
        x,y=site;
        supports=_trace_fusion_support(T[site...]);
        for leg in (1,3,iseven(x+y) ? 2 : 4)
            t.cache[(site,leg)]=supports[leg];
        end
    end
end

function _trace_support!(t,bond;initial=false)
    leg1,leg2=bond.axis==:horizontal ? (3,1) : (2,4);
    a=t.cache[(bond.site1,leg1)];b=t.cache[(bond.site2,leg2)];
    # Contracted legs are dual spaces. Compare labels in site2's orientation.
    left=Dict(HoneycombSU4.TensorKitSectors.dual(c)=>v for (c,v) in a);
    cs=union(keys(left),keys(b));compatible=0;overlap=0.;
    for c in cs
        l=get(left,c,0.);r=get(b,c,0.);
        compatible+=(l>0 && r>0);overlap+=l^2*r^2;
        _trace_line(t.streams[:support],_trace_prefix(t)...,bond.direction,
            string(bond.site1),string(bond.site2),string(c),Int(dim(c)),l,r);
    end
    _trace_metric(t,:support,"$(bond.direction)$(bond.site1)-$(bond.site2):overlap",overlap);
    if compatible==0
        initial && error("initial bond has no shared sector support: $bond");
        if !t.first_loss
            t.first_loss=true;
            _trace_snapshot(t,"first_lost_support";details=(affected_bond=bond,left,right=b));
            println("FIRST LOST SUPPORT ",(update=t.sequence,sweep=t.context.sweep,
                layer=t.context.layer,updated=t.context.bond,affected=bond));flush(stdout);
        end
    end
    return compatible
end

function _trace_snapshot(t,name;details=nothing)
    jldsave(joinpath(t.folder,name*".jld2");context=t.context,
        before_T=t.before[1],before_lx=t.before[2],before_ly=t.before[3],
        T_set=t.state[1],lambdax_set=t.state[2],lambday_set=t.state[3],
        stage=t.last_stage,values=t.last_values,details,
        julia_version=string(VERSION),tensorkit_version=string(Base.pkgversion(TensorKit)),
        sun_version=string(Base.pkgversion(SUNRepresentations)));
end

function (t::StarUpdateTrace)(stage;kwargs...)
    if stage==:begin_update
        t.sequence+=1;t.context=(;kwargs...);
        # The original routines replace tensors/weights; they do not mutate
        # input blocks. Shallow copies preserve the complete preceding state.
        t.before=map(copy,t.state);
    elseif stage==:input && haskey(kwargs,:gate)
        t.context=merge(t.context,(gate=kwargs[:gate],));
    elseif stage==:committed
        sites=(t.context.bond.site1,t.context.bond.site2);
        _trace_refresh!(t,sites);
        for bond in honeycomb_bonds(size(t.state[1])...)
            (bond.site1 in sites || bond.site2 in sites) || continue;
            _trace_support!(t,bond);
        end
        foreach(flush,values(t.streams));
        return nothing
    end
    t.last_stage=stage;t.last_values=(;kwargs...);
    for (name,A) in kwargs
        if A isa TensorKit.AbstractTensorMap
            n=_trace_tensor(t,stage,name,A);
            if !n.finite || !isfinite(n.norm) || n.norm==0
                _trace_snapshot(t,"first_zero_or_nonfinite";details=(name=name,stats=n));
                foreach(flush,values(t.streams));
                error("first zero/nonfinite object: stage=$stage name=$name update=$(t.sequence)");
            end
        elseif name==:lambdas
            for (k,B) in enumerate(A);_trace_tensor(t,stage,"lambda$k",B);end
        end
    end
    if stage in (:left_svd,:right_svd)
        S=kwargs[:s];_trace_spectrum(t,stage,S);
        horizontal=t.context.bond.axis==:horizontal;
        part=stage==:left_svd ? (horizontal ? ((1,2,4),(3,5)) : ((1,3,4),(2,5))) :
            (horizontal ? ((1,5),(2,3,4)) : ((4,5),(1,2,3)));
        A=kwargs[stage==:left_svd ? :T1 : :T2];
        M=TensorKit.permute(A,part);
        _trace_metric(t,stage,"svd_relative_residual",norm(M-kwargs[:u]*S*kwargs[:v])/norm(M));
    elseif stage==:bond
        M=TensorKit.permute(kwargs[:Tbond],((1,2),(3,4)));
        _,S,_=HoneycombSU4.tsvd(M);
        _trace_spectrum(t,:full_bond_svd,S);
        _trace_metric(t,stage,"full_D",dim(space(S,1)));
    elseif stage==:truncated
        S=kwargs[:s];_trace_spectrum(t,stage,S);
        M=TensorKit.permute(kwargs[:Tbond],((1,2),(3,4)));
        ratio=norm(S)/norm(M);
        _trace_metric(t,stage,"retained_norm_squared_fraction",ratio^2);
        _trace_metric(t,stage,"truncation_relative_residual",norm(M-kwargs[:u]*S*kwargs[:v])/norm(M));
        _trace_metric(t,stage,"kept_D",dim(space(S,1)));
    end
    return nothing
end

"""Trace every original two-site update, with native TensorKit dimension truncation.
Stops on the first zero/nonfinite intermediate, before normalization can hide it.
CSV files contain every substep; snapshots preserve the first lost shared support.
"""
function trace_star_simple_update(;folder,L=4,Dmax=24,dt=.01,sweeps=100,initial_checkpoint=nothing)
    ispath(folder) && error("trace output already exists: $folder");
    mkpath(folder);BLAS.set_num_threads(1);
    if isnothing(initial_checkpoint)
        T,lx,ly,_=initial_honeycomb_star_iPEPS(Lx=L,Ly=L);offset=0;
    else
        source=load(initial_checkpoint);
        T=source["T_set"];lx=source["lambdax_set"];ly=source["lambday_set"];
        offset=Int(source["sweep"]);
        size(T)==(L,L) || throw(DimensionMismatch("trace checkpoint cell does not match L"));
        _validate(T,lx,ly,Dmax);
    end
    streams=Dict{Symbol,IO}();
    prefix="update,sweep,layer,direction,site1,site2,";
    headers=(:numbers=>"stage,object,norm,min_nonzero,max_abs,stored,zeros,finite,sectors",
        :spectrum=>"stage,sector,irrep_dim,index,singular_value",
        :metrics=>"stage,metric,value",
        :support=>"bond_direction,bond_site1,bond_site2,sector,irrep_dim,left_relative_norm,right_relative_norm");
    for (name,header) in headers
        io=open(joinpath(folder,string(name)*".csv"),"w");streams[name]=io;println(io,prefix*header);
    end
    firstbond=first(honeycomb_bonds(L,L));
    tr=StarUpdateTrace(abspath(folder),streams,(sweep=offset,layer=0,bond=firstbond),
        (T,lx,ly),map(copy,(T,lx,ly)),Dict{Any,Any}(),false,offset*5*div(L*L,2),:initial,nothing);
    try
        _trace_refresh!(tr,Tuple.(CartesianIndices(T)));
        for bond in honeycomb_bonds(L,L);_trace_support!(tr,bond;initial=true);end
        jldsave(joinpath(folder,"state_initial.jld2");T_set=T,lambdax_set=lx,lambday_set=ly,sweep=offset,initial_checkpoint);
        println("TRACE START ",(L=L,Dmax=Dmax,dt=dt,sweeps=sweeps,truncation="TensorKit truncdim",
            start_sweep=offset,Julia=string(VERSION),TensorKit=string(Base.pkgversion(TensorKit)),SUN=string(Base.pkgversion(SUNRepresentations))));flush(stdout);
        for local_sweep=1:sweeps
            sweep=offset+local_sweep;
            HoneycombSU4.itebd_honeycomb(Dict("J"=>1.),T,lx,ly,dt,dt,Dmax;
                convention=:permutation,truncation=:legacy,method=:two_site,step_offset=sweep-1,verbose=false,trace=tr);
            println("TRACE SWEEP ",(sweep=sweep,updates=tr.sequence,
                min_tensor_norm=minimum(norm,T),max_tensor_norm=maximum(norm,T),lost_support=tr.first_loss));flush(stdout);
            if sweep%20==0 || local_sweep==sweeps
                jldsave(joinpath(folder,"state_step$(sweep).jld2");T_set=T,lambdax_set=lx,lambday_set=ly,sweep);
            end
        end
        return (status=:finished,updates=tr.sequence,T_set=T,lambdax_set=lx,lambday_set=ly)
    catch err
        _trace_snapshot(tr,"failure");
        open(joinpath(folder,"failure.log"),"w") do io;showerror(io,err,catch_backtrace());end
        showerror(stdout,err);println();flush(stdout);
        return (status=:failed,updates=tr.sequence,context=tr.context,T_set=T,lambdax_set=lx,lambday_set=ly)
    finally
        foreach(close,values(streams));
    end
end
