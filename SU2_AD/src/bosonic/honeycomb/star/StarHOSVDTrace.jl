# Read-only audit callbacks for the star kernel. Extra complete SVDs are
# diagnostic only; the update still uses its own native truncdim result.
_star_audit_label(c)=join(SUNRepresentations.dynkin_label(c),"-")
_star_audit_csv(x)=x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x)
_star_audit_line(io,x...)=println(io,join(_star_audit_csv.(x),","))

function _star_audit_site_support(A)
    result=[Dict{Any,Float64}() for _=1:5];n2=norm(A)^2;
    for (f,g) in fusiontrees(A)
        w=Float64(dim(f.coupled))*norm(A[f,g])^2/n2;
        for leg=1:5
            c=f.uncoupled[leg];result[leg][c]=get(result[leg],c,0.)+w;
        end
    end
    return result
end

"""Trace every internal mode and every incident bond, from the strict D=9 state.

Weights in the CSV are gauge-dependent SU diagnostics, not global Schmidt
probabilities. Spectrum fractions include irrep dimensions. No finite-D or
threshold rule is changed. An audit-off replay checks the output tensors.
"""
function trace_honeycomb_star_hosvd(folder;sweeps=3,dt=.01,Dmax=9,trun_tol=1e-10)
    ispath(folder) && error("use a new audit directory: $folder");mkpath(folder);
    T,lx,ly,info=initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    streams=Dict(name=>open(joinpath(folder,"$name.csv"),"w") for name in
        (:spectra,:bonds,:transitions,:updates,:sweeps));
    _star_audit_line(streams[:spectra],"update","sweep","layer","group","cx","cy","mode","stage",
        "dynkin","irrep_dim","copy","value","sector_fraction_per_copy","cutoff","space_dim");
    _star_audit_line(streams[:bonds],"update","sweep","layer","group","cx","cy","mode","stage","role",
        "x","y","leg","neighbor_x","neighbor_y","dynkin","irrep_dim","copy","weight",
        "weight_fraction_per_copy","site_sector_fraction","neighbor_dual_sector_fraction","space_dim");
    _star_audit_line(streams[:transitions],"update","sweep","layer","group","cx","cy","mode",
        "input_dim","full_dim","dimension_cut_dim","tolerance_cut_dim","output_dim",
        "dimension_retained_fraction","tolerance_retained_fraction","input_sectors","full_sectors",
        "dimension_sectors","tolerance_sectors","cutoff");
    _star_audit_line(streams[:updates],"update","sweep","layer","group","cx","cy","joint_retained_fraction");
    _star_audit_line(streams[:sweeps],"sweep","minimum_dim","maximum_dim","replay_max_relative_error");
    count=Ref(0);context=Ref{Any}(nothing);before=Ref{Any}(nothing);
    cache=Dict{Any,Any}();modes=Dict{Any,Any}();first_drop=Ref(false);
    refresh(sites)=foreach(s->cache[s]=_star_audit_site_support(T[s...]),sites);
    refresh([(x,y) for x=1:4 for y=1:4]);
    prefix(e)=(count[],e.sweep,e.layer,e.group,e.center...);
    function spectrum(e,stage,mode,S;cutoff=0.)
        n2=norm(S)^2;d=dim(space(S,1));labels=String[];
        for (c,b) in blocks(S)
            push!(labels,"$(_star_audit_label(c)):$(size(b,1))");
            for k=1:size(b,1)
                v=real(b[k,k]);_star_audit_line(streams[:spectra],prefix(e)...,mode,stage,
                    _star_audit_label(c),Int(dim(c)),k,v,Float64(dim(c))*abs2(v)/n2,cutoff,d);
            end
        end
        return (dim=d,norm2=n2,sectors=join(sort(labels),";"),cutoff=cutoff)
    end
    function bonds(e,stage)
        g=e.geometry;
        entries=[(g.center,g.legs[i],i,"internal") for i=1:3];
        for i=1:3,leg in g.outer[i];push!(entries,(g.leaves[i],leg,0,"external"));end
        for (site,leg,mode,role) in entries
            x,y=site;other=leg==1 ? (mod1(x-1,4),y) : leg==3 ? (mod1(x+1,4),y) :
                leg==2 ? (x,mod1(y-1,4)) : (x,mod1(y+1,4));
            opposite=(3,4,1,2)[leg];L=_star_weight(lx,ly,site,leg);n2=norm(L)^2;
            for (c,b) in blocks(L),k=1:size(b,1)
                v=real(b[k,k]);dual_c=TensorKitSectors.dual(c);
                _star_audit_line(streams[:bonds],prefix(e)...,mode,stage,role,site...,leg,other...,
                    _star_audit_label(c),Int(dim(c)),k,v,Float64(dim(c))*abs2(v)/n2,
                    get(cache[site][leg],c,0.),get(cache[other][opposite],dual_c,0.),dim(space(L,1)));
            end
        end
    end
    function audit(e)
        context[]=e;
        if e.stage==:input
            count[]+=1;empty!(modes);bonds(e,:input);
            before[]=deepcopy((T=T,lx=lx,ly=ly,geometry=e.geometry));
            for i=1:3
                L=_star_weight(lx,ly,e.center,e.geometry.legs[i]);
                modes[(i,:input)]=spectrum(e,:input_weight,i,L);
            end
        elseif e.stage==:before_gate
            theta=e.theta/norm(e.theta);
            for (i,left) in enumerate(((2,3),(4,5),(6,7)))
                right=Tuple(k for k=1:7 if !(k in left));
                _,S,_=tsvd(permute(theta,(left,right)));
                spectrum(e,:before_gate,i,S);
            end
        elseif e.stage==:mode_full
            _,S,_=tsvd(e.matrix);
            modes[(e.mode,:full)]=spectrum(e,:full,e.mode,S);
        elseif e.stage in (:dimension_cut,:tolerance_cut)
            modes[(e.mode,e.stage)]=spectrum(e,e.stage,e.mode,e.S;cutoff=e.cutoff);
        elseif e.stage==:output
            g=e.geometry;refresh((g.center,g.leaves...));bonds(e,:output);
            for i=1:3
                out=spectrum(e,:output_weight,i,_star_weight(lx,ly,e.center,g.legs[i]));
                a=modes[(i,:input)];b=modes[(i,:full)];c=modes[(i,:dimension_cut)];d=modes[(i,:tolerance_cut)];
                _star_audit_line(streams[:transitions],prefix(e)...,i,a.dim,b.dim,c.dim,d.dim,out.dim,
                    c.norm2/b.norm2,d.norm2/c.norm2,a.sectors,b.sectors,c.sectors,d.sectors,d.cutoff);
                if out.dim<a.dim && !first_drop[]
                    first_drop[]=true;
                    JLD2.jldsave(joinpath(folder,"first_dimension_drop.jld2");before=before[],
                        after=deepcopy((T=T,lx=lx,ly=ly)),context=prefix(e),mode=i,statistics=copy(modes));
                    println("FIRST STAR DIMENSION DROP ",(update=count[],sweep=e.sweep,layer=e.layer,
                        center=e.center,mode=i,before=a.dim,after=out.dim));flush(stdout);
                end
            end
            _star_audit_line(streams[:updates],prefix(e)...,e.retained);
            foreach(flush,values(streams));
        end
    end
    try
        for sweep=1:sweeps
            plain=deepcopy((T,lx,ly));
            itebd_honeycomb_star(Dict("J"=>1.),T,lx,ly,dt,dt,Dmax;
                convention=:permutation,step_offset=sweep-1,trun_tol,verbose=false,audit);
            itebd_honeycomb_star(Dict("J"=>1.),plain...,dt,dt,Dmax;
                convention=:permutation,step_offset=sweep-1,trun_tol,verbose=false);
            errors=[norm(A-B)/norm(B) for (xs,ys) in zip((T,lx,ly),plain) for (A,B) in zip(xs,ys)];
            err=maximum(errors);err<1e-11 || error("audit changed the numerical update: $err");
            dims=[dim(space(T[b.site1...],b.axis==:horizontal ? 3 : 2)) for b in honeycomb_bonds(4,4)];
            _star_audit_line(streams[:sweeps],sweep,minimum(dims),maximum(dims),err);flush(streams[:sweeps]);
            JLD2.jldsave(joinpath(folder,"sweep$sweep.jld2");T,lx,ly,sweep,dt,Dmax,trun_tol,info);
            println("STAR AUDIT SWEEP ",(sweep=sweep,dimensions=extrema(dims),replay_error=err));flush(stdout);
        end
    finally
        foreach(close,values(streams));
    end
    return T,lx,ly
end
