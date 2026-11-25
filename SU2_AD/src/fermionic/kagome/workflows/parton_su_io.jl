using SHA, MAT, JLD2

function read_parton_seed(path; family,model,name,source_hash,mat_hash,D)
    isfile(path) || error("missing D=$D input: $path")
    bytes2hex(open(sha256,path))==mat_hash || error("MAT SHA-256 mismatch: $path")
    fields=matopen(path) do f
        2^Int(read(f,"m"))==D || error("$name: expected D=$D")
        read(f,"source_sha256")==source_hash || error("$name: source hash mismatch")
        expected=family===:ipeps ? "cell_0" : "site_0_B"
        expected in keys(f) || error("$name: expected $family tensors")
        Tuple(vec(read(f,"fields")))
    end
    length(fields)==2 && all(isfinite,fields) || error("invalid chemical potentials in $name")
    (;path,fields,family,model,name,source_hash,mat_hash)
end

# Small text reports read only the two CTM measurements (no tensor contractions).
function write_table(writer,path)
    mkpath(dirname(path))
    mktemp(dirname(path)) do tmp,io
        writer(io);close(io)
        Base.Filesystem.rename(tmp,path)
    end
end
row(io,x...)=println(io,join(x,'\t'))
change(io,labels,old,new)=row(io,labels...,old,new,new-old)
dens(r)=(r.n_up_per_site+r.n_down_per_site)
mag(r)=(r.n_up_per_site-r.n_down_per_site)/2

function read_parton_measurements(dir,expected; mat_hash,source_hash,initial_D)
    function measurement(stage)
        jldopen(joinpath(dir,stage*".jld2"),"r") do f
            p=f["parameters"]; r=f["measurement"]
            p.provenance.mat_hash==mat_hash && p.provenance.source_hash==source_hash &&
                !p.provenance.projected || error("seed mismatch in $dir/$stage")
            all(getproperty(p,k)==v for (k,v) in pairs(expected)) ||
                error("model or settings differ in $dir/$stage")
            0<r.max_virtual_D<=p.Dmax && r.chi==p.chi || error("D/chi mismatch: $dir/$stage")
            all(isfinite,(r.energy_per_site,r.kinetic_per_site,r.n_up_per_site,r.n_down_per_site)) ||
                error("nonfinite observables: $dir/$stage")
            nblocks=prod(r.cell)
            for sites in (r.n_up_sites,r.n_down_sites)
                size(sites)==(nblocks,3) && all(isfinite,sites) || error("invalid site data: $dir/$stage")
            end
            for bonds in (r.up_hoppings,r.hoppings,r.up_bond_energy,r.down_bond_energy)
                size(bonds)==(nblocks,3,2) && all(isfinite,bonds) || error("invalid bond data: $dir/$stage")
            end
            (;r,p)
        end
    end
    first=measurement("initial");last=measurement("final")
    first.p==last.p && first.r.cell==last.r.cell || error("before/after settings or unit cell differ: $dir")
    first.r.max_virtual_D==initial_D || error("expected initial D=$initial_D in $dir")
    (;a=first.r,b=last.r,p=first.p)
end

function write_parton_comparison(dir,a,b,p)
    write_table(joinpath(dir,"comparison.tsv")) do io
        row(io,"observable","before","after","after_minus_before")
        for (label,fn) in (("grand_energy_per_site",r->r.energy_per_site),
                           ("kinetic_energy_per_site",r->r.kinetic_per_site),
                           ("n_up_per_site",r->r.n_up_per_site),("n_down_per_site",r->r.n_down_per_site),
                           ("n_per_site",dens),("Sz_per_site",mag),("M_over_Msat",r->2mag(r)),
                           ("max_virtual_D",r->r.max_virtual_D),("ctm_iterations",r->r.iterations),
                           ("ctm_error",r->r.ctm_error),("ctm_converged",r->Int(r.converged)))
            change(io,(label,),fn(a),fn(b))
        end
    end
    write_table(joinpath(dir,"sites.tsv")) do io
        row(io,"x","y","site","observable","before","after","after_minus_before")
        nx,ny=a.cell
        for y in 1:ny,x in 1:nx,j in 1:3
            i=x+(y-1)*nx
            up=a.n_up_sites[i,j],b.n_up_sites[i,j];dn=a.n_down_sites[i,j],b.n_down_sites[i,j]
            for (label,v) in (("n_up",up),("n_down",dn),("Sz",(up.-dn)./2))
                change(io,(x,y,("a","b","c")[j],label),v...)
            end
        end
    end
    write_table(joinpath(dir,"bonds.tsv")) do io
        row(io,"x","y","triangle","edge","spin","observable","before","after","after_minus_before")
        nx,ny=a.cell
        for y in 1:ny,x in 1:nx,up in (true,false),e in 1:3,σ in 1:2
            i=x+(y-1)*nx; key=(x,y,up ? "up" : "down",("ab","bc","ca")[e],σ==1 ? "up" : "down")
            h=map((a,b)) do r
                z=(up ? r.up_hoppings : r.hoppings)[i,e,σ]
                σ==1 ? z : -conj(z) # physical electron f↓ = b↓† on distinct sites
            end
            change(io,(key...,"Re_fdag_i_f_j"),real.(h)...)
            change(io,(key...,"Im_fdag_i_f_j"),imag.(h)...)
            energies=map(r->(up ? r.up_bond_energy : r.down_bond_energy)[i,e,σ],(a,b))
            change(io,(key...,"kinetic_on_bond"),energies...)
        end
    end
    steps=sum(round(Int,t/d) for (t,d) in p.schedule)
    status=a.converged && b.converged ? "complete" : "ctm_unconverged"
    (;status,energy_before=a.energy_per_site,energy_after=b.energy_per_site,
      values=(steps,p.chi,a.max_virtual_D,b.max_virtual_D,
       a.energy_per_site,b.energy_per_site,b.energy_per_site-a.energy_per_site,
       a.kinetic_per_site,b.kinetic_per_site,b.kinetic_per_site-a.kinetic_per_site,
       a.n_up_per_site,b.n_up_per_site,a.n_down_per_site,b.n_down_per_site,
       mag(a),mag(b),2mag(a),2mag(b),a.iterations,b.iterations,a.ctm_error,b.ctm_error,
       a.converged,b.converged))
end
