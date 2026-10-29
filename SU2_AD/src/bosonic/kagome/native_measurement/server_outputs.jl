# Included by the server task; numerical measurements live in the native module.
function server_dense_references(file=joinpath(SERVER_RUN_DIR,"data","dense_reference_h038.csv"))
    isfile(file) || return Dict[]
    lines=readlines(file);columns=split(first(lines),',');rows=Dict[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values=split(line,',');length(values)==length(columns) || error("invalid bundled dense reference")
        push!(rows,Dict(string(k)=>string(v) for (k,v) in zip(columns,values)))
    end
    rows
end
function server_collect_outputs(root;reference_file=joinpath(SERVER_RUN_DIR,"data","dense_reference_h038.csv"),evolution_symmetry=:su2)
    rows=Dict[];sites=Dict[];triangles=Dict[];bonds=Dict[];spaces=Dict[]
    refs=server_dense_references(reference_file)
    folders=filter(p->isdir(p) && startswith(basename(p),"D") && !isnothing(tryparse(Int,basename(p)[2:end])),readdir(root;join=true))
    sort!(folders;by=p->parse(Int,basename(p)[2:end]))
    for folder in folders
        file=joinpath(folder,"ctm","observables.toml");isfile(file) || continue
        D=parse(Int,basename(folder)[2:end]);r=TOML.parsefile(file)
        virtual=TOML.parsefile(joinpath(folder,"virtual_spaces.toml"))["bonds"]
        actual=[v[side*"_dimension"] for v in virtual for side in ("up","down")]
        candidates=filter(v->parse(Int,v["D"])==D && abs(parse(Float64,v["h"])-r["settings"]["h"])<1e-12,refs)
        good=filter(v->lowercase(v["ctm_converged"])=="true",candidates)
        chosen=isempty(candidates) ? nothing : first(sort(isempty(good) ? candidates : good;by=v->parse(Float64,v["energy"])))
        row=Dict{String,Any}("D"=>D,"actual_D_min"=>minimum(actual),"actual_D_max"=>maximum(actual),
            "energy_per_spin"=>r["energy_per_spin"],"exchange_energy_per_spin"=>r["exchange_energy_per_spin"],
            "field_energy_per_spin"=>r["field_energy_per_spin"],"m_normalized"=>r["m_normalized"],
            "distance_to_one_ninth"=>r["distance_to_one_ninth"],"chi"=>r["settings"]["chi"],
            "ctm_iterations"=>r["ctm"]["iterations"],"ctm_converged"=>r["ctm"]["converged"],
            "ctm_error"=>r["ctm"]["error"],"max_corner_dimension"=>maximum(r["actual_corner_dimensions"]),
            "seconds"=>r["seconds"],"representation"=>r["settings"]["representation"],
            "dense_reference_energy"=>isnothing(chosen) ? "" : parse(Float64,chosen["energy"]),
            "dense_reference_chi"=>isnothing(chosen) ? "" : chosen["chi"],
            "dense_reference_m"=>isnothing(chosen) ? "" : chosen["m"],
            "dense_reference_ctm_converged"=>isnothing(chosen) ? "" : chosen["ctm_converged"],
            "dense_reference_source"=>isnothing(chosen) ? "" : chosen["source"],
            "delta_E_$(evolution_symmetry)_minus_dense"=>isnothing(chosen) ? "" : r["energy_per_spin"]-parse(Float64,chosen["energy"]))
        push!(rows,row);groups=ServerNative.grouped_observables(r,D)
        append!(sites,groups.sites);append!(triangles,groups.triangles);append!(bonds,groups.bonds)
        append!(spaces,[merge(Dict("D"=>D),v) for v in virtual])
    end
    for (name,values) in (("summary",rows),("site_magnetization",sites),("triangle_observables",triangles),
                          ("bond_energies",bonds),("virtual_spaces",spaces))
        ServerNative.write_csv(joinpath(root,name*".csv"),values)
    end
    ServerNative.write_json(joinpath(root,"summary.json"),rows)
    server_atomic_toml(joinpath(root,"summary.toml"),Dict("results"=>rows))
    open(joinpath(root,"summary.txt"),"w") do io
        println(io,"Pure Julia TensorKit; evolution=",uppercase(string(evolution_symmetry)),"; sqrt3xsqrt3; H=Hex+h*sum(Sz); projected 1/9 sector.")
        println(io,"Historical dense references retain their own chi/initialization/convergence flags.")
        for r in rows
            println(io,"D=",r["D"]," actualD=",r["actual_D_min"],"..",r["actual_D_max"],
                " E=",r["energy_per_spin"]," m=",r["m_normalized"]," CTM=",r["ctm_iterations"],
                " converged=",r["ctm_converged"]," error=",r["ctm_error"])
        end
    end
    rows
end
