# All measurement/checkpoint transport is Julia-native; JSON is an optional
# human-readable export. TOML/JLD2 are used for restart, not a Python process.
function atomic_toml(path,data)
    mkpath(dirname(path));tmp=tempname(dirname(path))
    try
        open(io->TOML.print(io,data;sorted=true),tmp,"w");mv(tmp,path;force=true)
    finally
        isfile(tmp) && rm(tmp)
    end
end
json_escape(s)=replace(string(s),'\\'=>"\\\\",'"'=>"\\\"",'\n'=>"\\n",'\r'=>"\\r",'\t'=>"\\t")
function json_value(io,v)
    if v isa AbstractDict || v isa NamedTuple
        print(io,'{')
        for (k,(key,value)) in enumerate(pairs(v))
            k>1 && print(io,',');print(io,'"',json_escape(key),"\":");json_value(io,value)
        end
        print(io,'}')
    elseif v isa Tuple || v isa AbstractArray
        print(io,'[')
        for (k,value) in enumerate(v);k>1 && print(io,',');json_value(io,value);end
        print(io,']')
    elseif v isa Bool
        print(io,v ? "true" : "false")
    elseif v isa Real
        print(io,isfinite(v) ? string(v) : "null")
    elseif isnothing(v)
        print(io,"null")
    else
        print(io,'"',json_escape(v),'"')
    end
end
function write_json(path,data)
    tmp=tempname(dirname(path));open(io->json_value(io,data),tmp,"w");mv(tmp,path;force=true)
end
function write_csv(path,rows)
    isempty(rows) && return
    columns=collect(keys(first(rows)));tmp=tempname(dirname(path))
    quote_csv(v)="\""*replace(string(v),'"'=>"\"\"")*"\""
    open(tmp,"w") do io
        println(io,join(quote_csv.(columns),","))
        for row in rows;println(io,join([quote_csv(get(row,k,"")) for k in columns],","));end
    end
    mv(tmp,path;force=true)
end
function grouped_observables(result,D)
    sites=Dict[];triangles=Dict[];bonds=Dict[]
    for rep in sort(unique(c["representative"] for c in result["cells"]))
        cells=filter(c->c["representative"]==rep,result["cells"])
        average(field)=mean(c[field] for c in cells)
        for b in 1:3
            z=mean(c["Sz"][b] for c in cells)
            push!(sites,Dict("D"=>D,"representative"=>rep,"site"=>"B$b","Sz"=>z,"m_normalized"=>-2z,
                "equivalent_position_spread"=>maximum(abs(c["Sz"][b]-z) for c in cells)))
        end
        for (orientation,vertices) in (("up",("B1@R","B2@R","B3@R")),("down",("B2@R","B3@R+ey","B1@R+ex")))
            push!(triangles,Dict("D"=>D,"representative"=>rep,"orientation"=>orientation,
                "energy"=>average("E_$orientation"),"C_SzSzSz"=>average("C_$orientation")))
            for (i,(a,b)) in enumerate(((1,2),(2,3),(3,1)))
                push!(bonds,Dict("D"=>D,"representative"=>rep,"orientation"=>orientation,
                    "vertex_a"=>vertices[a],"vertex_b"=>vertices[b],"energy"=>mean(c["bonds_$orientation"][i] for c in cells)))
            end
        end
    end
    (;sites,triangles,bonds)
end
function save_converted_state(path,state;h=.38)
    if state isa KagomeU1Cells.U1KagomeState
        return KagomeU1Cells.save_u1_checkpoint(path,state,KagomeU1Cells.u1_heisenberg_triangle();
            metadata=Dict("h"=>h,"measurement_backend"=>"Julia TensorKit"))
    end
    # Also expose ordinary Julia arrays using the user's B1/B2/B3/Tup/Tdn axes.
    cells=[(B1=convert(Array,c.B_a),B2=convert(Array,c.B_b),B3=convert(Array,c.B_c),
        Tup=numind(c.T_u)==4 ? copy(selectdim(convert(Array,c.T_u),4,1)) : convert(Array,c.T_u),
        Tdn=convert(Array,c.T_d)) for c in state.cells]
    JLD2.jldsave(path;format="kagome_dense_native_v1",state,cells,h)
    path
end
function _measure_state(source;representation=:u1,component=1,h=.38,chi=40,iterations=50,tolerance=1e-6,verbose=true)
    state=conversion_state(source;representation,component)
    start=time();f=build_factors(state)
    env,info=native_ctmrg(f;chi,iterations,tolerance,verbose)
    result=native_observables(f,env;h)
    result["ctm"]=info;result["seconds"]=time()-start
    result["actual_corner_dimensions"]=info["actual_corner_dimensions"]
    result["actual_D_max"]=maximum(dim(space(getfield(c,n),leg)) for c in state.cells for n in BS.B_FIELDS for leg in 1:2)
    return result,state
end
function measure_state(source;device="cpu",kwargs...)
    prepare_device!(device)
    Base.invokelatest(_measure_state,source;kwargs...)
end
function measure_checkpoint(input,out;representation=:u1,component=1,h=.38,chi=40,iterations=50,
                            tolerance=1e-6,device="cpu",threads=10,D=0,verbose=true)
    settings=Dict("input_sha256"=>bytes2hex(sha256(read(input))),"representation"=>string(representation),
        "component"=>component,"h"=>h,"chi"=>chi,"ctm_iterations"=>iterations,"ctm_tolerance"=>tolerance,
        "device"=>device,"threads"=>threads,"method"=>"pure Julia TensorKit factorized CTMRG","schema"=>2)
    target=joinpath(out,"observables.toml")
    if isfile(target)
        previous=TOML.parsefile(target)
        previous["settings"]==settings || error("measurement settings/input changed; use a new output folder")
        println("SKIP completed Julia measurement: ",target);return previous
    end
    data=JLD2.load(input)
    source=data["format"]=="kagome_su2_charged_sqrt3_v1" ? first(KagomeChargedSU2.load_charged_checkpoint(input)) :
           data["format"]=="kagome_u1_cells_v1" ? first(KagomeU1Cells.load_u1_checkpoint(input)) :
           data["format"]=="kagome_dense_native_v1" ? data["state"] : error("unsupported state")
    BLAS.set_num_threads(threads)
    result,state=measure_state(source;representation,component,h,chi,iterations,tolerance,device,verbose)
    result["settings"]=settings
    all(isfinite(result[k]) for k in ("energy_per_spin","mean_Sz")) || error("nonfinite measurement")
    mkpath(out);save_converted_state(joinpath(out,"$(representation)_state.jld2"),state;h)
    rows=grouped_observables(result,D)
    for (name,values) in (("site_magnetization",rows.sites),("triangle_observables",rows.triangles),("bond_energies",rows.bonds))
        write_csv(joinpath(out,name*".csv"),values)
    end
    write_json(joinpath(out,"observables.json"),result)
    JLD2.jldsave(joinpath(out,"observables.jld2");result)
    atomic_toml(target,result) # complete marker is written last
    println("FINAL Julia D=$D E=",result["energy_per_spin"]," m=",result["m_normalized"]," CTM=",result["ctm"])
    result
end
