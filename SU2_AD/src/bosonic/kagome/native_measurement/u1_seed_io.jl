# Small, version-neutral U1 seeds, including lambdas. No CTM double layers.
# Included after server_workflow.jl, which supplies the existing state types/I/O.
function u1_seed_space_data(V)
    primal=isdual(V) ? V' : V
    Dict("dual"=>isdual(V),"sectors"=>[
        Dict("charge"=>Int(q.charge),"multiplicity"=>dim(primal,q)) for q in sectors(primal)])
end
function u1_seed_tensor_data(t)
    A=convert(Array,t)
    Dict("dims"=>collect(size(A)),"real"=>real.(vec(A)),"imag"=>imag.(vec(A)),
        "codomain"=>[u1_seed_space_data(V) for V in codomain(t)],
        "domain"=>[u1_seed_space_data(V) for V in domain(t)],
        "axis_charges"=>[ServerU1.u1_axis_charges(space(t,i)) for i in 1:numind(t)])
end
function u1_seed_restore_space(d)
    W=U1Space((q["charge"]=>q["multiplicity"] for q in d["sectors"])...)
    d["dual"] ? W' : W
end
function u1_seed_restore_tensor(d)
    cod=prod(u1_seed_restore_space.(d["codomain"]);init=one(U1Space(0=>1)))
    dom=prod(u1_seed_restore_space.(d["domain"]);init=one(U1Space(0=>1)))
    A=reshape(ComplexF64.(d["real"],d["imag"]),Tuple(d["dims"]))
    prototype=TensorMap(zeros(ComplexF64,size(A)),cod,dom;tol=1e-11)
    # Sector ordering can differ between TensorKit versions. Match each charge
    # and its multiplicity explicitly on every axis, including incoming duals.
    orders=Vector{Int}[]
    for i in 1:numind(prototype)
        old=Int.(d["axis_charges"][i]);new=ServerU1.u1_axis_charges(space(prototype,i))
        positions=Dict(q=>findall(==(q),old) for q in unique(old))
        counts=Dict(q=>0 for q in keys(positions));order=Int[]
        for q in new
            haskey(positions,q) || error("missing U1 charge in seed")
            counts[q]+=1;push!(order,positions[q][counts[q]])
        end
        sort(order)==collect(eachindex(old)) || error("U1 seed multiplicity mismatch")
        push!(orders,order)
    end
    expected=A[orders...];t=TensorMap(expected,cod,dom;tol=1e-11)
    norm(convert(Array,t)-expected)<1e-10*max(norm(expected),1) || error("U1 seed reconstruction changed coefficients")
    t
end
function export_u1_seed(path,s;metadata=Dict{String,Any}())
    ServerU1.check_u1(s)
    g=s.geometry;fields=fieldnames(ServerBase.KagomeSUTensors)
    data=Dict("format"=>"kagome_u1_seed_v1","charge_unit"=>"2Sz",
        "Lx"=>g.Lx,"Ly"=>g.Ly,"pattern"=>string(g.pattern),"metadata"=>metadata,
        "auxiliary_charges"=>Dict(string(k)=>v for (k,v) in s.auxiliary_charges),
        "cells"=>[Dict(string(n)=>u1_seed_tensor_data(getfield(c,n)) for n in fields
                      if !isnothing(getfield(c,n))) for c in s.cells])
    server_atomic_toml(path,data)
    path
end
function load_u1_seed(path)
    endswith(lowercase(path),".jld2") && return first(ServerU1.load_u1_checkpoint(path))
    data=TOML.parsefile(path)
    data["format"]=="kagome_u1_seed_v1" && data["charge_unit"]=="2Sz" || error("unknown U1 seed format")
    fields=fieldnames(ServerBase.KagomeSUTensors)
    cells=[ServerBase.KagomeSUTensors((haskey(c,string(n)) ? u1_seed_restore_tensor(c[string(n)]) : nothing
                                     for n in fields)...) for c in data["cells"]]
    g=ServerBase.KagomeSUCellGeometry(data["Lx"],data["Ly"];pattern=Symbol(data["pattern"]))
    charges=Dict(parse(Int,k)=>Int(v) for (k,v) in data["auxiliary_charges"])
    ServerU1.check_u1(ServerU1.U1KagomeState(g,cells,charges))
end
