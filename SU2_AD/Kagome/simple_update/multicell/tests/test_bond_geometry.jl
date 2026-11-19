# Independent kagome bond audit: Cartesian distances, bit-basis wavefunctions,
# and D=2 products of unequal complex three-spin cluster states.
# No production state optimization; no change to numerical measurement code.
using Test
include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","native_measurement","KagomeNativeMeasurement.jl"))
module KagomeBondAudit
using TensorKit, LinearAlgebra, Random, Test, TOML
import ..KagomeNativeMeasurement
const Native=KagomeNativeMeasurement
const BS=Native.BS
const a1=[2.0,0.0]
const a2=[1.0,sqrt(3.0)]
const translations=(a1+a2,-a1+2a2)
const basis=(-a1/2,zeros(2),-a2/2)  # B1, B2, B3 in the legacy Julia convention
const coordinates=[r*a1+basis[b] for r in 0:2 for b in 1:3]
edge(i,j)=minmax(i,j)
sub(id)=mod1(id,3)
cell(id)=div(id-1,3)+1

# This graph uses only Euclidean nearest-neighbor distance. In particular, it
# does NOT call representative(), triangle_sites(), or native bond lists.
function physical_graph()
    edges=Tuple{Int,Int}[]
    for i in 1:9,j in (i+1):9
        images=count(abs(norm(coordinates[j]+n*translations[1]+m*translations[2]-coordinates[i])-1)<1e-10
            for n in -3:3,m in -3:3)
        @test images<=1
        images==1 && push!(edges,(i,j))
    end
    @test length(edges)==18
    @test all(count(e->i in e,edges)==4 for i in 1:9)
    eset=Set(edges)
    triangles=[(i,j,k) for i in 1:9 for j in (i+1):9 for k in (j+1):9
        if edge(i,j) in eset && edge(i,k) in eset && edge(j,k) in eset]
    @test length(triangles)==6
    up=filter(t->length(unique(cell.(t)))==1,triangles)
    down=filter(t->length(unique(cell.(t)))!=1,triangles)
    @test length(up)==length(down)==3
    @test all(sort(collect(sub.(t)))==[1,2,3] for t in triangles)
    return edges,up,down
end

function identify(position)
    hits=[i for i in 1:9 if any(norm(position-coordinates[i]-n*translations[1]-m*translations[2])<1e-10
        for n in -3:3,m in -3:3)]
    only(hits)
end
function decode_vertex(rep,label)
    r=parse(Int,split(rep,',')[1]);b=parse(Int,string(label[2]))
    p=(r-1)*a1+basis[b]
    endswith(label,"+ex") && (p=p+a1)
    endswith(label,"+ey") && (p=p+a2)
    identify(p)
end

function production_graph(g)
    counts=Dict{Tuple{Int,Int},Int}()
    # Expand the actual 3x3 display BEFORE quotienting physical positions.
    # Count physical edges; each quotient bond should occur three times.
    for x in 1:g.Lx,y in 1:g.Ly
        p=(x-1)*a1+(y-1)*a2
        up=[identify(p+basis[b]) for b in 1:3]
        specs=((0,0,2),(0,1,3),(1,0,1))
        down=[identify(p+dx*a1+dy*a2+basis[b]) for (dx,dy,b) in specs]
        for triangle in (up,down),(i,j) in ((1,2),(2,3),(3,1))
            key=edge(triangle[i],triangle[j]);counts[key]=get(counts,key,0)+1
        end
    end
    counts
end

# Ordinary 512-component vector; reference observables do not use TensorKit,
# the production Hamiltonian, its physical_operators(), or its energy sum.
function exact_vector(amplitudes,clusters)
    psi=zeros(ComplexF64,512)
    for bits in 0:511
        psi[bits+1]=prod(amplitudes[r][(((bits>>(i-1))&1)+1 for i in ids)...] for (r,ids) in clusters)
    end
    psi/norm(psi)
end
function exact_bond(psi,i,j)
    result=0.0+0.0im
    for bits in 0:511
        a=(bits>>(i-1))&1;b=(bits>>(j-1))&1
        result+=abs2(psi[bits+1])*(.5-a)*(.5-b)
        if a!=b
            flipped=xor(bits,(1<<(i-1))|(1<<(j-1)))
            result+=.5conj(psi[flipped+1])*psi[bits+1]
        end
    end
    @test abs(imag(result))<1e-12
    real(result)
end
exact_sz(psi,i)=sum(abs2(psi[bits+1])*(.5-((bits>>(i-1))&1)) for bits in 0:511)

allout(A,spaces)=TensorMap(ComplexF64.(A),prod(spaces),one(first(spaces)))
function cluster_product(g,orientation,amplitudes)
    down=ComplexSpace(orientation==:down ? 2 : 1)
    up=ComplexSpace(orientation==:up ? 2 : 1)
    physical=ComplexSpace(2)
    B=zeros(ComplexF64,dim(down),dim(up),2)
    for p in 1:2
        B[orientation==:down ? p : 1,orientation==:up ? p : 1,p]=1
    end
    bond=allout(B,(down',up',physical))
    cells=BS.KagomeSUTensors[]
    for r in 1:3
        tu=allout(orientation==:up ? amplitudes[r] : ones(1,1,1),(up,up,up))
        td=allout(orientation==:down ? amplitudes[r] : ones(1,1,1),(down,down,down))
        push!(cells,BS.KagomeSUTensors(tu,td,copy(bond),copy(bond),copy(bond),ntuple(_->nothing,6)...))
    end
    Native.DenseKagomeState(g,cells,Dict{Int,Int}())
end

# Same product states as above, but with genuine U1 intertwiners and exactly
# ONE projected auxiliary charge +1 on Tu(1), as in the production 1/9 state.
# Charge routing is chosen explicitly; all physical axes remain [up,down].
function u1_cluster_product(g,orientation,amplitudes,clusters)
    U1=Native.KagomeU1Cells
    physical=U1Space(1=>1,-1=>1)
    @test U1.u1_axis_charges(physical)==[1,-1]
    fixed=orientation==:up ? [0 0 0;0 -1 0;1 0 0] : [0 -1 0;0 0 0;-1 1 0]
    downs=Matrix{Any}(undef,3,3);ups=similar(downs);bonds=similar(downs)
    active=similar(downs)
    for r in 1:3,b in 1:3
        flux=fixed[r,b]
        carrier=U1Space((q-flux=>1 for q in (1,-1))...)
        vacuum=U1Space(flux=>1)
        downs[r,b]=orientation==:down ? carrier : vacuum
        ups[r,b]=orientation==:up ? carrier : vacuum
        active[r,b]=carrier
        B=zeros(ComplexF64,dim(downs[r,b]),dim(ups[r,b]),2)
        charges=U1.u1_axis_charges(carrier)
        for (p,q) in enumerate((1,-1))
            k=only(findall(==(q-flux),charges))
            B[orientation==:down ? k : 1,orientation==:up ? k : 1,p]=1
        end
        bonds[r,b]=allout(B,(downs[r,b]',ups[r,b]',physical))
    end
    function core(r,ids,V,entangled,auxiliary)
        shape=Tuple(dim(v) for v in V)
        A=zeros(ComplexF64,auxiliary ? (shape...,1) : shape)
        if entangled
            for p in CartesianIndices((2,2,2))
                a=amplitudes[r][p];iszero(a) && continue
                index=ntuple(b->begin
                    id=ids[b];flux=fixed[cell(id),sub(id)];q=p[b]==1 ? 1 : -1
                    only(findall(==(q-flux),U1.u1_axis_charges(V[b])))
                end,3)
                A[(auxiliary ? (index...,1) : index)...]=a
            end
        else
            A.=1
        end
        allout(A,auxiliary ? (V...,U1Space(1=>1)) : V)
    end
    down_ids=Dict(cell(sort(collect(t);by=sub)[2])=>sort(collect(t);by=sub) for t in physical_graph()[3])
    cells=BS.KagomeSUTensors[]
    for r in 1:3
        uid=[3(r-1)+b for b in 1:3];did=down_ids[r]
        uv=Tuple(ups[r,b] for b in 1:3)
        dv=Tuple(downs[cell(id),sub(id)] for id in did)
        tu=core(r,uid,uv,orientation==:up,r==1)
        td=core(r,did,dv,orientation==:down,false)
        push!(cells,BS.KagomeSUTensors(tu,td,(bonds[r,b] for b in 1:3)...,ntuple(_->nothing,6)...))
    end
    U1.check_u1(U1.U1KagomeState(g,cells,Dict(1=>1)))
end

function run_audit(out)
    mkpath(out);BLAS.set_num_threads(2)
    edges,up,down=physical_graph()
    g=BS.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    counts=production_graph(g)
    @test Set(keys(counts))==Set(edges)
    @test all(==(3),values(counts))
    @test sum(values(counts))==54
    # Also compare the SU triangle geometry, without using it in the reference.
    for r in 1:3,orientation in (:up,:dn)
        sites=BS.triangle_sites(g,r,orientation)
        actual=sort([3(rr-1)+b for (rr,b) in sites])
        expected=orientation==:up ? up : down
        @test Tuple(actual) in expected
    end
    Native.write_csv(joinpath(out,"geometry_bonds.csv"),[
        Dict("i"=>i,"j"=>j,"site_i"=>"R$(cell(i))_B$(sub(i))","site_j"=>"R$(cell(j))_B$(sub(j))",
             "display_multiplicity"=>counts[(i,j)]) for (i,j) in edges])
    reports=Dict[];comparisons=Dict[]
    modes=get(ENV,"KAGOME_BOND_AUDIT_U1_ONLY","false")=="true" ?
        [(:u1,:up),(:u1,:down)] : [(:dense,:up),(:dense,:down),(:u1,:up),(:u1,:down)]
    for (representation,orientation) in modes
        rng=MersenneTwister(orientation==:up ? 1701 : 1839)
        amplitudes=[let a=randn(rng,ComplexF64,2,2,2);a/norm(a) end for _ in 1:3]
        if representation==:u1
            for r in 1:3
                target=(-1,-1,1)[r]
                for p in CartesianIndices((2,2,2))
                    sum(p[b]==1 ? 1 : -1 for b in 1:3)==target || (amplitudes[r][p]=0)
                end
                amplitudes[r]./=norm(amplitudes[r])
            end
        end
        triples=orientation==:up ? up : down
        clusters=[begin
            ordered=sort(collect(t);by=sub)
            r=orientation==:up ? cell(first(ordered)) : cell(ordered[2])
            (r,ordered)
        end for t in triples]
        psi=exact_vector(amplitudes,clusters)
        expected=Dict(e=>exact_bond(psi,e...) for e in edges)
        reference_energy=sum(values(expected))/9
        reference_sz=[exact_sz(psi,i) for i in 1:9]
        state=representation==:u1 ? u1_cluster_product(g,orientation,amplitudes,clusters) : cluster_product(g,orientation,amplitudes)
        println("AUDIT D2 ",representation," complex unequal ",orientation," cluster product; exact Eex=",reference_energy);flush(stdout)
        result,_=Native.measure_state(state;representation,h=.38,device="cpu",chi=16,
            iterations=20,tolerance=1e-10,verbose=true)
        groups=Native.grouped_observables(result,2)
        @test length(groups.bonds)==18
        measured=Dict{Tuple{Int,Int},Float64}()
        for row in groups.bonds
            key=edge(decode_vertex(row["representative"],row["vertex_a"]),decode_vertex(row["representative"],row["vertex_b"]))
            @test !haskey(measured,key)
            measured[key]=row["energy"]
            push!(comparisons,Dict("representation"=>string(representation),"cluster_product"=>string(orientation),"representative"=>row["representative"],
                "orientation"=>row["orientation"],"vertex_a"=>row["vertex_a"],"vertex_b"=>row["vertex_b"],
                "i"=>key[1],"j"=>key[2],"native"=>row["energy"],"exact"=>expected[key],"difference"=>row["energy"]-expected[key]))
            @test row["energy"]≈expected[key] atol=1e-9
        end
        @test Set(keys(measured))==Set(edges)
        for row in groups.sites
            id=decode_vertex(row["representative"],row["site"]*"@R")
            @test row["Sz"]≈reference_sz[id] atol=1e-9
        end
        @test result["exchange_energy_per_spin"]≈reference_energy atol=1e-9
        @test result["field_energy_per_spin"]≈.38sum(reference_sz)/9 atol=1e-9
        @test result["energy_per_spin"]≈reference_energy+.38sum(reference_sz)/9 atol=1e-9
        @test result["ctm"]["converged"]
        representation==:u1 && (@test result["m_normalized"]≈1/9 atol=1e-9)
        max_error=maximum(abs(measured[e]-expected[e]) for e in edges)
        push!(reports,Dict("representation"=>string(representation),"cluster_product"=>string(orientation),"exact_exchange_per_spin"=>reference_energy,
            "native_exchange_per_spin"=>result["exchange_energy_per_spin"],"max_bond_error"=>max_error,
            "ctm_converged"=>result["ctm"]["converged"],"ctm_iterations"=>result["ctm"]["iterations"]))
        Native.atomic_toml(joinpath(out,"audit_summary.toml"),Dict("results"=>reports))
        Native.write_csv(joinpath(out,"bond_comparison.csv"),comparisons)
        println("AUDIT ",orientation," max bond error=",max_error);flush(stdout)
    end
    println("Independent Cartesian geometry: 9 sites, 18 unique bonds, degree 4, 6 triangles; each bond appears 3 times in the 27-site display.")
    out
end
end

@testset "Cartesian bonds and independent exact cluster-product energies" begin
    out=get(ENV,"KAGOME_BOND_AUDIT_OUT",mktempdir())
    KagomeBondAudit.run_audit(out)
end
