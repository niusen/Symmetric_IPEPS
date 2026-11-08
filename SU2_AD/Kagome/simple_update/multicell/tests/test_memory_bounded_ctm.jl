# Small-D fixed-state comparisons only. Never starts production SU or large D.
using Test, TensorKit, LinearAlgebra, Random, TOML, SHA
root=get(ENV,"KAGOME_TEST_ROOT",normpath(joinpath(@__DIR__,"..","..","..","..")))
src=joinpath(root,"src","bosonic","kagome","native_measurement")
patch=get(ENV,"KAGOME_MEMORY_PATCH",joinpath(src,"memory_bounded_ctm.jl"))
report_dir=get(ENV,"KAGOME_MEMORY_TEST_OUT",@__DIR__)
mkpath(report_dir)

# Load precisely the original methods first, even after the patch is installed.
entry=joinpath(src,"KagomeNativeMeasurement.jl")
code=replace(read(entry,String),"include(\"memory_bounded_ctm.jl\")"=>"")
code=replace(code,r"include\(\"([^\"]+)\"\)"=>s->"include("*repr(joinpath(src,match(r"\"([^\"]+)\"",s)[1]))*")")
Base.include_string(Main,code,entry)
include(joinpath(src,"u1_server_workflow.jl"))
const N=KagomeNativeMeasurement
const C=N.CTM
BLAS.set_num_threads(10)
N.prepare_device!("cpu")

# Independent explicit local-AA reference: no CTM/Q/R five-factor expression.
module LocalAAReference
using TensorKit
function aa(q,r;charged=false)
    if charged
        @tensor a[:] := q[-1,1,-4,-5]*r[-2,-3,1]
    else
        @tensor a[:] := q[-1,1,-4]*r[-2,-3,1]
    end
    a
end
end
original=read(joinpath(src,"factorized_ctm.jl"),String)
original=original[findfirst("function factor_corner",original).start:findfirst("function split_corner",original).start-1]
Base.include_string(LocalAAReference,original,"legacy_corner_edge_reference.jl")
# Generate AA contractions by replacing q/r with their exact combined AA.
refcode=replace(original,"factor_corner"=>"corner_aa","factor_edge"=>"edge_aa")
refcode=replace(refcode,r"@tensor order=\([\d,]+\) value\[:\] := [^\n]+"=>function(s)
    charged=occursin("q[",s) && length(split(match(r"q\[([^\]]+)\]",s)[1],','))==4
    qi=split(match(r"q\[([^\]]+)\]",s)[1],',');ri=split(match(r"r\[([^\]]+)\]",s)[1],',')
    labels=[qi[1],ri[1],ri[2],qi[3]]
    charged && push!(labels,qi[4])
    expr=split(s,":=")[2]
    expr=replace(expr,r"q\[[^\]]+\]\*r\[[^\]]+\]"=>"combined["*join(labels,',')*"]")
    inds=sort(unique(parse.(Int,[m.match for m in eachmatch(r"(?<![\d-])[1-9]\d*",expr)])))
    "combined=aa(q,r;charged="*string(charged)*")\n        @tensor order=("*join(inds,',')*") value[:] :="*expr
end)
Base.include_string(LocalAAReference,refcode,"local_AA_corner_edge_reference.jl")

seed=joinpath(root,"Kagome","simple_update","multicell","data","u1_D4_optimized_seed.toml")
seed_sha=bytes2hex(sha256(read(seed)));u4=load_u1_seed(seed)
# A generic complex D2 state with exactly the same geometry/leg convention;
# not a product state and not artificially in the 1/9 sector.
rng=MersenneTwister(4102);V=ComplexSpace(2)
function random_dense_like(t;aux=false)
    spaces=[begin
        w=aux && i==4 ? ComplexSpace(1) : V
        isdual(space(t,i)) ? w' : w
    end for i in 1:numind(t)]
    co=prod(spaces[1:numout(t)];init=one(V))
    dom=prod([w' for w in spaces[numout(t)+1:end]];init=one(V))
    result=randn(rng,ComplexF64,co,dom);result/norm(result)
end
dcells=N.BS.KagomeSUTensors[]
for c in u4.cells
    tu=random_dense_like(c.T_u;aux=numind(c.T_u)==4)
    push!(dcells,N.BS.KagomeSUTensors(tu,random_dense_like(c.T_d),
        random_dense_like(c.B_a),random_dense_like(c.B_b),random_dense_like(c.B_c),
        nothing,nothing,nothing,nothing,nothing,nothing))
end
d2=N.DenseKagomeState(u4.geometry,dcells,copy(u4.auxiliary_charges))
sources=[("dense_D2_random",d2),("u1_D4_optimized",u4)]
baseline=Dict();factors=Dict();environments=Dict();ctminfo=Dict()
for (label,state) in sources
    println("BASELINE ",label);flush(stdout)
    f=N.build_factors(state);env,info=N.native_ctmrg(f;chi=4,iterations=12,tolerance=1e-6,verbose=true)
    result=N.native_observables(f,env;h=.38)
    baseline[label]=result;factors[label]=f;environments[label]=env;ctminfo[label]=info
    println("BASELINE E=",result["energy_per_spin"]," m=",result["m_normalized"]);flush(stdout)
end
Base.include(N,patch)
report=Dict{String,Any}("TensorKit"=>string(pkgversion(TensorKit)),"julia"=>string(VERSION),
    "patch_sha256"=>bytes2hex(sha256(read(patch))),"chi"=>4,"ctm_iterations_max"=>12,
    "h"=>.38,"seed_sha256"=>seed_sha)

@testset "all directions, charged corners, and explicit single-site AA" begin
    rng=MersenneTwister(47);v=CartesianSpace(3)
    tm(n)=randn(rng,ComplexF64,v^n,one(v))
    worst=0.0
    for mode in (:LU,:RU,:LD,:RD),d in 1:4,charged in (false,true)
        q=tm(charged ? 4 : 3);r=tm(3)
        es=mode==:LU ? (tm(2),tm(3),tm(3)) : mode==:RD ? (tm(3),tm(3),tm(2)) : (tm(3),tm(2),tm(3))
        a=LocalAAReference.factor_corner(q,r,es...,mode,d;charged)
        b=C.factor_corner(q,r,es...,mode,d;charged)
        aa=LocalAAReference.corner_aa(q,r,es...,mode,d;charged)
        delta=norm(a-b)/max(norm(a),eps());worst=max(worst,delta)
        @test delta<2e-12
        @test norm(aa-b)/max(norm(aa),eps())<2e-12
    end
    for d in 1:4
        t,pi,p,q,r=(tm(3) for _ in 1:5)
        a=LocalAAReference.factor_edge(t,pi,p,q,r,d)
        b=C.factor_edge(t,pi,p,q,r,d)
        aa=LocalAAReference.edge_aa(t,pi,p,q,r,d)
        @test norm(a-b)/norm(a)<2e-12
        @test norm(aa-b)/norm(aa)<2e-12
    end
    report["corner_relative_difference_max"]=worst
    q,r=tm(3),tm(3);Q=C.LayerGrid(((q,),));R=C.LayerGrid(((r,),))
    for d in 1:4
        @test norm(C.get_AA_direction(Q,R,d,(1,1))-
            C.rotate_AA_direction(LocalAAReference.aa(q,r),d))<1e-11
    end
end

@testset "contraction shape bounds, no large tensors allocated" begin
    pattern=r"@tensor order=\(([\d,]+)\) value\[:\] := ([^\n]+)"
    oldlines=collect(eachmatch(pattern,original));newlines=collect(eachmatch(pattern,read(patch,String)))
    @test length(oldlines)==length(newlines)==36
    function peak(m,chi,delta,edge)
        labels=[Set(parse.(Int,split(t[1],','))) for t in eachmatch(r"\[([\d,\-]+)\]",m[2])]
        size(a)=prod(i==-5 ? 1 : i in (edge ? (1,4,-1,-3) : (1,2,-1,-3)) ? chi : delta for i in a)
        largest=maximum(size.(labels))
        for k in parse.(Int,split(m[1],','))
            hits=findall(a->k in a,labels);isempty(hits) && continue
            @assert length(hits)==2
            a,b=labels[hits];out=setdiff(union(a,b),intersect(a,b))
            largest=max(largest,size(out));deleteat!(labels,hits);push!(labels,out)
        end
        @assert length(labels)==1
        largest
    end
    for i in 1:36,(chi,delta) in ((40,144),(144,144),(40,225))
        @test peak(newlines[i],chi,delta,i>32)<=chi^2*delta^2
    end
    # RU,d=3,neutral is expression 13, exactly the server's failing corner.
    report["D12_chi40_shape_analysis"]=Dict("old_max_elements"=>peak(oldlines[13],40,144,false),
        "new_max_elements"=>peak(newlines[13],40,144,false),"is_RSS_measurement"=>false)
end

function compare_observables(a,b)
    metrics=Dict{String,Float64}()
    for k in ("energy_per_spin","exchange_energy_per_spin","field_energy_per_spin","mean_Sz","m_normalized")
        metrics[k]=abs(a[k]-b[k]);@test metrics[k]<1e-8
    end
    @test length(a["cells"])==length(b["cells"])==9
    for k in ("Sz","bonds_up","bonds_down","E_up","E_down","C_up","C_down")
        deltas=Float64[]
        for (ca,cb) in zip(a["cells"],b["cells"])
            @test (ca["x"],ca["y"],ca["representative"])==(cb["x"],cb["y"],cb["representative"])
            push!(deltas,maximum(abs,ca[k].-cb[k]))
        end
        metrics[k]=maximum(deltas);@test metrics[k]<1e-8
    end
    metrics
end

@testset "same small-D state, old vs new full CTMRG and every observable" begin
    for (label,state) in sources
        println("OPTIMIZED ",label);flush(stdout)
        f=factors[label];old=baseline[label]
        # Same environment isolates operator insertion/contraction changes.
        fixed=N.native_observables(f,environments[label];h=.38)
        fixed_delta=compare_observables(old,fixed)
        env,info=N.native_ctmrg(f;chi=4,iterations=12,tolerance=1e-6,verbose=true)
        fresh=N.native_observables(f,env;h=.38)
        full_delta=compare_observables(old,fresh)
        @test info["iterations"]==ctminfo[label]["iterations"]
        @test info["converged"]==ctminfo[label]["converged"]
        @test info["actual_corner_dimensions"]==ctminfo[label]["actual_corner_dimensions"]
        groups=N.grouped_observables(fresh,occursin("D2",label) ? 2 : 4)
        @test length(groups.sites)==9 && length(groups.bonds)==18 && length(groups.triangles)==6
        @test sum(x["energy"] for x in groups.bonds)/9≈fresh["exchange_energy_per_spin"] atol=1e-12
        @test fresh["energy_per_spin"]≈fresh["exchange_energy_per_spin"]+.38fresh["mean_Sz"] atol=1e-12
        report[label]=Dict("baseline"=>old,"optimized"=>fresh,"fixed_environment_max_differences"=>fixed_delta,
            "full_ctm_max_differences"=>full_delta,"ctm_old"=>ctminfo[label],"ctm_new"=>info)
        println(label," max observable difference=",maximum(values(full_delta)));flush(stdout)
    end
    @test bytes2hex(sha256(read(seed)))==seed_sha
end
@testset "warm small dense contractions allocate less" begin
    # D4 -> delta=16; chi=4. No large D tensor is allocated in this test.
    vd=CartesianSpace(16);vc=CartesianSpace(4)
    randtm(vs)=randn(MersenneTwister(length(vs)),ComplexF64,prod(vs),one(vc))
    q=randtm([vd,vd,vd]);r=randtm([vd,vd,vd])
    e1=randtm([vc,vd,vc]);e2=randtm([vc,vc]);e3=randtm([vc,vd,vc])
    LocalAAReference.factor_corner(q,r,e1,e2,e3,:RU,3)
    C.factor_corner(q,r,e1,e2,e3,:RU,3)
    GC.gc();a=@allocated LocalAAReference.factor_corner(q,r,e1,e2,e3,:RU,3)
    GC.gc();b=@allocated C.factor_corner(q,r,e1,e2,e3,:RU,3)
    @test b<a
    report["allocation_check"]=Dict("D"=>4,"chi"=>4,"old_allocated_bytes"=>a,"new_allocated_bytes"=>b)
    println("Warm allocations D4 chi4: old=",a," new=",b);flush(stdout)
end
open(io->TOML.print(io,report;sorted=true),joinpath(report_dir,"memory_bounded_ctm_checks.toml"),"w")
println("ALL CHECKS PASSED; no large-D evolution or measurement was run.")
