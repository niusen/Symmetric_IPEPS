using Random
function product_factors(;quantum=false,seed=61)
    g=N.BS.KagomeSUCellGeometry(3,3;pattern=:sqrt3xsqrt3)
    vp=U1Space(1=>1,-1=>1);vv=U1Space(0=>1)
    configs=((1,1,2),(2,2,1),(2,1,2));rng=MersenneTwister(seed)
    Qs=Any[];Rs=Any[];fs=Any[];vectors=Vector{ComplexF64}[]
    for config in configs
        q=sum(i==1 ? 1 : -1 for i in config)
        values=zeros(ComplexF64,2,2,2)
        if quantum
            for inds in CartesianIndices(values)
                sum(i==1 ? 1 : -1 for i in Tuple(inds))==q && (values[inds]=randn(rng)+im*randn(rng))
            end
            values/=norm(values)
        else
            values[config...]=1
        end
        P=vv⊗vv⊗vv⊗vp⊗vp⊗vp⊗U1Space(-q=>1)
        push!(Qs,TensorMap(reshape(values,1,1,1,2,2,2,1),P,one(P)))
        push!(Rs,TensorMap(ones(ComplexF64,1,1,1),vv'⊗vv'⊗vv',one(vv)))
        push!(fs,[unitary(fuse(vv'⊗vv),vv'⊗vv) for _ in 1:3]);push!(vectors,vec(values))
    end
    N.NativeFactors((geometry=g,),Qs,Rs,fs),vectors
end

function exact_product_correlator(f,vectors,source,tx,ty,orientation,edge)
    g=f.state.geometry;sx,sy=g.representatives[source.representative]
    sites0=V.bond_sites(sx,sy,source.orientation,source.edge)
    sitesb=V.bond_sites(sx+tx,sy+ty,orientation,edge)
    function factors(sites)
        a,b=sites
        [(1.,((a,N.SZ),(b,N.SZ))),(.5,((a,N.SP),(b,N.SP'))),(.5,((a,N.SP'),(b,N.SP)))]
    end
    function evaluate(ordered)
        products=Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        for (site,op) in ordered
            x,y,i=site;p=(x,y)
            products[p]=get(products,p,Matrix{ComplexF64}(I,8,8))*N.embedded_operator(i,op)
        end
        value=1.0+0im
        for ((x,y),op) in products
            psi=vectors[N.owner(g,x,y)];value*=dot(psi,op*psi)
        end
        value
    end
    aa=factors(sites0);bb=factors(sitesb)
    ea=sum(w*evaluate(ops) for (w,ops) in aa)
    eb=sum(w*evaluate(ops) for (w,ops) in bb)
    hh=sum(wb*wa*evaluate([opsb...;opsa...]) for (wa,opsa) in aa for (wb,opsb) in bb)
    hh-ea*eb
end
