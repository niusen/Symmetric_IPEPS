using TensorKit, SUNRepresentations, JLD2, LinearAlgebra

"""Align bond duality conventions with paired inverse-transpose maps.
The two maps on each contracted bond cancel, so the PEPS is unchanged.
"""
function align_star_spaces(reference,T)
    nx,ny=size(T);maps=[Vector{Any}(undef,4) for x=1:nx,y=1:ny];
    for x=1:nx,y=1:ny
        u=mod1(x+1,nx);
        G=unitary(space(reference[x,y],3),space(T[x,y],3));
        H=TensorKit.permute(G',((2,),(1,)));
        @assert space(H,1)==space(reference[u,y],1) && space(H,2)'==space(T[u,y],1);
        maps[x,y][3]=G;maps[u,y][1]=H;
        v=mod1(y-1,ny);
        G=unitary(space(reference[x,y],2),space(T[x,y],2));
        H=TensorKit.permute(G',((2,),(1,)));
        @assert space(H,1)==space(reference[x,v],4) && space(H,2)'==space(T[x,v],4);
        maps[x,y][2]=G;maps[x,v][4]=H;
    end
    aligned=similar(T);
    for i in eachindex(T)
        G1,G2,G3,G4=maps[i];a=T[i];
        @tensor b[:]:=G1[-1,j]*G2[-2,k]*G3[-3,l]*G4[-4,m]*a[j,k,l,m,-5];
        @assert isapprox(norm(a),norm(b);atol=1e-12,rtol=1e-12);
        aligned[i]=b;
    end
    return aligned
end

function star_charge_weights(t,reallegs)
    a=TensorKit.permute(t,((1,2,3,4,5),()));
    weights=Dict{NTuple{3,Int},Float64}();
    for (f,g) in fusiontrees(a)
        q=ntuple(k->Int(SUNRepresentations.congruency(f.uncoupled[reallegs[k]])),3);
        weights[q]=get(weights,q,0.)+dim(f.coupled)*sum(abs2,a[f,g]);
    end
    @assert isapprox(sum(values(weights)),norm(a)^2;atol=1e-10,rtol=1e-10);
    return weights
end

function inspect_star_channels(folder=joinpath(@__DIR__,"../../data","star_SU_D9_L4"))
    for stage in (-1,0,1)
        T=load(joinpath(folder,"stage$(stage).jld2"),"T_set");
        outside=Float64[];
        for x=axes(T,1),y=axes(T,2)
            legs=iseven(x+y) ? (1,2,3) : (1,3,4);
            weights=star_charge_weights(T[x,y],legs);
            bad=sum(w for (q,w) in weights if !(q==(1,1,1) || sort(collect(q))==[0,0,3]);init=0.);
            push!(outside,bad/sum(values(weights)));
        end
        println("STAR CHARGES stage=",stage," maximum_outside_weight=",maximum(outside));
        flush(stdout);
    end
    initial=load(joinpath(folder,"stage-1.jld2"),"T_set");
    state=load(joinpath(folder,"stage1.jld2"));
    evolved=align_star_spaces(initial,state["T_set"]);
    jldsave(joinpath(folder,"stage1_aligned.jld2");T_set=evolved,
        history=state["history"],parameters=state["parameters"]);
    open(joinpath(folder,"star_channel_coefficients.csv"),"w") do io
        println(io,"x,y,q1,q2,q3,re,im,residual");
        max_residual=0.;
        for x=axes(initial,1),y=axes(initial,2)
            a=TensorKit.permute(initial[x,y],((1,2,3,4,5),()));
            b=TensorKit.permute(evolved[x,y],((1,2,3,4,5),()));
            space(a)==space(b) || error("virtual spaces changed; explicit gauge alignment is required");
            legs=iseven(x+y) ? (1,2,3) : (1,3,4);
            norms=Dict{NTuple{3,Int},Float64}();overlaps=Dict{NTuple{3,Int},ComplexF64}();
            for (f,g) in fusiontrees(a)
                q=ntuple(k->Int(SUNRepresentations.congruency(f.uncoupled[legs[k]])),3);
                norms[q]=get(norms,q,0.)+dim(f.coupled)*sum(abs2,a[f,g]);
                overlaps[q]=get(overlaps,q,0im)+dim(f.coupled)*dot(a[f,g],b[f,g]);
            end
            for (q,n) in norms
                n>1e-20 || continue;
                c=overlaps[q]/n;residual=0.;
                for (f,g) in fusiontrees(a)
                    qf=ntuple(k->Int(SUNRepresentations.congruency(f.uncoupled[legs[k]])),3);
                    qf==q || continue;
                    residual+=dim(f.coupled)*sum(abs2,b[f,g]-c*a[f,g]);
                end
                residual=sqrt(residual);max_residual=max(max_residual,residual);
                println(io,join((x,y,q...,real(c),imag(c),residual),","));
            end
        end
        println("STAR CHANNELS maximum_residual=",max_residual);flush(stdout);
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    inspect_star_channels();
end
