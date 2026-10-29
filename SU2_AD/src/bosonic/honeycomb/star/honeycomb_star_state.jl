"""Strict star-covering state: only (4,4,4) and cyclic (4bar,1,1)
virtual channels in the all-outgoing convention. No imaginary-time evolution.
"""
function _initial_honeycomb_star_tensors(; Lx=2, Ly=2, centers=nothing)
    honeycomb_bonds(Lx,Ly);
    V=honeycomb_virtual_space(["1"=>1,"4"=>1,"4⁺"=>1]);
    p=Rep[SU₄]("4"=>1); o=Rep[SU₄]("1"=>1);
    W=p←V'*V'*V'; rng=MersenneTwister(142);
    U=TensorMap(randn(rng,dim(W)),W);
    A=TensorKit.permute(U,((2,3,4,1),()));
    function projector(label)
        P=0.0*unitary(V,V);
        c=first(sectors(Rep[SU₄](label=>1)));
        TensorKit.block(P,c)[1,1]=1;
        return P
    end
    I0=projector("1"); I4=projector("4"); Ib=projector("4⁺");
    @tensor center[:]:=I4[-1,a]*I4[-2,b]*I4[-3,c]*A[a,b,c,-4];
    @tensor leaf[:]:=Ib[-1,a]*I0[-2,b]*I0[-3,c]*A[a,b,c,-4];
    center=TensorKit.permute(center,((1,2,3,4),()));
    leaf=TensorKit.permute(leaf,((1,2,3,4),()));
    rotate(t)=TensorKit.permute(t,((2,3,1,4),()));
    @assert norm(center)>1e-10 && norm(leaf)>1e-10;
    # Unit norm center is epsilon/sqrt(24); unit norm leaf is delta/2.
    # All covers contain the same numbers of each, so this scale is immaterial.
    center/=norm(center); leaf/=norm(leaf);
    A=center+leaf+rotate(leaf)+rotate(rotate(leaf));
    @assert norm(A-rotate(A))<1e-12;
    @assert norm(center+TensorKit.permute(center,((2,1,3,4),())))<1e-12;
    F=unitary(V',V); bar4=first(sectors(Rep[SU₄]("4⁺"=>1)));
    TensorKit.block(F,bar4).*= -1;
    B=TensorKit.permute(F,((1,2),()));
    @assert norm(B-TensorKit.permute(B,((2,1),())))<1e-12;
    trivial=TensorMap(ones(1),o,one(o*o));
    trivial_up=TensorMap(ones(1),o',one(o*o));
    @tensor even[:]:=A[-1,-2,a,-5]*B[-3,a]*trivial_up[-4];
    @tensor odd[:]:=A[-1,a,b,-5]*B[-3,a]*B[-4,b]*trivial[-2];
    even=(1+0im)*even/norm(even); odd=(1+0im)*odd/norm(odd);
    T=Matrix{Any}(undef,Lx,Ly);
    for x=1:Lx,y=1:Ly
        if isnothing(centers)
            T[x,y]=copy(iseven(x+y) ? even : odd);
        else
            left=(mod1(x-1,Lx),y);right=(mod1(x+1,Lx),y);
            vertical=(x,mod1(y+(iseven(x+y) ? -1 : 1),Ly));
            neighbors=iseven(x+y) ? (left,vertical,right) : (left,right,vertical);
            if (x,y) in centers
                @assert all(!(j in centers) for j in neighbors);
                local_A=center;
            else
                selected=findall(j->j in centers,neighbors);
                @assert length(selected)==1 "centers must form a disjoint star covering";
                # rotate() moves the occupied leaf leg 1 -> 3 -> 2.
                local_A=selected[1]==1 ? leaf : selected[1]==2 ? rotate(rotate(leaf)) : rotate(leaf);
            end
            if iseven(x+y)
                @tensor site[:]:=local_A[-1,-2,a,-5]*B[-3,a]*trivial_up[-4];
            else
                @tensor site[:]:=local_A[-1,a,b,-5]*B[-3,a]*B[-4,b]*trivial[-2];
            end
            T[x,y]=(1+0im)*site/norm(site);
        end
    end
    return T,(virtual_space=V,c3_error=norm(A-rotate(A)),center_norm=norm(center),leaf_norm=norm(leaf))
end

"""Exact D=9 star-covering SU(4) PEPS with identity SU bond factors.
Returns T, lambdax, lambday, info. Real virtual legs carry 1+4+4bar;
the unused brick-wall vertical leg carries the singlet. No random perturbation.
"""
function initial_honeycomb_star_iPEPS(;Lx=4,Ly=4,centers=nothing)
    T,info=_initial_honeycomb_star_tensors(;Lx,Ly,centers);
    lx=Matrix{Any}(undef,Lx,Ly);ly=similar(lx);
    for i in eachindex(T)
        vl=space(T[i],1);vd=space(T[i],2);
        lx[i]=DiagonalTensorMap(unitary(vl,vl));
        ly[i]=DiagonalTensorMap(unitary(vd',vd'));
    end
    return T,lx,ly,info
end
