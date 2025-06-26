using HalfIntegers

#Z2 symmetry
function sector_projector(V::GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
    V_odd=Rep[ℤ₂](1=>V.dims[2]);;
    V_even=Rep[ℤ₂](0=>V.dims[1]);

    if V.dual
        return V_odd',V_even'
    else
        return V_odd,V_even
    end
end

#SU2 symmetry
function sector_projector(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    L=length(V.dims.keys);
    Spin_odd_set=[];
    dim_odd_set=[];
    Spin_even_set=[];
    dim_even_set=[];
    for cc=1:L
        Spin=V.dims.keys[cc].j;
        if mod(Spin,1)==1/2
            Spin_odd_set=vcat(Spin_odd_set,Spin);
            dim_odd_set=vcat(dim_odd_set,V.dims.values[cc]);
        elseif mod(Spin,1)==0
            Spin_even_set=vcat(Spin_even_set,Spin);
            dim_even_set=vcat(dim_even_set,V.dims.values[cc]);
        end
    end

    if length(Spin_odd_set)>0
        V_odd=Rep[SU₂](Spin_odd_set[1]=>dim_odd_set[1]);
        for cc=2:length(Spin_odd_set)
            V_odd=V_odd ⊕ Rep[SU₂](Spin_odd_set[cc]=>dim_odd_set[cc]);
        end
    else
        V_odd=Rep[SU₂]();
    end

    if length(Spin_even_set)>0
        V_even=Rep[SU₂](Spin_even_set[1]=>dim_even_set[1]);
        for cc=2:length(Spin_even_set)
            V_even=V_even ⊕ Rep[SU₂](Spin_even_set[cc]=>dim_even_set[cc]);
        end
    else
        V_even=Rep[SU₂]();
    end

    if V.dual
        return V_odd',V_even'
    else
        return V_odd,V_even
    end
end

#U1 symmetry
function sector_projector(V::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    L=length(V.dims.keys);
    Charge_odd_set=[];
    dim_odd_set=[];
    Charge_even_set=[];
    dim_even_set=[];
    for cc=1:L
        Charge=V.dims.keys[cc].charge;
        if mod(Charge,2)==1
            Charge_odd_set=vcat(Charge_odd_set,Charge);
            dim_odd_set=vcat(dim_odd_set,V.dims.values[cc]);
        elseif mod(Charge,2)==0
            Charge_even_set=vcat(Charge_even_set,Charge);
            dim_even_set=vcat(dim_even_set,V.dims.values[cc]);
        end
    end

    if length(Charge_odd_set)>0
        V_odd=Rep[U₁](Charge_odd_set[1]=>dim_odd_set[1]);
        for cc=2:length(Charge_odd_set)
            V_odd=V_odd ⊕ Rep[U₁](Charge_odd_set[cc]=>dim_odd_set[cc]);
        end
    else
        V_odd=Rep[U₁]();
    end

    if length(Charge_even_set)>0
        V_even=Rep[U₁](Charge_even_set[1]=>dim_even_set[1]);
        for cc=2:length(Charge_even_set)
            V_even=V_even ⊕ Rep[U₁](Charge_even_set[cc]=>dim_even_set[cc]);
        end
    else
        V_even=Rep[U₁]();
    end

    if V.dual
        return V_odd',V_even'
    else
        return V_odd,V_even
    end
end

#U1xSU2 symmetry
function sector_projector(V::GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
    L=length(V.dims.keys);
    Charge_odd_set=[];
    Spin_odd_set=[];
    dim_odd_set=[];
    Charge_even_set=[];
    Spin_even_set=[];
    dim_even_set=[];
    for cc=1:L
        Spin=V.dims.keys[cc].sectors[2].j;
        
        if mod(Spin,1)==1/2
            Charge_odd_set=vcat(Charge_odd_set,V.dims.keys[cc].sectors[1].charge);
            Spin_odd_set=vcat(Spin_odd_set,Spin);
            dim_odd_set=vcat(dim_odd_set,V.dims.values[cc]);
        elseif mod(Spin,1)==0
            Charge_even_set=vcat(Charge_even_set,V.dims.keys[cc].sectors[1].charge);
            Spin_even_set=vcat(Spin_even_set,Spin);
            dim_even_set=vcat(dim_even_set,V.dims.values[cc]);
        end
    end

    if length(Spin_odd_set)>0
        V_odd=Rep[U₁ × SU₂]((Charge_odd_set[1],Spin_odd_set[1])=>dim_odd_set[1]);
        for cc=2:length(Spin_odd_set)
            V_odd=V_odd ⊕ Rep[U₁ × SU₂]((Charge_odd_set[cc],Spin_odd_set[cc])=>dim_odd_set[cc]);
        end
    else
        V_odd=Rep[U₁ × SU₂]();
    end

    if length(Spin_even_set)>0
        V_even=Rep[U₁ × SU₂]((Charge_even_set[1],Spin_even_set[1])=>dim_even_set[1]);
        for cc=2:length(Spin_even_set)
            V_even=V_even ⊕ Rep[U₁ × SU₂]((Charge_even_set[cc],Spin_even_set[cc])=>dim_even_set[cc]);
        end
    else
        V_even=Rep[U₁ × SU₂]();
    end

    if V.dual
        return V_odd',V_even'
    else
        return V_odd,V_even
    end
end

function projector_parity(V)
    V_odd,V_even=sector_projector(V);
    # P_odd=TensorMap(randn,V_odd,V);
    # P_even=TensorMap(randn,V_even,V);
    # function become_isometry(T::TensorMap)
    #     # for cc=1:length(T.data.values)
    #     #     mm=T.data.values[cc];
    #     #     mm_new=Matrix(I,size(mm,1),size(mm,2));
    #     #     T.data.values[cc]=mm_new;
    #     # end
    #     return T
    # end
    # P_odd=become_isometry(P_odd);
    # P_even=become_isometry(P_even);
    P_odd=isometry(V,V_odd)';
    P_even=isometry(V,V_even)';
    @assert norm(P_odd'*P_odd+P_even'*P_even-unitary(V,V))<1e-10;
    return P_odd,P_even
end




#SU2 symmetry
function sector_split(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    L=length(V.dims.keys);
    Spin_odd_set=[];
    dim_odd_set=[];
    
    Spin_even_set=[];
    dim_even_set=[];
    for cc=1:L
        Spin=V.dims.keys[cc].j;
        if mod(Spin,1)==1/2
            Spin_odd_set=vcat(Spin_odd_set,Spin);
            dim_odd_set=vcat(dim_odd_set,V.dims.values[cc]);
        elseif mod(Spin,1)==0
            Spin_even_set=vcat(Spin_even_set,Spin);
            dim_even_set=vcat(dim_even_set,V.dims.values[cc]);
        end
    end
    V_odd_set=Vector{Any}(undef,length(Spin_odd_set));
    V_even_set=Vector{Any}(undef,length(Spin_even_set));

    if length(Spin_odd_set)>0
        V_odd=Rep[SU₂](Spin_odd_set[1]=>dim_odd_set[1]);
        V_odd_set[1]=V_odd;
        for cc=2:length(Spin_odd_set)
            V_odd=Rep[SU₂](Spin_odd_set[cc]=>dim_odd_set[cc]);
            V_odd_set[cc]=V_odd;
        end
    else
        V_odd=Rep[SU₂]();
    end

    if length(Spin_even_set)>0
        V_even=Rep[SU₂](Spin_even_set[1]=>dim_even_set[1]);
        V_even_set[1]=V_even;
        for cc=2:length(Spin_even_set)
            V_even= Rep[SU₂](Spin_even_set[cc]=>dim_even_set[cc]);
            V_even_set[cc]=V_even;
        end
    else
        V_even=Rep[SU₂]();
    end

    if V.dual
        for cc=1:length(V_odd_set)
            V_odd_set[cc]=V_odd_set[cc]';
        end
        for cc=1:length(V_even_set)
            V_even_set[cc]=V_even_set[cc]';
        end
    end
    return V_odd_set,V_even_set
end

function projector_split(V)
    #split into sectors classified by quantum number. E.g., if V=Rep[SU₂](0=>4,1/2=>4,1=>2), this function will return three sectors
    V_odd_set,V_even_set=sector_split(V);
    P_odd_set=Vector{TensorMap}(undef,length(V_odd_set));
    P_even_set=Vector{TensorMap}(undef,length(V_even_set));
    # function become_isometry(T::TensorMap)
    #     for cc=1:length(T.data.values)
    #         mm=T.data.values[cc];
    #         mm_new=Matrix(I,size(mm,1),size(mm,2));
    #         T.data.values[cc]=mm_new;
    #     end
    #     return T
    # end

    for cc=1:length(P_odd_set)
        # P_odd=TensorMap(randn,V_odd_set[cc],V);
        # P_odd=become_isometry(P_odd);
        # P_odd_set[cc]=P_odd;
        P_odd_set[cc]=isometry(V,V_odd_set[cc])';
    end
    for cc=1:length(P_even_set)
        # P_even=TensorMap(randn,V_even_set[cc],V);
        # P_even=become_isometry(P_even);
        # P_even_set[cc]=P_even;
        P_even_set[cc]=isometry(V,V_even_set[cc])';
    end


    II=unitary(V,V)*0;
    for cc=1:length(P_even_set)
        II=II+P_even_set[cc]'*P_even_set[cc];
    end
    for cc=1:length(P_odd_set)
        II=II+P_odd_set[cc]'*P_odd_set[cc];
    end
    @assert norm(unitary(V,V)-II)<1e-15

    return P_odd_set,P_even_set
end

function projector_parity_divide(V)
    #devide space into minimal SU2 multiplets
    #E.g., if V=Rep[SU₂](0=>4,1/2=>4,1=>2), this function will return 10 sectors
    P_odd_set0,P_even_set0=projector_split(V);

    function further_divide(P0)
        v0=space(P0,1);
        @assert length(v0.dims.values)==1
        P_sub_set=Vector{TensorMap}(undef,v0.dims.values[1]);
        for cc=1:length(P_sub_set)
            V1=GradedSpace(v0.dims.keys[1]=>1);
            tt=TensorMap(randn,V1,V)*0;
            for (k,b) in blocks(tt)
                @assert  length(b)==length(P_sub_set);
                b[cc]=1;
                block(tt,k).=b;
            end
            P_sub_set[cc]=tt
        end
        return P_sub_set
    end
    P_odd_set=Vector{TensorMap}(undef,0);
    P_even_set=Vector{TensorMap}(undef,0);

    for p0 in (P_odd_set0)
        P_sub_set=further_divide(p0);
        P_odd_set=vcat(P_odd_set,P_sub_set);
    end
    for p0 in (P_even_set0)
        P_sub_set=further_divide(p0);
        P_even_set=vcat(P_even_set,P_sub_set);
    end
    #verify
    Id_tem=unitary(V,V)*0;
    for pp in (P_odd_set)
        Id_tem=Id_tem+pp'*pp;
    end
    for pp in (P_even_set)
        Id_tem=Id_tem+pp'*pp;
    end
    @assert norm(Id_tem-unitary(V,V))<1e-10;
    return P_odd_set,P_even_set
end



#Z2 symmetry
function get_Vspace_parity(V1::GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
    dm=V1.dims;
    oddlist1=vcat(0*ones(dm[1]),1*ones(dm[2]));
    return oddlist1
end

#U1 symmetry
function get_Vspace_parity(V1::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    oddlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;

    for cc in eachindex(Values)
        if V1.dual
            Qn=-Keys[cc].charge;
        else
            Qn=Keys[cc].charge;
        end
        Dim=Values[cc];

        if mod(Qn,1)==0 #integer Qn
            if mod(Qn,2)==0
                oddlist1=vcat(oddlist1,Int.(zeros(Dim)));
            elseif mod(Qn,2)==1
                oddlist1=vcat(oddlist1,Int.(ones(Dim)));
            end
        elseif mod(Qn,1)==1/2 #half integer Qn
            error("Qn not identified")
            # if V1.dual
            #     Qn_U1=-Qn+1/2;
            # else
            #     Qn_U1=Qn+1/2;
            # end

            # if mod(Qn_U1,2)==0
            #     oddlist1=vcat(oddlist1,Int.(zeros(Dim)));
            # elseif mod(Qn_U1,2)==1
            #     oddlist1=vcat(oddlist1,Int.(ones(Dim)));
            # end
        else
            error("Qn not identified")
        end
    end
    return oddlist1
end

#SU2 symmetry
function get_Vspace_parity(V1::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    oddlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;

    for cc in eachindex(Values)
        Spin=Keys[cc].j;
        Dim=Int(Values[cc]*(2*Spin+1));
        if mod(Spin*2,2)==0
            oddlist1=vcat(oddlist1,Int.(zeros(Dim)));
        elseif mod(Spin*2,2)==1
            oddlist1=vcat(oddlist1,Int.(ones(Dim)));
        end
    end
    return oddlist1
end

function get_Vspace_Spin(V1::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    Slist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;
    
    for cc in eachindex(Values)
        Spin=Keys[cc].j;
        Dim=Int(Values[cc]*(2*Spin+1))
        Slist1=vcat(Slist1,Int.(ones(Dim))*Spin);    
    end
    return Slist1
end

#U1 symmetry
function get_Vspace_Qn(V1::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    Qnlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;
    
    for cc in eachindex(Values)
        Sec1=Keys[cc];
        if V1.dual
            Qn=-Sec1.charge;
        else
            Qn=Sec1.charge;
        end
        Dim=Values[cc];
        Qnlist1=vcat(Qnlist1,Int.(ones(Dim))*Qn);
        
    end
    return Qnlist1
end

#U1 x SU2 symmetry
function get_Vspace_Qn(V1::GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
    Qnlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;
    
    for cc in eachindex(Values)
        Sec1=Keys[cc].sectors[1];
        Sec2=Keys[cc].sectors[2];

        if V1.dual
            Qn=-Sec1.charge;
        else
            Qn=Sec1.charge;
        end
        Dim=Values[cc];
        Spin=Sec2.j;
        Dim=Int(Dim*(2*Spin+1))
        Qnlist1=vcat(Qnlist1,Int.(ones(Dim))*Qn);
        
    end
    return Qnlist1
end
function get_Vspace_Spin(V1::GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
    Slist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;
    
    for cc in eachindex(Values)
        Sec1=Keys[cc].sectors[1];
        Sec2=Keys[cc].sectors[2];


        Dim=Values[cc];
        Spin=Sec2.j;
        Dim=Int(Dim*(2*Spin+1))
        Slist1=vcat(Slist1,Int.(ones(Dim))*Spin);
        
    end
    return Slist1
end

function get_Vspace_parity(V1::GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
    oddlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;
    
    for cc in eachindex(Values)
        Sec1=Keys[cc].sectors[1];
        Sec2=Keys[cc].sectors[2];

        if V1.dual
            Qn=-Sec1.charge;
        else
            Qn=Sec1.charge;
        end
        Dim=Values[cc];
        Spin=Sec2.j;
        Dim=Int(Dim*(2*Spin+1))
        if mod(Qn,2)==0
            oddlist1=vcat(oddlist1,Int.(zeros(Dim)));
        elseif mod(Qn,2)==1
            oddlist1=vcat(oddlist1,Int.(ones(Dim)));
        end
    end
    return oddlist1
end




# function parity_gate(A,p1)
#     V1=space(A,p1);
#     S=unitary( V1, V1);
#     S_dense=convert(Array,S);
#     oddlist1=get_Vspace_parity(V1);
#     for c1=1:length(oddlist1)
#         if (oddlist1[c1]==1)
#             S_dense[c1,c1]=-1;
#         end
#     end
#     S=TensorMap(S_dense,V1 ← V1);
#     return S
# end

function parity_gate(A,p1)
    V1=space(A,p1);
    P_odd,P_even=projector_parity(V1);
    if norm(P_odd)==0
        S=P_even'*P_even;
    elseif norm(P_even)==0
        S=-P_odd'*P_odd;
    else
        S=-P_odd'*P_odd+P_even'*P_even;
    end
    return S
end



function special_parity_gate(A,p1)
    #parity gate for fused space: V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1)
    #the sign act only on occu=2, which comes from swap gate before fusing
    V1=space(A,p1);
    S=unitary( V1, V1);


    S_dense=convert(Array,S);
    @assert size(S_dense)[1]==4
    Qnlist1=get_Vspace_Qn(V1);
    L=length(Qnlist1)
    for c1=1:L
        if (Qnlist1[c1]==2)|(Qnlist1[c1]==-2)
            S_dense[c1,c1]=-1;
        end
    end
    S=TensorMap(S_dense,V1 ← V1);
    return S
end



# function swap_gate(A,p1,p2)
#     V1=space(A,p1);
#     V2=space(A,p2);
#     S=unitary( V1 ⊗ V2, V1 ⊗ V2);

#     S_dense=convert(Array,S);
#     oddlist1=get_Vspace_parity(V1);
#     oddlist2=get_Vspace_parity(V2);
#     for c1=1:length(oddlist1)
#         for c2=1:length(oddlist2)
#             if (oddlist1[c1]==1)&(oddlist2[c2]==1)
#                 S_dense[c1,c2,c1,c2]=-1;
#             end
#         end
#     end
#     S=TensorMap(S_dense,V1 ⊗ V2 ← V1 ⊗ V2);
#     return S
# end


# function swap_gate(A,p1,p2) #faster
#     V1=space(A,p1);
#     V2=space(A,p2);
#     UU=unitary(fuse(V1*V2),V1*V2);
#     S=unitary( V1 ⊗ V2, V1 ⊗ V2);

#     S_dense=convert(Array,S);
#     oddlist1=get_Vspace_parity(V1);
#     oddlist2=get_Vspace_parity(V2);
#     for c1=1:length(oddlist1)
#         for c2=1:length(oddlist2)
#             if (oddlist1[c1]==1)&(oddlist2[c2]==1)
#                 S_dense[c1,c2,c1,c2]=-1;
#             end
#         end
#     end
#     UU_dense=convert(Array,UU);
#     @tensor S_dense[:]:=UU_dense[-1,1,2]*S_dense[1,2,3,4]*conj(UU_dense)[-2,3,4];
#     # S=TensorMap(S_dense,V1 ⊗ V2 ← V1 ⊗ V2);
#     S=TensorMap(S_dense,fuse(V1 ⊗ V2) ← fuse(V1 ⊗ V2));
#     @tensor S[:]:=UU'[-1,-2,1]*S[1,2]*UU[2,-3,-4];
#     S=permute(S,(1,2,),(3,4,));
#     return S
# end

function swap_gate(A,p1::Number,p2::Number) #much faster
    V1=space(A,p1);
    V2=space(A,p2);
    S=unitary( V1 ⊗ V2, V1 ⊗ V2);
    P1_odd,P1_even=projector_parity(V1);
    P2_odd,P2_even=projector_parity(V2);
    @tensor S_minus[:]:=P1_odd'[-1,1]*P1_odd[1,3]*P2_odd'[-2,2]*P2_odd[2,4]*S[3,4,-3,-4];
    S_minus=permute(S_minus,(1,2,),(3,4,));
    S=S-2*S_minus;
    return S
end

function swap_gate(V1::GradedSpace,V2::GradedSpace) #much faster
    S=unitary( V1 ⊗ V2, V1 ⊗ V2);
    P1_odd,P1_even=projector_parity(V1);
    P2_odd,P2_even=projector_parity(V2);
    @tensor S_minus[:]:=P1_odd'[-1,1]*P1_odd[1,3]*P2_odd'[-2,2]*P2_odd[2,4]*S[3,4,-3,-4];
    S_minus=permute(S_minus,(1,2,),(3,4,));
    S=S-2*S_minus;
    return S
end

function gauge_gate(A,p1,phase)
    V1=space(A,p1);
    S=unitary( V1, V1);

    
    S_dense=convert(Array,S)*(1+0*im);
    Qnlist1=get_Vspace_Qn(V1);
    for c1=1:length(Qnlist1)

        S_dense[c1,c1]=exp(Qnlist1[c1]*im*phase);
 
    end
    S=TensorMap(S_dense,V1 ← V1);
    return S
end

# function swap_operation(A,total_ind, p1,p2)
#     S=swap_gate(A,p1,p2);

#     indices=Array(1:total_ind);
#     indices=-indices;
#     indices[p1]=1;
#     indices[p2]=2;

#     @tensor A_new[:]:=A[indices...]*S[-p1,-p2,1,2];
#     return A_new
# end

