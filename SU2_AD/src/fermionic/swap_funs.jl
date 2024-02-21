using HalfIntegers

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




function parity_gate(A,p1)
    V1=space(A,p1);
    S=unitary( V1, V1);
    S_dense=convert(Array,S);
    oddlist1=get_Vspace_parity(V1);
    for c1=1:length(oddlist1)
        if (oddlist1[c1]==1)
            S_dense[c1,c1]=-1;
        end
    end
    S=TensorMap(S_dense,V1 ← V1);
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


function swap_gate(A,p1,p2) #faster
    V1=space(A,p1);
    V2=space(A,p2);
    UU=unitary(fuse(V1*V2),V1*V2);
    S=unitary( V1 ⊗ V2, V1 ⊗ V2);

    S_dense=convert(Array,S);
    oddlist1=get_Vspace_parity(V1);
    oddlist2=get_Vspace_parity(V2);
    for c1=1:length(oddlist1)
        for c2=1:length(oddlist2)
            if (oddlist1[c1]==1)&(oddlist2[c2]==1)
                S_dense[c1,c2,c1,c2]=-1;
            end
        end
    end
    UU_dense=convert(Array,UU);
    @tensor S_dense[:]:=UU_dense[-1,1,2]*S_dense[1,2,3,4]*conj(UU_dense)[-2,3,4];
    # S=TensorMap(S_dense,V1 ⊗ V2 ← V1 ⊗ V2);
    S=TensorMap(S_dense,fuse(V1 ⊗ V2) ← fuse(V1 ⊗ V2));
    @tensor S[:]:=UU'[-1,-2,1]*S[1,2]*UU[2,-3,-4];
    S=permute(S,(1,2,),(3,4,));
    return S
end

function gauge_gate(A,p1,phase)
    V1=space(A,p1);
    S=unitary( V1, V1);

    println(V1)
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

