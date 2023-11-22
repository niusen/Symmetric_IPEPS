function get_Vspace_parity(V1)
    oddlist1=[];

    for s in sectors(V1)
        str=string(s);
        str=str[end-1];
        Qn=parse(Int64, str)

        Dim=dim(V1, s)
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

function swap_gate(A,p1,p2)
    V1=space(A,p1);
    V2=space(A,p2);
    S=unitary( V1 ⊗ V2, V1 ⊗ V2);
    # for (a,b) in blocks(S)
    # println(a)
    # println(b)
    # end
    # for s in sectors(V1)
    #     println(s)
    #     println(dim(V, s))
    # end

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
    S=TensorMap(S_dense,V1 ⊗ V2 ← V1 ⊗ V2);
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

