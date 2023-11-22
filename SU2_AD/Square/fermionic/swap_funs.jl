function QN_str_search(Str)
    Leftb=Str[1];
    Rightb=Str[end];
    left_pos=[];
    right_pos=[];
    L=length(Str);
    for cc=1:L
        if Str[cc]==Leftb
            # println(cc)
            left_pos=vcat(left_pos,cc)
        end
    end

    for cc=1:L
        if Str[cc]==Rightb
            # println(cc)
            right_pos=vcat(right_pos,cc)
        end
    end

    xx=string(Irrep[U₁](1) ⊠ Irrep[SU₂](1/2));
    Slash=xx[end-3];
    slash_pos=[];
    for cc=1:L
        if Str[cc]==Slash
            # println(cc)
            slash_pos=vcat(slash_pos,cc)
        end
    end

    return left_pos,right_pos,slash_pos
end

function get_Vspace_Qn(V1)
    Qnlist1=[];
    
    for s in sectors(V1)
        st=replace(string(s), "Irrep[U₁]" => "a");
        st=replace(st, "⊠ Irrep[SU₂]" => "a");
        #println(st)
        left_pos,right_pos,slash_pos=QN_str_search(string(st));

        Qn=parse(Int64, st[left_pos[2]+1:right_pos[1]-1])
        if length(slash_pos)>0
            @assert length(slash_pos)==1
            Numerator=parse(Int64, st[left_pos[3]+1:slash_pos[1]-1])
            Denominator=parse(Int64, st[slash_pos[1]+1:right_pos[2]-1])
            Spin=Numerator/Denominator
        else
            Spin=Numerator=parse(Int64, st[left_pos[3]+1:right_pos[2]-1])
        end
        #println(Spin)
        Dim=dim(V1, s)
        Dim=Int(Dim*(2*Spin+1))
        Qnlist1=vcat(Qnlist1,Int.(ones(Dim))*Qn);
        
    end
    return Qnlist1
end

function get_Vspace_parity(V1)
    oddlist1=[];
    
    for s in sectors(V1)
        st=replace(string(s), "Irrep[U₁]" => "a");
        st=replace(st, "⊠ Irrep[SU₂]" => "a");
        #println(st)
        left_pos,right_pos,slash_pos=QN_str_search(string(st));

        Qn=parse(Int64, st[left_pos[2]+1:right_pos[1]-1])
        if length(slash_pos)>0
            @assert length(slash_pos)==1
            Numerator=parse(Int64, st[left_pos[3]+1:slash_pos[1]-1])
            Denominator=parse(Int64, st[slash_pos[1]+1:right_pos[2]-1])
            Spin=Numerator/Denominator
        else
            Spin=Numerator=parse(Int64, st[left_pos[3]+1:right_pos[2]-1])
        end
        #println(Spin)
        Dim=dim(V1, s)
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

