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

#SU2 symmetry
function get_Vspace_Spin(V1::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    oddlist1=[];
    Keys=V1.dims.keys;
    Values=V1.dims.values;

    for cc in eachindex(Values)
        Spin=Keys[cc].j;
        Dim=Int(Values[cc]*(2*Spin+1));
        oddlist1=vcat(oddlist1,Spin*Int.(ones(Dim)));

    end
    return oddlist1
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



