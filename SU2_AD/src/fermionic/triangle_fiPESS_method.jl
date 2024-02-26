function iPEPS_to_iPESS(A::Square_iPEPS)
    T=A.T;
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    u,s,v=tsvd(permute(T,(1,2,),(3,4,5,)));
    Tm=u*s;#|LU><M|
    Bm=v;#|Md><|RD
    A_new=Triangle_iPESS(Bm,Tm)
    return A_new
end

function iPEPS_to_iPESS(T::TensorMap)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    u,s,v=tsvd(permute(T,(1,2,),(3,4,5,)));
    Tm=u*s;#|LU><M|
    Bm=v;#|Md><|RD
    A_new=Triangle_iPESS(Bm,Tm)
    return A_new
end

function iPESS_to_iPEPS(A::Triangle_iPESS)
    Tm=A.Tm;#|LU><M|
    Bm=A.Bm;#|Md><|RD
    T=permute(Tm*Bm,(1,5,4,2,3,));#L,D,R,U,d,
    A_new=Square_iPEPS(T)
    return A_new
end


function iPEPS_to_iPESS_tuple(A_set::Tuple)    
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPEPS_to_iPESS(A_set[cx][cy]);
            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    return A_cell
end


function iPESS_to_iPEPS_tuple(A_set::Tuple)    
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPESS_to_iPEPS(A_set[cx][cy]::Triangle_iPESS);
            A_cell=fill_tuple(A_cell, A.T, cx,cy);
        end
    end
    return A_cell
end