function initial_tuple(L)
    M=[ones(1,L-1) "a"];
    M=(M...,);
    return M
end


function mps_update(mps_set::Tuple,T,posit)
    L=length(mps_set);
    mps_new=(mps_set[1:posit-1]...,T,mps_set[posit+1:L]...);
    return mps_new
end
function mps_update(mps_set::Vector,T,posit)
    L=length(mps_set);
    mps_new=(mps_set[1:posit-1]...,T,mps_set[posit+1:L]...);
    return mps_new
end


function vector_update(mps_set::Tuple,T,posit)
    L=length(mps_set);
    mps_new=(mps_set[1:posit-1]...,T,mps_set[posit+1:L]...);
    return mps_new
end
function vector_update(mps_set::Vector,T,posit)
    L=length(mps_set);
    mps_new=(mps_set[1:posit-1]...,T,mps_set[posit+1:L]...);
    return mps_new
end


function matrix_update(M,cx,cy,ele)
    Lx=size(M,1);
    Ly=size(M,2);

    row_old=M[cx,:];
    row_new=vcat(row_old[1:cy-1], ele, row_old[cy+1:Ly]);

    M_new=[M[1:cx-1,:]; reshape(row_new,(1,Ly)); M[cx+1:Lx,:]];
    return M_new
end
