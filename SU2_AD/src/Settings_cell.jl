function initial_tuple_cell(Lx,Ly)
    if (Lx==1)&(Ly==1)
        M=(1,);
    elseif (Lx==2)&(Ly==2)
        M=((1, 1), (1, 1));
    end
end

function fill_tuple(M0,a, cx,cy) #avoid mutating matrix, which is necessary for AD
    global Lx,Ly
    if (Lx==1)&(Ly==1)
        M=(a,);
    elseif (Lx==2)&(Ly==2)
        if (cx==1)&(cy==1)
            M=((a, M0[1][2]), (M0[2][1], M0[2][2]));
        elseif (cx==1)&(cy==2)
            M=((M0[1][1], a), (M0[2][1], M0[2][2]));;
        elseif (cx==2)&(cy==1)
            M=((M0[1][1], M0[1][2]), (a, M0[2][2]));
        elseif (cx==2)&(cy==2)
            M=((M0[1][1], M0[1][2]), (M0[2][1], a));
        end
    end
    return M
end