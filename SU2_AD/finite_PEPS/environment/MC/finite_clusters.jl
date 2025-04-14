#################################################################



function get_neighbours_kagome(Lx,Ly,boundary_condition)
    #@tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];

    #Lx,Ly denote number of up triangles, not number of spins
    #the below pattern is determined by contraction order of B_a,B_b,B_c,T_u,T_d
    #
        #                 /\ (1,3,b)
        #                /  \
        #    (1,3,a)     ----   (1,3,c)

                #              /\  (1,2,b)
                #             /  \
                #  (1,2,a)    ----   (1,2,c)
    
                    #              /\ (1,1,b)
                    #             /  \
                    #  (1,1,a)    ----   (1,1,c)

    #determine neighbours
    coord=reshape(Vector(1:Lx*Ly*3),(Lx,Ly,3));
    fnn_set=zeros(Int,Lx*Ly*3);
    snn_set=zeros(Int,Lx*Ly*3);
    tnn_set=zeros(Int,Lx*Ly*3);
    NN_matrix=zeros(Int,Lx*Ly*3,4);
    NNN_matrix=zeros(Int,Lx*Ly*3,4);
    NNNN_matrix=zeros(Int,Lx*Ly*3,4);

    coord_list=zeros(Int,Lx*Ly*3,3);
    for cx=1:Lx
        for cy=1:Ly
            coord_list[coord[cx,cy,1],:]=[cx,cy,1];
            coord_list[coord[cx,cy,2],:]=[cx,cy,2];
            coord_list[coord[cx,cy,3],:]=[cx,cy,3];
        end
    end

    up_triangles=Matrix{Vector{Int}}(undef,Lx,Ly);

    # if boundary_condition=="PBC"
        dn_triangles=Matrix{Vector{Int}}(undef,Lx,Ly);
        hexagons=Matrix{Vector{Int}}(undef,Lx,Ly);
    # elseif boundary_condition=="OBC"
    #     dn_triangles=Matrix{Vector{Int}}(undef,Lx-1,Ly-1);
    #     hexagons=Matrix{Vector{Int}}(undef,Lx-1,Ly-1);
    # end
    for cx=1:Lx
        for cy=1:Ly
            up_triangles[cx,cy]=coord[cx,cy,:];
            # if boundary_condition=="PBC"
                dn_triangles[cx,cy]=[coord[cx,mod1(cy+1,Ly),3], coord[mod1(cx+1,Lx),mod1(cy+1,Ly),1], coord[cx,cy,2]];
                hexagons[cx,cy]=[coord[cx,cy,2],coord[mod1(cx+1,Lx),mod1(cy+1,Ly),1],coord[mod1(cx+1,Lx),mod1(cy+1,Ly),3],coord[mod1(cx+1,Lx),cy,2],coord[mod1(cx+1,Lx),cy,1],coord[cx,cy,3]];
            # elseif boundary_condition=="OBC"
            #     if (cx<Lx)&&(cy<Ly)
            #         dn_triangles[cx,cy]=[coord[cx,cy+1,3], coord[cx+1,cy+1,1], coord[cx+1,cy,2]];
            #         hexagons[cx,cy]=[coord[cx,cy,2],coord[cx,cy+1,1],coord[cx,cy+1,3],coord[cx+1,cy,2],coord[cx+1,cy,1],coord[cx,cy,3]];
            #     end
            # end
        end
    end

    function check_BC(BC,pos1,pos2,coord_list,Lx,Ly)
        px1,py1,ind1=coord_list[pos1,:];
        px2,py2,ind2=coord_list[pos2,:];
        if BC=="OBC"
            if (abs(px1-px2) in [0,1])&&(abs(py1-py2) in [0,1])
                return true
            else
                return false
            end
        elseif BC=="PBC"
            if (abs(px1-px2) in [0,1,Lx-1])&&(abs(py1-py2) in [0,1,Ly-1])
                return true
            else
                return false
            end
        end
    end



    #find neighbours
    for cc in 1:Lx*Ly*3
        px1,py1,ind1=coord_list[cc,:];

        NN=[];
        NNN=[];
        NNNN=[];

        for triangle in up_triangles
            if cc in triangle
                for dd in triangle
                    if cc!=dd
                        if check_BC(boundary_condition,cc,dd,coord_list,Lx,Ly)
                            push!(NN,dd);
                        end
                    end
                end
            end
        end
        for triangle in dn_triangles
            if cc in triangle
                for dd in triangle
                    if cc!=dd
                        if check_BC(boundary_condition,cc,dd,coord_list,Lx,Ly)
                            push!(NN,dd);
                        end
                    end
                end
            end
        end

        # @show cc
        for hexagon in hexagons
            if cc in hexagon
                # @show cc, hexagon
                pos=findall(x->x.==cc,hexagon);
                if length(pos)>0
                    pos=pos[1];

                    dd=hexagon[mod1(pos+2,6)];
                    if check_BC(boundary_condition,cc,dd,coord_list,Lx,Ly)
                        push!(NNN,dd);
                    end
                    dd=hexagon[mod1(pos-2,6)];
                    if check_BC(boundary_condition,cc,dd,coord_list,Lx,Ly)
                        push!(NNN,dd);
                    end
                    dd=hexagon[mod1(pos+3,6)];
                    if check_BC(boundary_condition,cc,dd,coord_list,Lx,Ly)
                        push!(NNNN,dd);
                    end
                end
            end
        end
        # @show NNNN


        fnn_set[cc]=length(NN);
        snn_set[cc]=length(NNN);
        tnn_set[cc]=length(NNNN);
        NN_matrix[cc,1:length(NN)]=NN;
        NNN_matrix[cc,1:length(NNN)]=NNN;
        NNNN_matrix[cc,1:length(NNNN)]=NNNN;
            
        
    end

    
    function neighbour_convert_to_tuple(Lx,Ly,M)
        M_tuple=Vector{Tuple}(undef,Lx*Ly*3);
        for c1=1:Lx*Ly*3
            pos=findall(x->x.>0, M[c1,:]);
            M_tuple[c1]=Tuple(M[c1,pos]);
        end
        return M_tuple
    end

    NN_tuple=neighbour_convert_to_tuple(Lx,Ly,NN_matrix);
    NNN_tuple=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix);
    NNNN_tuple=neighbour_convert_to_tuple(Lx,Ly,NNNN_matrix);
    # @show NN_matrix
    
    NN_matrix_reduced=deepcopy(NN_matrix);
    NNN_matrix_reduced=deepcopy(NNN_matrix);
    NNNN_matrix_reduced=deepcopy(NNNN_matrix);
    #remove double counting
    for c1=1:Lx*Ly*3
        for c2=1:size(NN_matrix_reduced,2)
            p1=NN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NN_matrix_reduced[p1,:]
                    NN_matrix_reduced[c1,c2]=0
                end
            end
        end
        for c2=1:size(NNN_matrix_reduced,2)
            p1=NNN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NNN_matrix_reduced[p1,:]
                    NNN_matrix_reduced[c1,c2]=0
                end
            end
        end
        for c2=1:size(NNNN_matrix_reduced,2)
            p1=NNNN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NNNN_matrix_reduced[p1,:]
                    NNNN_matrix_reduced[c1,c2]=0
                end
            end
        end
    end


    NN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NN_matrix_reduced);
    NNN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix_reduced);
    NNNN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NNNN_matrix_reduced);

    return coord,coord_list,fnn_set,snn_set,tnn_set,NN_tuple,NNN_tuple,NNNN_tuple, NN_tuple_reduced,NNN_tuple_reduced,NNNN_tuple_reduced, up_triangles, dn_triangles, hexagons
end


function initial_Neel_config_kagome(Lx,Ly)
    #initial spin config, total sz=0
    @assert mod(Lx*Ly,2)==0;
    config=zeros(Int8,Lx,Ly,3);
    for cx=1:Lx
        for cy=1:Ly
            vs=(-1)^(cx+cy)*[1,1,-1];
            order=sortperm(rand(3))
            config[cx,cy,1]=vs[order[1]];
            config[cx,cy,2]=vs[order[2]];
            config[cx,cy,3]=vs[order[3]];
        end
    end
    @assert sum(sum(config))==0
    return config[:]
end
#################################################################
function get_neighbours_square(Lx,Ly,boundary_condition)
    #determine neighbours
    coord=reshape(Vector(1:Lx*Ly),(Lx,Ly));
    fnn_set=zeros(Int,Lx*Ly);
    snn_set=zeros(Int,Lx*Ly);
    NN_matrix=zeros(Int,Lx*Ly,4);
    NNN_matrix=zeros(Int,Lx*Ly,4);

    nn=[1 0;-1 0;0 1;0 -1];
    nnn=[1 1;1 -1;-1 1;-1 -1];

    for px in 1:Lx
        for py in 1:Ly
            NN=[];
            NNN=[];
            if boundary_condition in ("OBC",)
                for cn =1:4
                    if (px+nn[cn,1] in 1:Lx) && (py+nn[cn,2] in 1:Ly)
                        push!(NN,coord[px+nn[cn,1], py+nn[cn,2]]);
                    end
                    if (px+nnn[cn,1] in 1:Lx) && (py+nnn[cn,2] in 1:Ly)
                        push!(NNN,coord[px+nnn[cn,1], py+nnn[cn,2]]);
                    end
                end
            elseif boundary_condition in ("PBC",)
                for cn =1:4
                    push!(NN,coord[mod1(px+nn[cn,1],Lx), mod1(py+nn[cn,2],Ly)]);
                    push!(NNN,coord[mod1(px+nnn[cn,1],Lx), mod1(py+nnn[cn,2],Ly)]);
                end
            end
            fnn_set[coord[px,py]]=length(NN);
            snn_set[coord[px,py]]=length(NNN);
            NN_matrix[coord[px,py],1:length(NN)]=NN;
            NNN_matrix[coord[px,py],1:length(NNN)]=NNN;
            
        end
    end

    
    function neighbour_convert_to_tuple(Lx,Ly,M)
        M_tuple=Vector{Tuple}(undef,Lx*Ly);
        for c1=1:Lx*Ly
            pos=findall(x->x.>0, M[c1,:]);
            M_tuple[c1]=Tuple(M[c1,pos]);
        end
        return M_tuple
    end

    NN_tuple=neighbour_convert_to_tuple(Lx,Ly,NN_matrix);
    NNN_tuple=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix);
    # @show NN_matrix
    
    NN_matrix_reduced=deepcopy(NN_matrix);
    NNN_matrix_reduced=deepcopy(NNN_matrix);
    #remove double counting
    for c1=1:Lx*Ly
        for c2=1:size(NN_matrix_reduced,2)
            p1=NN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NN_matrix_reduced[p1,:]
                    NN_matrix_reduced[c1,c2]=0
                end
            end
        end
        for c2=1:size(NNN_matrix_reduced,2)
            p1=NNN_matrix_reduced[c1,c2];
            if p1>0
                if c1 in NNN_matrix_reduced[p1,:]
                    NNN_matrix_reduced[c1,c2]=0
                end
            end
        end
    end


    NN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NN_matrix_reduced);
    NNN_tuple_reduced=neighbour_convert_to_tuple(Lx,Ly,NNN_matrix_reduced);

    return coord,fnn_set,snn_set,NN_tuple,NNN_tuple, NN_tuple_reduced,NNN_tuple_reduced
end


function initial_Neel_config_square(Lx,Ly,sign)
    #initial spin config, total sz=0
    @assert sign in (1,-1);
    config=zeros(Int8,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            config[cx,cy]=(-1)^(cx+cy)*sign;
        end
    end
    @assert sum(sum(config))==0
    return config[:]
end