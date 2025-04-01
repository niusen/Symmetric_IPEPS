#################################################################
function kagome_lattice(Lx,Ly)
    #Lx,Ly denote number of up triangles, not number of spins
    #
        #                 /\ (1,3,c)
        #                /  \
        #    (1,3,a)     ----   (1,3,b)

        #              /\  (1,2,c)
        #             /  \
        #  (1,2,a)    ----   (1,2,b)
    
    #              /\ (1,1,c)
    #             /  \
    #  (1,1,a)    ----   (1,1,b)


    sites=reshape(1:Lx*Ly*3,(Lx,Ly,3));
    @assert unique(sort(sites[:]))==1:3*Lx*Ly;

    
    NN_set=Vector{Vector}(undef,0);
    NNN_set=Vector{Vector}(undef,0);
    NNNN_set=Vector{Vector}(undef,0);
    for cx=1:Lx
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            #inside an up triangles 
            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 1], sites[triangle_coord1[1],triangle_coord1[2], 2]])
            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord1[1],triangle_coord1[2], 3]])
            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 3], sites[triangle_coord1[1],triangle_coord1[2], 1]])
        end
    end


    for cx=1:Lx-1
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            #up triangles connected by x bond
            triangle_coord2=[cx+1,cy];

            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 1]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 3]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 3], sites[triangle_coord2[1],triangle_coord2[2], 1]])
            push!(NNNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 3], sites[triangle_coord2[1],triangle_coord2[2], 3]])


        end
    end

    for cx=1:Lx
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            #up triangles connected by y bond
            triangle_coord2=[cx,mod1(cy-1,Ly)];

            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 1], sites[triangle_coord2[1],triangle_coord2[2], 3]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 1], sites[triangle_coord2[1],triangle_coord2[2], 2]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 3]])
            push!(NNNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 2]])

        end
    end

    for cx=1:Lx-1
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            #up triangles connected by right-bot bond
            triangle_coord2=[cx+1,mod1(cy-1,Ly)];

            push!(NN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 3]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 1], sites[triangle_coord2[1],triangle_coord2[2], 3]])
            push!(NNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 2], sites[triangle_coord2[1],triangle_coord2[2], 1]])
            push!(NNNN_set, [sites[triangle_coord1[1],triangle_coord1[2], 1], sites[triangle_coord2[1],triangle_coord2[2], 1]])
        end
    end

    up_triangle_set=Matrix{Vector}(undef,Lx,Ly);
    dn_triangle_set=Matrix{Vector}(undef,Lx-1,Ly);
    for cx=1:Lx
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            triangle_coord2=[cx,cy];
            triangle_coord3=[cx,cy];
            p1=sites[triangle_coord1[1],triangle_coord1[2], 1];
            p2=sites[triangle_coord2[1],triangle_coord2[2], 2];
            p3=sites[triangle_coord3[1],triangle_coord3[2], 3];
            up_triangle_set[cx,cy]=[p1,p2,p3];
        end
    end
    for cx=1:Lx-1
        for cy=1:Ly
            triangle_coord1=[cx,cy];
            triangle_coord2=[cx+1,mod1(cy-1,Ly)];
            triangle_coord3=[cx+1,cy];
            p1=sites[triangle_coord1[1],triangle_coord1[2], 2];
            p2=sites[triangle_coord2[1],triangle_coord2[2], 3];
            p3=sites[triangle_coord3[1],triangle_coord3[2], 1];
            dn_triangle_set[cx,cy]=[p1,p2,p3];
        end
    end

    # matwrite("Kagome_"*string(Lx)*"_"*string(Ly)*".mat", Dict(
    # "coord"=>coord,
    # "sites"=>sites,
    # "NN_set"=>NN_set,
    # "NNN_set"=>NNN_set,
    # "NNNN_set"=>NNNN_set,
    # "up_triangle_set"=>up_triangle_set,
    # "dn_triangle_set"=>dn_triangle_set 
    # ); compress = false)     

    return NN_set,NNN_set,NNNN_set, up_triangle_set, dn_triangle_set,sites,coord
end


function get_neighbours_kagome(Lx,Ly,boundary_condition)
    #@tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];

    #Lx,Ly denote number of up triangles, not number of spins
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


function initial_Neel_config_kagome(Lx,Ly)
    #initial spin config, total sz=0
    @assert mod(Lx*Ly,2)==0;
    config=zeros(Int8,Lx,Ly,3);
    for cx=1:Lx
        for cy=1:Ly
            config[cx,cy,1]=(-1)^(cx+cy)*(1);
            config[cx,cy,2]=(-1)^(cx+cy)*(-1);
            config[cx,cy,3]=(-1)^(cx+cy)*(1);
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