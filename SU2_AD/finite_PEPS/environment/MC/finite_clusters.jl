#################################################################
function get_neighbours_kagome(Lx,Ly,boundary_condition)
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


function initial_Neel_config_kagome(Lx,Ly,sign)
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