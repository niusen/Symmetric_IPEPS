function flip_config(config0::Vector,pos1::Int,pos2::Int)
    config=deepcopy(config0);
    config[pos1]=config0[pos2];
    config[pos2]=config0[pos1];
    return config
end

function SU2_space_to_U1_space(V0)
    V1=U₁Space(0=>0);
    V0_comp=[];
    for cc=1:length(V0.dims.keys)
        for ccc=1:(V0.dims.values[cc])
            push!(V0_comp,V0.dims.keys[cc]);
            J=V0.dims.keys[cc].j;
            szs=Vector(J:-1:-J);
            V_=U₁Space(0=>0);
            for dd=1:length(szs)
                V_=oplus(V_,U₁Space(szs[dd]=>1));
            end
            V1=oplus(V1,V_);
        end
    end
    # @show V0_comp
    # @show V1

    V1_history=zeros(Int,length(V1.dims.keys));
    V1_charges=zeros(length(V1.dims.keys));
    V1_dims=zeros(Int,length(V1.dims.keys));
    for cc=1:length(V1_charges)
        V1_charges[cc]=V1.dims.keys[cc].charge;
        V1_dims[cc]=V1.dims.values[cc];
    end


    total_dim=TensorKit.dim(V1);
    R=zeros(total_dim,total_dim);#ind1:U1 space; ind2:SU2 space
    step=1;
    for cc=1:length(V0_comp)
        J=V0_comp[cc].j;
        szs=Vector(J:-1:-J)
        for dd=1:length(szs)
            pos=findfirst(x->x.==szs[dd],V1_charges);
            
            R[sum(V1_dims[1:pos-1])+V1_history[pos]+1,step]=1;
            V1_history[pos]=V1_history[pos]+1;
            step=step+1;
        end
    end
    @assert norm(R'*R-I(total_dim))<1e-10;

    if V0.dual
        V1=V1';
    end
    return V1,R
end

function SU2_tensor_to_U1_tensor(T0)
    if Rank(T0)==3
        V1=space(T0,1);
        V2=space(T0,2);
        V3=space(T0,3);
        V1p,R1=SU2_space_to_U1_space(V1);
        V2p,R2=SU2_space_to_U1_space(V2);
        V3p,R3=SU2_space_to_U1_space(V3);

        T_dense=convert(Array,T0);
        @tensor T_dense[:]:=T_dense[1,2,3]*R1[-1,1]*R2[-2,2]*R3[-3,3];
        T1=TensorMap(T_dense,V1p*V2p,V3p');
        T1=permute(T1,(1,2,3,));


    elseif Rank(T0)==4
        V1=space(T0,1);
        V2=space(T0,2);
        V3=space(T0,3);
        V4=space(T0,4);
        V1p,R1=SU2_space_to_U1_space(V1);
        V2p,R2=SU2_space_to_U1_space(V2);
        V3p,R3=SU2_space_to_U1_space(V3);
        V4p,R4=SU2_space_to_U1_space(V4);

        T_dense=convert(Array,T0);
        @tensor T_dense[:]:=T_dense[1,2,3,4]*R1[-1,1]*R2[-2,2]*R3[-3,3]*R4[-4,4];
        T1=TensorMap(T_dense,V1p*V2p*V3p,V4p');
        T1=permute(T1,(1,2,3,4,));
    elseif Rank(T0)==5
        V1=space(T0,1);
        V2=space(T0,2);
        V3=space(T0,3);
        V4=space(T0,4);
        V5=space(T0,5);
        V1p,R1=SU2_space_to_U1_space(V1);
        V2p,R2=SU2_space_to_U1_space(V2);
        V3p,R3=SU2_space_to_U1_space(V3);
        V4p,R4=SU2_space_to_U1_space(V4);
        V5p,R5=SU2_space_to_U1_space(V5);

        T_dense=convert(Array,T0);
        @tensor T_dense[:]:=T_dense[1,2,3,4,5]*R1[-1,1]*R2[-2,2]*R3[-3,3]*R4[-4,4]*R5[-5,5];
        T1=TensorMap(T_dense,V1p*V2p*V3p*V4p,V5p');
        T1=permute(T1,(1,2,3,4,5,));
    end
    return T1
end
function symmetry_space_to_dense_space(V0)
    dd=dim(V0);
    if V0.dual 
        return  (ℂ^dd)'
    else
        return  ℂ^dd
    end
end
function to_dense_tensor(A)
    if Rank(A)==5
        A=permute(A,(1,2,3,4,5,));
        V1=space(A,1);
        V2=space(A,2);
        V3=space(A,3);
        V4=space(A,4);
        V5=space(A,5);
        A_dense=convert(Array,A);
        siz=size(A_dense);

        V1=symmetry_space_to_dense_space(V1);
        V2=symmetry_space_to_dense_space(V2);
        V3=symmetry_space_to_dense_space(V3);
        V4=symmetry_space_to_dense_space(V4);
        V5=symmetry_space_to_dense_space(V5);

        A=TensorMap(A_dense, V1*V2*V3*V4, V5');
        A=permute(A,(1,2,3,4,5,));
    elseif Rank(A)==4
        A=permute(A,(1,2,3,4,));
        V1=space(A,1);
        V2=space(A,2);
        V3=space(A,3);
        V4=space(A,4);
        A_dense=convert(Array,A);
        siz=size(A_dense);

        V1=symmetry_space_to_dense_space(V1);
        V2=symmetry_space_to_dense_space(V2);
        V3=symmetry_space_to_dense_space(V3);
        V4=symmetry_space_to_dense_space(V4);

        A=TensorMap(A_dense, V1*V2*V3, V4');
        A=permute(A,(1,2,3,4,));
    elseif Rank(A)==3
        A=permute(A,(1,2,3,));
        V1=space(A,1);
        V2=space(A,2);
        V3=space(A,3);
        A_dense=convert(Array,A);
        siz=size(A_dense);

        V1=symmetry_space_to_dense_space(V1);
        V2=symmetry_space_to_dense_space(V2);
        V3=symmetry_space_to_dense_space(V3);

        A=TensorMap(A_dense, V1*V2, V3');
        A=permute(A,(1,2,3,));
    end
    return A
end

function load_fPEPS_from_kagome_iPESS(Lx::Int,Ly::Int,filenm::String,to_dense)
    data=load(filenm*".jld2");
    B_a=data["B_a"];
    B_b=data["B_b"];
    B_c=data["B_c"];
    T_u=data["T_u"];
    T_d=data["T_d"];

    # A=TensorMap(A.data,A.codom,A.dom);
    if isa(space(B_a,1),ComplexSpace)
    else
        if to_dense
            B_a=to_dense_tensor(B_a);
            B_b=to_dense_tensor(B_b);
            B_c=to_dense_tensor(B_c);
            T_u=to_dense_tensor(T_u);
            T_d=to_dense_tensor(T_d);

        else
            B_a=SU2_tensor_to_U1_tensor(B_a);
            B_b=SU2_tensor_to_U1_tensor(B_b);
            B_c=SU2_tensor_to_U1_tensor(B_c);
            T_u=SU2_tensor_to_U1_tensor(T_u);
            T_d=SU2_tensor_to_U1_tensor(T_d);
        end
    end



    @tensor A[:] := B_a[-1,1,-5]*B_b[4,3,-7]*B_c[-4,2,-6]*T_u[1,3,2]*T_d[-3,4,-2];#note that the physical order is (-5,-7,-6)
    

    A=permute(A,(1,2,3,4,5,6,7,));
    A=A/norm(A);
    
    Vv=space(A,1);
    Vp=space(A,5);
    
    #psi=generate_obc_from_iPEPS(A,Lx,Ly);
    psi=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            psi[cx,cy]=A;
        end
    end
    return psi,Vp,Vv
end

function load_fPEPS_from_iPEPS(Lx::Int,Ly::Int,filenm::String,to_dense)
    data=load(filenm*".jld2");
    A=data["A"];

    # A=TensorMap(A.data,A.codom,A.dom);

    if to_dense
        A_dense=convert(Array,A);
        siz=size(A_dense);
        A=TensorMap(A_dense, ℂ^siz[1]*ℂ^siz[2]*(ℂ^siz[3])'*(ℂ^siz[4])', ℂ^siz[5]);
        Vv=space(A,1);
        Vp=space(A,5);
    else
        Vv=U₁Space(0=>1,1/2=>1,-1/2=>1);
        Vp=U₁Space(1/2=>1,-1/2=>1);
        # Vv=ℤ₂Space(0=>1,1=>2);
        # Vp=ℤ₂Space(1=>2);
        A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);
    end
    

    A=permute(A,(1,2,3,4,5,));
    A=A/norm(A);
    
    
    #psi=generate_obc_from_iPEPS(A,Lx,Ly);
    psi=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            psi[cx,cy]=A;
        end
    end
    return psi,Vp,Vv
end

function load_fPEPS(Lx::Int,Ly::Int,filenm::String)
    data=load("saved_states/"*filenm*".jld2");
    if haskey(data,"E")
        println("Double layer method gives energy "*string(data["E"]));
    end
    if haskey(data,"E_sz0")
        println("Double layer method gives energy "*string(data["E_sz0"])*" in sz=0 sector");
    end
    if haskey(data,"psi")
        psi0=data["psi"];
    elseif haskey(data,"T_set")
        psi0=data["T_set"];
    end
    @assert Lx==size(psi0,1);
    @assert Ly==size(psi0,2);
    psi=Matrix{TensorMap}(undef,Lx,Ly);
    psi[:]=psi0[:];
    Vp=space(psi[2,2],5);
    return psi,Vp
end

function generate_obc_from_iPEPS(A,Lx,Ly)

    
    P=zeros(1,3);P[1,1]=1;
    # P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
    # P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));
    P_L=TensorMap(P,Rep[U₁](0=>1),space(A,1));
    P_D=TensorMap(P,Rep[U₁](0=>1),space(A,2));

    psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
    for cx=2:Lx-1
        for cy=2:Ly-1
            psi[cx,cy]=A;
        end
    end

    cx=1;
    for cy=2:Ly-1
        @tensor T[:]:=A[1,-2,-3,-4,-5]*P_L[-1,1];
        psi[cx,cy]=T;
    end

    cx=Lx;
    for cy=2:Ly-1
        @tensor T[:]:=A[-1,-2,1,-4,-5]*P_L'[1,-3];
        psi[cx,cy]=T;
    end

    cy=1;
    for cx=2:Lx-1
        @tensor T[:]:=A[-1,1,-3,-4,-5]*P_D[-2,1];
        psi[cx,cy]=T;
    end

    cy=Ly;
    for cx=2:Lx-1
        @tensor T[:]:=A[-1,-2,-3,1,-5]*P_D'[1,-4];
        psi[cx,cy]=T;
    end

    cx=1;
    cy=1;
    @tensor T[:]:=A[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
    psi[cx,cy]=T;

    cx=Lx;
    cy=1;
    @tensor T[:]:=A[-1,2,1,-4,-5]*P_L'[1,-3]*P_D[-2,2];
    psi[cx,cy]=T;

    cx=1;
    cy=Ly;
    @tensor T[:]:=A[1,-2,-3,2,-5]*P_L[-1,1]*P_D'[2,-4];
    psi[cx,cy]=T;

    cx=Lx;
    cy=Ly;
    @tensor T[:]:=A[-1,-2,1,2,-5]*P_L'[1,-3]*P_D'[2,-4];
    psi[cx,cy]=T;

    return psi
end







function decompose_physical_legs(fPEPS0::Matrix{TensorMap},Vp::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    
    Lx,Ly=size(fPEPS0);
    if Lattice=="square"
        fPEPS=Array{TensorMap}(undef,Lx,Ly,TensorKit.dim(Vp));
    elseif Lattice=="kagome"
        fPEPS=Array{TensorMap}(undef,Lx,Ly,TensorKit.dim(Vp),TensorKit.dim(Vp),TensorKit.dim(Vp));
    end
    # Vp=U₁Space(1/2=>1,-1/2=>1);
    # Vup=U₁Space(1/2=>1);
    # Vdn=U₁Space(-1/2=>1);
    Vup=GradedSpace(Vp.dims.keys[1]=>1);
    Vdn=GradedSpace(Vp.dims.keys[2]=>1);
    if Vp.dual
        Vup=Vup';
        Vdn=Vdn';
    end
    Pup=TensorMap([1,0]',Vup,Vp);
    Pdn=TensorMap([0,1]',Vdn,Vp);

    # println(space(Pup))
    projectors=[Pup,Pdn];
    @assert TensorKit.dim(Vp)==2;
    if Lattice=="square"
        for cp=1:TensorKit.dim(Vp)
            for cx=1:Lx
                for cy=1:Ly
                    T=fPEPS0[cx,cy];
                    @tensor T[:]:=T[-1,-2,-3,-4,1]*projectors[cp][-5,1];
                    fPEPS[cx,cy,cp]=T;
                end
            end
        end
    elseif Lattice=="kagome"
        for cp1=1:TensorKit.dim(Vp)
            for cp2=1:TensorKit.dim(Vp)
                for cp3=1:TensorKit.dim(Vp)
                    for cx=1:Lx
                        for cy=1:Ly
                            T=fPEPS0[cx,cy];
                            @tensor T[:]:=T[-1,-2,-3,-4,1,2,3]*projectors[cp1][-5,1]*projectors[cp2][-6,2]*projectors[cp3][-7,3];
                            U=unitary(fuse(space(T,5)*space(T,6)*space(T,7)), space(T,5)*space(T,6)*space(T,7));
                            @tensor T[:]:=T[-1,-2,-3,-4,1,2,3]*U[-5,1,2,3];
                            fPEPS[cx,cy,cp1,cp2,cp3]=T;
                        end
                    end
                end
            end
        end
    end
    return fPEPS
end



function apply_sampling_projector(fPEPS::Matrix{TensorMap},config::Array,sample::Matrix{TensorMap}, Vp::GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
    fPEPS=deepcopy(fPEPS);
    Lx,Ly=size(fPEPS);
    #Vp=U₁Space(1/2=>1,-1/2=>1);
    Vup=GradedSpace(Vp.dims.keys[1]=>1);
    Vdn=GradedSpace(Vp.dims.keys[2]=>1);
    if Vp.dual
        Vup=Vup';
        Vdn=Vdn';
    end

    Pup=TensorMap([1,0]',Vup,Vp);
    Pdn=TensorMap([0,1]',Vdn,Vp);

    if Lattice=="square"
        for cx=1:Lx
            for cy=1:Ly
                T=fPEPS[cx,cy];
                if config[cx,cy]==1
                    @tensor T[:]:=T[-1,-2,-3,-4,1]*Pup[-5,1];
                elseif config[cx,cy]==-1
                    @tensor T[:]:=T[-1,-2,-3,-4,1]*Pdn[-5,1];
                end
                fPEPS[cx,cy]=T;
            end
        end
    elseif Lattice=="kagome"
        projector_set=[Pup,Pdn];
        
        for cx=1:Lx
            for cy=1:Ly
                T=fPEPS[cx,cy];
                ind=Int.((config[cx,cy,:]*(-1) .+1)/2 .+1);
                @tensor T[:]:=T[-1,-2,-3,-4,1,2,3]*projector_set[ind[1]][-5,1]*projector_set[ind[2]][-6,2]*projector_set[ind[3]][-7,3];
                U=unitary(fuse(space(T,5)*space(T,6)*space(T,7)), space(T,5)*space(T,6)*space(T,7));
                @tensor T[:]:=T[-1,-2,-3,-4,1,2,3]*U[-5,1,2,3];
                fPEPS[cx,cy]=T;
            end
        end
    end
    return fPEPS
end


function decompose_physical_legs(fPEPS0::Matrix{TensorMap},Vp::ComplexSpace)
    Lx,Ly=size(fPEPS0);
    if Lattice=="square"
        fPEPS_decomposed=Array{TensorMap}(undef,Lx,Ly,TensorKit.dim(Vp));
    elseif Lattice=="kagome"
        fPEPS_decomposed=Array{TensorMap}(undef,Lx,Ly,TensorKit.dim(Vp),TensorKit.dim(Vp),TensorKit.dim(Vp));
    end
    @assert TensorKit.dim(Vp)==2;
    if Lattice=="square"
        for cp=1:TensorKit.dim(Vp)
            for cx=1:Lx
                for cy=1:Ly
                    T=fPEPS0[cx,cy];

                    if Rank(T)==3
                        fPEPS_decomposed[cx,cy,cp]=TensorMap(T[:,:,cp],space(T,1)*space(T,2),ProductSpace{ComplexSpace, 0}());
                    elseif Rank(T)==4
                        fPEPS_decomposed[cx,cy,cp]=TensorMap(T[:,:,:,cp],space(T,1)*space(T,2)*space(T,3),ProductSpace{ComplexSpace, 0}());
                    elseif Rank(T)==5
                        fPEPS_decomposed[cx,cy,cp]=TensorMap(T[:,:,:,:,cp],space(T,1)*space(T,2)*space(T,3)*space(T,4),ProductSpace{ComplexSpace, 0}());
                    end
                end
            end
        end
    elseif Lattice=="kagome"
        for cp1=1:TensorKit.dim(Vp)
            for cp2=1:TensorKit.dim(Vp)
                for cp3=1:TensorKit.dim(Vp)
                    for cx=1:Lx
                        for cy=1:Ly
                            T=fPEPS0[cx,cy];

                            fPEPS_decomposed[cx,cy,cp1,cp2,cp3]=TensorMap(T[:,:,:,:,cp1,cp2,cp3],space(T,1)*space(T,2)*space(T,3)*space(T,4),ProductSpace{ComplexSpace, 0}());
                            
                        end
                    end
                end
            end
        end
    end
    return fPEPS_decomposed
end

function apply_sampling_projector(fPEPS::Matrix{TensorMap},config::Array,sample::Matrix{TensorMap},Vp::ComplexSpace)
    sample=deepcopy(fPEPS);
    Lx,Ly=size(fPEPS);

    if Lattice=="square"
        for cx=1:Lx
            for cy=1:Ly
                T=fPEPS[cx,cy];
                if config[cx,cy]==1
                    pind=1
                elseif config[cx,cy]==-1
                    pind=2;
                end

                if Rank(T)==3
                    sample[cx,cy]=TensorMap(T[:,:,pind],space(T,1)*space(T,2),ProductSpace{ComplexSpace, 0}());
                elseif Rank(T)==4
                    sample[cx,cy]=TensorMap(T[:,:,:,pind],space(T,1)*space(T,2)*space(T,3),ProductSpace{ComplexSpace, 0}());
                elseif Rank(T)==5
                    sample[cx,cy]=TensorMap(T[:,:,:,:,pind],space(T,1)*space(T,2)*space(T,3)*space(T,4),ProductSpace{ComplexSpace, 0}());
                end
            end
        end
    elseif Lattice=="kagome"
        for cx=1:Lx
            for cy=1:Ly
                
                T=fPEPS[cx,cy];
                ind=Int.((config[cx,cy,:]*(-1) .+1)/2 .+1);
                 
                T_dense=T[:,:,:,:,ind[1],ind[2],ind[3]];
                sample[cx,cy]=TensorMap(T_dense,space(T,1)*space(T,2)*space(T,3)*space(T,4),ProductSpace{ComplexSpace, 0}());

            end
        end
    end
    return sample
end


function apply_sampling_projector(fPEPS,Lx::Int,Ly::Int,config::Vector,sample,Vp)
    if Lattice=="square"
        return apply_sampling_projector(fPEPS,reshape(config,Lx,Ly),sample,Vp)
    elseif Lattice=="kagome"
        return apply_sampling_projector(fPEPS,reshape(config,Lx,Ly,3),sample,Vp)
    end
end


function pick_sample(fPEPS_decomposed::Array{TensorMap},config0::Vector, sample::Matrix{TensorMap})
    if Lattice=="square"
        Lx,Ly,Lp=size(fPEPS_decomposed);
        config=reshape(config0,Lx,Ly)
        @assert Lp==2;#spin model
        # fPEPS=Matrix{TensorMap}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                if config[cx,cy]==1
                    sample[cx,cy]=fPEPS_decomposed[cx,cy,1];
                elseif config[cx,cy]==-1
                    sample[cx,cy]=fPEPS_decomposed[cx,cy,2];
                end
            end
        end
    elseif Lattice=="kagome"
        Lx,Ly,Lp1,Lp2,Lp3=size(fPEPS_decomposed);
        config=reshape(config0,Lx,Ly,3)
        @assert Lp1==2;#spin model
        # fPEPS=Matrix{TensorMap}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                ind=Int.((config[cx,cy,:]*(-1) .+1)/2 .+1);#map config (1,-1) to index (1,2)
                sample[cx,cy]=fPEPS_decomposed[cx,cy,ind[1],ind[2],ind[3]];
            end
        end
    end
    return sample
end

function normalize_PEPS!(psi::Matrix{TensorMap},Vp,contract_fun::Function)
    Lx,Ly=size(psi);
    #find a good initial config through random flip. Such step seems to be necessary when amplitude of intial config is close to zero.
    nsteps=1000;
    if Lattice=="kagome"
        @show config=initial_Neel_config_kagome(Lx,Ly);
    elseif Lattice=="square"
        @show config=initial_Neel_config_square(Lx,Ly,1);
    end
    config_max=deepcopy(config);
    Norm_max=0;
    psi_sample=Matrix{TensorMap}(undef,Lx,Ly);
    #coord,coord_list,fnn_set,snn_set,tnn_set,NN_tuple,NNN_tuple,NNNN_tuple, NN_tuple_reduced,NNN_tuple_reduced,NNNN_tuple_reduced, up_triangles, dn_triangles, hexagons=get_neighbours_kagome(Lx,Ly,"PBC");
    for ci=1:nsteps
        psi_sample=apply_sampling_projector(psi,Lx,Ly,config,psi_sample, Vp);
        if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
            psi_sample=shift_pleg(psi_sample);
        end
        chi__=30;
        Norm,trun_err=contract_fun(psi_sample,chi__);

        # @show config
        # @show Norm
        if abs(Norm)>Norm_max
            Norm_max=abs(Norm);
            config_max=deepcopy(config);
        end

        pos1=rand(1:L);
        pos2=rand(1:L);

        config=flip_config(config,pos1,pos2);


    end
    @show config_max
    @show Norm_max
    coe=Norm_max^(1/(Lx*Ly));
    for cc in eachindex(psi)
        setindex!(psi,psi[cc]/coe,cc);
    end
    #error("..")
    return config_max
end

# function normalize_PEPS_square!(psi::Matrix{TensorMap},Vp,contract_fun::Function)
#     Lx,Ly=size(psi);
#     config=initial_Neel_config_square(Lx,Ly,1);
#     psi_sample=Matrix{TensorMap}(undef,Lx,Ly);
#     psi_sample=apply_sampling_projector(psi,Lx,Ly,config,psi_sample, Vp);
#     if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
#         psi_sample=shift_pleg(psi_sample);
#     end
#     chi__=30;
#     Norm,trun_err=contract_fun(psi_sample,chi__);
#     Norm=norm(Norm);
#     coe=Norm^(1/(Lx*Ly));
#     for cc in eachindex(psi)
#         setindex!(psi,psi[cc]/coe,cc);
#     end
# end



function contract_sample(psi::Matrix{TensorMap},Lx::Int,Ly::Int,config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::GradedSpace,contract_fun::Function)
    psi_sample=apply_sampling_projector(psi,Lx,Ly,config,psi_sample_old, Vp);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    Norm,trun_err=contract_fun(psi_sample,chi);
    return Norm,psi_sample, trun_err
end

function contract_sample(psi_decomposed::Array{TensorMap},Lx::Int,Ly::Int,config::Vector,psi_sample_old, Vp::GradedSpace,contract_fun::Function)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    # Norm,trun_err,m,n=contract_fun(psi_sample,chi);
    # return Norm,trun_err,m,n
    Norm,trun_err=contract_fun(psi_sample,chi);
    return Norm,psi_sample, trun_err
end

function contract_sample(psi_decomposed::Array{TensorMap},Lx::Int,Ly::Int,config::Vector,psi_sample_old, Vp::ComplexSpace,contract_fun::Function)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);

    # Norm,trun_err,m,n=contract_fun(psi_sample,chi);
    # return Norm,trun_err,m,n
    Norm,trun_err=contract_fun(psi_sample,chi);
    return Norm,psi_sample, trun_err
end

# function contract_sample(psi::Matrix{TensorMap},Lx::Int,Ly::Int,config::Vector,Vp,contract_fun::Function)
#     return contract_sample(psi,Lx,Ly,reshape(config,Lx,Ly),Vp,contract_fun)
# end

function partial_contract_sample(psi_decomposed::Array{TensorMap},config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::Union{GradedSpace,ComplexSpace},contract_history_::disk_contract_history)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    Norm,trun_errs, contract_history_new=contract_partial_disk(psi_sample,config,contract_history_, chi)

    #################################
    #for verification, need to comment later
    # jldsave("test2.jld2";psi_decomposed,config,contract_history_,contract_history_new,chi)
    # verify_contract_history(psi_sample,contract_history_new, chi);
    #################################
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return Norm,psi_sample, trun_errs, contract_history_new
end

function partial_contract_sample(psi_decomposed::Array{TensorMap},config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::GradedSpace,contract_history_::torus_contract_history)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);
    if isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        psi_sample=shift_pleg(psi_sample);
    end
    #jldsave("test.jld2";psi_sample,config,contract_history_, chi)
    Norm,trun_errs, contract_history_new=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history_, chi)

    #################################
    #for verification, need to comment later
    # jldsave("test2.jld2";psi_decomposed,config,contract_history_,contract_history_new,chi)
    # verify_contract_history(psi_sample,contract_history_new, chi);
    #################################
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return Norm,psi_sample, trun_errs, contract_history_new
end

function partial_contract_sample(psi_decomposed::Array{TensorMap},config::Vector,psi_sample_old::Matrix{TensorMap}, Vp::ComplexSpace,contract_history_::torus_contract_history)
    psi_sample=pick_sample(psi_decomposed,config, psi_sample_old);

    Norm,trun_errs, contract_history_new=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history_, chi)

    #################################
    #for verification, need to comment later
    # jldsave("test2.jld2";psi_decomposed,config,contract_history_,contract_history_new,chi)
    # verify_contract_history(psi_sample,contract_history_new, chi);
    #################################
    global ite_num
    if mod(ite_num,GC_spacing)==0
        GC.gc(true);
    end
    return Norm,psi_sample, trun_errs, contract_history_new
end