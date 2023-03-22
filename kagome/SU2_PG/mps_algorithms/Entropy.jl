function QN_str_search(Str)

    Leftb=Str[2];
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

    xx=string("1/2");
    Slash=xx[end-1];
    slash_pos=[];
    for cc=1:L
        if Str[cc]==Slash
            # println(cc)
            slash_pos=vcat(slash_pos,cc)
        end
    end

    return left_pos,right_pos,slash_pos
end

function get_Vspace_S(V1)
    Slist1=[];
    
    for s in sectors(V1)
        st=replace(string(s), "Irrep[SU₂]" => "a");
        left_pos,right_pos,slash_pos=QN_str_search(string(st));
        if length(slash_pos)>0
            @assert length(slash_pos)==1
            Numerator=parse(Int64, st[left_pos[1]+1:slash_pos[1]-1])
            Denominator=parse(Int64, st[slash_pos[1]+1:right_pos[1]-1])
            Spin=Numerator/Denominator
        else
            Spin=parse(Int64, st[left_pos[1]+1:right_pos[1]-1])
        end
        #println(Spin)
        Dim=dim(V1, s)
        Dim=Int(Dim*(2*Spin+1))
        Slist1=vcat(Slist1,Int.(ones(Dim))*Spin);
        
    end
    return Slist1
end


function parity_gate(A,p1)
    V1=space(A,p1);
    S=unitary( V1, V1);


    S_dense=convert(Array,S);
    Slist1=get_Vspace_S(V1);
    for c1=1:length(Slist1)
        if (mod(Slist1[c1],1)==0.5)
            S_dense[c1,c1]=-1;
        end
    end
    S=TensorMap(S_dense,V1 ← V1);
    return S
end


function Entropy_finite_size(filenm,parameters,D,chi,N,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol)

    println("D="*string(D));
    println("chi="*string(chi));
    println("N="*string(N));flush(stdout);



    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);


    mpo_type="OO";#"O_O" or "OO", in my test "OO" is faster for large bond dimension


    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

    #filenm="LS_D_"*string(D)*"_chi_40.json"
    json_dict=read_json_state(filenm);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];


    CTM,U_L,U_D,U_R,U_U=do_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,U_phy, A_unfused, A_fused);


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];




    gate=parity_gate(O1,2);
    gate_dense=convert(Array,gate);
    Id=Matrix(I, size(gate_dense)[1],size(gate_dense)[1]);
    P_even=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_even[1,:,1,:]=Id;
    P_even[2,:,2,:]=gate_dense;
    P_odd=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_odd[1,:,1,:]=Id;
    P_odd[2,:,2,:]=-gate_dense;

    P_even=TensorMap(P_even, Rep[SU₂](0=>2)⊗Rep[SU₂](0=>1, 1/2=>1), Rep[SU₂](0=>2)⊗Rep[SU₂](0=>1, 1/2=>1));
    P_odd=TensorMap(P_odd, Rep[SU₂](0=>2)⊗Rep[SU₂](0=>1, 1/2=>1), Rep[SU₂](0=>2)⊗Rep[SU₂](0=>1, 1/2=>1));


    println("calculate entropy for N="*string(N));

    ###########################################
    Projector=P_even;

    @tensor OO_P[:]:=OO[-1,1,-3,2]*Projector[-2,2,-4,1];

    @tensor OO_OO_P[:]:=OO[-1,2,-4,1]*OO[-2,3,-5,2]*Projector[-3,1,-6,3];

    OO_P=permute(OO_P,(1,2,),(3,4,));
    OO_OO_P=permute(OO_OO_P,(1,2,3,),(4,5,6,));

    Norm=deepcopy(OO_P);
    Renyi2=deepcopy(OO_OO_P);
    for cc=1:N-2
        Norm=Norm*OO_P;
        Renyi2=Renyi2*OO_OO_P;
    end
    @tensor Norm[:]:=Norm[1,2,3,4]*OO_P[3,4,1,2];
    @tensor Renyi2[:]:=Renyi2[1,2,3,4,5,6]*OO_OO_P[4,5,6,1,2,3];

    Norm=blocks(Norm)[Irrep[SU₂](0)][1]/2;
    Renyi2=blocks(Renyi2)[Irrep[SU₂](0)][1]/2;
    Renyi2_even=-log(Renyi2/Norm^2);

    ###########################################
    Projector=P_odd;

    @tensor OO_P[:]:=OO[-1,1,-3,2]*Projector[-2,2,-4,1];

    @tensor OO_OO_P[:]:=OO[-1,2,-4,1]*OO[-2,3,-5,2]*Projector[-3,1,-6,3];

    OO_P=permute(OO_P,(1,2,),(3,4,));
    OO_OO_P=permute(OO_OO_P,(1,2,3,),(4,5,6,));

    Norm=deepcopy(OO_P);
    Renyi2=deepcopy(OO_OO_P);
    for cc=1:N-2
        Norm=Norm*OO_P;
        Renyi2=Renyi2*OO_OO_P;
    end
    @tensor Norm[:]:=Norm[1,2,3,4]*OO_P[3,4,1,2];
    @tensor Renyi2[:]:=Renyi2[1,2,3,4,5,6]*OO_OO_P[4,5,6,1,2,3];

    Norm=blocks(Norm)[Irrep[SU₂](0)][1]/2;
    Renyi2=blocks(Renyi2)[Irrep[SU₂](0)][1]/2;
    Renyi2_odd=-log(Renyi2/Norm^2);


    ES_filenm="Entropy_finite_size"*"_D"*string(D)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    matwrite(ES_filenm, Dict(
        "Renyi2_odd" => Renyi2_odd,
        "Renyi2_even" => Renyi2_even
    ); compress = false)


end




function Topo_entropy_Renyi2(filenm,parameters,D,chi,N_eu,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol)

    println("D="*string(D));
    println("chi="*string(chi));




    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);


    mpo_type="OO";#"O_O" or "OO", in my test "OO" is faster for large bond dimension


    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

    #filenm="LS_D_"*string(D)*"_chi_40.json"
    json_dict=read_json_state(filenm);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];


    CTM,U_L,U_D,U_R,U_U=do_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,U_phy, A_unfused, A_fused);


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];




    gate=parity_gate(O1,2);
    gate_dense=convert(Array,gate);
    Id=Matrix(I, size(gate_dense)[1],size(gate_dense)[1]);
    P_even=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_even[1,:,1,:]=Id;
    P_even[2,:,2,:]=gate_dense;
    P_odd=zeros(2,size(gate_dense)[1],2,size(gate_dense)[1]);
    P_odd[1,:,1,:]=Id;
    P_odd[2,:,2,:]=-gate_dense;

    P_even=TensorMap(P_even, Rep[SU₂](0=>2)⊗space(gate,1), Rep[SU₂](0=>2)⊗space(gate,1));
    P_odd=TensorMap(P_odd, Rep[SU₂](0=>2)⊗space(gate,1), Rep[SU₂](0=>2)⊗space(gate,1));


    

    ###########################################
    println("calculate topo entropy for even sector");
    Projector=P_even;

    Renyi2=trace_boundary_H(N_eu,OO,Projector,"OO_OO_P")/2;
    Norm=trace_boundary_H(N_eu,OO,Projector,"OO_P")/2;
    Renyi2_even=-log(Renyi2/Norm^2);

    println("calculate topo entropy for odd sector");
    Projector=P_odd;

    Renyi2=trace_boundary_H(N_eu,OO,Projector,"OO_OO_P")/2;
    Norm=trace_boundary_H(N_eu,OO,Projector,"OO_P")/2;
    Renyi2_odd=-log(Renyi2/Norm^2);

    ES_filenm="Topo_entropy_Renyi2"*"_D"*string(D)*"_chi"*string(chi)*".mat";
    matwrite(ES_filenm, Dict(
        "Renyi2_odd" => Renyi2_odd,
        "Renyi2_even" => Renyi2_even
    ); compress = false)


end
  

function trace_boundary_H(N_eu,OO,Projector,type)
    if type=="OO_OO_P"
        println("Trace H^2")
    elseif type=="OO_P"
        println("Trace H")
    end
    Sectors=[0,1/2,1,3/2,2,5/2];
    euL_set=Vector(undef,length(Sectors));
    evL_set=Vector(undef,length(Sectors));
    euR_set=Vector(undef,length(Sectors));
    evR_set=Vector(undef,length(Sectors));
    for sps=1:length(Sectors)
        if type=="OO_OO_P"
            vr_init=TensorMap(randn, space(OO,3)'*space(OO,3)'*space(Projector,3)',SU₂Space(Sectors[sps]=>1));
            vr_init=permute(vr_init,(1,2,3,4,),());
            Rcontraction_fun1(x)=R_action_OO_OO_P(OO,Projector,x);

            vl_init=TensorMap(randn, space(OO,3)*space(OO,3)*space(Projector,3),SU₂Space(Sectors[sps]=>1)');
            vl_init=permute(vl_init,(4,1,2,3,),());
            Lcontraction_fun1(x)=L_action_OO_OO_P(OO,Projector,x);
        elseif type=="OO_P"
            vr_init=TensorMap(randn, space(OO,3)'*space(Projector,3)',SU₂Space(Sectors[sps]=>1));
            vr_init=permute(vr_init,(1,2,3,),());
            Rcontraction_fun2(x)=R_action_OO_P(OO,Projector,x);

            vl_init=TensorMap(randn, space(OO,3)*space(Projector,3),SU₂Space(Sectors[sps]=>1)');
            vl_init=permute(vl_init,(3,1,2,),());
            Lcontraction_fun2(x)=L_action_OO_P(OO,Projector,x);
        end
            if norm(vl_init)<1e-12
                euR_set[sps]=[];
                euR_set[sps]=[];
                euL_set[sps]=[];
                euL_set[sps]=[];
                continue;
            end
            
        if type=="OO_OO_P"
            @time eur,evr=eigsolve(Rcontraction_fun1, vr_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
            @time eul,evl=eigsolve(Lcontraction_fun1, vl_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
        elseif type=="OO_P"
            @time eur,evr=eigsolve(Rcontraction_fun2, vr_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
            @time eul,evl=eigsolve(Lcontraction_fun2, vl_init, N_eu,:LM,Arnoldi(krylovdim=N_eu*5));
        end


            if length(eul)<length(eur)
                eur=eur[1:length(eul)];
                evr=evr[1:length(evl)];
            elseif length(eul)>length(eur)
                eul=eul[1:length(eur)];
                evl=evl[1:length(evr)];
            end

            @assert norm(abs.(eul)-abs.(eur))<1e-12
            euR_set[sps]=eur;
            evR_set[sps]=evr;
            euL_set[sps]=eul;
            evL_set[sps]=evl;
    
            println("Spin="*string(Sectors[sps]));flush(stdout);
            println("Eigenvalues:"*string(eur));flush(stdout);

    end


    #check that the leading eigenvalue is in S=0 sector
    for cc=2:length(euL_set)
        if length(euR_set[cc])>0
            @assert maximum(abs.(euR_set[cc]))/maximum(abs.(euR_set[1]))<(1-1e-6)
        end
    end

    #take only S=0 sector
    euL=euL_set[1];
    evL=evL_set[1];
    euR=euR_set[1];
    evR=evR_set[1];
    #truncate 
    N_keep=1;
    for cc=2:length(euL)
        if abs(euL[cc])/abs(euL[1])>(1-1e-6)
            N_keep=N_keep+1;
        end
    end
    @assert length(euL)>N_keep; #ensure that all degenerate largest eigenvalues are obtained

    euL=euL[1:N_keep];
    evL=evL[1:N_keep];
    euR=euR[1:N_keep];
    evR=evR[1:N_keep];

    println("largest eigenvalues:"*string(euL));flush(stdout);

    #choose correct gauge of left and right eigenvectors;
    M=zeros(length(evL),length(evR))*(1+0*im);
    for ca=1:length(evL)
        for cb=1:length(evR)
            if type=="OO_OO_P"
                @tensor ov[:]:=evL[ca][4,1,2,3]*evR[cb][1,2,3,4];
            elseif type=="OO_P"
                @tensor ov[:]:=evL[ca][3,1,2]*evR[cb][1,2,3];
            end
            ov=blocks(ov)[Irrep[SU₂](0)][1];
            M[ca,cb]=ov;
            
        end
    end
    M_inv=pinv(M);
    #H=evR*euR*(M_inv*evL);

    #compute total sum 
    tot=0;
    for ca=1:length(evR)
        for cb=1:length(evL)
            if type=="OO_OO_P"
                @tensor ov[:]:=evR[ca][1,2,3,4]*evL[cb][4,1,2,3];
            elseif type=="OO_P"
                @tensor ov[:]:=evR[ca][1,2,3]*evL[cb][3,1,2];
            end
            ov=blocks(ov)[Irrep[SU₂](0)][1];
            #tot=tot+ov*euR[ca]*M_inv[ca,cb];
            tot=tot+ov*M_inv[ca,cb];
        end
    end
    return tot
end




function R_action_OO_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[-1,4,5,6]*OO[-2,2,3,4]*Projector[-3,6,1,2]*v0[5,3,1,-4];
    return v_new
end

function L_action_OO_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[5,4,-2,6]*OO[3,2,-3,4]*Projector[1,6,-4,2]*v0[-1,5,3,1];
    return v_new
end

function R_action_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[-1,2,3,4]*Projector[-2,4,1,2]*v0[3,1,-3];
    return v_new
end

function L_action_OO_P(OO,Projector,v0)
    @tensor v_new[:]:=OO[3,2,-2,4]*Projector[1,4,-3,2]*v0[-1,3,1];
    return v_new
end

function cal_ES(filenm,parameters,D,chi,W,N,kset,EH_n,Dtrun_init,Dtrun_max,Dtrun_tol,Dtrun_method,unitcell_size=1,save_mat_tensors=false)
    # D=8;
    # chi=20;
    # W=20;
    # N=20;
    # kset=0:N-1;
    # EH_n=3;#number of entanglement spectrum per k point
    # Dtrun_method="svds";
    # Dtrun_init=400;
    # Dtrun_max=400;


    println("D="*string(D));
    println("chi="*string(chi));
    println("W="*string(W));
    println("N="*string(N));flush(stdout);



    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);

    CTM_conv_tol=1e-6;

    CTM_ite_nums=50;
    CTM_trun_tol=1e-12;
    group_size=Int(round((10^8)/(chi*chi*W*W*D)));

    mpo_type="OO";#"O_O" or "OO", in my test "OO" is faster for large bond dimension

    pow=Int((N-2)/2);



    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

    #filenm="LS_D_"*string(D)*"_chi_40.json"
    json_dict=read_json_state(filenm);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];



    CTM,U_L,U_D,U_R,U_U=try_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,U_phy, A_unfused, A_fused);

    Ag,O1,O2=try_ITEBD(D,chi,W,CTM,U_L,U_R,unitcell_size);

    if save_mat_tensors
        Tensors_filenm="Ag_mpo_tensors_D"*string(D)*"_chi"*string(chi)*"_W"*string(W)*".mat";
        matwrite(Tensors_filenm, Dict(
            "Ag" => convert(Array,Ag),
            "O1" => convert(Array,O1),
            "O2" => convert(Array,O2),
            "C1" => convert(Array,CTM["Cset"][1]),
            "C2" => convert(Array,CTM["Cset"][2]),
            "C3" => convert(Array,CTM["Cset"][3]),
            "C4" => convert(Array,CTM["Cset"][4]),
            "T1" => convert(Array,CTM["Tset"][1]),
            "T2" => convert(Array,CTM["Tset"][2]),
            "T3" => convert(Array,CTM["Tset"][3]),
            "T4" => convert(Array,CTM["Tset"][4])
        ); compress = false)
    end
    println("space of Ag:")
    println(space(Ag))




    space_AOA=fuse(space(Ag,1)'⊗space(O2,1)'⊗space(O1,1)⊗ space(Ag,1));
    space_AA=fuse(space(Ag,1)'⊗ space(Ag,1));

    AOA_sec=collect(sectors(space_AOA))
    AA_sec=collect(sectors(space_AA))

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];




    #normalize the MPO
    euL_set,_,_,_,_=FLR_eig(Ag,OO,20,space_AOA,AOA_sec);
    norm_coe=maximum(abs.(group_numbers(euL_set)));
    OO=OO/norm_coe;
    O1=O1/norm_coe;



    if Dtrun_method=="eigs"
        euR_set,evL_set,evR_set,SPIN_eig_set=TransfOp_decom(Ag,OO,space_AOA,AOA_sec,pow,Dtrun_init,Dtrun_max,Dtrun_tol,"eigenvalue_FLR");
        # println(euR_set)
        eur_set,evl_set,evr_set,spin_eig_set=TransfOp_decom(Ag,OO,space_AA,AA_sec,pow,Dtrun_init,Dtrun_max,Dtrun_tol,"eigenvalue_GLR");
        # println(eur_set)
    elseif Dtrun_method=="svds"
        S_set,U_set,Vh_set,SPIN_svd_set=TransfOp_decom(Ag,OO,space_AOA,AOA_sec,pow,Dtrun_init,Dtrun_max,Dtrun_tol,"svd_FLR");
        # println(S_set)
        s_set,u_set,vh_set,spin_svd_set=TransfOp_decom(Ag,OO,space_AA,AA_sec,pow,Dtrun_init,Dtrun_max,Dtrun_tol,"svd_GLR");
        # println(s_set)
    end


    check_truncated_decomp_error=false;

    if mpo_type=="O_O"
        OO_transform=true;
    elseif mpo_type=="OO"
        OO_transform=false;
    end

    if Dtrun_method=="eigs"
        euR_set_combined,evL_set_combined,evR_set_combined,SPIN_eig_set_combined=combine_singlespin_sector(euR_set,evL_set,evR_set,SPIN_eig_set,true);
        euR_set_grouped,evL_set_grouped,evR_set_grouped,SPIN_eig_set_grouped,DTrun_FLR_eig=group_singlespin_sector(group_size,euR_set_combined,evL_set_combined,evR_set_combined,SPIN_eig_set_combined,OO_transform,U_fuse_chichi)
        println("group information:");flush(stdout);
        println(DTrun_FLR_eig);flush(stdout);

        eur_set_combined,evl_set_combined,evr_set_combined,spin_eig_set_combined=combine_singlespin_sector(eur_set,evl_set,evr_set,spin_eig_set,true)
        eur_set_grouped,evl_set_grouped,evr_set_grouped,spin_eig_set_grouped,Dtrun_GLR_eig=group_singlespin_sector(group_size,eur_set_combined,evl_set_combined,evr_set_combined,spin_eig_set_combined,false,[])
        println("group information:");flush(stdout);
        println(Dtrun_GLR_eig);flush(stdout);

    elseif Dtrun_method=="svds"
        S_set_combined,Vh_set_combined,U_set_combined,SPIN_svd_set_combined=combine_singlespin_sector(S_set,Vh_set,U_set,SPIN_svd_set,false)
        S_set_grouped,Vh_set_grouped,U_set_grouped,SPIN_svd_set_grouped,DTrun_FLR_svd=group_singlespin_sector(group_size,S_set_combined,Vh_set_combined,U_set_combined,SPIN_svd_set_combined,OO_transform,U_fuse_chichi)
        println("group information:");flush(stdout);
        println(DTrun_FLR_svd);flush(stdout);

        s_set_combined,vh_set_combined,u_set_combined,spin_svd_set_combined=combine_singlespin_sector(s_set,vh_set,u_set,spin_svd_set,false)
        s_set_grouped,vh_set_grouped,u_set_grouped,spin_svd_set_grouped,Dtrun_GLR_svd=group_singlespin_sector(group_size,s_set_combined,vh_set_combined,u_set_combined,spin_svd_set_combined,false,[])
        println("group information:");flush(stdout);
        println(Dtrun_GLR_svd);flush(stdout);
    end




    ES_Sectors=[0,1/2,1,3/2,2,5/2];

    #kset=0:0
    Eset=[];
    Trun_err=0;
    DTrun=0;
    println("calculate ES for N="*string(N));
    println("kset="*string(kset));flush(stdout);
    pow=round((N-2)/2);



    if Dtrun_method=="eigs"
        DTrun=length(group_numbers(SPIN_eig_set));
        println("DTrun="*string(DTrun));

        euRs=abs.(group_numbers(euR_set));
        Trun_err=(minimum(euRs)/maximum(euRs))^pow;


        euR_pow=deepcopy(euR_set_grouped);
        for ca=1:length(euR_pow)
            for cb=1:length(euR_pow[ca])
                euR_pow[ca][cb]=euR_pow[ca][cb]^Int(pow);
            end
        end

        kset,Eset=solve_ITEBD_excitation_TrunTransOp_iterative(Ag,O1,O2,OO,EH_n,N,kset,ES_Sectors,pow,evR_set_grouped,euR_pow,evL_set_grouped,SPIN_eig_set_grouped,DTrun_FLR_eig,mpo_type,multi_threads)

    elseif Dtrun_method=="svds"
        DTrun=length(group_numbers(SPIN_svd_set));
        println("DTrun="*string(DTrun));

        Ss=abs.(group_numbers(S_set));
        Trun_err=(minimum(Ss)/maximum(Ss));

        kset,Eset=solve_ITEBD_excitation_TrunTransOp_iterative(Ag,O1,O2,OO,EH_n,N,kset,ES_Sectors,pow,U_set_grouped,S_set_grouped,Vh_set_grouped,SPIN_svd_set_grouped,DTrun_FLR_svd,mpo_type,multi_threads)
    end

    ES_filenm="ES_"*Dtrun_method*"_D"*string(D)*"_chi"*string(chi)*"_W"*string(W)*"_N"*string(N)*"_DTrun"*string(DTrun)*"_kset"*string(kset[1])*"to"*string(kset[end])*".mat";
    matwrite(ES_filenm, Dict(
        "kset" => convert(Vector,kset),
        "ES_Sectors" => ES_Sectors,
        "Eset" => Eset,
        "Trun_err"=>Trun_err,
        "DTrun"=>DTrun
    ); compress = false)


end




function solve_ITEBD_excitation_TrunTransOp_iterative(Ag,O1,O2,OO,n_E,N,kset,ES_Sectors,pow,U,S,Vt,SPIN_group,DTrun_group,mpo_type,multi_threads)

    Eset=Matrix{Any}(undef, length(kset),length(ES_Sectors));

    for kk=1:length(kset)

        ck=kset[kk]
        k=2*pi*ck/N

        norm_eff=excitation_TrunTransOp_iterative_norm_eff(Ag,pow,N,k) # put it on cpu because this matrix maybe large
        norm_eff=permute(norm_eff,(1,2,3,),(4,5,6,))
        uu,ss,vvt=tsvd(norm_eff, trunc=truncerr(0.0000001));
        norm_eff=[]#clear this big matrix
        input_transform=vvt';
        output_transform=uu';
        output_transform=output_transform;
        output_transform=pinv(ss)*output_transform;

        for sector_ind=1:length(ES_Sectors)
            SPIN=ES_Sectors[sector_ind];
            sectr=Irrep[SU₂](SPIN);
            println("sector "*"k="*string(ck)*", spin="*string(SPIN)*":");;flush(stdout);

            if dim(fuse(domain(input_transform,1)),Irrep[SU₂](SPIN))==0
                println("MPS decomposition does not have this sector, skip it");flush(stdout);
                Eset[kk,sector_ind]=[];
                continue;
            end

            v_init=TensorMap(randn,domain(input_transform), SU₂Space(SPIN=>1));

            excitation_iterative(x)=excitation_TrunTransOp_iterative_H_eff(x,input_transform,output_transform,O1,O2,OO,Ag,pow,U,S,Vt,SPIN_group,N,k,DTrun_group,mpo_type,multi_threads)

            @time Es,_=eigsolve(excitation_iterative, v_init, n_E,:LM,Arnoldi(krylovdim=minimum([10,n_E*3])));
            Eset[kk,sector_ind]=Es
            println(Es);println(" ");flush(stdout);
        end

    end


    return kset, Eset
end


function excitation_TrunTransOp_iterative_H_eff(x,input_transform,output_transform,O1,O2,OO,Ag,pow,VR,EU,VL,SPIN,N,k,DTrun_list,mpo_type,multi_threads)
    @assert pow==round((N-2)/2)
    x=input_transform*x;# do this calculation on cpu because the matrix 'input_transform' maybe large
    x=permute(x,(1,2,3,4,),());
    H_eff_x=x*0;
    cm=1;

    H_eff_x_set=Vector{Any}(undef, N);
    if multi_threads
        Threads.@threads for cn=1:N
            println("Thread id: "*string(Threads.threadid()));flush(stdout);
            coe=exp(-im*k*(cm-cn))
            H_eff_x0=H_eff_x*0;
            for c_sector=1:length(DTrun_list)
                for  c_comp=1:length(DTrun_list[c_sector])
                    #println(DTrun_list[c_sector][c_comp]);flush(stdout);

                    H_eff_x0=H_eff_x0+coe*contract_H_eff_trun(x,O1,O2,OO,Ag,VR[c_sector][c_comp],EU[c_sector][c_comp],VL[c_sector][c_comp],SPIN[c_sector],cm,cn,pow,N,DTrun_list[c_sector][c_comp],mpo_type);
                end
            end
            H_eff_x_set[cn]=H_eff_x0
        end
    else
        for cn=1:N
            coe=exp(-im*k*(cm-cn))
            H_eff_x0=H_eff_x*0;
            for c_sector=1:length(DTrun_list)
                for  c_comp=1:length(DTrun_list[c_sector])
                    #println(DTrun_list[c_sector][c_comp]);flush(stdout);

                    H_eff_x0=H_eff_x0+coe*contract_H_eff_trun(x,O1,O2,OO,Ag,VR[c_sector][c_comp],EU[c_sector][c_comp],VL[c_sector][c_comp],SPIN[c_sector],cm,cn,pow,N,DTrun_list[c_sector][c_comp],mpo_type);
                end
            end
            H_eff_x_set[cn]=H_eff_x0
        end
    end

    for cn=1:N
        H_eff_x=H_eff_x+H_eff_x_set[cn];
    end
    x=permute(H_eff_x,(1,2,3,),(4,));
    x_output=output_transform*x;
    return x_output
end

function excitation_TrunTransOp_iterative_norm_eff(Ag,pow,N,k)
    @assert pow==round((N-2)/2)
    norm_eff=permute(TensorMap(randn,codomain(Ag)←codomain(Ag))*0,(1,2,3,4,5,6,),());
    cm=1;
    for cn=1:N
            coe=exp(-im*k*(cm-cn))
            norm_eff=norm_eff+coe*contract_norm_eff(Ag,cm,cn,N)
    end
    return norm_eff
end



function create_puMPS(A,N)
    mps=Vector{Any}(undef, N);
    for c1=1:N
        mps[c1]=A
    end
    return mps
end


function contiguous_TM(mps1,mps2,positions)
    if length(positions)==1
        @tensor E[:]:=mps1[positions[1]]'[-1,-3,3]*mps2[positions[1]][-2,-4,3];
    elseif length(positions)>1
        @tensor E[:]:=mps1[positions[1]]'[-1,-3,3]*mps2[positions[1]][-2,-4,3];
        for ci=positions[2]:positions[end]
            @tensor E[:]:=E[-1,-2,1,2]*mps1[ci]'[1,-3,3]*mps2[ci][2,-4,3];
        end
    end
    return E
end


function contract_norm_eff(Ag,cm,cn,N)
    mps=create_puMPS(Ag,N);
    mps1=mps;
    mps2=mps;
    if cm==cn # cm=cn=1
        E=contiguous_TM(mps1,mps2,2:N)
        Id=id(space(Ag,3));
        @tensor norm_eff[:]:=E[-2,-5,-1,-4]*Id[-3,-6];
    elseif cn==2
        E=contiguous_TM(mps1,mps2,3:N);
        @tensor norm_eff[:]:=mps2[1][1,-4,-3]*mps1[2]'[-2,2,-6]*E[2,-5,-1,1];
    elseif cn==N
        E=contiguous_TM(mps1,mps2,2:N-1);
        @tensor norm_eff[:]:=mps2[1][-5,1,-3]*E[-2,1,2,-4]*mps1[N]'[2,-1,-6]
    else
        T1=mps2[1];
        T2=contiguous_TM(mps1,mps2,2:(cn-1))
        T3=mps1[cn]';
        T4=contiguous_TM(mps1,mps2,(cn+1):N)
        @tensor norm_eff[:]:=T1[6,1,-3]*T2[-2,1,3,-4]*T3[3,4,-6]*T4[4,-5,-1,6];
    end
    return norm_eff
end


function contract_H_eff_trun(x,O1,O2,OO,Ag,VR_comp,EU_comp,VL_comp,spin_comp,cm,cn,pow,N,D_set,mpo_type)
    H_eff_x=x*0;
    L_Ea=cn-cm-1
    L_Eb=N-cn
    if L_Ea>=L_Eb
        L=L_Ea
        decomp_posit="Ea"
    elseif L_Ea<L_Eb
        L=L_Eb
        decomp_posit="Eb"
    end

    VL_comp=v_Transop(VL_comp,Ag,O1,O2,OO,L-pow,mpo_type);
    VR_comp=VR_comp*EU_comp;

    if mpo_type=="O_O"
        if cm==cn # cm=cn=0
            @tensor ttt[:]:=VL_comp[9,-1,5,3,1]*O2[5,4,6,-3]*O1[3,2,7,4]*x[1,8,2,-4]*VR_comp[-2,6,7,8,9];
            H_eff_x=H_eff_x+ttt;
        elseif cn==2
            @tensor T1[:]:=VL_comp[-1,-2,5,3,1]*O2[5,4,-3,-6]*O1[3,2,-4,4]*Ag[1,-5,2];
            @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,-5,5,4]*VR_comp[1,3,5,-4,-6];
            @tensor T1T2x[:]:=T1[6,-1,3,4,5,-3]*T2[-2,3,4,2,1,6]*x[5,2,1,-4];
            H_eff_x=H_eff_x+T1T2x;
        elseif cn==N
            @tensor T1[:]:=VL_comp[-1,1,3,5,-2]*Ag'[1,-3,2]*O2[3,4,-4,2]*O1[5,-6,-5,4];
            @tensor T2[:]:=O2[-1,4,5,-5]*O1[-2,2,3,4]*Ag[-3,1,2]*VR_comp[-4,5,3,1,-6];
            @tensor T1xT2[:]:=T1[6,1,-1,3,4,2]*x[1,5,2,-4]*T2[3,4,5,-2,-3,6];
            H_eff_x=H_eff_x+T1xT2;
        else
            if  decomp_posit=="Ea"
                @tensor T1[:]:=VL_comp[-1,1,3,5,7]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*x[7,-5,6,-6];
                @tensor T2[:]:=O2[-1,4,5,-5]*O1[-2,2,3,4]*Ag[-3,1,2]*VR_comp[-4,5,3,1,-6];
                @assert L_Eb>=1
                for cr=1:L_Eb
                    @tensor T1[:]:=T1[-1,1,3,5,7,-6]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*Ag[7,-5,6];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,3,-4]*T2[1,2,3,-2,-3,4];
                H_eff_x=H_eff_x+T1T2;
            elseif decomp_posit=="Eb"
                @tensor T1[:]:=VL_comp[-1,-2,5,3,1]*O2[5,4,-3,-6]*O1[3,2,-4,4]*Ag[1,-5,2];
                @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,6,5,4]*x[-4,7,6,-6]*VR_comp[1,3,5,7,-5];
                @assert L_Ea>=1
                for cr=1:L_Ea
                    @tensor T2[:]:=Ag'[-1,1,2]*O2[-2,4,3,2]*O1[-3,6,5,4]*Ag[-4,7,6]*T2[1,3,5,7,-5,-6];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,3,-3]*T2[-2,1,2,3,4,-4];
                H_eff_x=H_eff_x+T1T2;
            end
        end
    elseif mpo_type=="OO"
        if cm==cn # cm=cn=0
            @tensor ttt[:]:=VL_comp[6,-1,2,1]*OO[2,3,4,-3]*x[1,5,3,-4]*VR_comp[-2,4,5,6];
            H_eff_x=H_eff_x+ttt;
        elseif cn==2
            @tensor T1[:]:=VL_comp[-1,-2,3,1]*OO[3,2,-3,-5]*Ag[1,-4,2];
            @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,-4,3,2]*VR_comp[1,3,-3,-5];
            @tensor T1T2x[:]:=T1[5,-1,4,3,-3]*T2[-2,4,2,1,5]*x[3,2,1,-4];
            H_eff_x=H_eff_x+T1T2x;
        elseif cn==N
            @tensor T1[:]:=VL_comp[-1,1,3,-2]*Ag'[1,-3,2]*OO[3,-5,-4,2];
            @tensor T2[:]:=OO[-1,2,3,-4]*Ag[-2,1,2]*VR_comp[-3,3,1,-5];
            @tensor T1xT2[:]:=T1[5,1,-1,3,2]*x[1,4,2,-4]*T2[3,4,-2,-3,5];
            H_eff_x=H_eff_x+T1xT2;
        else
            if  decomp_posit=="Ea"
                @tensor T1[:]:=VL_comp[-1,1,3,5]*Ag'[1,-2,2]*OO[3,4,-3,2]*x[5,-4,4,-5];
                @tensor T2[:]:=OO[-1,2,3,-4]*Ag[-2,1,2]*VR_comp[-3,3,1,-5];
                @assert L_Eb>=1
                for cr=1:L_Eb
                    @tensor T1[:]:=T1[-1,1,3,5,-5]*Ag'[1,-2,2]*OO[3,4,-3,2]*Ag[5,-4,4];
                end
                @tensor T1T2[:]:=T1[4,-1,1,2,-4]*T2[1,2,-2,-3,4];
                H_eff_x=H_eff_x+T1T2;
            elseif decomp_posit=="Eb"
                @tensor T1[:]:=VL_comp[-1,-2,3,1]*OO[3,2,-3,-5]*Ag[1,-4,2];
                @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,4,3,2]*x[-3,5,4,-5]*VR_comp[1,3,5,-4];
                @assert L_Ea>=1
                for cr=1:L_Ea
                    @tensor T2[:]:=Ag'[-1,1,2]*OO[-2,4,3,2]*Ag[-3,5,4]*T2[1,3,5,-4,-5];
                end
                @tensor T1T2[:]:=T1[3,-1,1,2,-3]*T2[-2,1,2,3,-4];
                H_eff_x=H_eff_x+T1T2;
            end
        end
    end
    H_eff_x=H_eff_x*(2*spin_comp+1);
    return H_eff_x
end



function v_Transop(VL_comp,Ag,O1,O2,OO,L,mpo_type)
    VL_comp=deepcopy(VL_comp)
    if mpo_type=="O_O"
        for cl=1:L
            @tensor VL_comp[:]:=VL_comp[-1,1,3,5,7]*Ag'[1,-2,2]*O2[3,4,-3,2]*O1[5,6,-4,4]*Ag[7,-5,6];
        end
    elseif mpo_type=="OO"
        for cl=1:L
            @tensor VL_comp[:]:=VL_comp[-1,1,3,5]*Ag'[1,-2,2]*OO[3,4,-3,2]*Ag[5,-4,4];
        end
    end
    return VL_comp
end
