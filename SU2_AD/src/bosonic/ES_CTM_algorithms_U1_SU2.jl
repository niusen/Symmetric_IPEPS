function Space_decomp(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
    Dims=V.dims;
    Spins=Vector{Float64}(undef,length(Dims.values));
    Degeneracy=Vector{Float64}(undef,length(Dims.values));
    for cc in eachindex(Dims.keys)
        Spins[cc]=Dims.keys[cc].j;
        Degeneracy[cc]=Dims.values[cc];
    end
    return Spins, Degeneracy
end
function ES_CTMRG_ED_SU2(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,decomp=false,y_anti_pbc=false)
    k_phase=[];
    eu=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);
    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    global gate_O1,gate_O2

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));



    order=findall(x -> x<3, S_Sectors);
    S_Sectors=S_Sectors[order];
    Degeneracy=Degeneracy[order];

    println("Space of ES:")
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);


    for sps in eachindex(S_Sectors)
        if N==3
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
        elseif N==4
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==5
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==6
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==7
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==8
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),Rep[SU₂](S_Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
        if mod(S_Sectors[sps],1)==0.5
            parity="odd";
        elseif mod(S_Sectors[sps],1)==0
            parity="even";
        end

        contraction_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,false,nothing,decomp);

        @time eu_,ev=eigsolve(contraction_fun, v_init, min(EH_n,Int(Degeneracy[sps])),:LM,Arnoldi(krylovdim=EH_n*2+5));

        @time ks=calculate_k(ev,N,U_DD)

        println("Spin="*string(S_Sectors[sps]));flush(stdout);
        println(eu_)
        println(ks)

        k_phase=vcat(k_phase,ks);
        eu=vcat(eu,eu_);
        Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));
    end


    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Spin=Float64.(Spin);



    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu/sum(eu);
    k_phase=k_phase[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Spin"=>Spin
    ); compress = false)


end


function Space_decomp(V1)

    Qnlist=[];
    Spinlist=[];
    Degeneracy=[];
    
    for s in sectors(V1)
        # println(s)
        # println(dim(V1,s))
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
        #Dim=Int(Dim*(2*Spin+1))
        
        Qnlist=vcat(Qnlist,Int(Qn));
        Spinlist=vcat(Spinlist,Spin);
        Degeneracy=vcat(Degeneracy,Dim);

        

    end

    return Qnlist,Spinlist,Degeneracy
end

function ES_CTMRG_ED_Kprojector(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,decomp=false,y_anti_pbc=false)
    k_phase=[];
    eu=[];
    Qn=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);

    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    global gate_O1,gate_O2, U_DD

    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    Q_Sectors,S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));



    order=findall(x -> x<3, S_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    println("Space of ES:")
    println("Qn:")
    println(Q_Sectors);
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);
    
    Ks=collect(0:N-1)

    for sps=1:length(S_Sectors)
        for kk=1:length(Ks)
            println("Qn="*string(Q_Sectors[sps])*", Spin="*string(S_Sectors[sps])*", kn="*string(kk));flush(stdout);
            kphase=exp(-im*(2*pi*kk/N));
            normalize=false;
            if N==3
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
            elseif N==4
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,5,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
            elseif N==5
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,5,6,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
            elseif N==6
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,5,6,7,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
            elseif N==7
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
            elseif N==8
                v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
                v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
                norm0=norm(v_init);
                v_init=k_projection(v_init,N,kphase,normalize);
                norm1=norm(v_init);
                @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
            end

            if mod(S_Sectors[sps],1)==0.5
                parity="odd";
            elseif mod(S_Sectors[sps],1)==0
                parity="even";
            end

            if norm1/norm0<1e-13#this sector doesn't has such momentum
                println("this momentum is skipped")
                continue;
            end
            
            contraction_group_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,true,kphase,decomp);
            @time eu_,ev=eigsolve(contraction_group_fun, v_init, min(EH_n,Degeneracy[sps]),:LM,Arnoldi(krylovdim=EH_n*2+5));

            
            println(eu_);flush(stdout);
    
            k_phase=vcat(k_phase,kphase*ones(length(eu_)));
            eu=vcat(eu,eu_);
            Qn=vcat(Qn,Q_Sectors[sps]*ones(length(eu_)));
            Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));


        end
    end

    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Qn=Int.(Qn);
    Spin=Float64.(Spin);

    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu/sum(eu);
    k_phase=k_phase[order];
    Qn=Qn[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_APBC_CTMite_decomp_Kprojector_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_PBC_CTMite_decomp_Kprojector_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_APBC_CTMite_Kprojector_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_PBC_CTMite_Kprojector_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn,
        "Spin"=>Spin
    ); compress = false)


end


function k_projection(v_unprojected,N,kphase,normalize)
    vnorm=dot(v_unprojected,v_unprojected);

    v_projected=deepcopy(v_unprojected);
    for cc=1:N-1
        v_unprojected=fermi_translate(v_unprojected,N);
        v_projected=v_projected+(kphase^(-cc))*v_unprojected;
    end
    #dot(v_projected,permute(v_projected,(2,3,4,1,5,),()))/dot(v_projected,v_projected);#check momentum
    if normalize
        v_projected=v_projected/sqrt(dot(v_projected,v_projected))*sqrt(vnorm);
    end
    return v_projected
end



function ES_CTMRG_ED(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,decomp=false,y_anti_pbc=false)
    k_phase=[];
    eu=[];
    Qn=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);
    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    global gate_O1,gate_O2

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    Q_Sectors,S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));



    order=findall(x -> x<3, S_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    println("Space of ES:")
    println("Qn:")
    println(Q_Sectors);
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);


    for sps=1:length(S_Sectors)
        if N==3
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
        elseif N==4
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==5
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==6
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==7
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==8
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
        if mod(S_Sectors[sps],1)==0.5
            parity="odd";
        elseif mod(S_Sectors[sps],1)==0
            parity="even";
        end

        contraction_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,false,nothing,decomp);
        @time eu_,ev=eigsolve(contraction_fun, v_init, min(EH_n,Degeneracy[sps]),:LM,Arnoldi(krylovdim=EH_n*2+5));

        @time ks=calculate_k(ev,N,U_DD)

        println("Qn="*string(Q_Sectors[sps])*", Spin="*string(S_Sectors[sps]));flush(stdout);
        println(eu_)
        println(ks)

        k_phase=vcat(k_phase,ks);
        eu=vcat(eu,eu_);
        Qn=vcat(Qn,Q_Sectors[sps]*ones(length(eu_)));
        Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));
    end


    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Qn=Int.(Qn);
    Spin=Float64.(Spin);



    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu/sum(eu);
    k_phase=k_phase[order];
    Qn=Qn[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn,
        "Spin"=>Spin
    ); compress = false)


end



function calculate_k(ev,N,U_DD)
    k_phase=Array{ComplexF64,1}(undef, length(ev));
    if N==3
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,-3,-4]*U_DD'[-1,-2,1];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
            #println(phase)

            k_phase[cc]=phase;
        end
    elseif N==4
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,2,-5]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
        
            k_phase[cc]=phase;
        end
    elseif N==5
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,2,-5,-6]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
        
            k_phase[cc]=phase;
        end
    elseif N==6
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,2,3,-7]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
        
            k_phase[cc]=phase;
        end
    elseif N==7
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,2,3,-7,-8]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
        
            k_phase[cc]=phase;
        end
    elseif N==8
        for cc=1:length(ev)
            v=ev[cc];
            @tensor v[:]:=v[1,2,3,4,-9]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3]*U_DD'[-7,-8,4];
            vp=fermi_translate(v,N);
            phase=dot(v,vp)/dot(v,v);
        
            k_phase[cc]=phase;
        end
    end
    return k_phase
end


function CTM_T_action(O1,O2,O1O1,O2O2,v0,N,parity,gate_O1,gate_O2,k_projector,kphase,decomp)
    #firstly apply O2, then O1
    normalize=true;
    if N==3
        if parity=="even"
            @tensor v_new[:]:=O2O2[5,4,3,-1]*O2[3,2,1,-2]*v0[4,2,-3]*gate_O2[5,1];
            @tensor v_new[:]:=O1O1[5,4,3,-1]*O1[3,2,1,-2]*v_new[4,2,-3]*gate_O1[5,1];
        elseif parity=="odd"
            @tensor v_new[:]:=O2O2[5,4,3,-1]*O2[3,2,5,-2]*v0[4,2,-3];
            @tensor v_new[:]:=O1O1[5,4,3,-1]*O1[3,2,5,-2]*v_new[4,2,-3];
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,-3,-4]*U_DD'[-1,-2,1];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,-2,-3]*U_DD[-1,1,2];
        end
    elseif N==4
        if parity=="even"
            @tensor v_new[:]:=O2O2[5,4,3,-1]*O2O2[3,2,1,-2]*v0[4,2,-3]*gate_O2[5,1];
            @tensor v_new[:]:=O1O1[5,4,3,-1]*O1O1[3,2,1,-2]*v_new[4,2,-3]*gate_O1[5,1];
        elseif parity=="odd"
            @tensor v_new[:]:=O2O2[5,4,3,-1]*O2O2[3,2,5,-2]*v0[4,2,-3];
            @tensor v_new[:]:=O1O1[5,4,3,-1]*O1O1[3,2,5,-2]*v_new[4,2,-3];
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,2,-5]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        end
    elseif N==5
        if decomp
            v_new=deepcopy(v0)*0;
            if parity=="even"
                siz=length(O2O2_L);
                for cp=1:siz
                    @tensor v_temp[:]:=O2O2_L[cp][6,2,3,-1]*O2O2[3,4,5,-2]*O2[5,7,1,-3]*v0[2,4,7,-4]*gate_O2_R[cp][6,1];
                    v_new=v_new+v_temp;
                end
                siz=length(O1O1_L);
                v0=deepcopy(v_new);
                v_new=deepcopy(v0)*0;
                for cp=1:siz
                    @tensor v_temp[:]:=O1O1_L[cp][6,2,3,-1]*O1O1[3,4,5,-2]*O1[5,7,1,-3]*v0[2,4,7,-4]*gate_O1_R[cp][6,1];
                    v_new=v_new+v_temp;
                end
            elseif parity=="odd"
                siz=length(O2O2_L);
                for cp=1:siz
                    @tensor v_temp[:]:=O2O2_L[cp][6,1,2,-1]*O2O2[2,3,4,-2]*O2_R[cp][4,5,6,-3]*v0[1,3,5,-4];
                    v_new=v_new+v_temp;
                end
                siz=length(O1O1_L);
                v0=deepcopy(v_new);
                v_new=deepcopy(v0)*0;
                for cp=1:siz
                    @tensor v_temp[:]:=O1O1_L[cp][6,1,2,-1]*O1O1[2,3,4,-2]*O1_R[cp][4,5,6,-3]*v0[1,3,5,-4];
                    v_new=v_new+v_temp;
                end
            end
        else
            if parity=="even"
                @tensor v_new[:]:=O2O2[6,2,3,-1]*O2O2[3,4,5,-2]*O2[5,7,1,-3]*v0[2,4,7,-4]*gate_O2[6,1];
                @tensor v_new[:]:=O1O1[6,2,3,-1]*O1O1[3,4,5,-2]*O1[5,7,1,-3]*v_new[2,4,7,-4]*gate_O1[6,1];
            elseif parity=="odd"
                @tensor v_new[:]:=O2O2[6,1,2,-1]*O2O2[2,3,4,-2]*O2[4,5,6,-3]*v0[1,3,5,-4];
                @tensor v_new[:]:=O1O1[6,1,2,-1]*O1O1[2,3,4,-2]*O1[4,5,6,-3]*v_new[1,3,5,-4];
            end
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,2,-5,-6]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        end
    elseif N==6
        if decomp
            v_new=deepcopy(v0)*0;
            siz=length(O1O1_L);
            if parity=="even"
                siz=length(O2O2_L);
                for cp=1:siz
                    @tensor v_temp[:]:=O2O2_L[cp][6,2,3,-1]*O2O2[3,4,5,-2]*O2O2[5,7,1,-3]*v0[2,4,7,-4]*gate_O2_R[cp][6,1];
                    v_new=v_new+v_temp;
                end
                siz=length(O1O1_L);
                v0=deepcopy(v_new);
                v_new=deepcopy(v0)*0;
                for cp=1:siz
                    @tensor v_temp[:]:=O1O1_L[cp][6,2,3,-1]*O1O1[3,4,5,-2]*O1O1[5,7,1,-3]*v0[2,4,7,-4]*gate_O1_R[cp][6,1];
                    v_new=v_new+v_temp;
                end
            elseif parity=="odd"
                siz=length(O2O2_L);
                for cp=1:siz
                    @tensor v_temp[:]:=O2O2_L[cp][6,1,2,-1]*O2O2[2,3,4,-2]*O2O2_R[cp][4,5,6,-3]*v0[1,3,5,-4];
                    v_new=v_new+v_temp;
                end
                siz=length(O1O1_L);
                v0=deepcopy(v_new);
                v_new=deepcopy(v0)*0;
                for cp=1:siz
                    @tensor v_temp[:]:=O1O1_L[cp][6,1,2,-1]*O1O1[2,3,4,-2]*O1O1_R[cp][4,5,6,-3]*v0[1,3,5,-4];
                    v_new=v_new+v_temp;
                end
            end
        else
            if parity=="even"
                @tensor v_new[:]:=O2O2[6,2,3,-1]*O2O2[3,4,5,-2]*O2O2[5,7,1,-3]*v0[2,4,7,-4]*gate_O2[6,1];
                @tensor v_new[:]:=O1O1[6,2,3,-1]*O1O1[3,4,5,-2]*O1O1[5,7,1,-3]*v_new[2,4,7,-4]*gate_O1[6,1];
            elseif parity=="odd"
                @tensor v_new[:]:=O2O2[6,1,2,-1]*O2O2[2,3,4,-2]*O2O2[4,5,6,-3]*v0[1,3,5,-4];
                @tensor v_new[:]:=O1O1[6,1,2,-1]*O1O1[2,3,4,-2]*O1O1[4,5,6,-3]*v_new[1,3,5,-4];
            end
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,2,3,-7]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        end
    elseif N==7
        if parity=="even"
            @tensor v_new[:]:=O2O2[9,2,3,-1]*O2O2[3,4,5,-2]*O2O2[5,6,7,-3]*O2[7,8,1,-4]*v0[2,4,6,8,-5]*gate_O2[9,1];
            @tensor v_new[:]:=O1O1[9,2,3,-1]*O1O1[3,4,5,-2]*O1O1[5,6,7,-3]*O1[7,8,1,-4]*v_new[2,4,6,8,-5]*gate_O1[9,1];
        elseif parity=="odd"
            @tensor v_new[:]:=O2O2[8,1,2,-1]*O2O2[2,3,4,-2]*O2O2[4,5,6,-3]*O2[6,7,8,-4]*v0[1,3,5,7,-5];
            @tensor v_new[:]:=O1O1[8,1,2,-1]*O1O1[2,3,4,-2]*O1O1[4,5,6,-3]*O1[6,7,8,-4]*v_new[1,3,5,7,-5];
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,2,3,-7,-8]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        end
    elseif N==8
        if parity=="even"
            @tensor v_new[:]:=O2O2[9,2,3,-1]*O2O2[3,4,5,-2]*O2O2[5,6,7,-3]*O2O2[7,8,1,-4]*v0[2,4,6,8,-5]*gate_O2[9,1];
            @tensor v_new[:]:=O1O1[9,2,3,-1]*O1O1[3,4,5,-2]*O1O1[5,6,7,-3]*O1O1[7,8,1,-4]*v_new[2,4,6,8,-5]*gate_O1[9,1];
        elseif parity=="odd"
            @tensor v_new[:]:=O2O2[8,1,2,-1]*O2O2[2,3,4,-2]*O2O2[4,5,6,-3]*O2O2[6,7,8,-4]*v0[1,3,5,7,-5];
            @tensor v_new[:]:=O1O1[8,1,2,-1]*O1O1[2,3,4,-2]*O1O1[4,5,6,-3]*O1O1[6,7,8,-4]*v_new[1,3,5,7,-5];
        end
        if k_projector 
            #momentum projector
            @tensor v_new[:]:=v_new[1,2,3,4,-9]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3]*U_DD'[-7,-8,4];
            v_new=k_projection(v_new,N,kphase,normalize);
            @tensor v_new[:]:=v_new[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
    end
    return v_new
end


function fermi_translate(v,N)
    v=deepcopy(v);
    if N==3
        v=permute_neighbour_ind(v,1,2,4);#L2',L1',L3',dummy
        v=permute_neighbour_ind(v,2,3,4);#L2',L3',L1',dummy
    elseif N==4
        v=permute_neighbour_ind(v,1,2,5);#L2',L1',L3',L4',dummy
        v=permute_neighbour_ind(v,2,3,5);#L2',L3',L1',L4',dummy
        v=permute_neighbour_ind(v,3,4,5);#L2',L3',L4',L1',dummy
    elseif N==5
        v=permute_neighbour_ind(v,1,2,6);#L2',L1',L3',L4',L5',dummy
        v=permute_neighbour_ind(v,2,3,6);#L2',L3',L1',L4',L5',dummy
        v=permute_neighbour_ind(v,3,4,6);#L2',L3',L4',L1',L5',dummy
        v=permute_neighbour_ind(v,4,5,6);#L2',L3',L4',L5',L1',dummy
    elseif N==6
        v=permute_neighbour_ind(v,1,2,7);#L2',L1',L3',L4',L5',L6',dummy
        v=permute_neighbour_ind(v,2,3,7);#L2',L3',L1',L4',L5',L6',dummy
        v=permute_neighbour_ind(v,3,4,7);#L2',L3',L4',L1',L5',L6',dummy
        v=permute_neighbour_ind(v,4,5,7);#L2',L3',L4',L5',L1',L6',dummy
        v=permute_neighbour_ind(v,5,6,7);#L2',L3',L4',L5',L6',L1',dummy
    elseif N==7
        v=permute_neighbour_ind(v,1,2,8);#L2',L1',L3',L4',L5',L6',L7',dummy
        v=permute_neighbour_ind(v,2,3,8);#L2',L3',L1',L4',L5',L6',L7',dummy
        v=permute_neighbour_ind(v,3,4,8);#L2',L3',L4',L1',L5',L6',L7',dummy
        v=permute_neighbour_ind(v,4,5,8);#L2',L3',L4',L5',L1',L6',L7',dummy
        v=permute_neighbour_ind(v,5,6,8);#L2',L3',L4',L5',L6',L1',L7',dummy
        v=permute_neighbour_ind(v,6,7,8);#L2',L3',L4',L5',L6',L7',L1',dummy
    elseif N==8
        v=permute_neighbour_ind(v,1,2,9);#L2',L1',L3',L4',L5',L6',L7',L8',dummy
        v=permute_neighbour_ind(v,2,3,9);#L2',L3',L1',L4',L5',L6',L7',L8',dummy
        v=permute_neighbour_ind(v,3,4,9);#L2',L3',L4',L1',L5',L6',L7',L8',dummy
        v=permute_neighbour_ind(v,4,5,9);#L2',L3',L4',L5',L1',L6',L7',L8',dummy
        v=permute_neighbour_ind(v,5,6,9);#L2',L3',L4',L5',L6',L1',L7',L8',dummy
        v=permute_neighbour_ind(v,6,7,9);#L2',L3',L4',L5',L6',L7',L1',L8',dummy
        v=permute_neighbour_ind(v,7,8,9);#L2',L3',L4',L5',L6',L7',L8',L1',dummy
    end
    return v

end









function ES_CTMRG_ED_fixedSpin(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,decomp=false,y_anti_pbc=false)
    k_phase=[];
    eu=[];
    Qn=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);
    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    global gate_O1,gate_O2

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    Q_Sectors,S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));


    #spin can only be 0 or 1/2
    order=findall(x -> x<1, S_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    println("Space of ES:")
    println("Qn:")
    println(Q_Sectors);
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);


    for sps=1:length(S_Sectors)
        if N==3
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
        elseif N==4
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==5
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==6
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==7
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==8
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
        if mod(S_Sectors[sps],1)==0.5
            parity="odd";
        elseif mod(S_Sectors[sps],1)==0
            parity="even";
        end

        contraction_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,false,nothing,decomp);
        @time eu_,ev=eigsolve(contraction_fun, v_init, min(EH_n,Degeneracy[sps]),:LM,Arnoldi(krylovdim=EH_n*2+5));

        @time ks=calculate_k(ev,N,U_DD)

        println("Qn="*string(Q_Sectors[sps])*", Spin="*string(S_Sectors[sps]));flush(stdout);
        println(eu_)
        println(ks)

        k_phase=vcat(k_phase,ks);
        eu=vcat(eu,eu_);
        Qn=vcat(Qn,Q_Sectors[sps]*ones(length(eu_)));
        Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));
    end


    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Qn=Int.(Qn);
    Spin=Float64.(Spin);



    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu;
    k_phase=k_phase[order];
    Qn=Qn[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_CTMite_APBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_CTMite_PBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn,
        "Spin"=>Spin
    ); compress = false)


end



function ES_CTMRG_ED_fixedQn(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,Q_min,Q_max,decomp=false,y_anti_pbc=false)
    k_phase=[];
    eu=[];
    Qn=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);
    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    global gate_O1,gate_O2

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    Q_Sectors,S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));


    
    order=findall(x -> x<3, S_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    order=findall(x -> x<=Q_max, Q_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    order=findall(x -> x>=Q_min, Q_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];


    println("Space of ES:")
    println("Qn:")
    println(Q_Sectors);
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);

    for sps=1:length(S_Sectors)
        if N==3
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
        elseif N==4
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==5
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==6
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==7
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==8
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
        if mod(S_Sectors[sps],1)==0.5
            parity="odd";
        elseif mod(S_Sectors[sps],1)==0
            parity="even";
        end

        contraction_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,false,nothing,decomp);
        @time eu_,ev=eigsolve(contraction_fun, v_init, min(EH_n,Degeneracy[sps]),:LM,Arnoldi(krylovdim=EH_n*2+5));

        @time ks=calculate_k(ev,N,U_DD)

        println("Qn="*string(Q_Sectors[sps])*", Spin="*string(S_Sectors[sps]));flush(stdout);
        println(eu_)
        println(ks)

        k_phase=vcat(k_phase,ks);
        eu=vcat(eu,eu_);
        Qn=vcat(Qn,Q_Sectors[sps]*ones(length(eu_)));
        Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));
    end


    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Qn=Int.(Qn);
    Spin=Float64.(Spin);



    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu/sum(eu);
    k_phase=k_phase[order];
    Qn=Qn[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_Qn_CTMite_APBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_Qn_CTMite_PBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_Qn_CTMite_APBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_Qn_CTMite_PBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn,
        "Spin"=>Spin
    ); compress = false)


end


function ES_CTMRG_ED_Qn_S(CTM,U_L,U_D,U_R,U_U,M,chi,N,EH_n,Qvalue,Svalue,decomp=false,y_anti_pbc=false)
    println("chi="*string(chi));
    if y_anti_pbc
        println("antiPBC");
    else
        println("PBC");
    end
    k_phase=[];
    eu=[];
    Qn=[];
    Spin=[];


    Tleft=CTM["Tset"][4];
    Tright=CTM["Tset"][2];
    @tensor O1[:]:=Tleft[-1,1,-3]*U_L[1,-4,-2];
    @tensor O2[:]:=Tright[-3,1,-1]*U_R[-2,-4,1];
    O1=O1/norm(O1);
    O2=O2/norm(O2);
    O1=O1*sqrt(chi);
    O2=O2*sqrt(chi);
    #firstly apply O2, then O1

    gate_O1=parity_gate(O1,3);
    gate_O2=parity_gate(O2,3);
    global gate_O1,gate_O2

    U_DD=unitary(fuse(space(O1,4)*space(O1,4)), space(O1,4)*space(O1,4));
    @tensor O1O1[:]:=O1[-1,4,1,2]*O1[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];
    @tensor O2O2[:]:=O2[-1,4,1,2]*O2[1,5,-3,3]*U_DD[-4,2,3]*U_DD'[4,5,-2];

    if decomp
        Ps=projector_general_SU2_U1(space(O1,3));
        siz=length(Ps);
        O1O1_L=Vector{Any}(undef, siz);
        O1O1_R=Vector{Any}(undef, siz);
        O1_R=Vector{Any}(undef, siz);
        gate_O1_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O1O1[1,-2,-3,-4]*Ps[cp]'[1,-1];
            O1O1_L[cp]=T;
            @tensor T[:]:=O1O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1O1_R[cp]=T;
            @tensor T[:]:=O1[-1,-2,1,-4]*Ps[cp][-3,1];
            O1_R[cp]=T;
            @tensor T[:]:=gate_O1[1,-2]*Ps[cp][-1,1];
            gate_O1_R[cp]=T;
        end

        Ps=projector_general_SU2_U1(space(O2,1));
        siz=length(Ps);
        O2O2_L=Vector{Any}(undef, siz);
        O2O2_R=Vector{Any}(undef, siz);
        O2_R=Vector{Any}(undef, siz);
        gate_O2_R=Vector{Any}(undef, siz);
        for cp=1:siz
            @tensor T[:]:=O2O2[1,-2,-3,-4]*Ps[cp][-1,1];
            O2O2_L[cp]=T;
            @tensor T[:]:=O2O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2O2_R[cp]=T;
            @tensor T[:]:=O2[-1,-2,1,-4]*Ps[cp]'[1,-3];
            O2_R[cp]=T;
            @tensor T[:]:=gate_O2[1,-2]*Ps[cp]'[1,-1];
            gate_O2_R[cp]=T;
        end
        global O1O1_L,O2O2_L,O1O1_R,O2O2_R,O1_R,O2_R,gate_O1_R,gate_O2_R
    end
    
    println("calculate ES for N="*string(N));
    if N==3
        V_ES=space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==4
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==5
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==6
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==7
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    elseif N==8
        V_ES=space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4);
    end
    Q_Sectors,S_Sectors,Degeneracy=Space_decomp(fuse(V_ES));


    
 

    order=findall(x -> x==Qvalue, Q_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];

    order=findall(x -> x==Svalue, S_Sectors);
    S_Sectors=S_Sectors[order];
    Q_Sectors=Q_Sectors[order];
    Degeneracy=Degeneracy[order];


    println("Space of ES:")
    println("Qn:")
    println(Q_Sectors);
    println("Spin:")
    println(S_Sectors);
    println("degeneracy:")
    println(Degeneracy);flush(stdout);

    for sps=1:length(S_Sectors)
        if N==3
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            @tensor v_init[:]:=v_init[1,2,-2,-3]*U_DD[-1,1,2];
        elseif N==4
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==5
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,),());
            @tensor v_init[:]:=v_init[1,2,3,4,-3,-4]*U_DD[-1,1,2]*U_DD[-2,3,4];
        elseif N==6
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==7
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6];
        elseif N==8
            v_init=TensorMap(randn, space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4)*space(O1,4),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Q_Sectors[sps],S_Sectors[sps])=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,8,9,),());
            @tensor v_init[:]:=v_init[1,2,3,4,5,6,7,8,-5]*U_DD[-1,1,2]*U_DD[-2,3,4]*U_DD[-3,5,6]*U_DD[-4,7,8];
        end
        if mod(S_Sectors[sps],1)==0.5
            parity="odd";
        elseif mod(S_Sectors[sps],1)==0
            parity="even";
        end

        contraction_fun(x)=CTM_T_action(O1,O2,O1O1,O2O2,x,N,parity,gate_O1,gate_O2,false,nothing,decomp);
        @time eu_,ev=eigsolve(contraction_fun, v_init, min(EH_n,Degeneracy[sps]),:LM,Arnoldi(krylovdim=EH_n*2+5));

        @time ks=calculate_k(ev,N,U_DD)

        println("Qn="*string(Q_Sectors[sps])*", Spin="*string(S_Sectors[sps]));flush(stdout);
        println(eu_)
        println(ks)

        k_phase=vcat(k_phase,ks);
        eu=vcat(eu,eu_);
        Qn=vcat(Qn,Q_Sectors[sps]*ones(length(eu_)));
        Spin=vcat(Spin,S_Sectors[sps]*ones(length(eu_)));
    end


    k_phase=ComplexF64.(k_phase);
    eu=ComplexF64.(eu);
    Qn=Int.(Qn);
    Spin=Float64.(Spin);



    order=sortperm(abs.(eu));
    eu=eu[order];
    eu=eu;
    k_phase=k_phase[order];
    Qn=Qn[order];
    Spin=Spin[order]

    if decomp
        if y_anti_pbc
            ES_filenm="ES_Qn"*string(abs(Qvalue))*"_S"*string(Svalue)*"_CTMite_APBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_Qn"*string(abs(Qvalue))*"_S"*string(Svalue)*"_CTMite_PBC_decomp_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    else
        if y_anti_pbc
            ES_filenm="ES_Qn"*string(abs(Qvalue))*"_S"*string(Svalue)*"_CTMite_APBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        else
            ES_filenm="ES_Qn"*string(abs(Qvalue))*"_S"*string(Svalue)*"_CTMite_PBC_Gutzwiller"*"_M"*string(M)*"_N"*string(N)*"_chi"*string(chi)*".mat";
        end
    end
    matwrite(ES_filenm, Dict(
        "k_phase" => k_phase,
        "eu" => eu,
        "Qn"=>Qn,
        "Spin"=>Spin
    ); compress = false)


end