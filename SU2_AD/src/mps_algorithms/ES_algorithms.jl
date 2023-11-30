function ES_CTMRG_ED(CTM::CTM_struc,D,chi,N,EH_n,group_index,vison)

    println("D="*string(D));
    println("chi="*string(chi));
    println("N="*string(N));flush(stdout);



    Tleft=CTM.Tset.T4;
    Tright=CTM.Tset.T2;
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];

    U_fuse_DD=unitary(fuse(space(O1,2)⊗ space(O1,2)),space(O1,2)'⊗ space(O1,2)');
    if group_index
       @tensor O1_O1[:]:=O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
       @tensor O2_O2[:]:=O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
       O1_O1=O1_O1/norm(O1_O1);
       O2_O2=O2_O2/norm(O2_O2);
       if N==8
            U_fuse_DD_D=unitary(fuse(space(O1_O1,2)⊗ space(O1,2)),space(O1_O1,2)'⊗ space(O1,2)');
            @tensor O1_O1_O1[:]:=O1_O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
            @tensor O2_O2_O2[:]:=O2_O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
            O1_O1_O1=O1_O1_O1/norm(O1_O1_O1);
            O2_O2_O2=O2_O2_O2/norm(O2_O2_O2);
            @tensor a_bcd_To_abc_d[:]:=U_fuse_DD_D[-1,3,4]*U_fuse_DD[3,-3,2]*U_fuse_DD'[2,4,1]*U_fuse_DD_D'[1,-2,-4];
       else
            U_fuse_DD_D=nothing;
            O1_O1_O1=nothing;
            O2_O2_O2=nothing;
            a_bcd_To_abc_d=nothing;
       end

    else
        U_fuse_DD_D=nothing;
        O1_O1_O1=nothing;
        O2_O2_O2=nothing;
        a_bcd_To_abc_d=nothing;
    end


    println("calculate ES for N="*string(N));
    Sectors=[0,1/2,1,3/2,2,5/2];

    eu_set=Vector(undef,length(Sectors));
    ks_set=Vector(undef,length(Sectors));

    for sps in eachindex(Sectors)
        if N==4
            v_init=TensorMap(randn, space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)',SU₂Space(Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,),());
            # v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
            if group_index
                @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4];
            end
        elseif N==6
            v_init=TensorMap(randn, space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)',SU₂Space(Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,5,6,7,),());
            # v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
            if group_index
                @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4]*U_fuse_DD[-3,5,6];
            end
        elseif N==8
            @assert group_index==true
            v_init=TensorMap(randn, fuse(space(OO,2)'*space(OO,2)'*space(OO,2)')*fuse(space(OO,2)'*space(OO,2)'*space(OO,2)')*fuse(space(OO,2)'*space(OO,2)'),SU₂Space(Sectors[sps]=>1));
            v_init=permute(v_init,(1,2,3,4,),());
            # v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
        end
        if norm(v_init)<1e-12
            eu_set[sps]=[];
            ks_set[sps]=[];
            continue;
        end

        ev=[];
        if group_index
            contraction_group_fun(x)=CTM_T_group_action(U_fuse_DD,O1_O1,O2_O2,U_fuse_DD_D,O1_O1_O1,O2_O2_O2,a_bcd_To_abc_d,x,N,[],vison);
            @time eu,ev=eigsolve(contraction_group_fun, v_init, EH_n,:LM,Arnoldi(krylovdim=EH_n*2+5));
            eu_set[sps]=eu;
            ks=calculate_k(ev,N,vison,group_index,U_fuse_DD,a_bcd_To_abc_d)
            ks_set[sps]=ks;
        else
            contraction_fun(x)=CTM_T_action(OO,x,N,vison);
            @time eu,ev=eigsolve(contraction_fun, v_init, EH_n,:LM,Arnoldi(krylovdim=EH_n*2+5));
            eu_set[sps]=eu;
            ks=calculate_k(ev,N,vison,group_index,U_fuse_DD,a_bcd_To_abc_d)
            ks_set[sps]=ks;
        end

        println("spin: "*string(Sectors[sps]));flush(stdout);
        println(eu);flush(stdout);

    end

    if vison
        ES_filenm="ES_vison"*"_D"*string(D)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    else
        ES_filenm="ES"*"_D"*string(D)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    end

    matwrite(ES_filenm, Dict(
        "eu_set" => eu_set,
        "Sectors" => Sectors,
        "ks_set" => ks_set
    ); compress = false)
end

function ES_CTMRG_ED_Kprojector(CTM::CTM_struc,D,chi,N,EH_n,group_index,vison)

    println("D="*string(D));
    println("chi="*string(chi));
    println("N="*string(N));flush(stdout);



    Tleft=CTM.Tset.T4;
    Tright=CTM.Tset.T2;
    @tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
    @tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];

    U_fuse_DD=unitary(fuse(space(O1,2)⊗ space(O1,2)),space(O1,2)'⊗ space(O1,2)');
    if group_index
       @tensor O1_O1[:]:=O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
       @tensor O2_O2[:]:=O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
       O1_O1=O1_O1/norm(O1_O1);
       O2_O2=O2_O2/norm(O2_O2);
       if N==8
            U_fuse_DD_D=unitary(fuse(space(O1_O1,2)⊗ space(O1,2)),space(O1_O1,2)'⊗ space(O1,2)');
            @tensor O1_O1_O1[:]:=O1_O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
            @tensor O2_O2_O2[:]:=O2_O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
            O1_O1_O1=O1_O1_O1/norm(O1_O1_O1);
            O2_O2_O2=O2_O2_O2/norm(O2_O2_O2);
            @tensor a_bcd_To_abc_d[:]:=U_fuse_DD_D[-1,3,4]*U_fuse_DD[3,-3,2]*U_fuse_DD'[2,4,1]*U_fuse_DD_D'[1,-2,-4];
       else
            U_fuse_DD_D=nothing;
            O1_O1_O1=nothing;
            O2_O2_O2=nothing;
            a_bcd_To_abc_d=nothing;
       end

    else
        U_fuse_DD_D=nothing;
        O1_O1_O1=nothing;
        O2_O2_O2=nothing;
        a_bcd_To_abc_d=nothing;
    end



    println("calculate ES for N="*string(N));
    Sectors=[0,1/2,1,3/2,2,5/2];
    Ks=collect(0:N-1)
    eu_set=Matrix(undef,length(Ks),length(Sectors));
    for kk=1:length(Ks)

        for sps=1:length(Sectors)
            if N==4
                v_init=TensorMap(randn, space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)',SU₂Space(Sectors[sps]=>1));
                v_init=permute(v_init,(1,2,3,4,5,),());
                v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
                if group_index
                    @tensor v_init[:]:=v_init[1,2,3,4,-3]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4];
                end
            elseif N==6
                v_init=TensorMap(randn, space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)'*space(OO,2)',SU₂Space(Sectors[sps]=>1));
                v_init=permute(v_init,(1,2,3,4,5,6,7,),());
                v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
                if group_index
                    @tensor v_init[:]:=v_init[1,2,3,4,5,6,-4]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4]*U_fuse_DD[-3,5,6];
                end
            elseif N==8
                @assert group_index==true
                v_init=TensorMap(randn, fuse(space(OO,2)'*space(OO,2)'*space(OO,2)')*fuse(space(OO,2)'*space(OO,2)'*space(OO,2)')*fuse(space(OO,2)'*space(OO,2)'),SU₂Space(Sectors[sps]=>1));
                v_init=permute(v_init,(1,2,3,4,),());
                v_init=k_projection(v_init,vison,N,Ks[kk],U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
            end
            if norm(v_init)<1e-12
                eu_set[kk,sps]=[];
                continue;
            end

            ev=[];
            if group_index
                contraction_group_fun(x)=CTM_T_group_action(U_fuse_DD,O1_O1,O2_O2,U_fuse_DD_D,O1_O1_O1,O2_O2_O2,a_bcd_To_abc_d,x,N,Ks[kk],vison);
                @time eu,ev=eigsolve(contraction_group_fun, v_init, EH_n,:LM,Arnoldi(krylovdim=EH_n*2+5));
                eu_set[kk,sps]=eu;
            else
                contraction_fun(x)=CTM_T_action(OO,x,N,vison);
                @time eu,ev=eigsolve(contraction_fun, v_init, EH_n,:LM,Arnoldi(krylovdim=EH_n*2+5));
                eu_set[kk,sps]=eu;
            end

            println("momentum: "*string(Ks[kk]));flush(stdout);
            println("spin: "*string(Sectors[sps]));flush(stdout);
            println(eu);flush(stdout);

            # for ccc=1:length(ev)
            #     v_projected=ev[ccc];
            #     @tensor v_projected[:]:=v_projected[1,2,-5]*U_fuse_DD'[-1,-2,1]*U_fuse_DD'[-3,-4,2];
            #     #print(space(v_projected))
            #     println(dot(v_projected,permute(v_projected,(2,3,4,1,5,),()))/dot(v_projected,v_projected));
            # end

        end
    end

    if vison
        ES_filenm="ES_Kprojector_vison"*"_D"*string(D)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    else
        ES_filenm="ES_Kprojector"*"_D"*string(D)*"_chi"*string(chi)*"_N"*string(N)*".mat";
    end
    matwrite(ES_filenm, Dict(
        "eu_set" => eu_set,
        "Sectors" => Sectors,
        "Ks" => Ks
    ); compress = false)


end


function vison_op(V)
    op=unitary(V,V);
    Keys=op.data.keys;
    for cc=1:length(Keys)
        if mod(Keys[cc].j,1)==1/2
            op.data.values[cc]=op.data.values[cc]*(-1);
        end
    end
    return op

end

function CTM_T_group_action(U_fuse_DD,O1_O1,O2_O2,U_fuse_DD_D,O1_O1_O1,O2_O2_O2,a_bcd_To_abc_d,v0,N,kn,vison)
    if N==4
        if vison
            op=vison_op(space(O1_O1,3));
            @tensor v_new[:]:=O1_O1[5,1,2,-1]*O1_O1[2,3,4,-2]*op[5,4]*v0[1,3,-3];
            op=vison_op(space(O2_O2,3));
            @tensor v_new[:]:=O2_O2[5,1,2,-1]*O2_O2[2,3,4,-2]*op[5,4]*v_new[1,3,-3];
        else
            @tensor v_new[:]:=O1_O1[4,1,2,-1]*O1_O1[2,3,4,-2]*v0[1,3,-3];
            @tensor v_new[:]:=O2_O2[4,1,2,-1]*O2_O2[2,3,4,-2]*v_new[1,3,-3];
        end 

        #momentum projector
        @tensor v_new[:]:=v_new[1,2,-5]*U_fuse_DD'[-1,-2,1]*U_fuse_DD'[-3,-4,2];
        if kn==[]
        else
            v_new=k_projection(v_new,vison,N,kn,U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
        end
        @tensor v_new[:]:=v_new[1,2,3,4,-3]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4];
    elseif N==6
        if vison
            op=vison_op(space(O1_O1,3));
            @tensor v_new[:]:=O1_O1[7,1,2,-1]*O1_O1[2,3,4,-2]*O1_O1[4,5,6,-3]*op[7,6]*v0[1,3,5,-4];
            op=vison_op(space(O2_O2,3));
            @tensor v_new[:]:=O2_O2[7,1,2,-1]*O2_O2[2,3,4,-2]*O2_O2[4,5,6,-3]*op[7,6]*v_new[1,3,5,-4];
        else
            @tensor v_new[:]:=O1_O1[6,1,2,-1]*O1_O1[2,3,4,-2]*O1_O1[4,5,6,-3]*v0[1,3,5,-4];
            @tensor v_new[:]:=O2_O2[6,1,2,-1]*O2_O2[2,3,4,-2]*O2_O2[4,5,6,-3]*v_new[1,3,5,-4];
        end

        #momentum projector
        @tensor v_new[:]:=v_new[1,2,3,-7]*U_fuse_DD'[-1,-2,1]*U_fuse_DD'[-3,-4,2]*U_fuse_DD'[-5,-6,3];
        if kn==[]
        else
            v_new=k_projection(v_new,vison,N,kn,U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
        end
        @tensor v_new[:]:=v_new[1,2,3,4,5,6,-4]*U_fuse_DD[-1,1,2]*U_fuse_DD[-2,3,4]*U_fuse_DD[-3,5,6];
    elseif N==8
        if vison
            op=vison_op(space(O1_O1,3));
            @tensor v_new[:]:=O1_O1_O1[7,1,2,-1]*O1_O1_O1[2,3,4,-2]*O1_O1[4,5,6,-3]*op[7,6]*v0[1,3,5,-4];
            op=vison_op(space(O2_O2,3));
            @tensor v_new[:]:=O2_O2_O2[7,1,2,-1]*O2_O2_O2[2,3,4,-2]*O2_O2[4,5,6,-3]*op[7,6]*v_new[1,3,5,-4];
        else
            @tensor v_new[:]:=O1_O1_O1[6,1,2,-1]*O1_O1_O1[2,3,4,-2]*O1_O1[4,5,6,-3]*v0[1,3,5,-4];
            @tensor v_new[:]:=O2_O2_O2[6,1,2,-1]*O2_O2_O2[2,3,4,-2]*O2_O2[4,5,6,-3]*v_new[1,3,5,-4];
        end

        #momentum projector
        if kn==[]
        else
            v_new=k_projection(v_new,vison,N,kn,U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d);
        end
    end
    return v_new
end

function k_projection(v_unprojected,vison,N,kn,U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d)
    vnorm=dot(v_unprojected,v_unprojected);
    v_projected=deepcopy(v_unprojected);
    for cc=1:N-1
        if N==4
            if vison #translation operator modified with existence of string
                op=vison_op(space(v_unprojected,1));
                @tensor v_unprojected[:]:=v_unprojected[1,-2,-3,-4,-5]*op[-1,1];
            end
            v_unprojected=permute(v_unprojected,(2,3,4,1,5),());
            sspin=space(v_unprojected,5).dims.keys[1].j;
            if (vison)&(mod(sspin,1)==1/2) #T^N=-1, where T is generalized translation operator
                v_projected=v_projected+exp(-im*((2*pi*kn+pi)/N)*cc)*v_unprojected;
            else
                v_projected=v_projected+exp(-im*((2*pi*kn)/N)*cc)*v_unprojected;
            end


        elseif N==6
            if vison #translation operator modified with existence of string
                op=vison_op(space(v_unprojected,1));
                @tensor v_unprojected[:]:=v_unprojected[1,-2,-3,-4,-5,-6,-7]*op[-1,1];
            end
            v_unprojected=permute(v_unprojected,(2,3,4,5,6,1,7),())

            sspin=space(v_unprojected,7).dims.keys[1].j;
            if (vison)&(mod(sspin,1)==1/2) #T^N=-1, where T is generalized translation operator
                v_projected=v_projected+exp(-im*((2*pi*kn+pi)/N)*cc)*v_unprojected;
            else
                v_projected=v_projected+exp(-im*((2*pi*kn)/N)*cc)*v_unprojected;
            end

        elseif N==8
            if vison #translation operator modified with existence of string
                op=vison_op(space(U_fuse_DD,2)');
                @tensor gate[:]:=U_fuse_DD[-1,2,1]*op[1,3]*U_fuse_DD'[2,3,-2];
                @tensor v_unprojected[:]:=gate[-3,1]*v_unprojected[-1,-2,1,-4];
            end
            v_unprojected=N8_permute(v_unprojected,U_fuse_DD,a_bcd_To_abc_d);


            sspin=space(v_unprojected,4).dims.keys[1].j;
            if (vison)&(mod(sspin,1)==1/2) #T^N=-1, where T is generalized translation operator
                v_projected=v_projected+exp(-im*((2*pi*kn+pi)/N)*cc)*v_unprojected;
            else
                v_projected=v_projected+exp(-im*((2*pi*kn)/N)*cc)*v_unprojected;
            end


        end
    end
    #dot(v_projected,permute(v_projected,(2,3,4,1,5,),()))/dot(v_projected,v_projected);#check momentum
    v_projected=v_projected/sqrt(dot(v_projected,v_projected))*sqrt(vnorm);
    return v_projected
end

function N8_permute(v_unpermuted,U_fuse_DD,a_bcd_To_abc_d)
    #initial group: (123)(456)(78)(9)
    @tensor v_permuted[:]:=v_unpermuted[-1,-2,1,-5]*U_fuse_DD'[-3,-4,1];#(123)(456)(7)(8)(9)
    v_permuted=permute(v_permuted,(4,1,2,3,5,),());#(8)(123)(456)(7)(9)
    @tensor v_permuted[:]:=v_permuted[1,2,-3,-4,-5]*a_bcd_To_abc_d[-1,-2,1,2];#(812)(3)(456)(7)(9)
    @tensor v_permuted[:]:=v_permuted[-1,1,2,-4,-5]*a_bcd_To_abc_d[-2,-3,1,2];#(812)(345)(6)(7)(9)
    @tensor v_permuted[:]:=v_permuted[-1,-2,1,2,-4]*U_fuse_DD[-3,1,2];
    return v_permuted
end




function calculate_k(ev,N,vison,group_index,U_fuse_DD,a_bcd_To_abc_d)
    #jldsave("test.jld2";ev,N,vison,group_index,U_fuse_DD,U_fuse_DD_D,a_bcd_To_abc_d)
    ks=Array{ComplexF64,1}(undef, length(ev));
    if N==4
        for cc=1:length(ev)
            v=ev[cc];
            if group_index
                @tensor v[:]:=v[1,2,-5]*U_fuse_DD'[-1,-2,1]*U_fuse_DD'[-3,-4,2];
            end
            if vison #translation operator modified with existence of string
                op=vison_op(space(v,1));
                @tensor vp[:]:=v[1,-2,-3,-4,-5]*op[-1,1];
                vp=permute(vp,(2,3,4,1,5),());
            else
                vp=permute(v,(2,3,4,1,5),());
            end
            
            phase=dot(vp,v)/dot(v,v);
            #println(phase)

            ks[cc]=phase;
        end
    elseif N==6
        for cc=1:length(ev)
            v=ev[cc];
            if group_index
                @tensor v[:]:=v[1,2,3,-7]*U_fuse_DD'[-1,-2,1]*U_fuse_DD'[-3,-4,2]*U_fuse_DD'[-5,-6,3];
            end
            if vison #translation operator modified with existence of string
                op=vison_op(space(v,1));
                @tensor vp[:]:=v[1,-2,-3,-4,-5,-6,-7]*op[-1,1];
                vp=permute(vp,(2,3,4,5,6,1,7),());
            else
                vp=permute(v,(2,3,4,5,6,1,7),());
            end
            
            phase=dot(vp,v)/dot(v,v);
            #println(phase)

            ks[cc]=phase;
        end
    elseif N==8
        for cc=1:length(ev)
            v=ev[cc];
            if group_index
                if vison #translation operator modified with existence of string
                    op=vison_op(space(U_fuse_DD,2)');
                    @tensor gate[:]:=U_fuse_DD[-1,2,1]*op[1,3]*U_fuse_DD'[2,3,-2];
                    @tensor vp[:]:=gate[-3,1]*v[-1,-2,1,-4];
                    vp=N8_permute(vp,U_fuse_DD,a_bcd_To_abc_d);
                else
                    vp=N8_permute(v,U_fuse_DD,a_bcd_To_abc_d);
                end
            else
                if vison #translation operator modified with existence of string
                    op=vison_op(space(v,1));
                    @tensor vp[:]:=v[1,-2,-3,-4,-5,-6,-7,-8,-9]*op[-1,1];
                    vp=permute(vp,(2,3,4,5,6,7,8,1,9),());
                else
                    vp=permute(v,(2,3,4,5,6,7,8,1,9),());
                end
                
            end
            
            phase=dot(vp,v)/dot(v,v);
            #println(phase)
            if (N==4)|(N==6)
                ks[cc]=phase;
            elseif (N==8) #definition of translation is opposite
                ks[cc]=phase';
            end
        end

    end
    return ks
end


function CTM_T_action(OO,v0,N,vison)
    if N==4
        if vison
            op=vison_op(space(OO,3));
            @tensor v_new[:]:=OO[9,1,2,-1]*OO[2,3,4,-2]*OO[4,5,6,-3]*OO[6,7,8,-4]*op[9,8]*v0[1,3,5,7,-5];
        else
            @tensor v_new[:]:=OO[8,1,2,-1]*OO[2,3,4,-2]*OO[4,5,6,-3]*OO[6,7,8,-4]*v0[1,3,5,7,-5];
        end
    end
    return v_new
end





