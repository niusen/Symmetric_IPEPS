using LinearAlgebra
using TensorKit

function plaquatte_ob(rho,op)
    @tensor ob[:]:=rho[1,2,3,4,5,6,7,8]*op[5,6,7,8,1,2,3,4];
    @tensor norm[:]:=rho[1,2,3,4,1,2,3,4];
    ob=ob/norm;
    ob=blocks(ob)[Irrep[SU₂](0)];
    return ob

end

function build_density_op(U_phy, A_unfused, AA_fused, U_L,U_D,U_R,U_U, CTM)

    AA_open_1, U_s_s=build_double_layer_open(A_unfused,"1",U_phy,U_L,U_D,U_R,U_U);
    @tensor AA1[:]:=AA_open_1[-1,-2,-3,-4,1]*U_s_s[-5,-6,1];
    @tensor AA1[:]:=AA1[-1,-2,-3,-4,1,1];
    @assert norm(AA1-permute(AA_fused,(1,2,3,4,)))/norm(AA1)<1e-10

    AA_open, U_s_s_,fuse_spin=build_double_layer_open(A_unfused,"12",U_phy,U_L,U_D,U_R,U_U);
    @tensor AA_[:]:=AA_open[-1,-2,-3,-4,1]*U_s_s_[2,3,1]*fuse_spin[2,a,b]*fuse_spin'[a,b,3]
    @assert norm(AA_-permute(AA_fused,(1,2,3,4,)))/norm(AA_)<1e-10


    AA_open_2, _=build_double_layer_open(A_unfused,"2",U_phy,U_L,U_D,U_R,U_U);

    rho=density_op(CTM,AA_fused,AA_open,AA_open_1,AA_open_2);#L,U,D,R in a plaquatte
    @tensor rho[:]:=rho[1,-3,-4]*U_s_s_[-1,-2,1];#L'U',LU,R'R,D'D
    @tensor rho[:]:=rho[-1,-2,1,2]*U_s_s[-3,-4,1]*U_s_s[-5,-6,2];#L'U',LU,R',R,D',D

    rho=permute(rho,(1,3,5,),(2,4,6,));#L'U',R',D',     LU,R,D
    

    @tensor rho[:]:=rho[1,-3,-4,2,-7,-8]*fuse_spin[1,-1,-2]*fuse_spin'[-5,-6,2];#
    rho=permute(rho,(1,2,3,4,),(5,6,7,8,));#L',U',R',D',     L,U,R,D
    @assert norm(rho'-rho)/norm(rho)<1e-10 #Hermitian density matrix

    return rho

end








function build_double_layer_open(A_unfused,inds,U_phy,U_L,U_D,U_R,U_U)
    #seperate the physical index that to be open
    if inds=="12"
        #display(space(A))
        #A=permute(A,(1,2,3,),(4,5,));
        V_D=space(A_unfused, 4);
        V_s=space(A_unfused,5);
        fuse_spin=unitary(fuse(V_s ⊗ V_s), V_s ⊗ V_s);
        @tensor A_fused[:]:=A_unfused[-1,-2,-3,-4,1,2]*fuse_spin[-5,1,2];
        V_s=space(A_fused,5);
        V_ss=fuse(V_s' ⊗ V_s);

        uM_dag,sM_dag,vM_dag=tsvd(permute(A_fused',(1,2,3,),(4,5,)));
        uM_dag=uM_dag*sM_dag;
        uM,sM,vM=tsvd(permute(A_fused,(1,2,),(3,4,5,)));
        uM=uM*sM;
        
        uM_dag=permute(uM_dag,(1,2,3,4,),());
        uM=permute(uM,(1,2,3,),());
        Vp=space(vM_dag,1);
        V=space(vM,1);
        U=unitary(fuse(Vp ⊗ V), Vp ⊗ V);
        @tensor double_LD[:]:=uM_dag[-1,-2,-3,1]*U'[1,-4,-5];
        @tensor double_LD[:]:=double_LD[-1,-3,-5,1,-6]*uM[-2,-4,1];

        vM_dag=permute(vM_dag,(1,2,3,),());
        vM=permute(vM,(1,2,3,4,),());
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vM_dag[1,-3,-5]*double_RU[-1,1,-2,-4,-6];

        double_LD=permute(double_LD,(1,2,),(3,4,5,6,));
        double_LD=U_L*double_LD;
        double_LD=permute(double_LD,(2,3,),(1,4,5,));
        double_LD=U_D*double_LD;
        double_LD=permute(double_LD,(2,1,3,4),());
    
        double_RU=permute(double_RU,(1,2,5,6,),(3,4,));
        double_RU=double_RU*U_U;
        U_s_s=unitary(V_ss, V_s' ⊗ V_s);
        @tensor double_RU[:]:=double_RU[-1,-2,1,2,-3]*U_s_s[-4,1,2];

        A_fused=[];#clear memory
        A_unfused=[];#clear memory
        @tensor double_RU[:]:=double_RU[-1,1,-4,-5]*U_R[-2,1,-3];

        @tensor AA_open_fused[:]:=double_LD[-1,-2,2,1]*double_RU[1,2,-3,-4,-5];

        return AA_open_fused, U_s_s',fuse_spin
    elseif inds in ["1","2"]
        #display(space(A))
        #A=permute(A,(1,2,3,),(4,5,));
        V_D=space(A_unfused, 4);
        V_s=space(A_unfused,6);

        if inds=="1"
            A_unfused=permute(A_unfused,(1,2,3,4,5,6,),());
        elseif inds=="2"
            A_unfused=permute(A_unfused,(1,2,3,4,6,5,),());
        end

        A_fused=A_unfused;
        V_ss=fuse(V_s' ⊗ V_s);

        uM_dag,sM_dag,vM_dag=tsvd(permute(A_fused',(1,2,3,),(4,5,6,)));
        uM_dag=uM_dag*sM_dag;
        uM,sM,vM=tsvd(permute(A_fused,(1,2,),(3,4,5,6,)));
        uM=uM*sM;
        
        uM_dag=permute(uM_dag,(1,2,3,4,),());
        uM=permute(uM,(1,2,3,),());
        Vp=space(vM_dag,1);
        V=space(vM,1);
        U=unitary(fuse(Vp ⊗ V), Vp ⊗ V);
        @tensor double_LD[:]:=uM_dag[-1,-2,-3,1]*U'[1,-4,-5];
        @tensor double_LD[:]:=double_LD[-1,-3,-5,1,-6]*uM[-2,-4,1];

        vM_dag=permute(vM_dag,(1,2,3,4,),());
        vM=permute(vM,(1,2,3,4,5,),());
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
        @tensor double_RU[:]:=vM_dag[1,-3,-5,2]*double_RU[-1,1,-2,-4,-6,2];

        double_LD=permute(double_LD,(1,2,),(3,4,5,6,));
        double_LD=U_L*double_LD;
        double_LD=permute(double_LD,(2,3,),(1,4,5,));
        double_LD=U_D*double_LD;
        double_LD=permute(double_LD,(2,1,3,4),());
    
        double_RU=permute(double_RU,(1,2,5,6,),(3,4,));
        double_RU=double_RU*U_U;
        U_s_s=unitary(V_ss, V_s' ⊗ V_s);
        @tensor double_RU[:]:=double_RU[-1,-2,1,2,-3]*U_s_s[-4,1,2];

        A_fused=[];#clear memory
        A_unfused=[];#clear memory
        @tensor double_RU[:]:=double_RU[-1,1,-4,-5]*U_R[-2,1,-3];

        @tensor AA_open_fused[:]:=double_LD[-1,-2,2,1]*double_RU[1,2,-3,-4,-5];

        return AA_open_fused, U_s_s'

    end
end



function density_op(CTM,AA_fused,AA_12,AA_1,AA_2)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_12[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_1[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_2[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(5,1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
    return rho
end


