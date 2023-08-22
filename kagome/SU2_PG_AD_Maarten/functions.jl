using LinearAlgebra
using TensorKit
using TensorKitAD


function construct_tensors(D)
    # #D=3
    # filenm="square_tensors_D"*string(D)*".mat"
    # vars = matread(filenm)
    # B1a=vars["B1a"]
    # B1b=vars["B1b"]
    # B2=vars["B2"]
    
    Va=SU2Space(0=>1, 1/2=>1,1=>1)
    Vb=SU2Space(0=>1, 1/2=>1,1=>1)

    B1a = TensorMap(randn, Va ⊗ Va ⊗ Va' ⊗ Va'← Vb);
    B1b = TensorMap(randn, Va ⊗ Va ⊗ Va' ⊗ Va'← Vb);
    B2 = TensorMap(randn, Va ⊗ Va ⊗ Va' ⊗ Va'← Vb);
    
    return B1a,B1b,B2
end

function construct_IPEPS(state_vec,B1a,B1b,B2)

    return state_vec[1]*B1a+state_vec[2]*B1b+state_vec[3]*B2
end




function build_double_layer(A,operator)
    #display(space(A))
    A=permute(A,(1,2,),(3,4,5));
    U_L = @ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D= @ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));


    U_R=U_L';
    U_U=U_D';
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uM,sM,vM=tsvd(A);
    uM=uM*sM

    uM=permute(uM,(1,2,3,),())
    V=space(vM,1);
    U = @ignore_derivatives unitary(fuse(V' ⊗ V), V' ⊗ V);
    @tensor double_LD[:]:=uM'[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vM=permute(vM,(1,2,3,4,),());
    if operator==[]
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vM'[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];
    else
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vM'[3,-2,-4,1]*operator[2,1]*double_RU[-1,3,-3,-5,2];
    end
    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;

    return AA_fused, U_L,U_D,U_R,U_U
end




function cost_fun(state_vec,B1a,B1b,B2)

    PEPS_tensor=construct_IPEPS(state_vec,B1a,B1b,B2);

    AA_fused, U_L,U_D,U_R,U_U=build_double_layer(PEPS_tensor,[]);
    
    ov = @tensor AA_fused[1,2,1,2];
    #@tensor ov[:]:= AA_fused[1,2,1,2];
    #ov=blocks(ov)[Irrep[SU₂](0)][1];
    
    return real(ov)

end



