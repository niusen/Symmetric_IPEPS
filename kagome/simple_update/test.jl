cd(@__DIR__)
T=CTM["Tset"][1];
T=permute(T,(1,3,2,),())

function HR(T,x)
    @tensor xp[:]:=T'[-1,1,2]*T[-2,3,2]*x[1,3,-3];
    return xp
end

function HR_conj(T,x)
    println(space(x))
    @tensor xp[:]:=T[1,-1,2]*T'[3,-2,2]*x[1,3,-3];
    return xp
end


vr_init=permute(TensorMap(randn, space(T',2)'*space(T,2)',SU₂Space(0=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
GLR_svd_R(x)=HR(T,x);
GLR_svd_R_conj(x)=HR_conj(T,x);
S0,U0,V0,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, 5,:LR, krylovdim=10);
#@assert info.converged >= minimum([n_eff,dim(full_space,sec)])

vr_init=permute(TensorMap(randn, space(T',2)'*space(T,2)',SU₂Space(0=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
GLR_svd_R(x)=HR(T,x);
GLR_svd_R_conj(x)=HR_conj(T,x);
Shalf,Uhalf,Vhalf,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, 5,:LR, krylovdim=10);
#@assert info.converged >= minimum([n_eff,dim(full_space,sec)])

vr_init=permute(TensorMap(randn, space(T',2)'*space(T,2)',SU₂Space(0=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
GLR_svd_R(x)=HR(T,x);
GLR_svd_R_conj(x)=HR_conj(T,x);
S1,U1,V1,info=svdsolve((GLR_svd_R,GLR_svd_R_conj), vr_init, 5,:LR, krylovdim=10);
#@assert info.converged >= minimum([n_eff,dim(full_space,sec)])


@tensor TT[:]:=T'[-1,-3,1]*T[-2,-4,1];
u,s,v=tsvd(TT,(1,2,),(3,4,))