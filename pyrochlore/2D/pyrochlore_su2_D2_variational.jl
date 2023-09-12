using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")

Random.seed!(1234)

J1=1;
J2=1;

D=2;

N_lambda=20;
lambdas=(0:N_lambda)/N_lambda;
N_theta=20;
thetas=(0:N_theta)/N_theta*pi;
Es=Matrix(undef,length(lambdas),length(thetas));

for ca=1:length(lambdas)
    for cb=1:length(thetas)
        lambda=lambdas[ca];
        theta=thetas[cb];

        coe=[cos(theta),sin(theta)];
        virtual_type="square";
        Irrep="A1+iB1";#"A1", "A1+iB1"
        PEPS_tensor,A_fused,U_phy=build_PEPS(D,coe,virtual_type,Irrep);

        CTM=[];
        U_L=[];
        U_D=[];
        U_R=[];
        U_U=[];

        init=Dict([("CTM", []), ("init_type", "PBC")]);
        conv_check="singular_value";
        CTM_ite_info=true;
        CTM_conv_info=true;
        CTM_conv_tol=1e-6;
        CTM_ite_nums=100;
        CTM_trun_tol=1e-12;
        chi=40;
        @time CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);

        rho=build_density_op(U_phy, PEPS_tensor, AA_fused, U_L,U_D,U_R,U_U, CTM);#L',U',R',D',  L,U,R,D

        Sigma=plaquatte_Heisenberg(J1,J2);
        AKLT=plaquatte_AKLT(Sigma);
        H=AKLT*(lambda)+(1-lambda)*Sigma;
        E=plaquatte_ob(rho,H);
        Es[ca,cb]=E;

    end
end

matwrite("D2_energy"*".mat", Dict(
    "Es" => real(Es),
    "lambdas"=>Vector(lambdas),
    "thetas"=>Vector(thetas)
); compress = false)