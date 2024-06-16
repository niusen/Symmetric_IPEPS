using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\ansatz\\square_lattice\\square_lattice.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")

Random.seed!(777)

global use_AD;
use_AD=false;



D=5;



Lx=8;
Ly=8;

data=load("CSL_D3.jld2");
A=data["A"];


A_new=zeros(5,5,5,5,2)*im;
A_new[[1,2,3],[1,2,3],[1,2,3],[1,2,3],:]=convert(Array,A);
A=TensorMap(A_new,Rep[SU₂](0=>1, 1/2=>2) ⊗ Rep[SU₂](0=>1, 1/2=>2) ⊗ Rep[SU₂](0=>1, 1/2=>2)' ⊗ Rep[SU₂](0=>1, 1/2=>2)', Rep[SU₂](1/2=>1));

A_noise=TensorMap(randn,codomain(A),domain(A))+TensorMap(randn,codomain(A),domain(A))*im;
A=A+A_noise/norm(A_noise)*0.1;
A=permute(A,(1,2,3,4,5,));



P=zeros(1,dim(space(A,1)));P[1,1]=1;
P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));

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


psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));
psi_double=construct_double_layer(psi,psi);

multiplet_tol=1e-5;
chi=100;







mps_bot=(psi_double[:,1]...,);
mps0=mps_bot;
mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
println(norm_1D(mps_bot,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_bot,mps_bot)))
for cy=2:Ly-2
    mpo=(psi_double[:,cy]...,);
    mps0,_=apply_mpo(mpo,mps_bot);
    mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
    println(norm_1D(mps_bot,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_bot,mps_bot)))
end


function treat_mps_top(mps_top)
    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];
    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end
    return mps_top
end

mps_top=(psi_double[:,Ly]...,);
mps_top=pi_rotate_mps(mps_top);
mps0=mps_top;
mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
println(norm_1D(mps_top,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_top,mps_top)))
for cy=Ly-1:-1:3
    mpo=pi_rotate_mpo((psi_double[:,cy]...,));
    mps0,_=apply_mpo(mpo,mps_top);
    mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
    println(norm_1D(mps_top,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_top,mps_top)))
end

global n_mps_sweep
n_mps_sweep=3;
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
mps_bot=(psi_double[:,1]...,);
mps0=mps_bot;
mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
println(norm_1D(mps_bot,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_bot,mps_bot)))
for cy=2:Ly-2
    mpo=(psi_double[:,cy]...,);
    mps0,_=apply_mpo(mpo,mps_bot);
    mps_bot,trun_errs=Zygote.checkpointed(simple_truncate_to_moddle, mpo, mps_bot);
    println(norm_1D(mps_bot,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_bot,mps_bot)))
end



mps_top=(psi_double[:,Ly]...,);
mps_top=pi_rotate_mps(mps_top);
mps0=mps_top;
mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
println(norm_1D(mps_top,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_top,mps_top)))
for cy=Ly-1:-1:3
    mpo=pi_rotate_mpo((psi_double[:,cy]...,));
    mps0,_=apply_mpo(mpo,mps_top);
    mps_top,trun_errs=Zygote.checkpointed(simple_truncate_to_moddle, mpo, mps_top);
    println(norm_1D(mps_top,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_top,mps_top)))
end




# include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
# mps_bot=(psi_double[:,1]...,);
# mpo=(psi_double[:,2]...,);

# mps_trun0,trun_errs=Zygote.checkpointed(simple_truncate_to_moddle, mpo, mps_bot);
# n_mps_sweep=3;
# mps_trun,trun_errs=Zygote.checkpointed(simple_truncate_to_moddle, mpo, mps_bot);


# mps0,_=apply_mpo(mpo,mps_bot);
# println(norm_1D(mps_trun0,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_trun0,mps_trun0)))
# println(norm_1D(mps_trun,mps0)/sqrt(norm_1D(mps0,mps0)*norm_1D(mps_trun,mps_trun)))



