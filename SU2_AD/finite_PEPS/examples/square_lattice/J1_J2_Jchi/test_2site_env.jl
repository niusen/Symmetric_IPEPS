using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\environment\\extend_bond\\extend_bond.jl")
include("..\\..\\..\\environment\\extend_bond\\environment_2site.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\line_search_lib_2site.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")
include("..\\..\\..\\optimization\\LineSearches\\My_Backtracking.jl")

# let
Random.seed!(888)
global use_AD;
use_AD=true;

global chi,Dmax
chi=200;
Dmax=8;
filenm="CSL_D3_Lx6_Ly6.jld2";



J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global backward_settings




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);
psi=add_global_noise(psi,1.3);

psi0=deepcopy(psi);



global Lx,Ly
Lx=size(psi,1);
Ly=size(psi,2);

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];


save_opt_filenm="optim_"*string(Lx)*"x"*string(Ly)*"_Dmax_"*string(Dmax)*"_chi_"*string(chi)*".jld2"
global save_opt_filenm

global starting_time
starting_time=now();


multiplet_tol=1e-5;



global chi,multiplet_tol

global px,py

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;

E_opt=real(cost_fun_global(psi0));

bond_coord_set=[];
for cx=1.5:1:Lx-0.5
    for cy=1:Ly
        px=cx;
        py=cy;
        bond_coord_set=vcat(bond_coord_set,(px,py,))
        # println([px,py])
        # Noise0=0;
        # trun_bond_type0="dD"
        # t_bond,psi_left=get_bond(psi,px,py,trun_bond_type0,Noise0);
        # E_opt1=real(cost_fun_bond(t_bond,psi_left,psi_double));
        # println(E_opt1-E_opt)
    end
end

for cx=1:Lx
    for cy=1.5:1:Ly-0.5
        px=cx;
        py=cy;
        bond_coord_set=vcat(bond_coord_set,(px,py,))
        # println([px,py])
        # Noise0=0;
        # trun_bond_type0="dD"
        # t_bond,psi_left=get_bond(psi,px,py,trun_bond_type0,Noise0);
        # E_opt1=real(cost_fun_bond(t_bond,psi_left,psi_double));
        # println(E_opt1-E_opt)
    end
end

println("initial energy: "*string(E_opt))
global E_opt

#n_mps_sweep=0;
#∂E=gradient(x ->cost_fun_global(x), psi)[1];
# ∂E=gradient(x ->cost_fun_bond(x,psi_left,psi_double), t_bond)[1];
# println(norm(∂E))
# E_tem,∂E=get_grad_2site(t_bond,psi_left,psi_double);

#########################################


# ls = BackTracking(order=3)
ls = BackTracking(c_1=0.0001,ρ_hi=0.5,ρ_lo=0.1,iterations=7,order=3,maxstep=Inf);
println(ls)

optim_maxiter=5;
LS_maxiter=8;#number of gradient optimization for each site
grad_tol=1e-2;


bond_noise=0;
trun_bond_type="dD"


for ite=1:1#optim_maxiter
    println("Optimization iteration: "*string(ite));
    for cp =1:length(bond_coord_set)
        global E_opt
        px,py=bond_coord_set[cp];
        println("bond: "*string([px,py]));

        n_mps_sweep=5;
        t_bond,psi_left=get_bond(psi,px,py,trun_bond_type,bond_noise);
        t_bond=permute(t_bond,(1,2,3,4,));
        # t_bond=t_bond/norm(t_bond);
        N_env,log_coe=env_2site(psi_left,px,py);
        Norm=@tensor N_env[3,4,5,6]*t_bond'[3,1,5,2]*t_bond[4,1,6,2];
        println(log(Norm)+log_coe)

      
        


    end
end


# end



