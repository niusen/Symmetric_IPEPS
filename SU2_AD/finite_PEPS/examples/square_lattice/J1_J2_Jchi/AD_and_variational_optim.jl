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
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\environment\\Variational\\oneD_contractions.jl")
include("..\\..\\..\\environment\\Variational\\variational_methods.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\environment\\Variational\\H_environment.jl")

global use_AD;


global chi,D
chi=100;
D=5;
filenm="sweep_LS_D_5_chi_100.jld2";


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);


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

Lx=size(psi,1);
Ly=size(psi,2);

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];

save_sweep_filenm="sweep_"*string(Lx)*"x"*string(Ly)*"_D_"*string(D)*"_chi_"*string(chi)*".jld2"
save_opt_filenm="optim_"*string(Lx)*"x"*string(Ly)*"_D_"*string(D)*"_chi_"*string(chi)*".jld2"
global save_sweep_filenm, save_opt_filenm

global starting_time
starting_time=now();


multiplet_tol=1e-5;



global chi,multiplet_tol

global psi,psi_double,px,py


px=1;py=1;
E_opt=real(cost_fun(psi[px,py]));

#########################################


ls = BackTracking(order=3)
println(ls)

optim_maxiter=5;
LS_maxiter=10;
grad_tol=1e-2;

for ite=1:optim_maxiter
    println("Optimization iteration: "*string(ite));
    for cx=1:Lx
        for cy=1:Ly
            px=cx;
            py=cy;
            println("coordinate: "*string([px,py]));

            #############################
            #variational optimization
            use_AD=false;
            psi_backup=deepcopy(psi);
            psi_new=variational_opt_site(psi,chi,only_nearest_plaquatte,px,py);
            E_tem=energy_disk(psi_new);
            E_tem=real(E_tem);
            println("accurate energy after variational optimization: "*string(E_tem))
            if E_tem<E_opt
                psi=psi_new;
                psi_double=construct_double_layer(psi,psi);
                E_opt=E_tem;
                println("energy improve from variational optimization");
                jldsave(save_opt_filenm; psi);
                continue
            else
                psi=psi_backup;
                println("energy not improved from variational optimization, do AD optimization");
            end

            #############################
            #AD optimization
            A0=psi[px,py];
            psi_double=construct_double_layer(psi,psi);
            use_AD=true;
            E_opt_new, T, iter_bt3 = gdoptimize(f, g!, fg!, A0, ls,LS_maxiter, 1e-8, grad_tol);
            if E_opt_new<E_opt
                psi[px,py]=T;
                psi_double=construct_double_layer(psi,psi);
                E_opt=E_opt_new;
                println("Energy of updated state: "*string(E_opt));flush(stdout);
                jldsave(save_opt_filenm; psi);
            else
                println("Energy not improved, move to next site")
            end
        end
    end
    
end





