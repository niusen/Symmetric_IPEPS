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
Dmax=6;
filenm="optim_4x4_D_3_chi_100_13.89035.jld2";

always_update=true;#update even if energy goes up


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters


svd_settings=Svd_settings();
svd_settings.svd_trun_method="chi";#chi" or "tol"
svd_settings.chi_max=500;
svd_settings.tol=1e-5;
dump(svd_settings);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global svd_settings, backward_settings




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);
psi0=deepcopy(psi);

global Lx,Ly
Lx=size(psi,1);
Ly=size(psi,2);

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];


save_opt_filenm="opt_2site_"*string(Lx)*"x"*string(Ly)*"_Dmax_"*string(Dmax)*"_chi_"*string(chi)*".jld2"
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
println("initial energy: "*string(E_opt))
global E_opt



bond_coord_set=Matrix{Int64}(undef,0,2);
for cx=1.5:1:Lx-0.5
    for cy=1:Ly
        px=cx;
        py=cy;
        bond_coord_set=vcat(bond_coord_set,[px py])
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
        bond_coord_set=vcat(bond_coord_set,[px py])
        # println([px,py])
        # Noise0=0;
        # trun_bond_type0="dD"
        # t_bond,psi_left=get_bond(psi,px,py,trun_bond_type0,Noise0);
        # E_opt1=real(cost_fun_bond(t_bond,psi_left,psi_double));
        # println(E_opt1-E_opt)
    end
end

#order from bulk to boundary
order=sortperm(abs.(bond_coord_set[:,1].-(Lx/2+0.5))+abs.(bond_coord_set[:,2].-(Ly/2+0.5)));
bond_coord_set=bond_coord_set[order,:];

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
grad_tol=1e-3;


bond_noise=0.1;
trun_bond_type="dD"


for ite=1:optim_maxiter
    println("Optimization iteration: "*string(ite));
    for cp =1:size(bond_coord_set,1)
        global E_opt
        px,py=bond_coord_set[cp,:];
        println("bond: "*string([px,py]));

        n_mps_sweep=5;
        t_bond,psi_left=get_bond(psi,px,py,trun_bond_type,bond_noise);

        E_opt_new, t_bond_new, iter_bt3 = gdoptimize_2site(psi_left,psi_double, f_2site, g!_2site, fg!_2site, t_bond, ls,LS_maxiter, 1e-8, grad_tol);
        N_env,log_coe=env_2site(psi_left,px,py);
        ####################
        T1a,T2a=bond_simple_trun(t_bond_new);
        psi_newa,psi_doublea=set_bond_cut(psi_left,psi_double,px,py,T1a,T2a);
        n_mps_sweep=5;
        E_cuta=real(cost_fun_global(psi_newa));
        println("space: "*string(space(T1a,3)))

        T1aa,T2aa=optimize_truncation(T1a,T2a,t_bond_new,N_env)
        psi_newaa,psi_doubleaa=set_bond_cut(psi_left,psi_double,px,py,T1aa,T2aa);
        n_mps_sweep=5;
        E_cutaa=real(cost_fun_global(psi_newaa));
        println("Energy simple cut: "*string(E_cuta)*" -> "*string(E_cutaa))
        if E_cutaa<E_cuta
            psi_newa=psi_newaa;
            psi_doublea=psi_doubleaa;
            E_cuta=E_cutaa;
        end
        #####################
        
        T1b,T2b=bond_gauge_fix_trun(t_bond_new,N_env);
        psi_newb,psi_doubleb=set_bond_cut(psi_left,psi_double,px,py,T1b,T2b);
        n_mps_sweep=5;
        E_cutb=real(cost_fun_global(psi_newb));
        println("space: "*string(space(T1b,3)))

        T1bb,T2bb=optimize_truncation(T1b,T2b,t_bond_new,N_env)
        psi_newbb,psi_doublebb=set_bond_cut(psi_left,psi_double,px,py,T1bb,T2bb);
        n_mps_sweep=5;
        E_cutbb=real(cost_fun_global(psi_newbb));
        println("Energy gauge fix cut= "*string(E_cutb)*" -> "*string(E_cutbb))

        if E_cutbb<E_cutb
            psi_newb=psi_newbb;
            psi_doubleb=psi_doublebb;
            E_cutb=E_cutbb;
        end
        #######################

        if E_cutb<E_cuta
            psi_tem=psi_newb;
            E_opt_new=E_cutb;
        else
            psi_tem=psi_newa;
            E_opt_new=E_cuta;
        end

        if (E_opt_new<E_opt)|(always_update)
            psi=psi_tem;
            psi_double=construct_double_layer(psi,psi);
            E_opt=E_opt_new;
            println("Energy of updated state: "*string(E_opt));flush(stdout);
            jldsave(save_opt_filenm; psi=psi);
        else
            psi=psi;
            psi_double=construct_double_layer(psi,psi);
            println("Energy not improved, change to next site")
        end

        #normalization of tensors
        for cc1=1:Lx
            for cc2=1:Ly
                psi[cc1,cc2]=psi[cc1,cc2]/norm(psi[cc1,cc2]);
            end
        end

    end
end


# end



