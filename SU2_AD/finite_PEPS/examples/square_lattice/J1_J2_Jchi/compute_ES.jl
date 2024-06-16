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
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk_excitation.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\environment\\Variational\\H_environment.jl")
include("..\\..\\..\\environment\\Variational\\oneD_contractions.jl")
include("..\\..\\..\\environment\\Variational\\variational_methods.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")
include("..\\..\\..\\environment\\full_update\\full_update_lib.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\triangle_terms.jl")
include("..\\..\\..\\ES\\ES_lib.jl")

Random.seed!(666)
global use_AD;
use_AD=true;

global chi,D
chi=100;
D=3;
filenm="optim_4x4_D_3_chi_100_13.89035.jld2";

import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_D"*string(D)*"_exc")
pid=getpid();
println("pid="*string(pid));


J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);

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
global backward_settings



global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=0;


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);


Lx=size(psi,1);
Ly=size(psi,2);
global U_phy
U_phy=space(psi[2,2],5);


multiplet_tol=1e-5;

global chi,multiplet_tol
###########################
psi=normalize_PEPS(psi);




###########################
E_total,E_set=energy_disk_(psi);
println(E_total)


psi=disk_to_torus(psi);

###########################
@assert Lx==4;
#contract virtual legs
psi_1=Matrix{TensorMap}(undef,Int(Lx/2),Ly);
for cx=1:Int(Lx/2)
    for cy=1:Ly
        T1=psi[2*cx-1,cy];
        T2=psi[2*cx,cy];
        U_u=unitary(fuse(space(T1,4)*space(T2,4)), space(T1,4)*space(T2,4));
        U_d=unitary(space(T1,2)'*space(T2,2)',fuse(space(T1,2)*space(T2,2)),);
        U_p=unitary(fuse(space(T1,5)*space(T2,5)), space(T1,5)*space(T2,5));
        @tensor T12[:]:=T1[-1,4,3,1,6]*T2[3,5,-3,2,7]*U_u[-4,1,2]*U_d[4,5,-2]*U_p[-5,6,7];
        psi_1[cx,cy]=T12;
    end
end

psi_2=Matrix{TensorMap}(undef,Int(Lx/4),Ly);
for cx=1:Int(Lx/4)
    for cy=1:Ly
        T1=psi_1[2*cx-1,cy];
        T2=psi_1[2*cx,cy];
        U_u=unitary(fuse(space(T1,4)*space(T2,4)), space(T1,4)*space(T2,4));
        U_d=unitary(space(T1,2)'*space(T2,2)',fuse(space(T1,2)*space(T2,2)),);
        U_p=unitary(fuse(space(T1,5)*space(T2,5)), space(T1,5)*space(T2,5));
        @tensor T12[:]:=T1[8,4,3,1,6]*T2[3,5,8,2,7]*U_u[-4,1,2]*U_d[4,5,-2]*U_p[-5,6,7];
        psi_2[cx,cy]=T12;
    end
end


@tensor psi_d[:]:=psi_2[1][-1,1,-2]*psi_2[2][1,-4,-3];
psi_d=permute(psi_d,(1,2,3,),(4,));
@tensor psi_u[:]:=psi_2[3][-1,1,-2]*psi_2[4][1,-4,-3];
psi_u=permute(psi_u,(1,),(2,3,4,));

psi_total=psi_d*psi_u;

u,s,v=tsvd(psi_total);
eu=s*s;

function get_ES(v0)
    eu0,ev0=eigen(v0);
    eu0_dense=convert(Array,eu0);
    order=sortperm(abs.(diag(eu0_dense)))


    Spin_set=Vector{Int64}(undef,0);
    eu_set=Vector{Int64}(undef,0);
    for cc=1:length(eu0.data.values)
      spa=eu0.data.keys[cc];

      Spin=spa.j;
      mm=diag(eu0.data.values[cc]);
      Spin_set=vcat(Spin_set,Spin*ones(length(mm)));
      eu_set=vcat(eu_set,mm);

    end

    eu_set=eu_set/sum(abs.(eu_set));
    pos=findall(x-> x.>1e-6, abs.(eu_set));
    Spin_set=Spin_set[pos];
    eu_set=eu_set[pos];

    order=sortperm(abs.(eu_set));
    Spin_set=Spin_set[order];
    eu_set=eu_set[order];
    return Spin_set,eu_set
  end


  Spin_set,eu_set=get_ES(eu);
  filenm_ = string(Lx)*"x"*string(Ly)*"_D"*string(D)*".mat";
  matwrite(filenm_, Dict(
    "E_total" => E_total,
    "Spin_set" => Spin_set,
    "eu_set" => eu_set
  ); compress = false)





