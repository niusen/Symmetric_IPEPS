using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)
include("..\\resource_codes\\kagome_load_tensor.jl")
include("..\\resource_codes\\kagome_CTMRG.jl")
include("..\\resource_codes\\kagome_model.jl")
include("..\\resource_codes\\kagome_IPESS.jl")
include("..\\resource_codes\\kagome_FiniteDiff.jl")
include("..\\resource_codes\\Settings.jl")


Random.seed!(12345)


D=3;
chi=40;


theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);





Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="A1_even";#"No", "A1_even"





ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=50;
ctm_setting.CTM_trun_tol=1e-8;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;

dump(ctm_setting);


optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"LS_A1even_U1_D_6_chi_60.json";#"nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"

dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_single_triangle";
energy_setting.E_up_method = "1x1";
energy_setting.cal_chiral_order = false;

dump(energy_setting);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb  
#run_FiniteDiff(parameters,D,chi,Bond_irrep,Triangle_irrep,nonchiral,ctm_setting,optim_setting,energy_setting);


json_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,"nothing",0)
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb)

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit =Hamiltonians(U_phy,J1,J2,J3,Jchi,Jtrip)

AA_H,_=build_double_layer(A_fused,H_triangle);
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(10,A_fused,"PBC",false,true);


function ob(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    Norm= @tensor envL[1,2,3]*envR[1,2,3];
    Norm=real(Norm);


    return Norm;
end
ob(CTM,AA_H)

function cfun(A_fused, CTM)
    
    
    function fun(A_fused)
        CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM_test(10,A_fused,"PBC",false,true);
        AA_H,_=build_double_layer(A_fused,H_triangle);
        AA,_=build_double_layer(A_fused,[]);
        E=ob(CTM,AA_H)/ob(CTM,AA_fused)
        println(E)
        return E
    end

    ∂E = fun'(A_fused)

    CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM_test(10,A_fused,"PBC",false,true);
    AA_H,_=build_double_layer(A_fused,H_triangle);
    AA,_=build_double_layer(A_fused,[]);
    E=ob(CTM,AA_H)/ob(CTM,AA_fused)
    

    @assert !isnan(norm(∂E))
    
    return E,∂E
end

a,b=cfun(A_fused,CTM)


function fuse_CTM_legs_test(CTM,U_L,U_D,U_R,U_U)
    #Tset_new=Vector{TensorMap}(undef,4);
    #fuse CTM legs
    Tset=CTM.Tset;

    #T4
    T4=permute(Tset.T4,(1,4,),(2,3,));
    T4=T4*U_R;
    T4=permute(T4,(1,3,2,),());
    #Tset_new[4]=T4
    #display(space(T4))

    #T3
    T3=permute(Tset.T3,(1,4,),(2,3,));
    T3=T3*U_U;
    T3=permute(T3,(1,3,2,),());
    #Tset_new[3]=T3
    #display(space(T3))

    #T2
    T2=permute(Tset.T2,(2,3,),(1,4,));
    T2=U_L*T2;
    T2=permute(T2,(2,1,3,),());
    #Tset_new[2]=T2
    #display(space(T2))

    #T1
    T1=permute(Tset.T1,(2,3,),(1,4,));
    T1=U_D*T1;
    T1=permute(T1,(2,1,3,),());
    #Tset_new[1]=T1
    #display(space(T1))

    CTM.Tset=Tset_struc(T1,T2,T3,T4);
    return CTM
end

function init_CTM_test(chi,A,type,CTM_ite_info,construct_double_layer)
    if CTM_ite_info
        display("initialize CTM")
    end
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    #Cset=Vector{TensorMap}(undef,4);
    #Tset=Vector{TensorMap}(undef,4);
    Cset=Cset_struc(A,A,A,A)
    Tset=Tset_struc(A,A,A,A)
    # Cset=(A,A,A,A)
    # Tset=(A,A,A,A)
    #space(A,1)
    
    if type=="PBC"
        for direction=1:4
            inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
            A_rotate=permute(A,inds);
            Ap_rotate=A_rotate';

            @tensor M[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];
            Cset=set_Cset(Cset,M,direction);
            @tensor M[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
            Tset=set_Tset(Tset,M,direction);
        end
        
        #fuse legs
        #ul_set=Vector{TensorMap}(undef,4);
        #ur_set=Vector{TensorMap}(undef,4);
        ul_set=Tset_struc(A,A,A,A)
        ur_set=Tset_struc(A,A,A,A)
        for direction=1:2
            ul_set_tem=@ignore_derivatives unitary(fuse(space(get_Cset(Cset,direction), 3) ⊗ space(get_Cset(Cset,direction), 4)), space(get_Cset(Cset,direction), 3) ⊗ space(get_Cset(Cset,direction), 4));
            ur_set_tem=@ignore_derivatives unitary(fuse(space(get_Tset(Tset,direction), 5) ⊗ space(get_Tset(Tset,direction), 6)), space(get_Tset(Tset,direction), 5) ⊗ space(get_Tset(Tset,direction), 6));
            ul_set=set_Tset(ul_set,ul_set_tem,direction);
            ur_set=set_Tset(ur_set,ur_set_tem,direction);
        end
        for direction=3:4
            ul_set_tem=@ignore_derivatives unitary(fuse(space(get_Cset(Cset,direction), 3) ⊗ space(get_Cset(Cset,direction), 4))', space(get_Cset(Cset,direction), 3) ⊗ space(get_Cset(Cset,direction), 4));
            ur_set_tem=@ignore_derivatives unitary(fuse(space(get_Tset(Tset,direction), 5) ⊗ space(get_Tset(Tset,direction), 6))', space(get_Tset(Tset,direction), 5) ⊗ space(get_Tset(Tset,direction), 6));
            ul_set=set_Tset(ul_set,ul_set_tem,direction);
            ur_set=set_Tset(ur_set,ur_set_tem,direction);
        end
        for direction=1:4
            C=get_Cset(Cset,direction);
            ul=get_Tset(ur_set,mod1(direction-1,4));
            ur=get_Tset(ul_set,direction);
            ulp=permute(ul',(3,),(1,2,));
            urp=permute(ur',(3,),(1,2,));
            #@tensor Cnew[(-1);(-2)]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]
            @tensor Cnew[:]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4];#put all indices in tone side so that its adjoint has the same index order
            Cset=set_Cset(Cset,Cnew,direction);
            

            T=get_Tset(Tset,direction);
            ul=get_Tset(ul_set,direction);
            ur=get_Tset(ur_set,direction);
            ulp=permute(ul',(3,),(1,2,));
            urp=permute(ur',(3,),(1,2,));
            #@tensor Tnew[(-1);(-2,-3,-4)]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]
            @tensor Tnew[:]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order
            Tset=set_Tset(Tset,Tnew,direction);
            
        end

    elseif type=="random"
    end

    CTM=CTM_struc(Cset, Tset);
    
    if construct_double_layer
        AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A,[]);
        
        CTM=fuse_CTM_legs_test(CTM,U_L,U_D,U_R,U_U);
        
    else
        U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
        U_R=(U_L)';
        U_U=(U_D)';
        AA_fused=[];
    end

    return CTM, AA_fused, U_L,U_D,U_R,U_U


end;



function get_Cset(Cset,direction)
    if direction==1
        return Cset.C1
    elseif direction==2
        return Cset.C2
    elseif direction==3
        return Cset.C3
    elseif direction==4
        return Cset.C4
    end

end

function get_Tset(Tset,direction)
    if direction==1
        return Tset.T1
    elseif direction==2
        return Tset.T2
    elseif direction==3
        return Tset.T3
    elseif direction==4
        return Tset.T4
    end

end

function set_Cset(Cset,M,direction)
    if direction==1 
        Cset.C1=M;
    elseif direction==2
        Cset.C2=M; 
    elseif direction==3
        Cset.C3=M; 
    elseif direction==4
        Cset.C4=M; 
    end
    return Cset
end

function set_Tset(Tset,M,direction)
    if direction==1 
        Tset.T1=M;
    elseif direction==2
        Tset.T2=M; 
    elseif direction==3
        Tset.T3=M; 
    elseif direction==4
        Tset.T4=M; 
    end
    return Tset
end



# function cfun(x)
#     (ψ,env) = x

#     function fun(peps)
#         env = leading_boundary(peps, alg_ctm, env)
#         x = H_expectation_value(peps, env, H)   
#         return x
#     end

#     ∂E = fun'(ψ)
#     env = leading_boundary(ψ, alg_ctm, env)
#     E = H_expectation_value(ψ, env, H)

#     @assert !isnan(norm(∂E))
#     return E,∂E
# end

# # my_retract is not an in place function which should not change x
# function my_retract(x,dx,α::Number)
#     (ϕ,env0) = x
#     ψ = deepcopy(ϕ)
#     env = deepcopy(env0)
#     ψ.A .+= dx.A .* α
#     #env = leading_boundary(ψ, alg_ctm,env)
#     return (ψ,env),dx
# end

# my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

# function my_add!(Y, X, a)
#     Y.A .+= X.A .* a
#     return Y
# end

# function my_scale!(η, β)
#     rmul!(η.A, β)
#     return η
# end


# function init_psi(d::Int, D::Int, Lx::Int, Ly::Int)
#     Pspaces = fill(ℂ^d,Lx,Ly)
#     Nspaces = fill(ℂ^D,Lx,Ly)
#     Espaces = fill(ℂ^D,Lx,Ly)

#     Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
#     Wspaces = adjoint.(circshift(Espaces, (0, -1)))

#     A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
#         return TensorMap(rand, ComplexF64, P ← N * E * S * W)
#     end

#     return InfinitePEPS(A)
# end


# alg_ctm = CTMRG(
#             verbose=10000,
#             tol=1e-10,
#             trscheme=truncdim(10),
#             miniter=4,
#             maxiter=200
#         )

# function main(;d=2,D=2,Lx=1,Ly=1)
#     ψ = init_psi(d,D,Lx,Ly)   
#     env = leading_boundary(ψ, alg_ctm) 
#     optimize(
#         cfun, 
#         (ψ,env),
#         ConjugateGradient(verbosity=3); 
#         inner=my_inner,
#         retract=my_retract,
#         scale! = my_scale!,
#         add! = my_add!
#     )
#     return ψ
# end

# main()
