using LinearAlgebra: norm, dot
import Base: +, -, *,/
import LinearAlgebra: norm,dot
#Define operations of groups of TensorMap
*(coe::Number,tt :: Matrix{T}) where T<:iPEPS_ansatz=add_group(tt,[],coe,0) ;
*(tt :: Matrix{T}, coe:: Number) where T<:iPEPS_ansatz=add_group(tt,[],coe,0);
/(tt :: Matrix{T}, coe::Number) where T<:iPEPS_ansatz=add_group(tt,[],1/coe,0);
+(tt1 :: Matrix{T}, tt2 :: Matrix{T}) where T<:iPEPS_ansatz=add_group(tt1,tt2,1,+1);
-(tt1 :: Matrix{T}, tt2 :: Matrix{T}) where T<:iPEPS_ansatz=add_group(tt1,tt2,1,-1);
norm(tt :: Matrix{T}) where T<:iPEPS_ansatz=norm_tensor_group(tt);
dot(tt1 :: Matrix{T}, tt2 :: Matrix{T}) where T<:iPEPS_ansatz=dot_tensor_group(tt1,tt2);



function C3_gdoptimize(f, g!, fg!, x0::Matrix{T}, linesearch, maxiter::Int = 500, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) where T<:iPEPS_ansatz
    global chi,D
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)

    # Univariate line search functions
    ϕ(α) = f(x + α*s)
    function dϕ(α)
        g!(gvec, x + α*s)
        return real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
    end
    function ϕdϕ(α)
        phi = fg!(gvec, x + α*s)
        dphi = real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
        return (phi, dphi)
    end

    s = similar(gvec) # Step direction

    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration "*string(iter))
        # x=normalize_tensor_group(x);
        x=normalize_ansatz(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = dot(s, gvec)
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/5, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end

function f(x::Matrix{T}) where T<:iPEPS_ansatz
    global CTM_tem,LS_ctm_setting,energy_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    if isa(x[1],Kagome_iPESS)
        E,E_up, E_down,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_up= "*string(E_up[:])*", "*"E_down= "*string(E_down[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x[1],Checkerboard_iPESS)
        E,E_plaquatte_cell,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_plaquatte_cell= "*string(E_plaquatte_cell[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x[1],Triangle_iPESS)
        if energy_setting.model == "spinful_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
        elseif energy_setting.model == "standard_triangle_Hubbard";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, triangle_up_set, triangle_dn_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("E= "*string(E));flush(stdout);
            println("ex_set= "*string(ex_set[:])); flush(stdout);
            println("ey_set= "*string(ey_set[:]));flush(stdout);
            println("e_diagonal1_set= "*string(e_diagonal1_set[:]));flush(stdout);
            println("e0_set= "*string(e0_set[:]));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
            println("eU_set= "*string(eU_set[:])); flush(stdout);
            println("triangle_up_set= "*string(triangle_up_set[:])); flush(stdout);
            println("triangle_dn_set= "*string(triangle_dn_set[:])); flush(stdout);
        elseif energy_setting.model == "standard_triangle_Hubbard_spiral";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, SSx_set, SSy_set, SSdiagonal_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("E= "*string(E));flush(stdout);
            println("ex_set= "*string(ex_set[:])); flush(stdout);
            println("ey_set= "*string(ey_set[:]));flush(stdout);
            println("e_diagonal1_set= "*string(e_diagonal1_set[:]));flush(stdout);
            println("e0_set= "*string(e0_set[:]));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
            println("eU_set= "*string(eU_set[:])); flush(stdout);
            println("SSx_set= "*string(SSx_set[:])); flush(stdout);
            println("SSy_set= "*string(SSy_set[:])); flush(stdout);
            println("SSdiagonal_set= "*string(SSdiagonal_set[:])); flush(stdout);
            
        end
    elseif isa(x[1],Square_iPEPS)
        if energy_setting.model =="triangle_J1_J2_Jchi"
            E,E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"E_LU_RU_LD= "*string(E_LU_RU_LD_set[:])*", "*"E_LD_RU_RD "*string(E_LD_RU_RD_set[:])*", "*"E_LU_LD_RD= "*string(E_LU_LD_RD_set[:])*", "*"E_LU_RU_RD= "*string(E_LU_RU_RD_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard";
            E, ex_set, ey_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard_pairing";
            E, ex_set, ey_set, px_set, py_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"px_set= "*string(px_set[:])*", "*"py_set= "*string(py_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinful_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        end

    end
    global E_history
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        # filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global save_filenm
        jldsave(save_filenm; x);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end

function g!(gvec::Matrix{T}, x) where T<:iPEPS_ansatz# this function changes the value of gvec
    println("compute grad")
    global E_tem, CTM_tem
    E_tem,∂E,CTM_tem=get_grad(x);


    ##############
    println("quality of C3 symmetry in grad:")
    Lx,Ly=size(gvec);
    B_set_grad=Matrix{TensorMap}(undef,Lx,Ly)
    T_set_grad=Matrix{TensorMap}(undef,Lx,Ly)
    for ca=1:Lx
        for cb=1:Ly
            B_set_grad[ca,cb]=∂E[ca,cb].Tm;
            T_set_grad[ca,cb]=∂E[ca,cb].Bm;
        end
    end
    
    B_set_grad_,T_set_grad_=to_C3_symmetric_fiPESS(B_set_grad,T_set_grad,0);
    B_set_grad_=B_set_grad_/3;
    T_set_grad_=T_set_grad_/3;
    
    C3_ovB=dot(B_set_grad_,B_set_grad)/sqrt(dot(B_set_grad_,B_set_grad_)*dot(B_set_grad,B_set_grad))
    C3_ovT=dot(T_set_grad_,T_set_grad)/sqrt(dot(T_set_grad_,T_set_grad_)*dot(T_set_grad,T_set_grad))
    @show C3_ovB;
    @show C3_ovT;

    ∂E_C3=deepcopy(∂E);
    for ca=1:Lx
        for cb=1:Ly
            ∂E_C3[ca,cb]=Triangle_iPESS(T_set_grad_[ca,cb],B_set_grad_[ca,cb]);
        end
    end
    
    ##############

    #gvec=∂E;#this will not change the input variable
    for cc in eachindex(gvec)
        setindex!(gvec,∂E_C3[cc],cc);#this will change the input variable
    end
    println("norm of grad: "*string(norm(gvec)));flush(stdout);

    return gvec
end


function NamedTuple_to_Struc_special(∂E,x)
    ∂E_new=deepcopy(x);
    Keys=keys(∂E);
    for cc in Keys
        setfield!(∂E_new,cc,getindex(∂E,cc))
    end
    return ∂E_new
end
function NamedTuple_to_Struc_cell(∂E,x)
    ∂E_new=deepcopy(x);
    for cc in eachindex(x)
        ∂E_new[cc]=NamedTuple_to_Struc_special(∂E[cc],x[cc])
    end
    return ∂E_new
end


function normalize_ansatz(x::Matrix{T}) where T<:iPEPS_ansatz
    for cc in eachindex(x)
        ansatz=x[cc];
        if isa(x[cc],Kagome_iPESS)
            B1=ansatz.B1;
            B2=ansatz.B2;
            B3=ansatz.B3;
            Tup=ansatz.Tup;
            Tdn=ansatz.Tdn;

            B1=B1/norm(B1);
            B2=B2/norm(B2);
            B3=B3/norm(B3);
            Tup=Tup/norm(Tup);
            Tdn=Tdn/norm(Tdn);
            ansatz_new=Kagome_iPESS(B1,B2,B3,Tup,Tdn);
        elseif isa(x[cc],Checkerboard_iPESS)
            BL=ansatz.B_L;
            BU=ansatz.B_U;
            Tm=ansatz.Tm;

            BL=BL/norm(BL);
            BU=BU/norm(BU);
            Tm=Tm/norm(Tm);
            ansatz_new=Checkerboard_iPESS(BL,BU,Tm);
        elseif isa(x[cc],Triangle_iPESS)
            iPEss=x[cc];
            bm=iPEss.Bm;
            tm=iPEss.Tm;
            bm=bm/norm(bm);
            tm=tm/norm(tm);
            ansatz_new=Triangle_iPESS(bm,tm);
        elseif isa(x[cc],Square_iPEPS)
            A=ansatz.T;
            A=A/norm(A);
            ansatz_new=Square_iPEPS(A);
        end
        x[cc]=ansatz_new;
    end
    return x
end

function get_grad(x0::Matrix{T}) where T<:iPEPS_ansatz
    global Lx,Ly
    x0=normalize_ansatz(x0);
    #∂E = cost_fun'(x);
    
    if isa(x0[1],Kagome_iPESS)
        x=Matrix{Kagome_iPESS_immutable}(undef,size(x0,1),size(x0,2));
    elseif isa(x0[1],Checkerboard_iPESS)
        x=Matrix{Checkerboard_iPESS_immutable}(undef,size(x0,1),size(x0,2));
    elseif isa(x0[1],Triangle_iPESS)
        x=Matrix{Triangle_iPESS_immutable}(undef,size(x0,1),size(x0,2));
    elseif isa(x0[1],Square_iPEPS)
        x=Matrix{Square_iPEPS_immutable}(undef,size(x0,1),size(x0,2));    
    end
    for cc in eachindex(x0)
        if isa(x0[cc],Kagome_iPESS)
            x[cc]=Kagome_iPESS_convert(x0[cc]);#convert to immutable ansatz
        elseif isa(x0[cc],Checkerboard_iPESS)
            x[cc]=Checkerboard_iPESS_convert(x0[cc]);#convert to immutable ansatz
        elseif isa(x0[cc],Triangle_iPESS)
            x[cc]=Triangle_iPESS_convert(x0[cc]);#convert to immutable ansatz
        elseif isa(x0[cc],Square_iPEPS)
            x[cc]=Square_iPEPS_convert(x0[cc]);#convert to immutable ansatz
        end
    end

    ∂E=gradient(x ->cost_fun(x), x)[1];#this works when x is a mutable structure. The output is a NamedTuple, not a structure, due to that the cost function takes out some fields of the input structure.

    ∂E=NamedTuple_to_Struc_cell(∂E,x0);
    #E=fun(state_vec)
    global E_tem, CTM_tem
    x_tem=x;

    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E,CTM_tem
end

function norm_tensor_group(x0:: Matrix{T}) where T<:iPEPS_ansatz
    global Lx,Ly
    Norm=0;
    for cc in eachindex(x0)
        Norm=Norm+norm(x0[cc])^2;
    end
    Norm=sqrt(Norm);
    return Norm
end



# function normalize_tensor_group(x0:: Matrix{T}) where T<:iPEPS_ansatz
#     x_new=deepcopy(x0);
#     Norm=norm_tensor_group(x0);
#     for cc in eachindex(x0)
#         setindex!(x_new,x0[cc]/Norm, cc)
#     end
#     return x_new
# end

# function normalize_tensor_group(x0:: Matrix{Triangle_iPESS}) 
#     x_new=deepcopy(x0);
#     for cc in eachindex(x0)
#         iPEss=x_new[cc];
#         bm=iPEss.Bm;
#         tm=iPEss.Tm;
#         bm=bm/norm(bm);
#         tm=tm/norm(tm);
#         iPEss=Triangle_iPESS(bm,tm);
#         setindex!(x_new,iPEss, cc)
#     end
#     return x_new
# end

function add_group(Tp1:: Matrix{T}, Tp2, coe1, coe2) where T<:iPEPS_ansatz
    x_new=deepcopy(Tp1);
    if Tp2==[]
        for cc in eachindex(Tp1)
            x_new[cc]=Tp1[cc]*coe1;
        end
    else
        for cc in eachindex(Tp1)
            x_new[cc]=Tp1[cc]*coe1+Tp2[cc]*coe2;
        end
    end
    return x_new
end

function dot_tensor_group(Tp1::Matrix{T},Tp2::Matrix{T}) where T<:iPEPS_ansatz
    y=0;
    for cc in eachindex(Tp1)
        y=y+dot(Tp1[cc],Tp2[cc])
    end
    if imag(y)/real(y)<1e-10
        y=real(y);
    end
    return y
end


function FD(state_vec::Matrix{TT}) where TT<:iPEPS_ansatz

    dt=0.000001

    E0=cost_fun_testt(state_vec);

    grad=similar(state_vec);
    for ct in eachindex(state_vec)
        grad[ct]=similar(state_vec[ct])*0;
    end

    for ct in eachindex(state_vec)
        Fields=fieldnames(typeof(state_vec[ct]));
        for fi in Fields
            for n_block in eachindex(getfield(state_vec[ct], fi).data.values)
                for elem in eachindex(getfield(state_vec[ct], fi).data.values[n_block])
                    state_vec_tem=deepcopy(state_vec);
                    tensor_=getfield(state_vec_tem[ct], fi);
                    T=tensor_.data.values[n_block];
                    T[elem]=T[elem]+dt;
                    tensor_.data.values[n_block]=T;
                    setfield!(state_vec_tem[ct],fi,tensor_);
                    real_part=(cost_fun_testt(state_vec_tem)-E0)/dt;

                    state_vec_tem=deepcopy(state_vec);
                    tensor_=getfield(state_vec_tem[ct], fi);
                    T=tensor_.data.values[n_block];
                    T[elem]=T[elem]+dt*im;
                    tensor_.data.values[n_block]=T;
                    setfield!(state_vec_tem[ct],fi,tensor_);
                    imag_part=(cost_fun_testt(state_vec_tem)-E0)/dt;

                    tensor__=getfield(grad[ct],fi);
                    tensor__.data.values[n_block][elem]=real_part+im*imag_part;
                    setfield!(grad[ct],fi,tensor__);
                end
            end
        end
    end
    return E0, grad
end


