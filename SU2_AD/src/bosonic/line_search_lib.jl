using LinearAlgebra: norm, dot
import Base: +, -, *,/
import LinearAlgebra: norm,dot
#Define operations of groups of TensorMap
*(coe::Number,tt :: Vector{TensorMap})=add_group(tt,[],coe,0);
*(coe::Number,tt :: iPEPS_ansatz)=add_group(tt,[],coe,0);
*(tt :: Vector{TensorMap}, coe:: Number)=add_group(tt,[],coe,0);
*(tt :: iPEPS_ansatz, coe:: Number)=add_group(tt,[],coe,0);
/(tt :: Vector{TensorMap}, coe::Number)=add_group(tt,[],1/coe,0);
/(tt:: iPEPS_ansatz, coe::Number)=add_group(tt,[],1/coe,0);
+(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=add_group(tt1,tt2,1,+1);
+(tt1 :: iPEPS_ansatz, tt2 :: iPEPS_ansatz)=add_group(tt1,tt2,1,+1);
-(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=add_group(tt1,tt2,1,-1);
-(tt1 :: iPEPS_ansatz, tt2 :: iPEPS_ansatz)=add_group(tt1,tt2,1,-1);
norm(tt :: Vector{TensorMap})=norm_tensor_group(tt);
norm(tt :: iPEPS_ansatz)=norm_tensor_group(tt);
dot(tt1 :: Vector{TensorMap}, tt2 :: Vector{TensorMap})=dot_tensor_group(tt1,tt2);
dot(tt1 :: iPEPS_ansatz, tt2 :: iPEPS_ansatz)=dot_tensor_group(tt1,tt2);


#function gdoptimize(f, g!, fg!, x0::Vector{TensorMap}, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
function gdoptimize(f, g!, fg!, x0::iPEPS_ansatz, linesearch, maxiter::Int = 500, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
    global chi,D,Dx,Dy
    if @isdefined(D)
        println("D="*string(D));flush(stdout);
    elseif @isdefined(Dx)&@isdefined(Dy)
        println("Dx,Dy="*string([Dx,Dy]));flush(stdout);
    end
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
        x=normalize_tensor_group(x);

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

function f(x::Kagome_iPESS)
    global CTM_tem,LS_ctm_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    E,E_up, E_down,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
    println("E= "*string(E)*", "*"E_up= "*string(real(E_up))*", "*"E_down= "*string(real(E_down))*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    global E_history
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        jldsave(filenm; B_a=x.B1,B_b=x.B2,B_c=x.B3,T_u=x.Tup,T_d=x.Tdn);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end

function f(x::Square_iPEPS)
    global CTM_tem,LS_ctm_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    if isa(energy_setting,Square_Energy_settings)
        E, E_T1, E_T2, E_T3, E_T4, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_triangle= "*string(real.([E_T1, E_T2, E_T3, E_T4]))*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(energy_setting,Square_2site_Energy_settings)
        E, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
        println("E= "*string(E)*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(energy_setting,Square_Hubbard_Energy_settings)
        if energy_setting.model=="spinless_Hubbard"
            E, ex,ey,e0, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex,ey,e0= "*string([ex,ey,e0])*", ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model=="spinless_Hubbard_pairing"
            E, ex,ey,px,py, e0, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex,ey,px,py,e0= "*string([ex,ey,px,py,e0])*", ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model=="spinless_t1_t2"
            E, ex,ey,e_diagonal2,e_diagonal1, e0, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex,ey,e_diagonal2,e_diagonal1,e0= "*string([ex,ey,e_diagonal2,e_diagonal1,e0])*", ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model=="spinful_triangle_lattice_2site"
            E, ex1,ex2,ey1,ey2,e_diagonal21,e_diagonal22, e01,e02,eU1,eU2, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex1,ex2,ey1,ey2,e_diagonal21,e_diagonal22,e01,e02,eU1,eU2= "*string([ex1,ex2,ey1,ey2,e_diagonal21,e_diagonal22,e01,e02,eU1,eU2])*", ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        end
    end
    global E_history,save_filenm
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        #save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

        jldsave(save_filenm; A=x.T);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end

function g!(gvec::Vector{TensorMap}, x)# this function changes the value of gvec
    println("compute grad")
    global E_tem, CTM_tem
    E_tem,∂E,CTM_tem=get_grad(x);
    #gvec=∂E;#this will not change the input variable
    for cc=1:length(gvec)
        setindex!(gvec,∂E[cc],cc);#this will change the input variable
    end
    # println("norm of grad: "*string(norm(gvec)))
    return gvec
end
function g!(gvec::iPEPS_ansatz, x)# this function changes the value of gvec
    println("compute grad")
    global E_tem, CTM_tem
    E_tem,∂E,CTM_tem=get_grad(x);
    #gvec=∂E;#this will not change the input variable
    Fields=fieldnames(typeof(gvec));
    for i in Fields
        Value=getfield(∂E, i)
        setfield!(gvec,i,Value)
    end
    # println("norm of grad: "*string(norm(gvec)))
    return gvec
end
function fg!(gvec, x)
    #println("one fg!")
    g!(gvec, x)
    f(x)
end






function define_tensor_group(b1,b2,b3,tup,tdn)
    state=Vector{TensorMap}(undef,5);
    state[1]=b1;
    state[2]=b2;
    state[3]=b3;
    state[4]=tup;
    state[5]=tdn;
    return state
end


function NamedTuple_to_Struc(∂E,x)
    ∂E_new=deepcopy(x);
    Keys=keys(∂E);
    for cc in Keys
        setfield!(∂E_new,cc,getindex(∂E,cc))
    end
    return ∂E_new
end
function get_grad(x)
    #∂E = cost_fun'(x);
    # if isa(x0,Kagome_iPESS)
    #     x=Kagome_iPESS_convert(x0);#convert to immutable ansatz
    # elseif isa(x0,Checkerboard_iPESS)
    #     x=Checkerboard_iPESS_convert(x0);#convert to immutable ansatz
    # elseif isa(x0[1],Square_iPEPS)
    #     x=Square_iPEPS_convert(x0);#convert to immutable ansatz 
    # end


    ∂E=gradient(x ->cost_fun(x), x)[1];#this works when x is a mutable structure. The output is a NamedTuple, not a structure, due to that the cost function takes out some fields of the input structure.
    ∂E=NamedTuple_to_Struc(∂E,x);
    #E=fun(state_vec)
    global E_tem, CTM_tem
    x_tem=x;

    println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E,CTM_tem
end

function norm_tensor_group(x0:: Vector{TensorMap})
    Norm=0;
    for tt in x0
        Norm=Norm+norm(tt)^2;
    end
    Norm=sqrt(Norm);
    return Norm
end

function norm_tensor_group(x0:: iPEPS_ansatz)
    Norm=0;
    Fields=fieldnames(typeof(x0));
    for i in Fields
        Value=getfield(x0, i)
        Norm=Norm+norm(Value)^2;
    end
    Norm=sqrt(Norm);
    return Norm
end

function normalize_tensor_group(x0:: Vector{TensorMap})
    Norm=norm_tensor_group(x0);
    #x_new=[[tt/Norm for tt in x0]...,]
    x_new=deepcopy(x0);
    for cc in eachindex(x0)
        x_new[cc]=x_new[cc]/Norm;
    end
    return x_new
end

function normalize_tensor_group(x0:: iPEPS_ansatz)
    x_new=deepcopy(x0);
    Norm=norm_tensor_group(x0);
    Fields=fieldnames(typeof(x_new));
    for i in Fields
        Value=getfield(x_new, i)
        setfield!(x_new,i,Value/Norm)
    end
    return x_new
end

function add_group(Tp1:: Vector{TensorMap}, Tp2, coe1, coe2)
    # if Tp2==[]
    #     x_new=[[tt*coe1 for tt in Tp1]...,];
    # else
    #     x_new=[[Tp1[cc]*coe1+Tp2[cc]*coe2 for cc in eachindex(Tp1)]...,];
    # end 
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
function add_group(Tp1:: iPEPS_ansatz, Tp2, coe1, coe2)
    x_new=deepcopy(Tp1);
    if Tp2==[]
        Fields=fieldnames(typeof(Tp1));
        for i in Fields
            setfield!(x_new,i,getfield(Tp1, i)*coe1)
        end

    else
        Fields=fieldnames(typeof(Tp1));
        for i in Fields
            setfield!(x_new,i,getfield(Tp1, i)*coe1+getfield(Tp2, i)*coe2)
        end
    end
    return x_new
end
function dot_tensor_group(Tp1::Vector{TensorMap},Tp2::Vector{TensorMap})
    y=0;
    for cc in eachindex(Tp1)
        y=y+dot(Tp1[cc],Tp2[cc])
    end
    if imag(y)/real(y)<1e-10
        y=real(y);
    end
    return y
end
function dot_tensor_group(Tp1::iPEPS_ansatz, Tp2::iPEPS_ansatz)
    y=0;
    Fields=fieldnames(typeof(Tp1));
    for i in Fields
        y=y+dot(getfield(Tp1, i), getfield(Tp2, i));
    end
    if imag(y)/real(y)<1e-10
        y=real(y);
    end
    return y
end

function FD(state_vec::Matrix)

    dt=0.000001

    E0=cost_fun(state_vec);

    grad=similar(state_vec);
    for ct in eachindex(state_vec)
        grad[ct]=similar(state_vec[ct])*0;
    end

    for ct in eachindex(state_vec)
        for n_block in eachindex(state_vec[ct].data.values)
            for elem in eachindex(state_vec[ct].data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem[ct].data.values[n_block];
                T[elem]=T[elem]+dt;
                state_vec_tem[ct].data.values[n_block]=T;
                real_part=(cost_fun(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem[ct].data.values[n_block];
                T[elem]=T[elem]+dt*im;
                state_vec_tem[ct].data.values[n_block]=T;
                imag_part=(cost_fun(state_vec_tem)-E0)/dt;

                grad[ct].data.values[n_block][elem]=real_part+im*imag_part;
            end
        end
    end
    return E0, grad
end

function FD(state_vec::iPEPS_ansatz) 

    dt=0.000001

    E0=cost_fun(state_vec);

    grad=similar(state_vec)*0;

    Fields=fieldnames(typeof(state_vec));

    for ct in Fields
        Tensor=getfield(state_vec, ct);
        for n_block in eachindex(Tensor.data.values)
            for elem in eachindex(Tensor.data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                Tensor_tem=getfield(state_vec_tem, ct);
                T=Tensor_tem.data.values[n_block];
                T[elem]=T[elem]+dt;
                Tensor_tem.data.values[n_block]=T;
                setfield!(state_vec_tem,ct,Tensor_tem)
                real_part=(cost_fun(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                Tensor_tem=getfield(state_vec_tem, ct);
                T=Tensor_tem.data.values[n_block];
                T[elem]=T[elem]+dt*im;
                Tensor_tem.data.values[n_block]=T;
                setfield!(state_vec_tem,ct,Tensor_tem)
                imag_part=(cost_fun(state_vec_tem)-E0)/dt;

                grad_Tenosr=getfield(grad, ct);
                grad_Tenosr.data.values[n_block][elem]=real_part+im*imag_part;
                setfield!(grad,ct,grad_Tenosr);
            end
        end
    end
    return E0, grad
end
