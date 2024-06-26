using LinearAlgebra: norm, dot
import LinearAlgebra: norm,dot,isless
import Base: <
import Base:min


import Base: +, -, *,/
#Define operations of groups of TensorMap
*(coe::Number,tt :: Matrix{TensorMap}) =add_group(tt,[],coe,0) ;
*(tt :: Matrix{TensorMap}, coe:: Number) =add_group(tt,[],coe,0);
/(tt :: Matrix{TensorMap}, coe::Number) =add_group(tt,[],1/coe,0);
+(tt1 :: Matrix{TensorMap}, tt2 :: Matrix{TensorMap}) =add_group(tt1,tt2,1,+1);
-(tt1 :: Matrix{TensorMap}, tt2 :: Matrix{TensorMap}) =add_group(tt1,tt2,1,-1);
norm(tt :: Matrix{TensorMap}) =norm_tensor_group(tt);
dot(tt1 :: Matrix{TensorMap}, tt2 :: Matrix{TensorMap}) =dot_tensor_group(tt1,tt2);




function norm_tensor_group(x0:: Matrix{TensorMap}) 
    global Lx,Ly
    Norm=0;
    for cc in eachindex(x0)
        Norm=Norm+norm(x0[cc])^2;
    end
    Norm=sqrt(Norm);
    return Norm
end
function norm_tensor_group(x0::Triangle_iPESS) 
    Norm=sqrt(norm(x0.Bm)^2+norm(x0.Tm)^2)
    return Norm
end
function normalize_tensor_group(x0:: Triangle_iPESS) 
    x0=deepcopy(x0);
    x0.Bm=x0.Bm/norm(x0.Bm);
    x0.Tm=x0.Tm/norm(x0.Tm);
    return x0
end

function normalize_tensor_group(x0:: Matrix{Triangle_iPESS}) 
    x0=deepcopy(x0);
    Lx,Ly=size(x0);
    for cx=1:Lx
        for cy=1:Ly
            x0[cx,cy]=normalize_tensor_group(x0[cx,cy]:: Triangle_iPESS)
        end
    end
    return x0
end



function normalize_tensor_group(x0:: Matrix{TensorMap}) 
    x_new=deepcopy(x0);
    for cc in eachindex(x0)
        setindex!(x_new,x0[cc]/norm(x0[cc]), cc)
    end
    return x_new
end

function add_group(Tp1:: Matrix{TensorMap}, Tp2, coe1, coe2) 
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

function dot_tensor_group(Tp1::Matrix{TensorMap},Tp2::Matrix{TensorMap}) 
    y=0;
    for cc in eachindex(Tp1)
        y=y+dot(Tp1[cc],Tp2[cc])
    end
    if imag(y)/real(y)<1e-10
        y=real(y);
    end
    return y
end



function min(a::ComplexF64,b::Float64)
    println(a)
    if imag(a)/real(a)<1e-14
        a=real(a);
    end
    if a<b
        return a
    else
        return b
    end
end

function isless(a::ComplexF64,b::Float64)
    if imag(a)/real(a)<1e-14
        a=real(a);
    end
    return a<b
end
<(a::ComplexF64,b::Float64)=isless(a::ComplexF64,b::Float64);

#function gdoptimize(f, g!, fg!, x0::Vector{TensorMap}, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
function gdoptimize(f, g!, fg!, x0::TensorMap, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
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
        x=x/norm(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = real(dot(s, gvec))
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/3, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end


function gdoptimize(f, g!, fg!, x0::Matrix{TensorMap}, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
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
        x=normalize_tensor_group(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = real(dot(s, gvec))
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/3, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end


function gdoptimize(f, g!, fg!, x0::Triangle_iPESS, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
    global chi,D
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm_tensor_group(gvec)
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
        x=x/norm_tensor_group(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = real(dot(s, gvec))
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/3, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x)
        gnorm = norm_tensor_group(gvec)
    end

    return (fx, x, iter)
end


function f(x::TensorMap)
    global n_mps_sweep
    n_mps_sweep=5;#for line search

    E=cost_fun_local(x); 

    println("E= "*string(E));flush(stdout);

    global E_history,save_opt_filenm
    global psi,px,py
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        #save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

        jldsave(save_opt_filenm; psi,px,py,x);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end


function g!(gvec::TensorMap, x)# this function changes the value of gvec
    println("compute grad")
    global n_mps_sweep
    n_mps_sweep=0;
    
    E_tem,∂E=get_grad(x);
    #gvec=∂E;#this will not change the input variable

    for (k,block) in blocks(gvec)
        copyto!(block,blocks(∂E)[k]);
    end

    println("norm of grad: "*string(norm(gvec)))
    return gvec
end


function get_grad(x::TensorMap)
    global n_mps_sweep
    n_mps_sweep=0;

    ∂E=gradient(x ->cost_fun_local(x), x)[1];

    #E=fun(state_vec)
    global E_tem
    # x_tem=x;

    # println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E
end


function f(x::Matrix{TensorMap})
    global n_mps_sweep
    n_mps_sweep=5;#for line search

    E=cost_fun_global(x); 

    println("E= "*string(E));flush(stdout);

    global E_history,save_opt_filenm
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        #save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

        x_save=Matrix{Any}(undef,size(x,1),size(x,2));#if sector is trivial, data can only be loaded in the format of Matrix{Any}
        x_save[:,:]=x[:,:];
        jldsave(save_opt_filenm; psi=x_save);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end


function g!(gvec::Matrix{TensorMap}, x)# this function changes the value of gvec
    println("compute grad")
    global n_mps_sweep
    n_mps_sweep=0;
    
    E_tem,∂E=get_grad(x);
    #gvec=∂E;#this will not change the input variable

    for cc in eachindex(gvec)
        setindex!(gvec,∂E[cc],cc);#this will change the input variable
    end

    println("norm of grad: "*string(norm(gvec)))
    return gvec
end



function get_grad(x::Matrix{TensorMap})
    global n_mps_sweep
    
    n_mps_sweep=0;

    ∂E0=gradient(x ->cost_fun_global(x), x)[1];
    Lx,Ly=size(∂E0);
    ∂E=Matrix{TensorMap}(undef,Lx,Ly);#convert to Matrix{TensorMap}
    for cc in eachindex(∂E0)
        ∂E[cc]=∂E0[cc];
    end

    #E=fun(state_vec)
    global E_tem
    # x_tem=x;

    # println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E
end


####################

function f(x::Triangle_iPESS)
    global n_mps_sweep
    n_mps_sweep=5;#for line search

    E=cost_fun_local(x::Triangle_iPESS); 

    println("E= "*string(E));flush(stdout);

    global E_history,save_opt_filenm
    global psi,px,py
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        #save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

        jldsave(save_filenm; psi,px,py,x);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end


function g!(gvec::Triangle_iPESS, x)# this function changes the value of gvec
    println("compute grad")
    global n_mps_sweep
    n_mps_sweep=0;
    
    E_tem,∂E=get_grad(x);
    #gvec=∂E;#this will not change the input variable

    # for (k,block) in blocks(gvec)
    #     copyto!(block,blocks(∂E)[k]);
    # end

    setfield!(gvec,:Bm,∂E.Bm)
    setfield!(gvec,:Tm,∂E.Tm)

    println("norm of grad: "*string(norm_tensor_group(gvec)))
    return gvec
end


function get_grad(x::Triangle_iPESS)
    global n_mps_sweep
    n_mps_sweep=0;

    ∂E=gradient(x ->cost_fun_local(x), x)[1];
    ∂E=Triangle_iPESS(∂E.Bm,∂E.Tm);

    #E=fun(state_vec)
    global E_tem
    # x_tem=x;

    # println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm_tensor_group(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm_tensor_group(elem))
        end
    end
    
    return E_tem,∂E
end
####################



function fg!(gvec, x)
    #println("one fg!")
    g!(gvec, x)
    f(x)
end









function FinteDiff(state_vec::TensorMap)

    dt=0.000001

    E0=cost_fun_local(state_vec);

    grad=similar(state_vec)*0;



        for n_block in eachindex(state_vec.data.values)
            for elem in eachindex(state_vec.data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt;
                state_vec_tem.data.values[n_block]=T;
                real_part=(cost_fun_local(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt*im;
                state_vec_tem.data.values[n_block]=T;
                imag_part=(cost_fun_local(state_vec_tem)-E0)/dt;

                grad.data.values[n_block][elem]=real_part+im*imag_part;
            end
        end
    return E0, grad
end




function FinteDiff_test(state_vec::TensorMap)

    dt=0.000001

    E0=cost_fun_local_test(state_vec);

    grad=similar(state_vec)*0;



        for n_block in eachindex(state_vec.data.values)
            for elem in eachindex(state_vec.data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt;
                state_vec_tem.data.values[n_block]=T;
                real_part=(cost_fun_local_test(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt*im;
                state_vec_tem.data.values[n_block]=T;
                imag_part=(cost_fun_local_test(state_vec_tem)-E0)/dt;

                grad.data.values[n_block][elem]=real_part+im*imag_part;
            end
        end
    return E0, grad
end
function get_grad_double_layer(x,px,py,psi_double_open,U_s_s,funtype)
    psi_double=contract_physical_all(psi_double_open, U_s_s);
    ∂E=gradient(x ->cost_fun_double_layer(x,px,py,psi_double_open,psi_double,U_s_s,funtype), x)[1];
    #E=fun(state_vec)
    global E_tem
    # x_tem=x;

    # println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E
end

