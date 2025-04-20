
import LinearAlgebra: norm,dot
norm(tt :: Matrix{TensorMap})=psi_norm(tt);


function to_Matrix_TensorMap(xx0::Matrix)
    xx=Matrix{TensorMap}(undef,size(xx0));
    xx[:]=xx0[:];
    return xx
end
function psi_norm(x::Matrix{TensorMap})
    Norm=0;
    for cc in x
        Norm=Norm+norm(cc)^2
    end
    return sqrt(Norm)
end

function stochastic_opt(x0::Matrix{TensorMap}, ls) 

    println("stochastic optimization");
    
    global save_filenm
    global contract_fun, Vp 
    ntask=nworkers();
    x = deepcopy(x0);
    x_min=deepcopy(x);
    E_min=10000;
    E_set=Vector{global_eltype}(undef,0);
    delta=ls.delta0;

    

    gvec = similar(x);
    gnorm=10000;
    iter = 0
    while iter < ls.maxiter && gnorm > ls.gtol
        println("optim iteration "*string(iter));flush(stdout);

        config_max=normalize_PEPS!(x,Vp,contract_whole_disk);#normalize psi such that the amplitude of a single config is close to 1
        #x=normalize_ansatz(x);


        #@time main(dir, 1, ntask)
        @sync begin
            for cp=1:ntask
                worker_id=workers()[cp]
                @spawnat worker_id compute_grad(Vp,x,config_max, dir, worker_id, ntask);
            end
        end

        Eterms_set, grads_set, E_grads_set=read_data(ntask);
        E_mean, Eterms_set, grad_mean, E_grad_mean, gvec=grad_analysis(Eterms_set, grads_set, E_grads_set);
        gnorm = norm(gvec);
        @show E_mean;flush(stdout);
        println("norm of grad:"*string(norm(gvec)))

        gvec=get_grad_conjugate(gvec);
        if ls.fix_delta
            if real(E_mean)<E_min
                E_min=real(E_mean);
                global starting_time
                Now=now();
                Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
                println("Time consumed: "*string(Time));flush(stdout);
    
            end
            



            
            x_min=x;
            push!(E_set,real(E_mean));
            jldsave(save_filenm; E_set, psi=x);


            x_norm=norm(x);
            x_updated=x-x_norm*get_random_grad(gvec,delta);#get random grad
            println("norm of random grad:"*string(norm(x_updated-x)));flush(stdout);
    
            x=to_Matrix_TensorMap(x_updated);
            iter += 1
        

        else
            if real(E_mean)<E_min

                global starting_time
                Now=now();
                Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
                println("Time consumed: "*string(Time));flush(stdout);



                E_min=real(E_mean);
                x_min=x;
                push!(E_set,E_min);
                jldsave(save_filenm; E_set, E_mean, psi=x);


                x_norm=norm(x);
                x_updated=x-x_norm*get_random_grad(gvec,delta);#get random grad
                println("norm of random grad:"*string(norm(x_updated-x)));flush(stdout);
        
                x=to_Matrix_TensorMap(x_updated);
                iter += 1
        
            else
                @show delta=delta*(ls.alpha);
                x=x_min;



                x_norm=norm(x);
                x_updated=x-x_norm*get_random_grad(gvec,delta);#get random grad
                println("norm of random grad:"*string(norm(x_updated-x)));flush(stdout);
        
                x=to_Matrix_TensorMap(x_updated);

            end
        end
        
        
    end
    return x
end


function get_grad_conjugate(grad::Matrix{TensorMap})
    for cc in eachindex(grad)
        T=grad[cc];
        if Rank(T)==3
            T=permute(T',(1,2,3,));
        elseif Rank(T)==4
            T=permute(T',(1,2,3,4,));
        elseif Rank(T)==5
            T=permute(T',(1,2,3,4,5,));
        else
            error("unknown case")
        end
        grad[cc]=T;

    end
    return grad
end




function random_tensor_sign(T::TensorMap)
    T=deepcopy(T);
    function generate_number(a)
        if a>0
            b=rand(1);
        elseif a<0
            b=-rand(1);
        elseif a==0
            b=0;
        end
        return b[1]
    end
    if sectortype(space(T,1)) == Trivial
        mm=T.data;
        for dd in eachindex(mm)
            a=mm[dd];
            if isa(a,Float64)
                a_new=generate_number(a);
            elseif isa(a,ComplexF64)
                a_new=generate_number(real(a))+im*generate_number(imag(a));
            else
                error("unknown number type")
            end
            mm[dd]=a_new;
        end
        T=TensorMap(mm,codomain(T),domain(T));
    else
        for cc=1:length(T.data.values)
            mm=T.data.values[cc];
            for dd in eachindex(mm)
                a=mm[dd];
                if isa(a,Float64)
                    a_new=generate_number(a);
                elseif isa(a,ComplexF64)
                    a_new=generate_number(real(a))+im*generate_number(imag(a));
                else
                    error("unknown number type")
                end
                mm[dd]=a_new;
            end
            T.data.values[cc]=mm;
        end
    end
    return T
end

function get_random_grad(x::Matrix,delta) 
    x_new=deepcopy(x);
    for cc in eachindex(x)
        ansatz=x[cc];
        if isa(x[cc],Kagome_iPESS)
            B1=ansatz.B1;
            B2=ansatz.B2;
            B3=ansatz.B3;
            Tup=ansatz.Tup;
            Tdn=ansatz.Tdn;

            B1=random_tensor_sign(B1)*delta;
            B2=random_tensor_sign(B2)*delta;
            B3=random_tensor_sign(B3)*delta;
            Tup=random_tensor_sign(Tup)*delta;
            Tdn=random_tensor_sign(Tdn)*delta;
            ansatz_new=Kagome_iPESS(B1,B2,B3,Tup,Tdn);
        elseif isa(x[cc],Checkerboard_iPESS)
            BL=ansatz.B_L;
            BU=ansatz.B_U;
            Tm=ansatz.Tm;

            BL=random_tensor_sign(BL)*delta;
            BU=random_tensor_sign(BU)*delta;
            Tm=random_tensor_sign(Tm)*delta;
            ansatz_new=Checkerboard_iPESS(BL,BU,Tm);
        elseif isa(x[cc],Triangle_iPESS)
            iPEss=x[cc];
            bm=iPEss.Bm;
            tm=iPEss.Tm;
            bm=random_tensor_sign(bm)*delta;
            tm=random_tensor_sign(tm)*delta;
            ansatz_new=Triangle_iPESS(bm,tm);
        elseif isa(x[cc],Square_iPEPS)
            A=ansatz.T;
            A=random_tensor_sign(A)*delta;
            ansatz_new=Square_iPEPS(A);
        elseif isa(x[cc],TensorMap)
            A=ansatz;
            A=random_tensor_sign(A)*delta;
            ansatz_new=A;
        else
            error("unknown type")
        end
        x_new[cc]=ansatz_new;
    end
    return x_new
end





