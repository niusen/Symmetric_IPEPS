import Base: similar
import LinearAlgebra: mul!,axpy!
import TensorKit: zerovector,scale!!,add!!
# using Setfield

mutable struct FinitePEPS_excitation
    Lx::Int
    Ly::Int
    Tensors::Matrix{TensorMap}
end

mutable struct FinitePEPS_excitation_artificial
    L::Int
    Tensors::Vector{T} where T<:Number
end


function initial_excitation(psi_g,spin,type)
    psi_ex=deepcopy(psi_g);
    Lx,Ly=size(psi_g);
    Real_tensor=nothing;
    if isreal(psi[1].data.values[1][1])
        Real_tensor=true;
    else
        Real_tensor=false;
    end
    
    if spin==0
        if type=="ground_state"
            return FinitePEPS_excitation(Lx,Ly,psi_ex);
        elseif type=="random"
            for cx=1:Lx
                for cy=1:Ly
                    if Real_tensor
                        T=TensorMap(randn,codomain(psi[cx,cy]),domain(psi[cx,cy]));
                    else
                        T=TensorMap(randn,codomain(psi[cx,cy]),domain(psi[cx,cy]))+TensorMap(randn,codomain(psi[cx,cy]),domain(psi[cx,cy]))*im;
                    end
                    T=T/norm(T);
                    psi_ex[cx,cy]=T;
                end
            end
            return FinitePEPS_excitation(Lx,Ly,psi_ex);
        end
    end
end


function initial_excitation_artificial(L)
    v=randn(L);
    return FinitePEPS_excitation_artificial(L,v);
end

function zerovector(a::FinitePEPS_excitation)
    a1=deepcopy(a);
    Lx=a1.Lx;
    Ly=a1.Ly;
    Ts=a1.Tensors;
    return FinitePEPS_excitation(Lx,Ly,Ts*0)
end

function zerovector(a::FinitePEPS_excitation_artificial)
    a1=deepcopy(a);
    L=a1.L;
    Ts=a1.Tensors;
    return FinitePEPS_excitation_artificial(L,Ts*0)
end

function dot(a::FinitePEPS_excitation,b::FinitePEPS_excitation)
    Lx=a.Lx;
    Ly=a.Ly;
    a=a.Tensors;
    b=b.Tensors;
    Lx,Ly=size(a);
    S=0;
    for cx=1:Lx
        for cy=1:Ly
            S=S+dot(a[cx,cy],b[cx,cy]);
        end
    end
    return S
end

function dot(a::FinitePEPS_excitation_artificial,b::FinitePEPS_excitation_artificial)
    aa=a.Tensors;
    bb=b.Tensors;

    return dot(aa,bb)
end


function norm(a::FinitePEPS_excitation)
    return real(sqrt(dot(a,a)))
end

function norm(a::FinitePEPS_excitation_artificial)
    return norm(a.Tensors)
end

function *(a::ComplexF64, b::FinitePEPS_excitation)
    Lx=b.Lx;
    Ly=b.Ly;
    Ts=b.Tensors;
    return FinitePEPS_excitation(Lx,Ly,Ts*a)
end

function *(a::ComplexF64, b::FinitePEPS_excitation_artificial)
    L=b.L;
    Ts=b.Tensors;
    return FinitePEPS_excitation_artificial(L,Ts*a);
end

function +(a::FinitePEPS_excitation, b::FinitePEPS_excitation)
    Lx=b.Lx;
    Ly=b.Ly;
    Tsa=a.Tensors;
    Tsb=b.Tensors;
    return FinitePEPS_excitation(Lx,Ly,Tsa+Tsb)
end

function +(a::FinitePEPS_excitation_artificial, b::FinitePEPS_excitation_artificial)
    L=b.L;
    Tsa=a.Tensors;
    Tsb=b.Tensors;
    return FinitePEPS_excitation_artificial(L,Tsa+Tsb)
end

function -(a::FinitePEPS_excitation, b::FinitePEPS_excitation)
    Lx=b.Lx;
    Ly=b.Ly;
    Tsa=a.Tensors;
    Tsb=b.Tensors;
    return FinitePEPS_excitation(Lx,Ly,Tsa-Tsb)
end

function -(a::FinitePEPS_excitation_artificial, b::FinitePEPS_excitation_artificial)
    L=b.L;
    Tsa=a.Tensors;
    Tsb=b.Tensors;
    return FinitePEPS_excitation_artificial(L,Tsa-Tsb)
end

function similar(a::FinitePEPS_excitation)
    a=deepcopy(a);
    Lx=a.Lx;
    Ly=a.Ly;
    Tsa=a.Tensors;
    if isreal(Tsa[1].data.values[1][1])
        Real_tensor=true;
    else
        Real_tensor=false;
    end
    for cx=1:Lx
        for cy=1:Ly
            tt=Tsa[cx,cy];
            if Real_tensor
                T=TensorMap(randn,codomain(tt),domain(tt));
            else
                T=TensorMap(randn,codomain(tt),domain(tt))+TensorMap(randn,codomain(tt),domain(tt))*im;
            end
            Tsa[cx,cy]=T;
        end
    end
    
    return FinitePEPS_excitation(Lx,Ly,Tsa)
end

function similar(a::FinitePEPS_excitation_artificial)
    L=a.L;
    Tsa=randn(L);

    return FinitePEPS_excitation_artificial(L,Tsa);
end


function mul!(a::FinitePEPS_excitation, b::FinitePEPS_excitation, c::Number)

    Ts=b.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end





function mul!(a::FinitePEPS_excitation_artificial, b::FinitePEPS_excitation_artificial, c::Number)

    Ts=b.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end


function scale!!(a::FinitePEPS_excitation, b::FinitePEPS_excitation, c::Number)

    Ts=b.Tensors;
    setfield!(a,:Tensors,Ts*c);
    return a
end


function scale!!(a::FinitePEPS_excitation_artificial, b::FinitePEPS_excitation_artificial, c::Number)

    Ts=b.Tensors;
    setfield!(a,:Tensors,Ts*c);
    return a
end


function scale!!(a::FinitePEPS_excitation, c::Number)

    Ts=a.Tensors;
    setfield!(a,:Tensors,Ts*c);
    return a
end


function scale!!(a::FinitePEPS_excitation_artificial, c::Number)

    Ts=a.Tensors;
    setfield!(a,:Tensors,Ts*c);
    return a
end

function mul!(a::FinitePEPS_excitation, c::Number, b::FinitePEPS_excitation)

    Ts=b.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function mul!(a::FinitePEPS_excitation_artificial, c::Number, b::FinitePEPS_excitation_artificial)

    Ts=b.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function rmul!(a::FinitePEPS_excitation, c::Number)

    Ts=a.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function rmul!(a::FinitePEPS_excitation_artificial, c::Number)

    Ts=a.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function lmul!(c::Number, a::FinitePEPS_excitation)

    Ts=a.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function lmul!(c::Number, a::FinitePEPS_excitation_artificial)

    Ts=a.Tensors;
    # a = @set a.Tensors = Ts*c
    #a=FinitePEPS_excitation(Lx,Ly,Ts*c);
    setfield!(a,:Tensors,Ts*c);
    return a
end

function axpy!(a::Number, X::FinitePEPS_excitation, Y::FinitePEPS_excitation)
    #Y need to be updated
    Tx=X.Tensors;
    Ty=Y.Tensors;
    setfield!(Y,:Tensors,a*Tx+Ty);
    return Y
end

function add!!(Y::FinitePEPS_excitation, X::FinitePEPS_excitation,  a::Number)
    #Y need to be updated
    Tx=X.Tensors;
    Ty=Y.Tensors;
    setfield!(Y,:Tensors,a*Tx+Ty);
    return Y
end

function axpy!(a::Number, X::FinitePEPS_excitation_artificial, Y::FinitePEPS_excitation_artificial)
    #Y need to be updated
    Tx=X.Tensors;
    Ty=Y.Tensors;
    setfield!(Y,:Tensors,a*Tx+Ty);
    return Y
end

function add!!(Y::FinitePEPS_excitation_artificial, X::FinitePEPS_excitation_artificial, a::Number, )
    #Y need to be updated
    Tx=X.Tensors;
    Ty=Y.Tensors;
    setfield!(Y,:Tensors,a*Tx+Ty);
    return Y
end

function apply_excitation_H_or_N(psi_g, psi_ex::FinitePEPS_excitation, psi_double_open,psi_double,op_type)
    println("apply Mx");flush(stdout)
    global U_s_s
    Lx,Ly=size(psi_g);
    psi_ex_old=psi_ex.Tensors;
    psi_ex_new=psi_ex_old.*0;#initialize
    N_operator_sites=1;

    for cx1=1:Lx
        for cy1=1:Ly
            pos_bra=[cx1,cy1];
            A_bra=psi_ex_old[pos_bra[1],pos_bra[2]];

            for cx2=1:Lx
                for cy2=1:Ly
                    pos_ket=[cx2,cy2];
                    A_ket=psi_ex_old[pos_ket[1],pos_ket[2]];

                    if op_type=="energy"
                        ∂E_real=gradient(x ->cost_fun_bra_ket(N_operator_sites,x,A_ket,pos_bra,pos_ket,deepcopy(psi_g),deepcopy(psi_g),deepcopy(psi_double_open),deepcopy(psi_double),U_s_s,op_type,"real"), A_bra)[1];
                        ∂E_imag=gradient(x ->cost_fun_bra_ket(N_operator_sites,x,A_ket,pos_bra,pos_ket,deepcopy(psi_g),deepcopy(psi_g),deepcopy(psi_double_open),deepcopy(psi_double),U_s_s,op_type,"imag"), A_bra)[1];
                        psi_ex_new[cx1,cy1]=psi_ex_new[cx1,cy1]+∂E_real+∂E_imag*im;
                    elseif op_type=="norm"
                        ∂N_real=gradient(x ->cost_fun_bra_ket(N_operator_sites,x,A_ket,pos_bra,pos_ket,deepcopy(psi_g),deepcopy(psi_g),deepcopy(psi_double_open),deepcopy(psi_double),U_s_s,op_type,"real"), A_bra)[1];
                        ∂N_imag=gradient(x ->cost_fun_bra_ket(N_operator_sites,x,A_ket,pos_bra,pos_ket,deepcopy(psi_g),deepcopy(psi_g),deepcopy(psi_double_open),deepcopy(psi_double),U_s_s,op_type,"imag"), A_bra)[1];
                        psi_ex_new[cx1,cy1]=psi_ex_new[cx1,cy1]+∂N_real+∂N_imag*im;
                    else
                        error("unknown type");
                    end
                end
            end
        end
    end
    return FinitePEPS_excitation(Lx,Ly,psi_ex_new)
end


function apply_excitation_H_or_N(M, psi_ex::FinitePEPS_excitation_artificial)
    println("apply Mx");flush(stdout)
    v=psi_ex.Tensors
    return FinitePEPS_excitation_artificial(L,M*v)
end
