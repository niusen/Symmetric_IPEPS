@with_kw struct TEBDOpts #<: Algorithm
	trscheme::TruncationScheme = TensorKit.truncdim(2)
	tol::Float64 = 1e-12
	maxiter::Integer = 100
	miniter::Integer = 4
	verbose::Integer = 1
	fixedspace::Bool = false
    nstep_iter::Integer = 100
end

function show_tensor(T)
    @show codomain(T) ← domain(T)
end

function PEPS_SimpleUpdate_V!(ir, ic, Ts, Ls, G, alg)
    ixp = ir + 1

    Ta = Ts[ir,ic]
    Tb = Ts[ixp,ic]
    for ith = [1,2,4]
        Ta = MergeLmda(Ta, ith, Ls[ith,ir,ic])
    end
    for ith = [2,3,4]
        Tb = MergeLmda(Tb, ith, Ls[ith,ixp,ic])
    end
    
    Ta = permute(Ta, (1,2,4,3), (5,))
    Tb = permute(Tb, (2,1,3,4), (5,))
    Ta, Tb, S, Err = TaTb_SimpleUpdate(Ta, Tb, G, alg)
    Ta = permute(Ta, (1,2,4,3), (5,))
    Tb = permute(Tb, (2,1,3,4), (5,))
    
    for ith = [1,2,4]
        Ta = MergeLmda(Ta, ith, sdiag_inv(Ls[ith,ir,ic]));
    end
    for ith = [2,3,4]
        Tb = MergeLmda(Tb, ith, sdiag_inv(Ls[ith,ixp,ic]));
    end
    
    normalize!(S) 
    normalize!(Ta)
    normalize!(Tb)

    Ls[3,ir,ic] = S
    Ls[1,ixp,ic] = permute(S, (2,), (1,))
    Ts[ir,ic] = Ta
    Ts[ixp,ic] = Tb
    return Err
end

function PEPS_SimpleUpdate_H!(ir, ic, Ts, Ls, G, alg)
    iyp = ic + 1

    Ta = Ts[ir,ic]
    Tb = Ts[ir,ic+1]
    for ith = 1:3
        Ta = MergeLmda(Ta, ith, Ls[ith,ir,ic])
    end
    for ith = [1,3,4]
        Tb = MergeLmda(Tb, ith, Ls[ith,ir,iyp])
    end
    
    Ta, Tb, S, Err = TaTb_SimpleUpdate(Ta, Tb, G, alg)
    
    for ith = 1:3
        Ta = MergeLmda(Ta, ith, sdiag_inv(Ls[ith,ir,ic]));
    end
    for ith = [1,3,4]
        Tb = MergeLmda(Tb, ith, sdiag_inv(Ls[ith,ir,iyp]));
    end
    
    normalize!(S) 
    normalize!(Ta)
    normalize!(Tb)

    Ls[4,ir,ic] = S
    Ls[2,ir,iyp] = permute(S, (2,), (1,))
    Ts[ir,ic] = Ta
    Ts[ir,iyp] = Tb
    return Err
end

function PEPS_SimpleUpdate_V1!(ir, ic, Ts, Ls, alg)
    ixp = ir + 1

    Ta = Ts[ir,ic]
    Tb = Ts[ixp,ic]
    for ith = [1,2,4]
        Ta = MergeLmda(Ta, ith, Ls[ith,ir,ic])
    end
    for ith = [2,3,4]
        Tb = MergeLmda(Tb, ith, Ls[ith,ixp,ic])
    end
    
    Ta = permute(Ta, (1,2,4,3), (5,))
    Tb = permute(Tb, (2,1,3,4), (5,))
    Ta, Tb, S, Err = truncate_ab(Ta, Tb, alg)
    Ta = permute(Ta, (1,2,4,3), (5,))
    Tb = permute(Tb, (2,1,3,4), (5,))
    
    for ith = [1,2,4]
        Ta = MergeLmda(Ta, ith, sdiag_inv(Ls[ith,ir,ic]));
    end
    for ith = [2,3,4]
        Tb = MergeLmda(Tb, ith, sdiag_inv(Ls[ith,ixp,ic]));
    end
    
    normalize!(S) 
    normalize!(Ta)
    normalize!(Tb)

    Ls[3,ir,ic] = S
    Ls[1,ixp,ic] = permute(S, (2,), (1,))
    Ts[ir,ic] = Ta
    Ts[ixp,ic] = Tb
    return Err
end

function PEPS_SimpleUpdate_H1!(ir, ic, Ts, Ls, alg)
    iyp = ic + 1

    Ta = Ts[ir,ic]
    Tb = Ts[ir,ic+1]
    for ith = 1:3
        Ta = MergeLmda(Ta, ith, Ls[ith,ir,ic])
    end
    for ith = [1,3,4]
        Tb = MergeLmda(Tb, ith, Ls[ith,ir,iyp])
    end
    
    Ta, Tb, S, Err = truncate_ab(Ta, Tb, alg)
    
    for ith = 1:3
        Ta = MergeLmda(Ta, ith, sdiag_inv(Ls[ith,ir,ic]));
    end
    for ith = [1,3,4]
        Tb = MergeLmda(Tb, ith, sdiag_inv(Ls[ith,ir,iyp]));
    end
    
    normalize!(S) 
    normalize!(Ta)
    normalize!(Tb)

    Ls[4,ir,ic] = S
    Ls[2,ir,iyp] = permute(S, (2,), (1,))
    Ts[ir,ic] = Ta
    Ts[ir,iyp] = Tb
    return Err
end
function gate_on_A(G, T)
    @tensor T1[-1,-11,-2,-3,-4,-14;-5] := G[-11,-14;-5,5]*T[-1,-2,-3,-4;5]
    u1 = isometry(fuse(space(T1,1)*space(T1,2)), space(T1,1)*space(T1,2))
    @tensor T1[-1,-2,-3,-4,-14;-5] := u1[-1,1,11]*T1[1,11,-2,-3,-4,-14;-5]
    u2 = isometry(fuse(space(T1,4)*space(T1,5)), space(T1,4)*space(T1,5))
    @tensor T1[-1,-2,-3,-4;-5] := u2[-4,4,14]*T1[-1,-2,-3,4,14;-5]
    return T1
end
function truncate_ab(T1::AbstractTensorMap, T2::AbstractTensorMap, alg)
    Q1,R1 = leftorth(T1, (1,2,3,5), (4,))
    R2,Q2 = rightorth(T2, (1,), (2,3,4,5))
    R1,S,R2,Err = tsvd(R1*R2,trunc=alg.trscheme)
    S = sdiag_sqrt(S)
    T1 = permute(Q1*(R1*S), (1,2,3,5), (4,))
    T2 = permute((S*R2)*Q2, (1,2,3,4), (5,))
    return T1,T2,S,Err
end
function PEPS_Loop_SimpleUpdate!(ir, ic, Ts, Ls, Gs, alg)
    iyp = ic + 1
    ixp = ir + 1

    Ta = Ts[ir,ic]
    Tb = Ts[ir,iyp]
    Tc = Ts[ixp,iyp]
    Td = Ts[ixp,ic]
    for ith = [1,2]
        Ta = MergeLmda(Ta, ith, Ls[ith,ir,ic])
    end
    for ith = [1,4]
        Tb = MergeLmda(Tb, ith, Ls[ith,ir,iyp])
    end
    for ith = [3,4]
        Tc = MergeLmda(Tc, ith, Ls[ith,ixp,iyp])
    end
    for ith = [2,3]
        Td = MergeLmda(Td, ith, Ls[ith,ixp,ic])
    end
    
    T4 = Vector{AbstractTensorMap}(undef, 4)
    T4[1] = permute(Ta, (3,2,1,4), (5,))
    T4[2] = permute(Tb, (2,1,4,3), (5,))
    T4[3] = permute(Tc, (1,4,3,2), (5,))
    T4[4] = permute(Td, (4,3,2,1), (5,))

    for i = 1:4
        T4[i] = gate_on_A(Gs[i], Ts[i])
    end
    
    T4[1] = permute(Ts[1], (3,2,1,4), (5,))
    T4[2] = permute(Ts[2], (2,1,4,3), (5,))
    T4[3] = permute(Ts[3], (1,4,3,2), (5,))
    T4[4] = permute(Ts[4], (4,3,2,1), (5,))

    Ls[4,ir,ic] = id(space(T4[1],4)')
    Ls[3,ir,ic] = id(space(T4[1],3)')

    Ls[4,ir,iyp] = id(space(T4[2],2)')
    Ls[3,ir,iyp] = id(space(T4[2],3)')

    Ls[1,ixp,iyp] = id(space(T4[3],1)')
    Ls[2,ixp,iyp] = id(space(T4[3],2)')

    Ls[1,ixp,ic] = id(space(T4[4],1)')
    Ls[4,ixp,ic] = id(space(T4[4],4)')

    for ith = [1,2]
        Ts[1] = MergeLmda(Ts[1], ith, sdiag_inv(Ls[ith,ir,ic]))
    end
    for ith = [1,4]
        Ts[2] = MergeLmda(Ts[2], ith, sdiag_inv(Ls[ith,ir,iyp]))
    end
    for ith = [3,4]
        Ts[3] = MergeLmda(Ts[3], ith, sdiag_inv(Ls[ith,ixp,iyp]))
    end
    for ith = [2,3]
        Ts[4] = MergeLmda(Ts[4], ith, sdiag_inv(Ls[ith,ixp,ic]))
    end

    Ts[ir,ic] = Ts[1] 
    Ts[ir,iyp] = Ts[2]
    Ts[ixp,iyp] = Ts[3]
    Ts[ixp,ic] = Ts[4]

    Err = zeros(4,1)
    Err[1] = PEPS_SimpleUpdate_H1!(ir,ic, Ts, Ls, alg)
    Err[2] = PEPS_SimpleUpdate_V1!(ir,iyp, Ts, Ls, alg)
    Err[3] = PEPS_SimpleUpdate_H1!(ixp,ic, Ts, Ls, alg)
    Err[4] = PEPS_SimpleUpdate_V1!(ir,ic, Ts, Ls, alg)
    Err = sum(Err)
    return Err
end



function TaTb_SimpleUpdate(Ta, Tb, G, alg)
    # Ta and Tb are rank-3 MPS tensor: l,r,p
    # S is a vector
    # G_ket_bra |ij><ji|
    Qa,Ra = leftorth(Ta, (1,2,3), (4,5));
    Rb,Qb = rightorth(Tb, (2,5), (1,3,4));

    @tensor Rab[-1,-2;-3,-4] := Ra[-1,1,5]*Rb[1,6,-4]*G[5,6,-2,-3]
    Ra,S,Rb,Err = tsvd(Rab,trunc=alg.trscheme)
    
    S = sdiag_sqrt(S) 
    @tensor Ta[-1,-2,-3,-4;-5] := Qa[-1,-2,-3,2]*(Ra[2,-5,1]*S[1,-4])
    @tensor Tb[-1,-2,-3,-4;-5] := S[-2,1]*Rb[1,-5,2]*Qb[2,-1,-3,-4]
    
    return Ta,Tb,S,Err
end

function MergeLmda(Tb::TensorMap, i::Integer, L::TensorMap)
    #show_tensor(Tb)
    #show_tensor(L)
    T = ncon([Tb,L], [[(-1:-1:(-i+1))...,i,(-i-1:-1:-5)...],[i,-i]], order=[i], output=[-1,-2,-3,-4,-5])
    return permute(T, (1,2,3,4), (5,))
end

function sdiag_inv(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.pinv(S.data));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.pinv(b));
        end
    end
    toret
end

function sdiag_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(1/2)));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(1/2)));
        end
    end
    toret
end

function fshow(T::TensorMap)
    show(T.codom → T.dom)
    println()
end

# no hole yet
function ham_expectation(H, As, Ls)
    nrow,ncol = size(As);
    Esh = zeros(ComplexF64,nrow,ncol);
    Esv = zeros(ComplexF64,nrow,ncol);
    for i=1:nrow
        ip = i+1
        for j = 1:ncol
            jp = j+1

            if jp<ncol+1
                Ta = As[i,j];
                Tb = As[i,jp];
                for ith = 1:3
                    Ta = MergeLmda(Ta, ith, Ls[ith,i,j]);
                end
                for ith = [1,3,4]
                    Tb = MergeLmda(Tb, ith, Ls[ith,i,jp]);
                end
                rho = @ncon([conj(Ta),Ta,conj(Tb),Tb], [[1,2,3,4,-1],[1,2,3,14,-2],[5,4,6,7,-3],[5,14,6,7,-4]];
                    order=[1,2,3,5,6,7,4,14], output=[-1,-3,-2,-4]);
                nn = @tensor rho[1,2,1,2]
                Esh[i,j] = @tensor rho[3,4,1,2]*H[1,2,3,4]/nn;
            end
            
            if ip<nrow + 1
                Ta = As[i,j];
                Tb = As[ip,j];
                for ith = [1,2,4]
                    Ta = MergeLmda(Ta, ith, Ls[ith,i,j]);
                end
                for ith = [2,3,4]
                    Tb = MergeLmda(Tb, ith, Ls[ith,ip,j]);
                end
                rho = @ncon([conj(Ta),Ta,conj(Tb),Tb], [[1,2,3,4,-1],[1,2,13,4,-2],[3,5,6,7,-3],[13,5,6,7,-4]];
                    order=[1,2,4,5,6,7,3,13], output=[-1,-3,-2,-4]);
                nn = @tensor rho[1,2,1,2]
                Esv[i,j] = @tensor rho[3,4,1,2]*H[1,2,3,4]/nn;
            end
        end
    end
    E = 2.0*sum(Esh[:]+Esv[:])/(2*nrow*ncol-nrow-ncol);
    return E,Esh,Esv
end