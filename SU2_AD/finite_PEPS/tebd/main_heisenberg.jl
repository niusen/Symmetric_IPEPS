using Printf, Parameters, TensorKit, JLD2, LinearAlgebra
cd(@__DIR__)
include("tebd.jl")

function initial_peps(D::Int, L1::Int, L2::Int)
    As = Array{TensorMap,2}(undef, L1, L2);
    Ls = Array{TensorMap,3}(undef, 4, L1, L2);
    p = ComplexSpace(2);
    v = ComplexSpace(D);
    v0 = ComplexSpace(1)
    ifset(ifx::Bool, a, b) = ifx ? a : b 

    for x = 1:L1
        for y = 1:L2
            v1 = ifset(x==1, v0, v)
            v2 = ifset(y==1, v0, v)
            v3 = ifset(x==L1, v0, v)
            v4 = ifset(y==L2, v0, v)

            As[x,y] = TensorMap(randn, v1*v2*v3'*v4', p)
            for i = 1:4
                Ls[i,x,y] = id(space(As[x,y],i)')
            end
        end
    end
    return As, Ls
end

function xxz_evo_gates(τ::Float64, g::Float64)
    v = ComplexSpace(2)
    X = TensorMap([0im 1.0; 1.0 0], v, v)*0.5
    Y = TensorMap([0.0 -im; im 0], v, v)*0.5
    Z = TensorMap([1.0 0; 0 -1.0], v, v)*0.5

    @tensor XX[-1 -2; -3 -4] := X[-1; -3] * X[-2; -4]
    @tensor YY[-1 -2; -3 -4] := Y[-1; -3] * Y[-2; -4]
    @tensor ZZ[-1 -2; -3 -4] := Z[-1; -3] * Z[-2; -4]
    H = XX + YY + g * ZZ
    #Sl = isometry(fuse(codomain(H)), codomain(H))
    #Sr = isometry(domain(H), fuse(domain(H)))
    #G = Sl'*exp(-τ*Sl*H*Sr)*Sr'
    G = exp(-τ*H)
    return G, H
end


function peps_loop_simple_upate(As, Ls, G, H, alg)
    # simple update

    nrow, ncol = size(As)
    Ls0 = deepcopy(Ls)
    E0 = 10.0
    dE = 1.0

    for iter = 1:alg.maxiter
        #tic();
        Err = 0.0
        for ix = [1:2:nrow-1...,2:2:nrow-1...]
            y0 = 1
            if mod(ix,2) ==0
                y0 = 2;
            end
            for iy = y0:2:ncol-1
                #@show ix,iy
                Err2 = PEPS_SimpleUpdate_H!(ix, iy, As, Ls, G, alg)
                Err = max(Err, Err2)
                #@show Err2
            end
        end

        #@show "start updateV:"
        for ix = [1:2:nrow-1..., 2:2:nrow-1...]
            y0 = 2
            if mod(ix,2) ==0
                y0 = 1;
            end
            for iy = y0:2:ncol-1
                #@show ix,iy
                Err1 = PEPS_SimpleUpdate_V!(ix, iy, As, Ls, G, alg)
                Err = max(Err, Err1)
            end
        end

        if mod(iter, alg.verbose) == 0
            @printf("%4d   %.2e\n", iter, Err)
        end

        if mod(iter, alg.nstep_iter) == 0
            E, Esh, Esv = ham_expectation(H, As, Ls)
            dE = abs((E - E0) / E0)
            @printf("%4d   %.2e   %.10e\n", iter, dE, E)
            E0 = E
        end

        if dE < alg.tol
            break
        end
        Ls0 = deepcopy(Ls)
    end

    return As, Ls
end


function peps_simple_upate(As, Ls, G, H, alg)
    # simple update

    nrow, ncol = size(As)
    Ls0 = deepcopy(Ls)
    E0 = 10.0
    dE = 1.0

    for iter = 1:alg.maxiter
        #tic();
        Err = 0.0
        for ix = 1:nrow
            for iy = [1:2:ncol-1..., 2:2:ncol-1...]
                #@show ix,iy
                Err2 = PEPS_SimpleUpdate_H!(ix, iy, As, Ls, G, alg)
                Err = max(Err, Err2)
                #@show Err2
            end
        end

        #@show "start updateV:"
        for ix = [1:2:nrow-1..., 2:2:nrow-1...]
            for iy = 1:ncol-1
                #@show ix,iy
                Err1 = PEPS_SimpleUpdate_V!(ix, iy, As, Ls, G, alg)
                Err = max(Err, Err1)
            end
        end

        if mod(iter, alg.verbose) == 0
            @printf("%4d   %.2e\n", iter, Err)
        end

        if mod(iter, alg.nstep_iter) == 0
            E, Esh, Esv = ham_expectation(H, As, Ls)
            dE = abs((E - E0) / E0)
            @printf("%4d   %.2e   %.10e\n", iter, dE, E)
            E0 = E
        end

        if dE < alg.tol
            break
        end
        Ls0 = deepcopy(Ls)
    end

    return As, Ls
end


function main_xxz_sq(;D=4)
    opts = TEBDOpts(
        tol = 1e-6,
        maxiter = 1000,
        trscheme = truncerr(1e-14)&truncdim(D),
        verbose = 20,
        nstep_iter = 100
    )

    g = 1.0
    As,Ls = initial_peps(2, 10, 10)

    for τ = [0.2, 0.1, 0.02]
        @show τ
        G,H = xxz_evo_gates(τ, g)
        As, Ls = peps_simple_upate(As, Ls, G, H, opts)
    end
    fname = string("xxz_D",D,".jld2")
    jldsave(fname; As, Ls, opts, g)
end





D=3;

opts = TEBDOpts(
    tol = 1e-6,
    maxiter = 1000,
    trscheme = truncerr(1e-14)&truncdim(D),
    verbose = 20,
    nstep_iter = 100
)

g = 1.0
As,Ls = initial_peps(D, 4, 4)

for τ = [0.2, 0.1, 0.02]
    @show τ
    G,H = xxz_evo_gates(τ, g)
    As, Ls = peps_simple_upate(As, Ls, G, H, opts)
end

psi=Matrix{Any}(undef,size(As,2),size(As,1));
psi[:,:]=As[:,:];

fname = string("SU_"*string(size(As,2))*"x"*string(size(As,1))*"_D",D,".jld2")
jldsave(fname; psi=psi);