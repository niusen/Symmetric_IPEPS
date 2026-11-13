"""Uniform triangle gate in the same index convention as gate_RU_LD_RD."""
function triangle_gates(model,dt;cell=(2,2))
    isfinite(dt) && dt>=0 || throw(ArgumentError("dt must be nonnegative"))
    h=model.h; identity=id(model.V)
    @tensor h12[-1,-2,-3,-4,-5,-6] := h[-1,-2,-4,-5]*identity[-3,-6]
    @tensor h13[-1,-2,-3,-4,-5,-6] := h[-1,-3,-4,-6]*identity[-2,-5]
    @tensor h23[-1,-2,-3,-4,-5,-6] := h[-2,-3,-5,-6]*identity[-1,-4]
    H=model.J*permute(h12+h13+h23,(1,2,3,),(4,5,6,))
    norm(H-H')<=1e-11*max(norm(H),1) || error("Triangle Hamiltonian is not Hermitian")
    # CG recoupling can leave O(eps) antisymmetric noise in a zero-energy block;
    # explicitly symmetrize before the blockwise Hermitian eigensolver.
    H=(H+H')/2
    eu,ev=eigh(H)
    gate=ev*exp(-dt*eu)*ev'
    return fill(gate,cell...)
end

"""Apply the original triangle_update_iPESS; Dmax is a total virtual dimension budget."""
function simple_update!(state;model=six_model(backend=sectortype(state.T[1])==Trivial ? :dense : :su4),
    Dmax=maximum(dim(space(b,k)) for b in state.B for k=1:3),dt=0.001,sweeps=1,trun_tol=1e-12)
    setup_cell(size(state.B)...)
    Dmax>0 && sweeps>=0 || throw(ArgumentError("invalid Dmax/sweeps"))
    gates=triangle_gates(model,dt;cell=size(state.B))
    for step=1:sweeps
        triangle_update_iPESS(Dmax,step,state.B,state.T,state.lambda1,state.lambda2,state.lambda3,gates,trun_tol)
    end
    return state
end
