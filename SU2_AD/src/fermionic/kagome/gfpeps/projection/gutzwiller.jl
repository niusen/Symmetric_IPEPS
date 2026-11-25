# This file belongs to KagomeGFPEPSValidation. No shared Hubbard code is changed.
export ProjectedSpinIPESS, gutzwiller_projector, gutzwiller_project,
       projected_spin_space, projected_spin_operators, blocked_spin_operators,
       measure_projected_spins, save_projected_checkpoint, load_projected_checkpoint

"""Spin basis (up,down), both even in the source b=(up,down-hole) grading."""
projected_spin_space()=Rep[ℤ₂](0=>2)

"""G maps (b vacuum,b pair,b up,b hole) to physical (up,down).
G†G is the physical-electron single-occupancy projector; GG†=I₂.
It commutes with total Sz but does not select a total-Sz sector. A general
Z2 Gaussian input can mix sectors, whose relative weights change on projection.
"""
function gutzwiller_projector()
    G=zeros(ComplexF64,2,4)
    G[1,2]=1; G[2,1]=1
    TensorMap(G,projected_spin_space(),KF.physical_space())
end

"""Projected spin iPESS: physical B legs have dimension 2; virtual grading stays fermionic.
The separate type prevents accidental use of the electron Hubbard state format.
"""
struct ProjectedSpinIPESS
    T_u::Matrix{TensorMap}
    T_d::Matrix{TensorMap}
    B::Array{TensorMap,3}
    lambda_u::Array{TensorMap,3}
    lambda_d::Array{TensorMap,3}
end

function gutzwiller_project(state::KF.KagomeFermionicState)
    KF.check_state(state)
    G=gutzwiller_projector()
    B=Array{TensorMap,3}(undef,size(state.B))
    for i in eachindex(B)
        B[i]=KF._leg_map(state.B[i],G,3)
        isfinite(norm(B[i])) && norm(B[i])>0 || throw(ArgumentError("Gutzwiller projection annihilates local B tensor $i"))
    end
    projected=ProjectedSpinIPESS(deepcopy(state.T_u),deepcopy(state.T_d),B,
        deepcopy(state.lambda_u),deepcopy(state.lambda_d))
    KF.check_state(projected)
end

function save_projected_checkpoint(path,state::ProjectedSpinIPESS;parameters=Dict(),metadata=Dict())
    KF.check_state(state);mkpath(dirname(abspath(path)))
    jldsave(path;format="kagome_gutzwiller_spin_z2_v1",T_u=state.T_u,T_d=state.T_d,
        B=state.B,lambda_u=state.lambda_u,lambda_d=state.lambda_d,parameters,metadata,
        physical_basis="(up,down); even b-mode parity",virtual_grading="fermionic Z2")
    path
end

KF.save_checkpoint(path,state::ProjectedSpinIPESS;kwargs...)=save_projected_checkpoint(path,state;kwargs...)

function load_projected_checkpoint(path)
    data=load(path)
    get(data,"format",nothing)=="kagome_gutzwiller_spin_z2_v1" || throw(ArgumentError("not a projected-spin checkpoint"))
    state=ProjectedSpinIPESS(data["T_u"],data["T_d"],data["B"],data["lambda_u"],data["lambda_d"])
    KF.check_state(state),data["parameters"],data["metadata"]
end

function KF.check_state(state::ProjectedSpinIPESS;Dmax=nothing)
    # Embed in the original four-state space to reuse all connectivity/weight
    # checks; this does not change the stored projected tensors.
    G=gutzwiller_projector()
    B=Array{TensorMap,3}(undef,size(state.B))
    for i in eachindex(B)
        space(state.B[i],3)==projected_spin_space() || throw(SpaceMismatch("expected projected spin space"))
        B[i]=KF._leg_map(state.B[i],G',3)
    end
    KF.check_state(KF.KagomeFermionicState(state.T_u,state.T_d,B,state.lambda_u,state.lambda_d);Dmax)
    state
end

function KF.build_A_kagome(state::ProjectedSpinIPESS,x,y)
    # Same geometric contraction and virtual crossings as the fermionic adapter.
    W=KF.fpermute(KF.triangle_wavefunction(state,x,y,:up;weighted=false),(1,2,4,5,6,3))
    T=KF.fpermute(state.T_d[x,y],(2,1,3))
    M=TensorKit.permute(W,((1,2,3,4,5),(6,)))*TensorKit.permute(T,((1,),(2,3)))
    A=KF.fpermute(M,(1,7,6,4,2,3,5))
    A=KF._crossing(KF._crossing(KF._crossing(A,1,2),1,3),2,3)
    P=projected_spin_space(); U=unitary(fuse(P*P*P),P*P*P)
    @tensor fused[-1,-2,-3,-4,-5] := A[-1,-2,-3,-4,1,2,3]*U[-5,1,2,3]
    A,KF._allout(fused),U
end

"""Project a blocked cell using its actual fusion map. Returns (A_spin,U_spin).
Projection reduces the physical dimension 64→8 and leaves all virtual spaces
unchanged. Input tensors are not mutated. A zero projected tensor is an error.
"""
function gutzwiller_project(A::Tuple,U;normalize::Bool=false)
    P=KF.physical_space(); domain(U)==P*P*P || throw(SpaceMismatch("expected three unprojected b-mode physical legs"))
    S=projected_spin_space(); Us=unitary(fuse(S*S*S),S*S*S)
    G=gutzwiller_projector(); K=Us*(G⊗G⊗G)*U'
    result=map(A) do column
        map(column) do a
            space(a,5)==space(U,1) || throw(SpaceMismatch("physical fusion map does not match tensor"))
            b=KF._leg_map(a,K,5)
            isfinite(norm(b)) && norm(b)>0 || throw(ArgumentError("Gutzwiller projection annihilates a local tensor"))
            normalize ? b/norm(b) : b
        end
    end
    result,Us
end

function measure_gaussian(state::ProjectedSpinIPESS,fields;kwargs...)
    throw(ArgumentError("a Gutzwiller-projected state is not Gaussian; use spin-model observables instead of the saved free-fermion energy reference"))
end

"""Physical spin-1/2 operators in (up,down), not the b-fermion occupations."""
function projected_spin_operators()
    S=projected_spin_space(); make(M)=TensorMap(ComplexF64.(M),S,S)
    Id=make([1 0;0 1]); Sp=make([0 1;0 0]); Sm=Sp'
    (;Id,Sp,Sm,Sx=(Sp+Sm)/2,Sy=(Sp-Sm)/(2im),Sz=make([0.5 0;0 -0.5]))
end

"""Three-site spin operators, with the same (a,b,c) ordering/fusion as A."""
function blocked_spin_operators(U)
    S=projected_spin_space(); domain(U)==S*S*S || throw(SpaceMismatch("expected three projected spin legs"))
    ops=projected_spin_operators()
    embed(O,i)=U*( (i==1 ? O : ops.Id) ⊗ (i==2 ? O : ops.Id) ⊗ (i==3 ? O : ops.Id) )*U'
    (;Id=U*(ops.Id⊗ops.Id⊗ops.Id)*U',
      Sx=ntuple(i->embed(ops.Sx,i),3),Sy=ntuple(i->embed(ops.Sy,i),3),
      Sz=ntuple(i->embed(ops.Sz,i),3),Sp=ntuple(i->embed(ops.Sp,i),3),Sm=ntuple(i->embed(ops.Sm,i),3))
end

"""CTMRG onsite spin measurements; no Hamiltonian or Gaussian reference assumed.
magnetization[x,y,site,component], components=(x,y,z). Fields returned by the
Gaussian importer are source metadata, not automatically spin-model fields.
"""
function measure_projected_spins(A,U;chi=16,maxiter=100,tol=1e-8,projector_tol=1e-10,verbose=true)
    op=blocked_spin_operators(U)
    CTM.Lx,CTM.Ly=length(A),length(A[1]); CTM.chi=chi; CTM.projector_trun_tol=projector_tol
    CTM.algrithm_CTMRG_settings.CTM_cell_ite_method="continuous_update"
    settings=CTM.LS_CTMRG_settings()
    settings.CTM_conv_tol=tol; settings.CTM_ite_nums=maxiter;settings.CTM_trun_tol=projector_tol
    settings.CTM_ite_info=verbose;settings.CTM_conv_info=true;settings.grad_checkpoint=false
    initial=CTM.initial_condition(init_type="PBC",reconstruct_CTM=true,reconstruct_AA=true)
    env,AA,_,_,_,_,iterations,error=CTM.Fermionic_CTMRG_cell(A,chi,initial,nothing,settings)
    magnetization=zeros(ComplexF64,length(A),length(A[1]),3,3)
    normalization=zeros(ComplexF64,length(A),length(A[1]))
    for x in 1:length(A), y in 1:length(A[1])
        normalization[x,y]=CTM.ob_onsite(env,op.Id,A,AA,x-1,y-1,settings)
        for site in 1:3, (k,component) in enumerate((:Sx,:Sy,:Sz))
            magnetization[x,y,site,k]=CTM.ob_onsite(env,getproperty(op,component)[site],A,AA,x-1,y-1,settings)
        end
    end
    (;magnetization,normalization,chi,iterations,ctm_error=error,converged=error<tol,CTM=env)
end
