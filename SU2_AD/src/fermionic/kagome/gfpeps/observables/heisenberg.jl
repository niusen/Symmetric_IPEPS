export measure_heisenberg

function _spin_layer(a,O)
    # Same even physical-leg insertion as the triangular Hubbard ob_onsite.
    @tensor ket[:] := a[-1,-2,-3,-4,1]*O[-5,1]
    CTM.build_double_layer_swap(a',ket)[1]
end

function _spin_patch(env,AA,cx,cy;insertions=())
    Lx,Ly=length(AA),length(AA[1])
    offsets=((1,1),(2,1),(1,2),(2,2)) # LU,RU,LD,RD
    patch=map(offsets) do (dx,dy)
        AA[mod1(cx+dx,Lx)][mod1(cy+dy,Ly)]
    end
    for (slot,tensor) in insertions
        patch=Base.setindex(patch,tensor,slot)
    end
    CTM.ob_2x2(env,patch...,cx,cy)
end

"""Nearest-neighbor spin-1/2 Heisenberg expectation in a fixed projected state.

H=J Σ_<ij> S_i⋅S_j, with a uniform -h Σ_i Sz_i term and no constant shift. Each kagome site
has four neighbors: six bonds per three-site primitive cell. Spin operators
are even: insert them independently, without hopping parity strings/minus signs.
down(x,y)=[a(x+1,y),b(x,y),c(x,y+1)], bond order=(ab,bc,ca).
"""
function measure_heisenberg(A,U;J::Real=1,h::Real=0,chi=16,maxiter=100,tol=1e-8,
        projector_tol=1e-10,verbose=true,CTM0=nothing,environment_path=nothing,update_method="continuous_update")
    all(isfinite,(J,h)) || throw(ArgumentError("J and h must be finite"))
    chi>0 && maxiter>=0 && tol>0 || throw(ArgumentError("invalid CTMRG settings"))
    op=blocked_spin_operators(U)
    Lx,Ly=length(A),length(A[1])
    CTM.Lx,CTM.Ly=Lx,Ly;CTM.chi=chi;CTM.projector_trun_tol=projector_tol
    update_method in ("continuous_update","together_update") || throw(ArgumentError("unknown CTMRG update method"))
    CTM.algrithm_CTMRG_settings.CTM_cell_ite_method=update_method
    settings=CTM.LS_CTMRG_settings()
    settings.CTM_conv_tol=tol;settings.CTM_ite_nums=maxiter;settings.CTM_trun_tol=projector_tol
    settings.CTM_ite_info=verbose;settings.CTM_conv_info=true;settings.grad_checkpoint=false
    initial=CTM.initial_condition(init_type="PBC",reconstruct_CTM=isnothing(CTM0),reconstruct_AA=true)
    env,AA,_,_,_,_,iterations,error=CTM.Fermionic_CTMRG_cell(A,chi,initial,CTM0,settings)
    isnothing(environment_path) || jldsave(environment_path;CTM=env,chi,iterations,error)
    components=(op.Sx,op.Sy,op.Sz)
    layers=Array{TensorMap,4}(undef,Lx,Ly,3,3)
    for x=1:Lx,y=1:Ly,site=1:3,k=1:3
        layers[x,y,site,k]=_spin_layer(A[x][y],components[k][site])
    end
    layer(x,y,site,k)=layers[mod1(x,Lx),mod1(y,Ly),site,k]
    norms=[_spin_patch(env,AA,x-1,y-1) for x=1:Lx,y=1:Ly]
    patchnorm(cx,cy)=norms[mod1(cx+1,Lx),mod1(cy+1,Ly)]
    expect(cx,cy,ins)=_spin_patch(env,AA,cx,cy;insertions=ins)/patchnorm(cx,cy)
    magnetization=zeros(ComplexF64,Lx,Ly,3,3)
    up=zeros(ComplexF64,Lx,Ly,3);down=similar(up)
    upops=[sum(components[k][i]*components[k][j] for k=1:3) for (i,j) in ((1,2),(2,3),(3,1))]
    for x=1:Lx,y=1:Ly
        verbose && (println("Measuring Heisenberg cell ($x,$y)");flush(stdout))
        for site=1:3,k=1:3
            magnetization[x,y,site,k]=expect(x-1,y-1,(1=>layer(x,y,site,k),))
        end
        for e=1:3
            up[x,y,e]=expect(x-1,y-1,(1=>_spin_layer(A[x][y],upops[e]),))
        end
        down[x,y,1]=sum(expect(x-1,y-1,(1=>layer(x,y,2,k),2=>layer(x+1,y,1,k))) for k=1:3)
        down[x,y,2]=sum(expect(x-2,y-1,(2=>layer(x,y,2,k),4=>layer(x,y+1,3,k))) for k=1:3)
        down[x,y,3]=sum(expect(x-1,y-1,(3=>layer(x,y+1,3,k),2=>layer(x+1,y,1,k))) for k=1:3)
    end
    nsites=3Lx*Ly
    m=[real(sum(magnetization[:,:,:,k]))/nsites for k=1:3]
    exchange=J*real(sum(up)+sum(down))/nsites
    zeeman=-h*m[3]
    max_imag=max(maximum(abs,imag.(magnetization)),maximum(abs,imag.(up)),maximum(abs,imag.(down)))
    local_m=dropdims(sqrt.(sum(real.(magnetization).^2;dims=4));dims=4)
    (;energy_per_site=exchange+zeeman,energy_per_bond=exchange/2,
      exchange_per_site=exchange,zeeman_per_site=zeeman,h,magnetization_per_site=m,
      uniform_magnitude=norm(m),mean_local_magnitude=sum(local_m)/nsites,
      mz_over_saturation=2m[3],magnetization,up_bonds=up,down_bonds=down,max_imag,
      J,chi,projector_tol,iterations,ctm_error=error,converged=error<tol,update_method,CTM=env)
end
