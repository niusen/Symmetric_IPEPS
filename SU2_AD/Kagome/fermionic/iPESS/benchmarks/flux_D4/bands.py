"""Pi/2 triangular Hofstadter model with the odd/odd sites disconnected.

Parent primitive vectors e1=(1,0), e2=(1/2,sqrt(3)/2).
Actual forward Hamiltonian coefficients (matching the legacy M=2 gauge):
H(e1)=-1, H(e2)=-(-1)^m, H(e2-e1)=exp(i*pi*(m-1/2)).
Kagome A1=2e1,A2=2e2, basis a=-e1,b=0,c=-e2.
Surviving up/down (t_ab,t_bc,t_ca)=(1,1,-i), with NO regauging.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/'results'/'flux_D4'
OUT.mkdir(parents=True,exist_ok=True)

def parent_hopping(start,end):
    m,n=start
    dx,dy=np.subtract(end,start)
    if (dx,dy)==(1,0): return -1.+0j
    if (dx,dy)==(0,1): return -complex((-1.)**m)
    if (dx,dy)==(-1,1): return np.exp(1j*np.pi*(m-.5))
    if (dx,dy) in ((-1,0),(0,-1),(1,-1)):
        return np.conj(parent_hopping(end,start))
    raise ValueError('not a parent nearest-neighbor bond')

def loop_phase(vertices):
    return np.angle(np.prod([parent_hopping(a,b) for a,b in zip(vertices,vertices[1:]+vertices[:1])]))

def bloch(k1,k2,chirality=1):
    k1,k2=np.broadcast_arrays(k1,k2)
    h=np.zeros(k1.shape+(3,3),complex)
    h[...,0,1]=-(1+np.exp(-1j*k1))
    h[...,1,2]=-(1+np.exp(1j*k2))
    h[...,2,0]=1j*chirality*(1+np.exp(1j*(k1-k2)))
    return h+np.swapaxes(h.conj(),-1,-2)

def parent_bloch(k1,k2):
    # Independent real-space edge enumeration in the 2x2 parent cell.
    sites=[(-1,0),(0,0),(0,-1),(-1,-1)]
    h=np.zeros((4,4),complex)
    for j,s in enumerate(sites):
        if j==3: continue
        for d in ((1,0),(0,1),(-1,1)):
            dest=tuple(np.add(s,d))
            if dest[0]%2 and dest[1]%2: continue
            i=next(i for i,r in enumerate(sites) if (dest[0]-r[0])%2==0 and (dest[1]-r[1])%2==0)
            shift=(np.array(dest)-sites[i])//2
            z=parent_hopping(s,dest)*np.exp(-1j*np.dot([k1,k2],shift))
            h[i,j]+=z; h[j,i]+=z.conjugate()
    return h

def band_check(mesh,chirality=1):
    k=np.arange(mesh)*2*np.pi/mesh
    k1,k2=np.meshgrid(k,k,indexing='ij')
    e,v=np.linalg.eigh(bloch(k1,k2,chirality))
    analytic=np.sqrt(6+2*np.cos(k1)+2*np.cos(k2)+2*np.cos(k1-k2))
    assert np.max(abs(e-np.stack([-analytic,np.zeros_like(analytic),analytic],axis=-1)))<1e-12
    ux=np.sum(v.conj()*np.roll(v,-1,axis=0),axis=-2)
    uy=np.sum(v.conj()*np.roll(v,-1,axis=1),axis=-2)
    ux/=abs(ux); uy/=abs(uy)
    curvature=np.angle(ux*np.roll(uy,-1,axis=0)/np.roll(ux,-1,axis=1)/uy)
    chern=curvature.sum(axis=(0,1))/(2*np.pi)
    return dict(mesh=mesh,chirality=chirality,chern=chern.tolist(),
                band_min=e.min(axis=(0,1)).tolist(),band_max=e.max(axis=(0,1)).tolist(),
                flat_band_max_abs=float(abs(e[...,1]).max()),gap=float(analytic.min()),
                kinetic_spinful_per_site=float(2*e[...,0].mean()/3))

def main():
    for m in range(-2,3):
        for n in range(-2,3):
            up=[(m,n),(m+1,n),(m,n+1)]
            dn=[(m+1,n),(m+1,n+1),(m,n+1)]
            assert abs(loop_phase(up)-np.pi/2)<1e-12
            assert abs(loop_phase(dn)-np.pi/2)<1e-12
    # The six surviving neighbors of the disconnected (1,1) site.
    hexagon=[(2,1),(1,2),(0,2),(0,1),(1,0),(2,0)]
    assert all(not (m%2 and n%2) for m,n in hexagon)
    assert abs(abs(loop_phase(hexagon))-np.pi)<1e-12
    for s in ((-1,0),(0,0),(0,-1)):
        neighbors=[np.add(s,d) for d in ((1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1))]
        assert sum(not (m%2 and n%2) for m,n in neighbors)==4
    for k1,k2 in np.random.default_rng(27).uniform(-np.pi,np.pi,(20,2)):
        hp=parent_bloch(k1,k2)
        assert np.max(abs(hp[:3,:3]-bloch(k1,k2)))<1e-12
        assert np.max(abs(hp[3,:]))==0
    checks=[band_check(n) for n in (30,60,120,300,600)]
    inverse=band_check(60,-1)
    assert np.allclose(checks[-1]['chern'],checks[0]['chern'],atol=1e-10)
    assert sorted(np.rint(checks[-1]['chern']).astype(int))==[-1,0,1]
    assert abs(checks[-1]['chern'][1])<1e-10
    assert np.allclose(inverse['chern'],-np.array(checks[-1]['chern']),atol=1e-10)
    mu=-np.sqrt(3)/2
    result=dict(convention='Actual H_ij loop products have +pi/2 CCW flux, matching legacy M=2 Hofstadter hopping',
        basis=[[-1,0],[0,0],[0,-1]],bravais=[[2,0],[0,2]],
        hexagon_flux_mod_2pi=float(loop_phase(hexagon)),spin_degeneracy=2,
        chemical_potential=float(mu),density=2/3,
        grand_energy_per_site=checks[-1]['kinetic_spinful_per_site']-mu*2/3,
        checks=checks,opposite_flux=inverse)
    (OUT/'bands.json').write_text(json.dumps(result,indent=2)+'\n')
    # Coordinates are k.A_i; this is Gamma-K-M-Gamma in the triangular BZ.
    corners=np.array([[0,0],[4*np.pi/3,2*np.pi/3],[np.pi,np.pi],[0,0]])
    path=np.concatenate([np.linspace(a,b,151)[:-1] for a,b in zip(corners[:-1],corners[1:])]+[corners[-1:]])
    e=np.linalg.eigvalsh(bloch(path[:,0],path[:,1]))
    cart=np.column_stack([path[:,0]/2,(path[:,1]-path[:,0]/2)/np.sqrt(3)])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(cart,axis=0),axis=1))]
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for b in range(3):
        ax.plot(distance,e[:,b],label=f'C = {round(checks[-1]["chern"][b])}')
    ax.axhline(mu,color='black',ls='--',lw=1,label=r'$\mu=-\sqrt{3}/2$')
    ax.set_xticks(distance[[0,150,300,450]],[r'$\Gamma$',r'$K$',r'$M$',r'$\Gamma$'])
    ax.set_ylabel('Single-particle energy / t')
    ax.set_title(r'Kagome inherited flux: $\Phi_\triangle=\pi/2$, $\Phi_{hexagon}=\pi$')
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5)); ax.grid(alpha=.2)
    fig.savefig(OUT/'bands.png',dpi=180)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
