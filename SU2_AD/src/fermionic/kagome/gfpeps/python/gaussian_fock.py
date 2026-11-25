"""Pure Gaussian Nambu projectors -> explicit, parity-resolved Fock tensors.

Convention: P_ij=<z_j^dagger z_i>, z=(c_0,...,c_n-1,c_0^dagger,...).
Bit i labels c_i^dagger; |bits>=prod_i(increasing) c_i^dagger |vac>.
No spin or anomalous blocks are discarded.
"""
import numpy as np


def standard_frame(w, physical):
    """Source rows (p,pdag,v,vdag) -> (p,v,pdag,vdag)."""
    n=len(w)//2; v=n-physical
    order=np.r_[0:physical,2*physical:2*physical+v,
                physical:2*physical,2*physical+v:2*n]
    return w[order]


def particle_hole_rows(w, modes):
    n=len(w)//2; order=np.arange(2*n)
    for i in modes: order[i],order[n+i]=order[n+i],order[i]
    return w[order]


def fock_vector(w):
    """Pfaffian expansion in a well-conditioned particle/hole vacuum chart."""
    n=len(w)//2
    if w.shape!=(2*n,n): raise ValueError('a pure full Nambu frame is required')
    diag=np.sum(abs(w[:n])**2,axis=1)
    base=sum((1<<i) for i in range(n) if diag[i]>.5)
    rng=np.random.default_rng(917)
    candidates=[base,0]+[base^(1<<i) for i in range(n)]
    candidates+=rng.integers(0,1<<n,size=32).tolist()
    best=None
    for bits in candidates:
        t=particle_hole_rows(w,[i for i in range(n) if bits>>i&1])
        singular=np.linalg.svd(t[n:],compute_uv=False)
        if best is None or singular[-1]>best[0]: best=(singular[-1],bits,t)
    smallest,reference,t=best
    if smallest<1e-10: raise ValueError(f'no nonsingular vacuum chart: {smallest}')
    z=np.linalg.solve(t[n:].T,t[:n].T).T
    skew=np.max(abs(z+z.T))
    if skew>1e-8: raise ValueError(f'non-Gaussian or inconsistent Nambu frame: {skew}')
    z=(z-z.T)/2
    pf=np.zeros(1<<n,complex); pf[0]=1
    for mask in range(1,1<<n):
        if mask.bit_count()%2: continue
        first=mask&-mask; i=first.bit_length()-1; rest=mask^first
        walk=rest; sign=1; value=0j
        while walk:
            bit=walk&-walk; j=bit.bit_length()-1
            value+=sign*z[i,j]*pf[rest^bit]
            walk^=bit; sign=-sign
        pf[mask]=value
    result=np.zeros_like(pf)
    # The reference vacuum is its ordinary occupation basis vector. Apply
    # d_i^dagger in descending i, including every Jordan-Wigner sign.
    for mask in range(1<<n):
        if mask.bit_count()%2: continue
        occupation=reference; sign=1
        for i in range(n-1,-1,-1):
            if mask>>i&1:
                sign*=(-1)**((occupation&((1<<i)-1)).bit_count())
                occupation^=1<<i
        result[occupation]=sign*pf[mask]
    result/=np.linalg.norm(result)
    return result,dict(reference=reference,reference_parity=reference.bit_count()%2,
                       smallest_chart_singular_value=float(smallest),skew_error=float(skew))


def act(vector,i,creation=False):
    nbits=np.arange(len(vector),dtype=np.int64)
    selected=nbits[((nbits>>i)&1)==(0 if creation else 1)]
    signs=np.array([1-2*((int(j)&((1<<i)-1)).bit_count()%2) for j in selected])
    result=np.zeros_like(vector)
    result[selected^(1<<i)]=signs*vector[selected]
    return result


def covariance(vector):
    n=(len(vector)-1).bit_length()
    responses=np.array([act(vector,i,c) for c in (False,True) for i in range(n)])
    return responses@responses.conj().T


def parity_order(m):
    return np.array(sorted(range(1<<m),key=lambda x:(x.bit_count()%2,x)))


def leg_tensor(vector,mode_counts):
    """Group consecutive modes into legs, with even states first on each leg."""
    if len(vector)!=1<<sum(mode_counts): raise ValueError('wrong mode counts')
    tensor=vector.reshape(tuple(1<<m for m in mode_counts),order='F')
    for i,m in enumerate(mode_counts): tensor=np.take(tensor,parity_order(m),axis=i)
    return tensor


def bond_diagonal(m,ph_left):
    """Projecting bond has covariance C_in: occupied (left+right)/sqrt(2).

    Transform exactly one endpoint of each mode to holes. The resulting
    even Gaussian bond is diagonal in the two endpoint occupation bases.
    Its bra coefficients can be absorbed into a tensor before index summation.
    """
    orbitals=np.vstack((np.eye(m),np.eye(m)))/np.sqrt(2)
    empty=np.vstack((np.eye(m),-np.eye(m)))/np.sqrt(2)
    w=np.zeros((4*m,2*m),complex)
    w[:2*m,:m]=orbitals; w[2*m:,m:]=empty
    modes=[j if ph_left[j] else m+j for j in range(m)]
    w=particle_hole_rows(w,modes)
    vector,_=fock_vector(w)
    tensor=leg_tensor(vector,(m,m))
    diag=np.diag(tensor)
    assert np.max(abs(tensor-np.diag(diag)))<1e-10
    assert np.max(abs(abs(diag)-abs(diag[0])))<1e-10
    return (diag/diag[0]).conj()
