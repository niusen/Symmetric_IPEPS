"""Independent Gaussian contraction of a simplex vs the imported Fock tensors."""
from prepare import *
from scipy.io import loadmat
from parity_fit import double_matrix
from gaussian_fock import parity_order


def gaussian_triangle(z,m):
    frames=[standard_frame(z[f'site_0_{s}'],2) for s in 'ABC']+[standard_frame(z['triangle_0_up'],0)]
    sizes=[len(w)//2 for w in frames]; N=sum(sizes)
    total=np.zeros((2*N,N),complex); offset=0
    for w,n in zip(frames,sizes):
        total[np.ix_(np.r_[offset:offset+n,N+offset:N+offset+n],np.arange(offset,offset+n))]=w
        offset+=n
    starts=np.cumsum([0]+sizes)
    keep=np.r_[np.concatenate([starts[i]+np.arange(2) for i in range(3)]),
               np.concatenate([starts[i]+2+m+np.arange(m) for i in range(3)])]
    contract=np.r_[np.concatenate([starts[i]+2+np.arange(m) for i in range(3)]),starts[3]+np.arange(3*m)]
    order=np.r_[keep,N+keep,contract,N+contract]
    c=np.block([[np.zeros((3*m,3*m)),np.eye(3*m)],[np.eye(3*m),np.zeros((3*m,3*m))]])
    rho=output(total[order],double_matrix(c,-c)[None])[0]
    if m%2:
        perm=np.arange(len(rho)); n=len(rho)//2
        perm[6],perm[n+6]=perm[n+6],perm[6]
        rho=rho[np.ix_(perm,perm)]
    return rho


def fermi_permute(t,counts,order):
    # Each input axis is in the graded, even-then-odd occupation basis.
    q=[np.array([j.bit_count()%2 for j in parity_order(m)]) for m in counts]
    sign=np.ones(t.shape,dtype=np.int8)
    for i in range(len(order)):
        for j in range(i+1,len(order)):
            a,b=order[i],order[j]
            if a>b:
                ashape=[1]*t.ndim;ashape[a]=len(q[a]); bshape=[1]*t.ndim;bshape[b]=len(q[b])
                sign*=1-2*q[a].reshape(ashape)*q[b].reshape(bshape)
    return (t*sign).transpose(order)


def check(m):
    directory=OUT/f'ipess_D{2**m}'; z=np.load(selected('iPESS',m)['path'])
    target=gaussian_triangle(z,m)
    # Native axes (external B,physical B,external A,physical A,external C,physical C).
    t=np.fromfile(directory/'triangle_native.bin',dtype=np.complex128).reshape((2**m,4)*3,order='F')
    for axis,s in ((0,'B'),(2,'A'),(4,'C')):
        mask=[m%2==1 and s=='A']+[False]*(m-1)
        shape=[1]*6;shape[axis]=2**m
        t=t/bond_diagonal(m,mask).reshape(shape)
    t=fermi_permute(t,[m,2,m,2,m,2],[3,1,5,2,0,4])
    for axis,c in enumerate([2,2,2,m,m,m]):t=np.take(t,np.argsort(parity_order(c)),axis=axis)
    vector=t.reshape(-1,order='F');vector/=np.linalg.norm(vector)
    actual=covariance(vector)
    error=np.max(abs(actual-target)); purity=np.max(abs(actual@actual-actual))
    assert max(error,purity)<1e-10
    print('TRIANGLE PATCH',m,'covariance error',error,'purity error',purity,flush=True)
    savemat(directory/'triangle_check.mat',dict(actual=actual,target=target,vector=vector))
    (directory/'triangle_check.json').write_text(json.dumps(dict(covariance_error=float(error),purity_error=float(purity)),indent=2))
    return error

if __name__=='__main__':check(int(sys.argv[1]) if len(sys.argv)>1 else 2)
