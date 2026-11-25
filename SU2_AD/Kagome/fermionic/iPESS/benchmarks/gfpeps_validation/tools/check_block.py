"""Derive the square-block import convention from a native five-tensor block."""
from check_patch import *


def gaussian_block(z,m):
    frames=[standard_frame(z[f'site_0_{s}'],2) for s in 'ABC']+[
        standard_frame(z['triangle_0_up'],0),standard_frame(z['triangle_0_down'],0)]
    sizes=[len(w)//2 for w in frames];N=sum(sizes);starts=np.cumsum([0]+sizes)
    total=np.zeros((2*N,N),complex)
    for j,(w,n) in enumerate(zip(frames,sizes)):
        o=starts[j]
        total[np.ix_(np.r_[o:o+n,N+o:N+o+n],np.arange(o,o+n))]=w
    keep=np.r_[np.concatenate([starts[i]+np.arange(2) for i in (1,0,2)]),
        starts[1]+2+m+np.arange(m),starts[4]+2*m+np.arange(m),
        starts[4]+m+np.arange(m),starts[2]+2+m+np.arange(m)]
    pairs=[(starts[i]+2,starts[3]+i*m) for i in range(3)]+[(starts[0]+2+m,starts[4])]
    contract=np.concatenate([start+np.arange(m) for pair in pairs for start in pair])
    c=np.zeros((len(contract),len(contract)))
    for j in range(0,len(contract),2*m):
        a=j+np.arange(m);c[a,a+m]=c[a+m,a]=1
    order=np.r_[keep,N+keep,contract,N+contract]
    rho=output(total[order],double_matrix(c,-c)[None])[0]
    # The exposed simplex B/C legs are PH transformed for both D=4 and D=8.
    p=np.arange(len(rho));n=len(rho)//2
    for i in range(6+m,6+3*m):p[i],p[n+i]=p[n+i],p[i]
    return rho[np.ix_(p,p)]


def check_block(m=2):
    directory=OUT/f'ipess_D{2**m}';z=np.load(selected('iPESS',m)['path'])
    target=gaussian_block(z,m)
    t=np.fromfile(directory/'block_native.bin',dtype=np.complex128).reshape((2**m,)*4+(4,)*3,order='F')
    # Export has already undone the legacy three crossings.
    t=fermi_permute(t,[m,m,m,m,2,2,2],[4,5,6,0,1,2,3])
    for axis in (3,6):
        shape=[1]*7;shape[axis]=2**m
        t=t/bond_diagonal(m,[False]*m).reshape(shape)
    for axis,c in enumerate([2,2,2,m,m,m,m]):t=np.take(t,np.argsort(parity_order(c)),axis=axis)
    v=t.reshape(-1,order='F');v/=np.linalg.norm(v)
    actual=covariance(v)
    error=float(np.max(abs(actual-target)));purity=float(np.max(abs(actual@actual-actual)))
    assert error<1e-10 and purity<1e-10
    print('BLOCK',error,'purity',purity)
    savemat(directory/'block_check.mat',dict(actual=actual,target=target))
    (directory/'block_check.json').write_text(json.dumps(dict(covariance_error=error,purity_error=purity),indent=2))


if __name__=='__main__':check_block(int(sys.argv[1]) if len(sys.argv)>1 else 2)
