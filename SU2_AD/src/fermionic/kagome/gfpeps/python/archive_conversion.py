"""Read archived Gaussian tensors without changing the source workspace."""
import os,sys
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[5]
if not os.environ.get('KAGOME_GAUSSIAN_SOURCE'):
    raise RuntimeError('Set KAGOME_GAUSSIAN_SOURCE to the original kagome Gaussian archive workspace; pre-exported MAT server runs do not need it.')
SOURCE=Path(os.environ['KAGOME_GAUSSIAN_SOURCE']).expanduser().resolve()
OUT=ROOT/'Kagome/fermionic/iPESS/results/gfpeps_validation'
sys.path[:0]=[str(ROOT/'src/fermionic/kagome/gfpeps/python'),str(SOURCE/'gfpeps_parton')]
from gaussian_fock import standard_frame,particle_hole_rows,fock_vector,covariance,leg_tensor,bond_diagonal
from gfpeps import output,bloch,mesh
from parity_fit import incoming


def selected(family,m,model="orig"):
    manifest=json.loads((SOURCE/'parton_summary/data/es_manifest.json').read_text())
    return next(s for s in manifest['states'] if s['family']==family and s['model']==model and s['D']==2**m)


def export_ipess(m, *, item=None, directory=None):
    item=selected('iPESS',m) if item is None else item
    path=Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    z=np.load(path); data={'m':m,'fields':z['fields'],'source_sha256':item['sha256']}
    checks=[]
    for x in range(3):
        for s in 'ABC':
            key=f'site_{x}_{s}'; w=standard_frame(z[key],2)
            # Bell bonds: for odd m, move mode 0's PH from both simplices to A.
            ph=[2,2+m] if m%2 and s=='A' else []
            w=particle_hole_rows(w,ph); v,info=fock_vector(w)
            error=float(np.max(abs(covariance(v)-w@w.conj().T)))
            assert error<1e-10 and info['reference_parity']==0
            t=leg_tensor(v,(2,m,m))
            mask=[m%2==1 and s=='A']+[False]*(m-1)
            diagonal=bond_diagonal(m,mask)
            # Absorb the two projecting bond bras into the physical tensor.
            t=t*diagonal[None,:,None]*diagonal[None,None,:]
            data[key]=t; checks.append(dict(key=key,covariance_error=error,**info))
        for direction in ('up','down'):
            key=f'triangle_{x}_{direction}'; w=standard_frame(z[key],0)
            ph=[i for i in range(3*m) if not (m%2 and i==0)]
            w=particle_hole_rows(w,ph); v,info=fock_vector(w)
            error=float(np.max(abs(covariance(v)-w@w.conj().T)))
            assert error<1e-10 and info['reference_parity']==0
            data[key]=leg_tensor(v,(m,m,m)); checks.append(dict(key=key,covariance_error=error,**info))
    directory=OUT/f'ipess_D{2**m}' if directory is None else Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,directory/'source_archive.npz')
    savemat(directory/'tensors.mat',data)
    (directory/'source.json').write_text(json.dumps(dict(**item,checks=checks,
        physical_basis='b=(up,down-hole); Fock basis (vac,up+hole,up,hole)',
        virtual_basis='even bitstrings first, then odd; bit 0 is first mode'),indent=2))
    print('exported',directory,'max covariance error',max(r['covariance_error'] for r in checks),flush=True)
    return directory,z


def export_ipeps(m, *, item=None, directory=None):
    item=selected('iPEPS',m) if item is None else item
    path=Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    z=np.load(path); data={'m':m,'fields':z['fields'],'source_sha256':item['sha256']}
    checks=[]
    for x in range(3):
        w=standard_frame(z[f'Wsite{x}'],6); n=len(w)//2
        # Physical site order B,A,C. Read connectivity from primitive_fit.bonds:
        # source U has the same outgoing y bond as the native site's C-down
        # leg, which build_A_kagome calls U. The CTM D/U labels are not Cartesian
        # +/-y labels. Thus current L,D,R,U = source R,D,L,U.
        order=np.r_[1,4,0,3,2,5,
            np.concatenate([6+d*m+np.arange(m) for d in (1,2,0,3)])]
        w=w[np.r_[order,n+order]]
        # Match the validated iPESS blocking: PH on current D/R endpoints.
        w=particle_hole_rows(w,list(range(6+m,6+3*m)))
        v,info=fock_vector(w)
        error=float(np.max(abs(covariance(v)-w@w.conj().T)))
        assert error<1e-10 and info['reference_parity']==0
        t=leg_tensor(v,(2,2,2,m,m,m,m))
        diagonal=bond_diagonal(m,[False]*m)
        t=t*diagonal.reshape((1,1,1,2**m,1,1,1))*diagonal.reshape((1,1,1,1,1,1,2**m))
        data[f'cell_{x}']=t
        checks.append(dict(key=f'Wsite{x}',covariance_error=error,**info))
    directory=OUT/f'ipeps_D{2**m}' if directory is None else Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,directory/'source_archive.npz')
    savemat(directory/'tensors.mat',data)
    (directory/'source.json').write_text(json.dumps(dict(**item,checks=checks,
        physical_basis='b=(up,down-hole); site order B,A,C; each (vac,up+hole,up,hole)',
        exported_axes='pa,pb,pc,L,D,R,U; requires legacy crossing convention'),indent=2))
    print('exported',directory,'max covariance error',max(r['covariance_error'] for r in checks),flush=True)
    return directory,z


def reference(z,n):
    """Independent exact Gaussian contraction and BZ integration of THIS saved state."""
    m=int(z['mx']); mu=z['fields']; sums=np.zeros(3)
    local=np.zeros((3,3,4,4),complex)
    hopping=np.zeros((3,2,3,2),complex)
    points=mesh(n,(.173,.331))
    for k in np.array_split(points,max(1,len(points)//256)):
        rho=output(z['W'],incoming(k,m,3*m))
        b=rho[:,:18,:18]
        h=bloch(k); hh=-bloch(-k).conj()
        kinetic=(np.einsum('kij,kji->k',h,b[:,:9,:9])+np.einsum('kij,kji->k',hh,b[:,9:,9:])).real
        nu=np.trace(b[:,:9,:9],axis1=1,axis2=2).real
        nd=9-np.trace(b[:,9:,9:],axis1=1,axis2=2).real
        sums+=np.array([kinetic.sum(),nu.sum(),nd.sum()])
        for x in range(3):
            for s in range(3):
                i=3*x+s; idx=[i,9+i,18+i,27+i]
                local[x,s]+=rho[:,idx][:,:,idx].sum(axis=0)
            for o,(dx,dy) in enumerate(((0,0),(-1,-1))):
                sites=[(x,0,0),(x+dx,0,1),(x,dy,2)]
                # Current (a,b,c)=(source B,A,C); bond order ab,bc,ca.
                for edge,(src,dst) in enumerate(((sites[0],sites[1]),(sites[2],sites[0]),(sites[1],sites[2]))):
                    a,b_,s=src; aa,bb,t=dst
                    delta=np.array([aa//3-a//3,bb-b_])
                    phase=np.exp(-1j*k@delta)
                    for spin in range(2):
                        hopping[x,o,edge,spin]+=np.sum(phase*rho[:,3*(a%3)+s+9*spin,3*(aa%3)+t+9*spin])
    kinetic,nu,nd=sums/len(points)
    grand=(kinetic-mu[0]*nu-mu[1]*nd)/9
    return dict(mesh=n,kinetic_per_site=kinetic/9,grand_per_site=grand,
        n_up_per_site=nu/9,n_down_per_site=nd/9),local/len(points),hopping/len(points)


def export_charge_audit(family, m, n=16, *, directory=None):
    """Export source Gaussian covariances only; the charge/projection audit is Julia.

    Read the frozen archive, without reselecting/replacing the imported state.
    The zero-momentum covariance is the complete nine-site periodic torus state.
    """
    directory=OUT/f'{family}_D{2**m}' if directory is None else Path(directory)
    path=directory/'source_archive.npz'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==json.loads((directory/'source.json').read_text())['sha256']
    with np.load(path) as z:
        points=mesh(n,(0.,0.))
        rho=output(z['W'],incoming(points,m,3*m))
        zero=output(z['W'],incoming(np.zeros((1,2)),m,3*m))[0]
    savemat(directory/'charge_audit_input.mat',dict(
        rho_zero=zero,rho_mesh=np.moveaxis(rho,0,-1),mesh_n=n,source_sha256=digest))
    print('exported charge audit',family,2**m,n,flush=True)

