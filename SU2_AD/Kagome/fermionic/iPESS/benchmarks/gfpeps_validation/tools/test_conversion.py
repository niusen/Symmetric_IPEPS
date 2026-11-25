"""Small independent Fock-space tests of the Gaussian conversion conventions."""
from prepare import *
from scipy.linalg import expm
from parity_fit import LocalChart,double_matrix
from gaussian_fock import act


def random_frame(n,seed):
    rng=np.random.default_rng(seed)
    vacuum=np.eye(2*n,dtype=complex)[:,n:]
    chart=LocalChart(vacuum,n)
    a=rng.normal(size=(2*n,2*n)); a=a-a.T
    return chart.u.conj().T@expm(a)@chart.base


def test_covariances():
    errors=[]
    for n in (2,4,7):
        for seed in (17,43):
            w=random_frame(n,seed)
            for flips in ([],[0],[0,n-1]):
                t=particle_hole_rows(w,flips);v,info=fock_vector(t)
                errors.append(float(np.max(abs(covariance(v)-t@t.conj().T))))
                assert errors[-1]<1e-12
                assert info['reference_parity']==len(flips)%2
    return max(errors)


def test_projection():
    # Two physical and two virtual modes. Explicitly project the virtual pair;
    # no tensor-network implementation or fitted energy enters this check.
    w=random_frame(4,29);v,_=fock_vector(w)
    order=[0,1,4,5,2,3,6,7]
    c=np.array([[0.,1.],[1.,0.]])
    errors=[]
    for sign in (1,-1):
        bond=np.array([0,1,sign,0])/np.sqrt(2)
        projected=v.reshape(4,4,order='F')@bond
        projected/=np.linalg.norm(projected)
        expected=output(w[order],double_matrix(sign*c,-sign*c)[None])[0]
        errors.append(float(np.max(abs(covariance(projected)-expected))))
        assert errors[-1]<1e-12
    return max(errors)


if __name__=='__main__':
    result=dict(random_full_nambu_covariance_error=test_covariances(),
                explicit_bond_projection_error=test_projection())
    print(result)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'conversion_tests.json').write_text(json.dumps(result,indent=2))
