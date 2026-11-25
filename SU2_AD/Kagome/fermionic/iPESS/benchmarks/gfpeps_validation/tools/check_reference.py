"""Reconstruct Gaussian energy from individual bonds, independently of Tr(h P)."""
from prepare import *
from scipy.io import loadmat


def check():
    rows=[]
    for path in sorted(OUT.glob('*_D*/reference_N*.mat')):
        r=loadmat(path);hop=r['hoppings'];local=r['onsite_nambu']
        mu=r['fields'].ravel();theta=-np.pi/24;flux=2*np.pi/3
        energy=0.
        for x in range(3):
            for o in range(2):
                phases=(theta,theta+(flux*x if o else 0),theta-(flux*x if o else 0))
                for edge,phase in enumerate(phases):
                    t=np.exp(1j*phase)
                    energy+=2*np.real(t*hop[x,o,edge,0]-t.conjugate()*hop[x,o,edge,1])
        nu=local[:,:,0,0].real.mean();nd=1-local[:,:,1,1].real.mean()
        kinetic=energy/9;grand=kinetic-mu[0]*nu-mu[1]*nd
        error=max(abs(kinetic-r['kinetic_per_site'].item()),abs(grand-r['grand_per_site'].item()),
                  abs(nu-r['n_up_per_site'].item()),abs(nd-r['n_down_per_site'].item()))
        assert error<1e-11,(path,error)
        rows.append(dict(file=str(path.relative_to(OUT)),max_energy_density_error=float(error)))
    print('Reference bond/trace consistency:',max(r['max_energy_density_error'] for r in rows))
    (OUT/'reference_consistency.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':check()
