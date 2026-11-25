"""Check that each saved magnetic-cell frame equals its saved local tensors."""
from prepare import *
from gaussian_ipess import SimplexCell
from primitive_parity import MagneticCell
from primitive_fit import assemble,bonds
from parity_fit import double_matrix


def check():
    rng=np.random.default_rng(91307)
    k=rng.uniform(-np.pi,np.pi,size=(11,2))
    rows=[]
    for family in ('iPESS','iPEPS'):
        for m in (2,3):
            item=selected(family,m);z=np.load(item['path'])
            blocked=output(z['W'],incoming(k,m,3*m))
            if family=='iPESS':
                cell=SimplexCell(m)
                direct=cell.direct([z[key] for key in cell.keys],k)
            else:
                frames=[z[f'Wsite{x}'] for x in range(3)]
                cell=MagneticCell(m,frames)
                c=bonds(k,m)
                direct=output(assemble(frames,cell.rows,cell.cols),double_matrix(c,-c))
            error=float(np.max(abs(direct-blocked)))
            assert error<1e-10
            row=dict(family=family,D=2**m,probe_count=len(k),max_full_nambu_error=error)
            rows.append(row);print(row,flush=True)
    (OUT/'archive_consistency.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':check()
