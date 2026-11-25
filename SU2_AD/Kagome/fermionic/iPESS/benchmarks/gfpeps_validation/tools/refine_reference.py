"""Check Brillouin-zone integration on finer meshes without reexporting tensors."""
from prepare import *


if __name__=='__main__':
    for directory in sorted(OUT.glob('*_D*')):
        path=directory/'source_archive.npz'
        if not path.exists():continue
        z=np.load(path)
        rows={int(r['mesh']):r for r in json.loads((directory/'gaussian_reference.json').read_text())}
        for n in (384,768):
            row,local,hops=reference(z,n)
            rows[n]=row
            savemat(directory/f'reference_N{n}.mat',dict(**row,onsite_nambu=local,hoppings=hops,fields=z['fields']))
            (directory/'gaussian_reference.json').write_text(json.dumps([rows[k] for k in sorted(rows)],indent=2))
            print(directory.name,row,flush=True)
