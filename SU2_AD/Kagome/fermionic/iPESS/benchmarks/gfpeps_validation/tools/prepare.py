import argparse
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[6]/"src/fermionic/kagome/gfpeps/python"))
from archive_conversion import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--m',type=int,default=2)
    parser.add_argument('--meshes',type=int,nargs='+',default=[48,96,192])
    parser.add_argument('--family',choices=['ipess','ipeps'],default='ipess')
    parser.add_argument('--su-seed',choices=['orig','mild','zero'],
                        help='export only the selected native MAT seed for SU into a fresh, model-specific directory')
    parser.add_argument('--charge-audit',action='store_true',help='only export frozen source covariances for audit_spin_sector.jl')
    args=parser.parse_args()
    if args.su_seed:
        if args.charge_audit: parser.error('--su-seed and --charge-audit are mutually exclusive')
        item=selected('iPEPS' if args.family=='ipeps' else 'iPESS',args.m,args.su_seed)
        directory=OUT/f"{args.family}_D{2**args.m}_{args.su_seed}"
        directory.mkdir(parents=True,exist_ok=False)
        (export_ipess if args.family=='ipess' else export_ipeps)(args.m,item=item,directory=directory)
        return
    if args.charge_audit:
        export_charge_audit(args.family,args.m)
        raise SystemExit(0)
    directory,z=(export_ipess if args.family=='ipess' else export_ipeps)(args.m)
    rows=[]
    for n in args.meshes:
        row,local,hops=reference(z,n);rows.append(row)
        (directory/'gaussian_reference.json').write_text(json.dumps(rows,indent=2))
        savemat(directory/f'reference_N{n}.mat',dict(**row,onsite_nambu=local,hoppings=hops,fields=z['fields']))
        print(row,flush=True)

if __name__ == "__main__":
    main()
