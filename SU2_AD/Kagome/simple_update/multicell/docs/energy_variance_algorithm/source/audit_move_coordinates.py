"""Check physical-coordinate tables against the rotated-index symbolic kernel."""
import hashlib
import json
from pathlib import Path
from trace_ctm_dependencies import Trace

root=Path(__file__).resolve().parents[6]
report={'date':'2026-11-07','kind':'symbolic coordinate audit, not a tensor numerical test',
        'directions':[], 'source_sha256':{}}
count=0
for d in (1,2,3,4):
    t=Trace(12);before=t.grid.copy();j=5
    t.move(j,d,1)
    expected={}
    for v in range(1,13):
        if d==1:
            entries=[(('C',j,v,1),[('C',j-1,v,1),('T',j,v,1)]),
                     (('T',j,v,4),[('T',j-1,v,4),('a',j,v)]),
                     (('C',j,v,4),[('C',j-1,v,4),('T',j,v,3)])]
        elif d==3:
            entries=[(('C',j,v,3),[('C',j+1,v,3),('T',j,v,3)]),
                     (('T',j,v,2),[('T',j+1,v,2),('a',j,v)]),
                     (('C',j,v,2),[('C',j+1,v,2),('T',j,v,1)])]
        elif d==2:
            entries=[(('C',v,j,2),[('C',v,j-1,2),('T',v,j,2)]),
                     (('T',v,j,1),[('T',v,j-1,1),('a',v,j)]),
                     (('C',v,j,1),[('C',v,j-1,1),('T',v,j,4)])]
        else:
            entries=[(('C',v,j,4),[('C',v,j+1,4),('T',v,j,4)]),
                     (('T',v,j,3),[('T',v,j+1,3),('a',v,j)]),
                     (('C',v,j,3),[('C',v,j+1,3),('T',v,j,2)])]
        expected.update(entries)
    changed={k for k in before if before[k]!=t.grid[k]}
    assert changed==set(expected)
    for key,parents in expected.items():
        ids=t.nodes[t.grid[key]]['parents']
        actual=[t.nodes[p]['label'] for p in ids if t.nodes[p]['kind']!='projector']
        assert set(actual)==set(parents),(d,key,actual,parents)
        count+=1
    report['directions'].append({'direction':d,'updated_tensors':len(changed),'passed':True})
report['verified_output_and_direct_input_formulas']=count
for rel in ('src/bosonic/kagome/native_measurement/energy_variance/variance_array_sweeps.jl',
            'src/bosonic/kagome/native_measurement/energy_variance/variance_contraction.jl',
            'src/bosonic/kagome/native_measurement/energy_variance/variance_workflow.jl',
            'src/bosonic/kagome/native_measurement/factorized_ctm.jl',
            'src/bosonic/kagome/native_measurement/memory_bounded_ctm.jl',
            'src/fermionic/Fermionic_CTMRG_unitcell_iPESS.jl'):
    report['source_sha256'][rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
out=Path(__file__).with_name('move_coordinate_audit.json')
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
