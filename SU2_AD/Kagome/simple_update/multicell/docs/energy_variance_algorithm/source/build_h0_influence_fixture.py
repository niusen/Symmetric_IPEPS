"""Structural dependence on aH(6,6)/aH(7,6), not a response magnitude."""
from pathlib import Path
import json
from trace_ctm_dependencies import Trace

root=Path(__file__).parent
t=Trace(12)
flags=[n['label'] in (('a',6,6),('a',7,6)) for n in t.nodes]
schedule=[(3,7,0),(4,6,0)]
for s in (1,2):
    for d,js in ((1,range(5,12)),(3,range(8,1,-1)),(2,range(5,12)),(4,range(7,1,-1))):
        schedule.extend((d,j,s) for j in js)
snapshots=[[]];pending=[]
for d,j,s in schedule:
    before=t.grid.copy();start=len(t.nodes);t.move(j,d,s)
    for n in t.nodes[start:]:flags.append(any(flags[p] for p in n['parents']))
    marked=[];written=[]
    for key,n in t.grid.items():
        kind,x,y,k=key
        if flags[n]:
            marked.append(f'{kind}{k},{x},{y}')
            if n!=before[key]:written.append(f'{kind}{k},{x},{y}')
    snapshots.append(sorted(marked));pending.append(sorted(written))
first=snapshots[1]
assert sorted(first)==sorted(['C2,7,5','C2,7,6','T2,7,5','T2,7,6','T2,7,7','C3,7,6','C3,7,7'])
result=dict(description='Exact structural read-dependence on the two inserted local factors; no tensor response values',
            sources=[[6,6],[7,6]],snapshots=snapshots,pending=pending)
(root/'h0_influence_expected.json').write_text(json.dumps(result),encoding='utf-8')
print('Verified first-move dependence:',first)
print('Affected environment record counts at moves 0,1,2,17,30,56:',[len(snapshots[i]) for i in (0,1,2,17,30,56)])
