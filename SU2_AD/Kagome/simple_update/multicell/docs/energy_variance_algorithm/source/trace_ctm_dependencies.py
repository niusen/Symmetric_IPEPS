"""Symbolic provenance of C/T and 4-corner projectors; no tensor calculation.

Mirrors variance_array_sweeps.jl, split_corner, convert_cell_posit and
observables.enlarged_corner. All outputs in one move read the same old grid.
The square-paper schedule is distinguished from the kagome warm-start schedule.
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path


class Trace:
    def __init__(self, L=12):
        self.L = L
        self.nodes = []
        self.grid = {}
        self.moves = []
        self.layers = {}
        for x in range(1, L+1):
            for y in range(1, L+1):
                self.layers[x,y] = self.node(('a',x,y), [])
                for kind in ('C','T'):
                    for k in range(1,5):
                        key=(kind,x,y,k)
                        self.grid[key] = self.node(key, [])

    def node(self, label, parents, move=None, kind='tensor'):
        n=len(self.nodes)
        self.nodes.append(dict(label=label, parents=list(dict.fromkeys(parents)),
                               move=move, kind=kind))
        return n

    def pos(self,cx,cy,a,b,d):
        xy={1:(cx+a,cy+b),2:(cy-b,cx+a),3:(cx-a,cy-b),4:(cy+b,cx-a)}[d]
        return tuple((v-1)%self.L+1 for v in xy)

    def move(self,j,d,sweep):
        mid=len(self.moves)
        self.moves.append(dict(id=mid,j=j,d=d,sweep=sweep))
        cx=j-1 if d in (1,2) else j+1
        old=self.grid.copy()
        P={};Pi={}
        def at(cy,kind,a,b,k):
            x,y=self.pos(cx,cy,a,b,d)
            return old[kind,x,y,(k-1)%4+1]
        for cy in range(1,self.L+1):
            specs=[('C',0,0,d),('T',1,0,d),('T',0,1,d-1),
                   ('T',0,2,d-1),('C',0,3,d-1),('T',1,3,d-2),
                   ('T',2,0,d),('C',3,0,d+1),('T',3,1,d+1),
                   ('T',3,2,d+1),('T',2,3,d-2),('C',3,3,d-2)]
            parents=[at(cy,*s) for s in specs]
            parents += [self.layers[self.pos(cx,cy,a,b,d)]
                        for a,b in ((1,1),(1,2),(2,1),(2,2))]
            p=self.node(('projectors',mid,cy),parents,mid,'projector')
            P[self.pos(cx,cy,0,2,d)]=p
            Pi[self.pos(cx,cy,0,1,d)]=p
        updates={}
        for cy in range(1,self.L+1):
            def pos(a,b):return self.pos(cx,cy,a,b,d)
            edge=('T',*pos(1,2),(d-2)%4+1)
            upper=('C',*pos(1,0),(d-1)%4+1)
            lower=('C',*pos(1,3),(d-2)%4+1)
            edge_parents=[at(cy,'T',0,2,d-1),Pi[pos(0,2)],P[pos(0,2)],self.layers[pos(1,2)]]
            up_parents=[at(cy,'C',0,0,d),at(cy,'T',1,0,d),Pi[pos(0,0)]]
            low_parents=[at(cy,'C',0,3,d-1),at(cy,'T',1,3,d-2),P[pos(0,3)]]
            for key,parents in ((edge,edge_parents),(upper,up_parents),(lower,low_parents)):
                assert key not in updates
                updates[key]=self.node(key,parents,mid)
        assert len(updates)==3*self.L
        self.grid.update(updates)

    def measurement(self,x=10,y=9,height=2):
        # 2x2 is the actual kagome measurement patch. A 2x1 horizontal-bond
        # closure is also reported to make the pedagogical square example clear.
        top=y-1;bottom=y+height;left=x-1;right=x+2
        keys=[('C',left,top,1),('C',right,top,2),
              ('C',right,bottom,3),('C',left,bottom,4)]
        keys += [('T',xx,top,1) for xx in (x,x+1)]
        keys += [('T',right,yy,2) for yy in range(y,y+height)]
        keys += [('T',xx,bottom,3) for xx in (x,x+1)]
        keys += [('T',left,yy,4) for yy in range(y,y+height)]
        roots=[self.grid[key] for key in keys]
        roots += [self.layers[xx,yy] for xx in (x,x+1) for yy in range(y,y+height)]
        return keys,roots

    def ancestors(self,roots,include_projectors=True):
        seen=set();todo=list(roots)
        while todo:
            n=todo.pop()
            if n in seen:continue
            seen.add(n)
            if self.nodes[n]['kind']=='projector' and not include_projectors:continue
            todo.extend(self.nodes[n]['parents'])
        return seen

    def source_path(self,root,source=(6,6)):
        # One physical-absorption path; projector inputs treated as fixed.
        target=self.layers[source]
        def find(n):
            if n==target:return [n]
            if self.nodes[n]['kind']=='projector':return None
            for p in self.nodes[n]['parents']:
                ans=find(p)
                if ans is not None:return [n]+ans
            return None
        return find(root)

    def signatures(self, roots):
        """Structural replay check, not a numerical tensor comparison."""
        signatures=[]
        for n in self.nodes:
            label=n['label']
            if n['kind']=='projector':label=('projectors',label[-1])
            move=None if n['move'] is None else self.moves[n['move']]
            if move is not None:move={k:v for k,v in move.items() if k!='id'}
            payload=(label,move,[signatures[p] for p in n['parents']])
            signatures.append(hashlib.sha256(json.dumps(payload).encode()).hexdigest())
        return [signatures[r] for r in roots]


def run(sweeps=2,kagome_warm=False,height=2):
    t=Trace()
    t.move(7,3,0)
    if kagome_warm:t.move(6,4,0)
    for sweep in range(1,sweeps+1):
        for d,js in ((1,range(5,12)),(3,range(8,1,-1)),
                     (2,range(5,12)),(4,range(7,1,-1))):
            for j in js:t.move(j,d,sweep)
    keys,roots=t.measurement(height=height)
    seen=t.ancestors(roots)
    mids={t.nodes[n]['move'] for n in seen if t.nodes[n]['move'] is not None}
    replay=Trace()
    for m in t.moves:
        if m['id'] in mids:replay.move(m['j'],m['d'],m['sweep'])
    _,replay_roots=replay.measurement(height=height)
    assert t.signatures(roots)==replay.signatures(replay_roots)
    groups=defaultdict(list)
    for m in t.moves:
        if m['id'] in mids:groups[m['sweep'],m['d']].append(m['j'])
    result=dict(sweeps=sweeps,kagome_warm=kagome_warm,patch_height=height,
                total_moves=len(t.moves),ancestor_moves=len(mids),
                pruned_replay_same_expression=True,
                groups={f'sweep{s}_d{d}':v for (s,d),v in groups.items()},
                last_writers=[],physical_paths={})
    for key in keys:
        n=t.grid[key];m=t.nodes[n]['move']
        result['last_writers'].append(dict(tensor=key,move=None if m is None else t.moves[m]))
        path=t.source_path(n)
        if path is not None:
            result['physical_paths'][str(key)]=[
                dict(tensor=t.nodes[p]['label'],move=None if t.nodes[p]['move'] is None else t.moves[t.nodes[p]['move']])
                for p in path[::-1]]
    return result


if __name__=='__main__':
    results=[run(sweeps=s,kagome_warm=k,height=h)
             for s in (1,2) for k in (False,True) for h in (1,2)]
    out=Path(__file__).with_name('ctm_dependency_trace.json')
    out.write_text(json.dumps(results,indent=2),encoding='utf-8')
    for r in results:
        print({k:v for k,v in r.items() if k not in ('last_writers','physical_paths')})
    r=results[5] # two paper sweeps, 2x2 patch
    print('LAST WRITERS',json.dumps(r['last_writers']))
    print('PATH to C1(9,8)',json.dumps(r['physical_paths'].get("('C', 9, 8, 1)")))
