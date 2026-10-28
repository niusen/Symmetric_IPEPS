"""Bosonic U(1) Q/R double layers and observables; reuse existing CTM moves."""
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
import yastn
from ctmrg import Fermionic_CTMRG_unitcell_iPESS as core
from model.kagome_spin import enlarged_corner, close_corners, embedded_spins


def block_tensor(config, array, signatures, charges, total=0):
    """Pack labelled coefficients into genuine U(1) blocks, checking all entries."""
    array = torch.as_tensor(array, dtype=torch.complex128, device=config.default_device)
    labels = [sorted(set(map(int, q))) for q in charges]
    positions = [{q: [i for i, value in enumerate(axis) if value == q] for q in qs}
                 for axis, qs in zip(charges, labels)]
    legs = [yastn.Leg(config, s=s, t=qs, D=[len(pos[q]) for q in qs])
            for s, qs, pos in zip(signatures, labels, positions)]
    tensor = yastn.zeros(config=config, legs=legs, n=total)
    reconstructed = torch.zeros_like(array)
    for ts in tensor.get_blocks_charge():
        indices = [torch.tensor(pos[q], device=array.device) for pos, q in zip(positions, ts)]
        mesh = torch.meshgrid(*indices, indexing='ij')
        values = array[mesh]
        tensor.set_block(ts=ts, val=values)
        reconstructed[mesh] = values
    error = torch.linalg.vector_norm(array-reconstructed).item()
    scale = max(torch.linalg.vector_norm(array).item(), 1.)
    if error > 1e-12*scale:
        raise ValueError('nonzero coefficients violate supplied U(1) charge metadata')
    return tensor


class U1State:
    def __init__(self, data, device):
        if data['format'] != 'kagome_u1_portable_v1' or data['charge_unit'] != '2Sz':
            raise ValueError('not a supported U(1) kagome export')
        self.Lx, self.Ly = data['Lx'], data['Ly']
        self.device = torch.device(device)
        self.dtype = torch.complex128
        self.config = yastn.make_config(sym='U1', fermionic=False, backend='torch',
            default_dtype='complex128', default_device=str(self.device))
        self.site_map = data['site_map']
        self.unique = {}
        for rep, values in data['cells'].items():
            cell = {}
            for name, item in values.items():
                array = (np.asarray(item['real'])+1j*np.asarray(item['imag'])).reshape(item['dims'], order='F')
                tensor = block_tensor(self.config, array, item['signatures'],
                                      item['axis_charges'], item['total_charge'])
                cell[name] = tensor/tensor.norm()
            self.unique[rep] = cell
        self.inequivalent_cells = self.unique
        self.cells = OrderedDict((self.key(x,y), self.unique[self.site_map[self.key(x,y)]])
                                 for x in range(1,self.Lx+1) for y in range(1,self.Ly+1))
        self.D = max(t.get_shape()[i] for c in self.unique.values() for name,t in c.items()
                     for i in range(2 if name.startswith('B') else 3))
        self.factors = {}
        for rep,c in self.unique.items():
            Q = yastn.ncon([c['B1'],c['B2'],c['B3'],c['Tup']],
                [[-1,1,-4],[-3,2,-5],[-2,3,-6],[1,2,3]])
            self.factors[rep] = Q,c['Tdn'].transpose(axes=(2,0,1))

    def key(self,x,y):
        return f'{1+(x-1)%self.Lx},{1+(y-1)%self.Ly}'

    def factor(self,x,y):
        return self.factors[self.site_map[self.key(x,y)]]


def double_q(Q, operator=None):
    ket = Q if operator is None else yastn.ncon([operator,Q],
        [[-4,-5,-6,1,2,3],[-1,-2,-3,1,2,3]])
    layer = yastn.ncon([Q.conj(),ket],
                      [[-1,-5,-3,1,2,3],[-2,-6,-4,1,2,3]])
    return layer.fuse_legs(((0,1),(2,3),(4,5)),mode='hard')


def double_r(R):
    layer = yastn.ncon([R.conj(),R],[[-1,-3,-5],[-2,-4,-6]])
    return layer.fuse_legs(((0,1),(2,3),(4,5)),mode='hard')


def ctmrg_u1(state, settings):
    unique = {r:(double_q(Q),double_r(R)) for r,(Q,R) in state.factors.items()}
    left = OrderedDict((k,unique[r][0]) for k,r in state.site_map.items())
    right = OrderedDict((k,unique[r][1]) for k,r in state.site_map.items())
    corners,edges = OrderedDict(),OrderedDict()
    for key in state.cells:
        corners[key],edges[key] = core.init_CTM_swap(right[key],left[key])
    args = SimpleNamespace(Lx=state.Lx,Ly=state.Ly,device=str(state.device))
    previous = {}
    error = float('inf')
    for iteration in range(1,settings.CTM_ite_nums+1):
        for direction in (3,4,1,2):
            corners,edges = core.CTM_ite_cell_continuous_update(
                corners,edges,left,right,direction,settings,args)
        current,differences = {},[]
        for key in state.cells:
            for label,C in corners[key].items():
                spectrum = torch.linalg.svdvals(C.to_dense())
                spectrum = spectrum/spectrum[0]
                index = key,label
                if index in previous:
                    old = previous[index]
                    size = max(old.numel(),spectrum.numel())
                    diff = torch.zeros(size,device=spectrum.device,dtype=spectrum.dtype)
                    diff[:old.numel()] -= old
                    diff[:spectrum.numel()] += spectrum
                    differences.append(diff.norm().item())
                current[index] = spectrum.clone()
        error = max(differences) if differences else float('inf')
        previous = current
        if settings.CTM_ite_info:
            print(f'U1 CTMRG {iteration}: corner-spectrum error={error:.3e}',flush=True)
        if error < settings.CTM_conv_tol:
            break
    return {'Cset':corners,'Tset':edges},left,right,dict(
        iterations=iteration,error=error,converged=error<settings.CTM_conv_tol)


def u1_energy(state,env,left,right,h):
    spins = embedded_spins(state.dtype,state.device)
    heis = sum(spins[a][k]@spins[b][k] for a,b in [(0,1),(1,2),(2,0)] for k in range(3))
    labels = [[1,-1]]*6  # Canonical physical order, independently packed by charge.
    def operator(value,charge=0):
        return block_tensor(state.config,value.reshape([2]*6),[-1]*3+[1]*3,labels,charge)
    up = operator(heis)
    z = [operator(s[2]) for s in spins]
    plus = [operator(s[0]+1j*s[1],-2) for s in spins]
    minus = [operator(s[0]-1j*s[1],2) for s in spins]
    field = operator(h*sum(s[2] for s in spins))
    terms = []
    for x in range(1,state.Lx+1):
        for y in range(1,state.Ly+1):
            positions = [(x,y),(x+1,y),(x,y+1),(x+1,y+1)]
            names = ('LU','RU','LD','RD')
            factors = [(left[state.key(*p)],right[state.key(*p)]) for p in positions]
            corners = [enlarged_corner(env,*pair,name,x-1,y-1,state)
                       for pair,name in zip(factors,names)]
            norm = close_corners(corners)
            def inserted(index,op):
                Q,_ = state.factor(*positions[index])
                return enlarged_corner(env,double_q(Q,op),factors[index][1],names[index],x-1,y-1,state)
            def expectation(replacements):
                return close_corners([replacements.get(i,c) for i,c in enumerate(corners)])/norm
            e_up = expectation({0:inserted(0,up)}).real
            e_field = expectation({0:inserted(0,field)}).real
            # U(1) Sx/Sy individually contain two charges. Use Sz and S+/-.
            specs = ((0,1),(2,2),(1,0)) # B2(LU), B3(LD), B1(RU)
            local = [(inserted(i,z[s]),inserted(i,plus[s]),inserted(i,minus[s])) for i,s in specs]
            e_down = norm.real*0
            for a,b in ((0,1),(1,2),(2,0)):
                ca,cb = specs[a][0],specs[b][0]
                e_down += expectation({ca:local[a][0],cb:local[b][0]}).real
                e_down += .5*expectation({ca:local[a][1],cb:local[b][2]}).real
                e_down += .5*expectation({ca:local[a][2],cb:local[b][1]}).real
            mz = torch.stack([expectation({0:inserted(0,op)}).real for op in z])
            terms.append((e_up,e_down,e_field,mz))
    upvals = torch.stack([t[0] for t in terms]).reshape(state.Lx,state.Ly)
    downvals = torch.stack([t[1] for t in terms]).reshape(state.Lx,state.Ly)
    fields = torch.stack([t[2] for t in terms]).reshape(state.Lx,state.Ly)
    mz = torch.stack([t[3] for t in terms]).reshape(state.Lx,state.Ly,3)
    return (upvals+downvals+fields).mean()/3,dict(E_up=upvals,E_down=downvals,E_field=fields,Sz=mz)
