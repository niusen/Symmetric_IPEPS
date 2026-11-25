from pathlib import Path
import tomllib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]/'results'/'flux_D4'
if len(sys.argv)>1: root=root/sys.argv[1]
d=tomllib.loads((root/'comparison.toml').read_text())
ms=d['methods']; colors=['#4c78a8','#f58518','#54a24b']
fig,grid=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
axes=grid.ravel()
axes[0].bar(['SU','FU','AD'],[ms[m]['energy_error'] for m in ('su','fu','ad')],color=colors)
axes[0].set_ylabel(r'$E-E_{exact}$ per site')
axes[0].set_title('Error against the exact filled band')
h=ms['ad']['history']
axes[1].plot([r['iteration'] for r in h],[r['energy']-ms['su']['energy'] for r in h],color=colors[2],label='AD objective')
for r in ms['ad']['budget_comparison']:
    axes[1].scatter(r['iteration'],r['energy']-ms['su']['energy'],marker='x',color='black')
axes[1].scatter(ms['ad']['iterations'],ms['ad']['energy']-ms['su']['energy'],marker='x',color='black',label='Strict CTM measurement')
axes[1].axhline(0,ls='--',lw=1,color=colors[0],label='SU')
axes[1].set_xlabel('L-BFGS iteration'); axes[1].set_ylabel('Energy change from SU per site')
axes[1].set_title('AD: '+ms['ad']['status'].replace('_',' ')); axes[1].legend(frameon=False,fontsize=8)
stages=ms['fu']['stages']
tau=np.cumsum([r['tau'] for r in stages])
axes[2].plot(np.r_[0,tau],np.r_[0,[r['energy']-ms['su']['energy'] for r in stages]],
             '-o',color=colors[1],label='FU stage measurement')
axes[2].scatter(tau[-1],ms['fu']['energy']-ms['su']['energy'],marker='x',color='black',label='Strict CTM measurement')
if 'best_stage_measurement' in ms['fu']:
    idx=ms['fu']['best_stage_selection']['stage']-1
    axes[2].scatter(tau[idx],ms['fu']['best_stage_measurement']['energy']-ms['su']['energy'],marker='x',color='black')
axes[2].axhline(0,ls='--',lw=1,color=colors[0],label='SU')
axes[2].set_xlabel('Imaginary time after SU'); axes[2].set_ylabel('Energy change from SU per site')
axes[2].set_title('FU continuation'); axes[2].legend(frameon=False,fontsize=8)
for j,m in enumerate(('su','fu','ad')):
    axes[3].bar(np.arange(3)+(j-1)*.25,ms[m]['site_densities'],width=.24,color=colors[j],label=m.upper())
axes[3].axhline(2/3,color='black',ls='--',lw=1,label='Exact')
axes[3].set_xticks([0,1,2],['a','b','c']); axes[3].set_ylabel('Site density')
axes[3].set_ylim(0,.8)
axes[3].set_title('Sublattice nonuniformity'); axes[3].legend(loc='upper center',ncol=4,frameon=False,fontsize=8)
for ax in axes: ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.suptitle(r'Kagome inherited Hofstadter flux: $D=4$, $U=0$, $\mu=-\sqrt{3}/2$, two spins'
             +'\nInitial seed '+str(d['seed']))
fig.savefig(root/'comparison.png',dpi=180)
print(root/'comparison.png')
