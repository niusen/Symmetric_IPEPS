"""SERVER ENTRY: measure pre-exported dense states with Juraj; no Julia call."""
from pathlib import Path
from measure import run_plan

HERE = Path(__file__).resolve().parent

# All server options are here. Point this to YOUR existing multicell Juraj checkout.
JURAJ_REPOSITORY = Path.home() / 'python_code' / 'tn-torch_dev'
SETTINGS = dict(
    states=[
        dict(D=9, state_json=str(HERE / 'states' / 'D9.json'),
             julia_reference=str(HERE / 'states' / 'D9_julia_chi40.toml')),
        dict(D=10, state_json=str(HERE / 'states' / 'D10.json'),
             julia_reference=str(HERE / 'states' / 'D10_julia_chi40.toml')),
    ],
    chis=[40],                # Same chi as the bundled Julia reference.
    h=0.38,                   # H=Hex+h*sum(Sz); J=1.
    threads=10,
    device='cpu',             # Optional GPU: 'cuda:1'.
    ctm_iterations=50,
    ctm_tolerance=1e-6,
    projector_cutoff=1e-12,
    juraj_repository=str(JURAJ_REPOSITORY),
    out=str(HERE / 'results'),
)

if __name__ == '__main__':
    run_plan(SETTINGS)
