"""Prepare a normal hierarchical run directory with its Julia dependencies.
Python is used only to copy the files; running the directory needs Julia/sh.
"""
import argparse
from pathlib import Path
import re
import shutil


def prepare_star_run(output=None):
    run = Path(__file__).resolve().parents[1]
    project = run.parents[1]
    source = project / 'src'
    pending = [source / 'bosonic/honeycomb/star/StarSimpleUpdateRun.jl']
    sources = set()
    while pending:
        file = pending.pop().resolve()
        file.relative_to(source.resolve())  # Reject dependencies outside the source tree.
        if file in sources:
            continue
        if not file.is_file():
            raise FileNotFoundError(file)
        sources.add(file)
        for name in re.findall(r'^\s*include\("([^"\n]+)"\)',
                               file.read_text(encoding='utf-8'),re.MULTILINE):
            pending.append(file.parent / name)
    output = (Path(output) if output else run / 'deploy/su4_star_run').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite an existing run directory: {output}. Use --output for a new directory.')
    if source.resolve() in output.parents or output == source.resolve():
        raise ValueError('The exported run directory must be outside the maintained src tree')
    files = {Path('src') / file.relative_to(source.resolve()):file for file in sources}
    files[Path('run_star_su.sh')] = run / 'run_star_su.sh'
    files[Path('server/run_star_simple_update.jl')] = run / 'server/run_star_simple_update.jl'
    for file in (run / 'config').glob('*.toml'):
        files[Path('config') / file.name] = file
    output.mkdir(parents=True,exist_ok=True)
    for relative,file in sorted(files.items()):
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target)
    (output/'README.txt').write_text(
            'Copy this directory anywhere. From inside it: sh run_star_su.sh\n'
            'Edit config/StarSimpleUpdate.toml for Dmax, schedule and CTMRG.\n'
            'One background Julia job runs SU and then CTMRG observables.\n'
            'Repeat the same sh command to resume. Logs: logs/. Checkpoints: server_runs/.\n'
            'Use the existing compatible Julia/TensorKit/SUNRepresentations environment.\n'
            'JULIA_BIN and JULIA_PROJECT are optional environment overrides.\n',encoding='utf-8')
    print(f'Prepared directory {output}: {len(files)} code/config files, {len(sources)} Julia source dependencies')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    prepare_star_run(parser.parse_args().output)
