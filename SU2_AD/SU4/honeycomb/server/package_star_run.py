"""Build a movable run folder with only its transitive Julia source dependencies.
Python is used to package locally; running the archive requires only Julia/sh.
"""
import argparse
from pathlib import Path
import re
import zipfile


def package_star_run(output=None):
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
    output = Path(output) if output else run / 'dist/su4_star_run.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    files = {Path('src') / file.relative_to(source.resolve()):file for file in sources}
    files[Path('run_star_su.sh')] = run / 'run_star_su.sh'
    files[Path('server/run_star_simple_update.jl')] = run / 'server/run_star_simple_update.jl'
    for file in (run / 'config').glob('*.toml'):
        files[Path('config') / file.name] = file
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative,file in sorted(files.items()):
            info = zipfile.ZipInfo('su4_star_run/' + relative.as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if file.suffix == '.sh' else 0o100644) << 16
            archive.writestr(info,file.read_bytes())
        archive.writestr('su4_star_run/README.txt',
            'Unpack anywhere; cd su4_star_run; sh run_star_su.sh\n'
            'Edit config/StarSimpleUpdate.toml for Dmax, schedule and CTMRG.\n'
            'One background Julia job runs SU and then CTMRG observables.\n'
            'Repeat the same sh command to resume. Logs: logs/. Checkpoints: server_runs/.\n'
            'Use the existing compatible Julia/TensorKit/SUNRepresentations environment.\n'
            'JULIA_BIN and JULIA_PROJECT are optional environment overrides.\n')
    print(f'Created {output}: {len(files)} code/config files, {len(sources)} Julia source dependencies')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    package_star_run(parser.parse_args().output)
