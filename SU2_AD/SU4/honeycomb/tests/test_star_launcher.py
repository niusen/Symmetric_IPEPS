"""Test sh process/argument/log routing with a fake Julia; no TN calculation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def shell_path(path):
    path = Path(path).resolve()
    value = path.as_posix()
    return '/' + value[0].lower() + value[2:] if path.drive else value


def main():
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if git_bash.exists() else shutil.which('bash')
    assert bash, 'bash is needed for this launcher test'
    launcher = Path(__file__).resolve().parents[1] / 'run_star_su.sh'
    with tempfile.TemporaryDirectory(prefix='star_launcher_') as directory:
        folder = Path(directory)
        binary = folder / 'bin'
        binary.mkdir()
        fake = binary / 'fake_julia'
        fake.write_bytes(b'#!/bin/sh\nprintf "%s\\n" "$@"\n')
        # Git Bash lacks setsid; the shim tests routing only, not setsid itself.
        (binary / 'setsid').write_bytes(b'#!/bin/sh\nexec "$@"\n')
        subprocess.run([bash, '-c', 'chmod +x "$1"/*', 'test', shell_path(binary)], check=True)
        env = os.environ.copy()
        for key in ('SLURM_JOB_ID','PBS_JOBID','LSB_JOBID','SU4_STAR_FOREGROUND'):
            env.pop(key, None)
        env['JULIA_BIN'] = shell_path(fake)

        def run(args, name, extra=None):
            current = env.copy()
            current['SU4_STAR_LOG_DIR'] = name
            current.update(extra or {})
            result = subprocess.run([bash, '-c',
                'PATH="$1:/usr/bin:/bin:$PATH"; export PATH; shift; exec sh "$@"',
                'test', shell_path(binary), shell_path(launcher), *args],
                cwd=folder,env=current, text=True, encoding='utf-8', errors='replace',
                capture_output=True, timeout=15)
            assert result.returncode == 0, result.stderr
            return result

        result = run(['--config','plan with spaces.toml','--output','result folder'], 'background')
        assert 'Started PID ' in result.stdout
        outputs = list((folder / 'background').glob('*.out'))
        assert len(outputs) == 1
        for _ in range(100):
            lines = outputs[0].read_text().splitlines()
            if '--auto-resume' in lines:
                break
            time.sleep(.02)
        assert lines[0] == '--startup-file=no'
        assert lines[1].endswith('/run_star_simple_update.jl')
        assert lines[2:] == ['--auto-resume','--config','plan with spaces.toml','--output','result folder']
        assert list((folder / 'background').glob('*.pid'))[0].read_text().strip().isdigit()
        for args,name,extra in [(['--foreground','--resume'],'foreground',{}),
                                (['--dry-run'],'preview',{}),
                                (['--help'],'help',{}),
                                (['--resume'],'scheduler',{'SLURM_JOB_ID':'test'})]:
            result = run(args,name,extra)
            assert '--auto-resume' in result.stdout
            assert 'Started PID ' not in result.stdout
            assert '--foreground' not in result.stdout
            assert not (folder / name).exists()
    print('PASS: background logs/PID, quoted arguments, foreground, preview/help, scheduler routing')


if __name__ == '__main__':
    main()
