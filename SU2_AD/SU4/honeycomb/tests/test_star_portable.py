"""Verify a normal exported run directory outside the repository, without evolving."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    run = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('star_exporter',run/'server/prepare_star_run.py')
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    julia = os.environ.get('JULIA_BIN') or shutil.which('julia')
    assert julia, 'Set JULIA_BIN to test the directory'
    with tempfile.TemporaryDirectory(prefix='star_portable_') as directory:
        root = Path(directory)
        portable = packager.prepare_star_run(root/'my_calculation')
        assert (portable/'src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl').is_file()
        try:
            packager.prepare_star_run(portable)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing run directories must not be overwritten')
        env = os.environ.copy()
        for key in ('SU4_SRC_DIR','SU2_AD_ROOT','SU4_STAR_ENTRY'):
            env.pop(key,None)
        result = subprocess.run([julia,'--startup-file=no','server/run_star_simple_update.jl','--dry-run'],
            cwd=portable,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        assert result.returncode==0,result.stderr
        assert 'automatic CTMRG: true' in result.stdout
        bash_path=Path('C:/Program Files/Git/bin/bash.exe')
        bash=str(bash_path) if bash_path.exists() else shutil.which('bash')
        assert bash, 'bash is needed to test the portable sh entry'
        env['JULIA_BIN']=str(julia)
        result=subprocess.run([bash,'-c',
            'PATH="/usr/bin:/bin:$PATH"; export PATH; exec sh run_star_su.sh --dry-run'],
            cwd=portable,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        assert result.returncode==0,result.stderr
        assert 'automatic CTMRG: true' in result.stdout
        # Load all library dependencies from the copied folder, not the original checkout.
        code = ('include("src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl");'
                '@assert isdefined(HoneycombStarRun,:run_star_server);'
                '@assert isdefined(HoneycombStarRun,:measure_star_server);'
                'println("Portable library: OK")')
        result = subprocess.run([julia,'--startup-file=no','-e',code],cwd=portable,env=env,
            capture_output=True,text=True,encoding='utf-8',timeout=180)
        assert result.returncode==0,result.stderr
        print(result.stdout.strip())
    print('PASS: isolated portable folder, automatic source/config lookup, complete source dependency load')


if __name__=='__main__':
    main()
