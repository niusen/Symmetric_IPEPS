"""Check a separate run directory using the existing shared src, without copies."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    run=Path(__file__).resolve().parents[1]
    shared=run.parents[1]/'src'
    julia=os.environ.get('JULIA_BIN') or shutil.which('julia')
    assert julia,'Set JULIA_BIN for the test'
    with tempfile.TemporaryDirectory(prefix='star_shared_') as directory:
        job=Path(directory)/'my_job'
        (job/'server').mkdir(parents=True)
        shutil.copy2(run/'run_star_su.sh',job/'run_star_su.sh')
        shutil.copy2(run/'server/run_star_simple_update.jl',job/'server/run_star_simple_update.jl')
        shutil.copytree(run/'config',job/'config')
        assert not (job/'src').exists()
        env=os.environ.copy()
        env.pop('SU2_AD_ROOT',None)
        env.pop('SU4_STAR_ENTRY',None)
        env['SU4_SRC_DIR']=str(shared)
        env['JULIA_BIN']=str(julia)
        result=subprocess.run([julia,'--startup-file=no','server/run_star_simple_update.jl','--dry-run'],
            cwd=job,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        assert result.returncode==0,result.stderr
        assert 'automatic CTMRG: true' in result.stdout
        git_bash=Path('C:/Program Files/Git/bin/bash.exe')
        bash=str(git_bash) if git_bash.exists() else shutil.which('bash')
        assert bash,'bash is needed for the launcher test'
        result=subprocess.run([bash,'-c',
            'PATH="/usr/bin:/bin:$PATH"; export PATH; exec sh run_star_su.sh --dry-run'],
            cwd=job,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        assert result.returncode==0,result.stderr
        code=('include("server/run_star_simple_update.jl");source=star_source_dir();'
              '@assert source==normpath(joinpath(ENV["SU4_SRC_DIR"],"bosonic/honeycomb/star"));'
              'include(joinpath(source,"StarSimpleUpdateRun.jl"));'
              '@assert isdefined(HoneycombStarRun,:run_star_server);'
              '@assert isdefined(HoneycombStarRun,:measure_star_server);'
              'println("Existing shared source: OK")')
        result=subprocess.run([julia,'--startup-file=no','-e',code],cwd=job,env=env,
            capture_output=True,text=True,encoding='utf-8',timeout=180)
        assert result.returncode==0,result.stderr
        assert not (job/'src').exists()
        print(result.stdout.strip())
    print('PASS: separate run directory uses the existing src; no source copy or generated library')


if __name__=='__main__':
    main()
