"""Executes a path-rewritten window COPY using ONLY local command shims.
No system manager, HTTP connection, model, production path or real service touched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]


SHIM = r'''import json,os,sys
from pathlib import Path
args=sys.argv[1:]; name=Path(sys.argv[0]).name
if name in ('systemctl','systemd-run'):args=[a for a in args if a!='--user']
record=Path(os.environ['MOCK_RECORD']); statefile=Path(os.environ['MOCK_STATE'])
with record.open('a') as f:f.write(json.dumps([name]+args)+'\n')
state=json.loads(statefile.read_text())
if name=='git':
    if args[:2]==['rev-parse','--show-toplevel']:print(os.environ['MOCK_TEST'])
elif name=='systemctl':
    action=args[0];unit=args[1]
    if action=='is-active':print(state.get(unit,'inactive'));sys.exit(0 if state.get(unit)=='active' else 3)
    if action=='show':
        prop=args[args.index('-p')+1] if '-p' in args else ''
        if prop=='LoadState':print('loaded' if unit in state else 'not-found')
        elif prop=='ActiveState':print(state.get(unit,'inactive'))
        elif prop=='KillMode':print('control-group')
        else:print('ActiveState='+state.get(unit,'inactive'))
    elif action in ('start','stop'):
        state[unit]='active' if action=='start' else 'inactive';statefile.write_text(json.dumps(state))
elif name=='systemd-run':
    unit=next(a.split('=',1)[1] for a in args if a.startswith('--unit='))
    cg=next(a for a in args if a.endswith('/cgroup.txt'))
    Path(cg).write_text('0::/nonexistent-CPU-MOCK/'+unit+'.service\n')
    sys.exit(int(os.environ.get('MOCK_WORKER_RC','0')))
elif name=='curl':
    started=any(json.loads(x)[:3]==['systemctl','start','strata.service'] for x in record.read_text().splitlines())
    images=not(os.environ.get('MOCK_VISION_AFTER_FALSE')=='1' and started)
    print(json.dumps(dict(service='strata',status='ok',loaded=True,images=images)))
elif name=='python-shim':
    if 'tools/q36_multimodal_cache_gate.py' in args: print('CPU_MOCK_PREFLIGHT_ONLY');sys.exit(0)
    if args[0]=='-c' and 'no_models' in args[1]:sys.exit(0)
    if args[0]=='-' and os.environ.get('MOCK_CGROUP_FAIL')=='1':sys.exit(1)
    os.execv(os.environ['REAL_PYTHON'],[os.environ['REAL_PYTHON']]+args)
'''


def setup(tmp_path):
    test=tmp_path/'review';test.mkdir();stage=tmp_path/'stage';stage.mkdir();bin=tmp_path/'bin';bin.mkdir()
    record=tmp_path/'commands.jsonl';state=tmp_path/'state.json'
    for name in ('git','systemctl','systemd-run','curl','python-shim'):
        path=bin/name;path.write_text('#!'+sys.executable+'\n'+SHIM);path.chmod(0o755)
    source=(ROOT/'tools/q36_multimodal_cache_window.sh').read_text()
    # Host cgroup availability is simulated; the production script keeps its check.
    source=source.replace('test -f /sys/fs/cgroup/cgroup.controllers', ': CPU_MOCK_CGROUP_V2')
    script=tmp_path/'window-copy.sh';script.write_text(source)
    env=dict(os.environ,PATH=str(bin)+':'+os.environ['PATH'],REAL_PYTHON=sys.executable,MOCK_RECORD=str(record),MOCK_STATE=str(state),MOCK_TEST=str(test),
             TEST=str(test),STAGE=str(stage),PYTHON=str(bin/'python-shim'),MM_CACHE_STOP_APPROVED='1',MM_CACHE_ENGINE=str(test/'build-mm-cache-mock/strata-q36'),SCOPE='user',HEALTH='CPU_MOCK_NO_HTTP',MODEL='CPU_MOCK_NOT_A_MODEL',FIXTURES=str(stage/'fixture'))
    return script,env,record,state


@pytest.mark.parametrize('llama,worker_rc',[('active','0'),('inactive','1')])
def test_mock_window_restores_both_exact_prior_states_after_success_or_failure(tmp_path,llama,worker_rc):
    script,env,record,state=setup(tmp_path);state.write_text(json.dumps({'strata.service':'active','llama-server.service':llama}))
    env['MOCK_WORKER_RC']=worker_rc
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==int(worker_rc),result.stderr
    assert json.loads(state.read_text())['strata.service']=='active'
    assert json.loads(state.read_text())['llama-server.service']==llama
    commands=[json.loads(x) for x in record.read_text().splitlines()]
    preflight=next(i for i,c in enumerate(commands) if '--preflight' in c)
    stop=next(i for i,c in enumerate(commands) if c[:3]==['systemctl','stop','strata.service'])
    assert preflight<stop
    run=next(c for c in commands if c[0]=='systemd-run')
    assert '--property=RuntimeMaxSec=300s' in run and '--property=KillMode=control-group' in run
    assert sum(c[:3]==['systemctl','start','llama-server.service'] for c in commands)==int(llama=='active')


def test_mock_restoration_does_not_pass_without_loaded_vision_health(tmp_path):
    script,env,record,state=setup(tmp_path);state.write_text(json.dumps({'strata.service':'active','llama-server.service':'active'}))
    env['MOCK_VISION_AFTER_FALSE']='1'
    script.write_text(script.read_text().replace('$(seq 1 90)', '$(seq 1 1)').replace('sleep 2', 'true'))
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0
    commands=[json.loads(x) for x in record.read_text().splitlines()]
    assert any(c[:3]==['systemctl','start','strata.service'] for c in commands)
    assert json.loads(state.read_text())['strata.service']=='active'  # active alone is NOT a successful recovery


def test_mock_cgroup_inspection_failure_blocks_production_restart(tmp_path):
    script,env,record,state=setup(tmp_path);state.write_text(json.dumps({'strata.service':'active','llama-server.service':'active'}))
    env['MOCK_CGROUP_FAIL']='1'
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0
    commands=[json.loads(x) for x in record.read_text().splitlines()]
    assert not any(c[:2]==['systemctl','start'] for c in commands)
    assert json.loads(state.read_text())['strata.service']=='inactive'
