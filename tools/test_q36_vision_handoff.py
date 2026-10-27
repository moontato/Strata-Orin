"""Execute ONLY the documented shell against FAKE service/cgroup/health commands.
No real services, CUDA, encoder or model operation. Fixtures remain temporary.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
MOCK='''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
r=Path(os.environ['MOCK_ROOT']);mode=os.environ['MOCK_MODE'];args=sys.argv[1:];name=Path(sys.argv[0]).name

def event(text):
 with (r/'events').open('a') as f:f.write(text+'\\n')
if name=='pgrep':sys.exit(1)
if name=='curl':
 after=(r/'ran').exists();event('health-after' if after else 'health-before')
 print(json.dumps(dict(service='strata',status='ok',loaded=not(after and mode=='health-fail'))));sys.exit(0)
if name=='systemd-run':
 assert (r/'strata').read_text()=='inactive' and (r/'llama').read_text()!='active'
 event('run-test');(r/'test').touch();(r/'ran').touch()
 unit=next(x.split('=',1)[1] for x in args if x.startswith('--unit='))
 if mode!='missing-cgroup-record':Path(os.environ['CGRECORD']).write_text('0::/mock-vision-review/'+unit+'.service\\n')
 sys.exit(3 if mode=='test-fail' else 143 if mode=='cancel' else 0)
args=[x for x in args if x!='--user'];cmd=args[0];unit=next(x for x in args if x.endswith('.service'))
test=unit.startswith('strata-vision-review-');key='llama' if unit=='llama-server.service' else 'strata'
if cmd=='show':
 if 'KillMode' in args:print('control-group')
 elif 'LoadState' in args:print('loaded' if not test or (r/'test').exists() else 'not-found')
 elif 'ActiveState' in args:print((r/key).read_text())
elif cmd=='is-active':
 active=(r/key).read_text()=='active'
 if '--quiet' not in args:print('active' if active else 'inactive')
 sys.exit(0 if active else 3)
elif cmd=='stop':
 event('stop-test' if test else 'stop-'+key)
 if test:
  if mode=='cleanup-fail':sys.exit(1)
  if (r/'test').exists():(r/'test').unlink()
 else:(r/key).write_text('inactive')
elif cmd=='start':
 assert not (r/'test').exists()
 event('start-'+key);(r/key).write_text('active')
else:raise AssertionError(args)
'''
class Handoff(unittest.TestCase):
    def run_window(self,mode='ok',llama=True):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'bin').mkdir();(root/'stage').mkdir()
            for arm in ('baseline','feature'):
                target=root/('build-'+arm+'-vision');target.mkdir();exe=target/'strata-q36';exe.write_text('MOCK --kv-vision-test');exe.chmod(0o700)
            target=root/'build-vision-encoder/bin';target.mkdir(parents=True);exe=target/'strata-vision';exe.touch();exe.chmod(0o700)
            for name in ('model','mmproj'): (root/name).touch()
            (root/'strata').write_text('active');(root/'llama').write_text('active' if llama else 'inactive')
            for name in ('systemctl','systemd-run','curl','pgrep'):
                path=root/'bin'/name;path.write_text(MOCK);path.chmod(0o700)
            source=re.findall(r'```bash\n(.*?)```',(ROOT/'docs/Q36_VISION_PREALLOCATION.md').read_text(),re.S)[-1]
            source=source.replace('SCOPE=system # confirmed actual scope: system OR user, not a guess','SCOPE=user # MOCK ONLY')
            source=source.replace('test -f /sys/fs/cgroup/cgroup.controllers',': # MOCK host preflight')
            source=source.replace('seq 1 90','seq 1 1').replace('sleep 2','sleep 0')
            script=root/'window.sh';script.write_text(source)
            env=dict(os.environ,MOCK_ROOT=str(root),MOCK_MODE=mode,TEST=str(root),STAGE=str(root/'stage'),MODEL=str(root/'model'),MMPROJ=str(root/'mmproj'),
                     PYTHON=sys.executable,VISION_STOP_APPROVED='1',PATH=str(root/'bin')+':'+os.environ['PATH'])
            run=subprocess.run(['bash',str(script)],env=env,capture_output=True,timeout=30)
            events=(root/'events').read_text().splitlines() if (root/'events').exists() else []
            return run,events,(root/'strata').read_text(),(root/'llama').read_text()

    def test_both_services_and_entire_test_reaped_before_prior_states_restored(self):
        run,events,strata,llama=self.run_window()
        self.assertEqual(run.returncode,0,run.stderr.decode());self.assertEqual((strata,llama),('active','active'))
        self.assertLess(events.index('stop-strata'),events.index('run-test'));self.assertLess(events.index('stop-llama'),events.index('run-test'))
        self.assertLess(events.index('stop-test'),events.index('start-strata'));self.assertLess(events.index('health-after'),events.index('start-llama'))
        self.assertEqual(events[-1],'health-after')

    def test_initially_inactive_llama_not_started(self):
        run,events,strata,llama=self.run_window(llama=False)
        self.assertEqual(run.returncode,0,run.stderr.decode());self.assertNotIn('start-llama',events);self.assertEqual((strata,llama),('active','inactive'))

    def test_failure_or_cancel_nonzero_survives_restoration(self):
        for mode in ('test-fail','cancel'):
            run,events,strata,llama=self.run_window(mode)
            self.assertNotEqual(run.returncode,0);self.assertEqual((strata,llama),('active','active'));self.assertLess(events.index('stop-test'),events.index('start-strata'))

    def test_unproven_cgroup_cleanup_blocks_production_overlap(self):
        for mode in ('cleanup-fail','missing-cgroup-record'):
            run,events,_,_=self.run_window(mode);self.assertNotEqual(run.returncode,0)
            self.assertNotIn('start-strata',events);self.assertNotIn('start-llama',events)

    def test_failed_loaded_health_keeps_failure_but_restores_llama(self):
        run,events,_,llama=self.run_window('health-fail');self.assertNotEqual(run.returncode,0);self.assertIn('start-llama',events);self.assertEqual(llama,'active')

if __name__=='__main__':unittest.main()
