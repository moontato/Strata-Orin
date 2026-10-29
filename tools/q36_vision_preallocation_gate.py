#!/usr/bin/env python3
"""Operator-approved isolated vision review worker. NO service/config operations.
Owns one encoder and at most one LLM process. Must run inside an operator-managed
transient cgroup AFTER production Strata/llama were stopped/reaped. All staging
and results stay under the configured test staging directory. No downloads or HTTP listener.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.q36_vision_fixture import make_images,prepare,sve


def no_models(exclude=()):
    names={'strata','strata-q36','strata-vision','llama-server','llama-cli','llama-bench'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in exclude:continue
        try:name=(p/'comm').read_text().strip()
        except FileNotFoundError:continue
        if name in names:raise RuntimeError('model/encoder still running outside this worker: '+p.name+' '+name)


def reap(process):
    if process is None:return
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=20)
    else:process.wait()


def exact(a,b):
    with Path(a).open('rb') as x,Path(b).open('rb') as y:
        while True:
            left=x.read(65536);right=y.read(65536)
            if left!=right:return False
            if not left:return True


def protocol(path):
    out=[];done=[]
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0]=='ERR' or parts[0]=='BERR':raise ValueError('native error: '+line)
        if parts[0] in ('T','BT','BADM'):out.append(parts)
        elif parts[0]=='BDONE':out.append(parts[:4])
        elif parts[0]=='DONE':
            done.append(parts);out.append(parts[:3]+parts[5:9]+parts[14:16]) # no timing fields
    if len(done)<2 or list(map(int,done[-2][6:8]))!=[0,0] or int(done[-1][7])<=0:
        raise ValueError('missing image-sampled exclusion / later text sampled-MTP offers')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',required=True);p.add_argument('--baseline',required=True);p.add_argument('--encoder',required=True)
    p.add_argument('--model',required=True);p.add_argument('--mmproj',required=True);p.add_argument('--out',required=True);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args()
    if os.environ.get('STRATA_VISION_REVIEW_WINDOW')!='1':p.error('an explicit isolated model-test window is required')
    out=Path(a.out).resolve();stage=Path(os.environ.get('STRATA_Q36_STAGE',ROOT/'logs/q36-tests')).resolve()
    if not out.is_relative_to(stage) or out==stage or out.exists():p.error('use a NEW results directory under the configured test stage')
    for binary in (a.engine,a.baseline):
        if b'--kv-vision-test' not in Path(binary).read_bytes():p.error('build diagnostic native binary first')
    no_models();out.mkdir(parents=True);make_images(out/'images')
    (out/'arguments.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    encoder=None;model=None;results={}
    def interrupted(signum,_frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (out/'encoder.stderr').open('w') as err:
            encoder=subprocess.Popen([a.encoder,'--mmproj',a.mmproj,'--model',a.model,'--gpu','--max-tokens','1024'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,start_new_session=True,cwd=out)
            replies=queue.Queue()
            def reader():
                for line in encoder.stdout:replies.put(line.rstrip())
                replies.put(None)
            threading.Thread(target=reader,daemon=True).start()
            def reply(prefix):
                deadline=time.monotonic()+a.timeout
                while True:
                    line=replies.get(timeout=max(0.01,deadline-time.monotonic()))
                    with (out/'encoder.stdout').open('a') as log:log.write(str(line)+'\n')
                    if line is None or line.startswith('ERR'):raise RuntimeError('encoder failed: '+str(line))
                    if line.startswith(prefix):return line
            if reply('READY ')!='READY 2048':raise RuntimeError('encoder width is not2048')
            for name in ('red-square','blue-wide'):
                target=out/(name+'.sve');encoder.stdin.write(f'ENC {out/"images"/(name+".png")} {target}\n');encoder.stdin.flush();reply('OK ');sve(target)
            manifest=prepare(a.model,out/'red-square.sve',out/'blue-wide.sve',out/'fixture')
            (out/'encoder-resident-status.txt').write_text(Path(f'/proc/{encoder.pid}/status').read_text()) # host RSS, NOT a GPU workspace measurement
            common=['--model',a.model,'--serve','--vision','--max-context','262144','--batch','4','--kv-pool-tokens','524288',
                    '--kv','int8','--mtp','2','--batch-mtp','--mtp-sampled','--chunk','512','--kv-checkpoint-mib','1024','--kv-checkpoint-interval','1024',
                    '--kv-vision-test',manifest['fixture']]
            streams={}
            for name,binary,extra in [('baseline',a.baseline,[]),('lazy',a.engine,[]),('preallocated',a.engine,['--kv-preallocate','--kv-vision-max-tokens','1024'])]:
                no_models({encoder.pid})
                dump=out/(name+'.bin');env=dict(os.environ,LD_LIBRARY_PATH=str(Path(binary).resolve().parent)+((':'+os.environ['LD_LIBRARY_PATH']) if os.environ.get('LD_LIBRARY_PATH') else ''))
                with (out/(name+'.log')).open('w') as log:
                    model=subprocess.Popen([binary,*common,*extra,'--test-state-dump',str(dump)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    rc=model.wait(timeout=a.timeout);model=None
                results[name]=rc;(out/(name+'.exit')).write_text(str(rc)+'\n')
                if rc:raise RuntimeError(name+' native gate failed; all artifacts retained, no mismatch waived')
                data=(out/(name+'.log')).read_text()
                if 'VISION_PREALLOCATION_MODEL PASS' not in data:raise RuntimeError('missing native pass marker')
                if name=='preallocated' and ('KVVISION fixed=1' not in data or 'KV_PREALLOCATION_MODEL PASS' not in data):raise RuntimeError('missing strict ancillary/bank accounting marker')
                streams[name]=protocol(out/(name+'.log'))
                tokens=[int(line.split()[-1]) for line in data.splitlines() if line.startswith(('T ','BT '))]
                from tools.strata_tokenizer import Tokenizer
                (out/(name+'-decoded-tokens.txt')).write_text(Tokenizer.from_gguf(a.model).decode(tokens))
            if not exact(out/'baseline.bin',out/'lazy.bin') or not exact(out/'baseline.bin',out/'preallocated.bin') or streams['baseline']!=streams['lazy'] or streams['baseline']!=streams['preallocated']:
                raise RuntimeError('baseline/lazy/preallocated raw state/logit or emitted-token protocol differs (EXACT required)')
            (out/'result.json').write_text(json.dumps(dict(passed=True,statuses=results,exact_state_and_logits=True,exact_tokens=True,
                semantic_image_review='PENDING_MANUAL',note='one fixture/config only; inspect decoded red/blue answers and individual bank diagnostics; not broad vision/sampler/performance qualification'),indent=2)+'\n')
            print('Q36_VISION_PREALLOCATION_ORIN PASS',out)
    except BaseException as error:
        (out/'result.json').write_text(json.dumps(dict(passed=False,statuses=results,error=str(error)),indent=2)+'\n')
        raise
    finally:
        # Only children owned by this test. Never kill production names/PIDs.
        signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:reap(model)
        finally:reap(encoder)

if __name__=='__main__':main()
