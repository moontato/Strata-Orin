"""Focused integration contracts; no GPU, native binary, or production service."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import q36_image_http_smoke as h
from tools import q36_image_http_geometry as g

SOLO = ('strata serve: session 0: prompt 12 tokens = 8 reused + 4 read in 3 ms '
        '(1.3 tok/s), 2 generated in 5 ms (4.0 tok/s), drafts accepted 0 of 0 (length)\n')
BATCH = 'strata batch: slot 1 takes a prompt of 13 tokens = 0 reused + 13 read in 4 ms (3.0 tok/s), done at once\n'
CACHE = 'KVCACHE session=0 generation=2 epoch=2 state=active tokens=8 bytes=100 reserved_tokens=13 result=checkpoint replay=0\n'
RESTORE = 'KVREPLAY session=0 generation=2 epoch=2 tokens=8 reason=prefix_checkpoint\n'


def answer(step='s', prompt=12):
    return dict(step=step, usage=dict(prompt_tokens=prompt))


def test_real_worker_uses_only_spans():
    import inspect
    code = inspect.getsource(h.worker)
    assert 'attribute_spans(' in code and 'correlate(' not in code


def test_request_local_restore_is_proven_only_before_its_summary():
    a = h.audit_log_span((RESTORE+CACHE+SOLO).encode(), [answer()])
    assert a['exact'] and a['restored'] is True and a['result'] == 'checkpoint'
    a = h.audit_log_span(SOLO.encode(), [answer()])
    assert a['exact'] and a['restored'] is False
    assert not h.audit_log_span((SOLO+RESTORE).encode(), [answer()])['exact']


def test_actual_http_stderr_restore_needs_no_stdout_protocol_lines():
    raw=('Qwen3.6 prefix checkpoint: source=2 session=0 restored_tokens=8\n'+SOLO).encode()
    a=h.audit_log_span(raw,[answer()])
    assert a['exact'] and a['restored'] and a['result'] is None
    assert a['events'][0]['source']==2
    wrong=raw.replace(b'restored_tokens=8',b'restored_tokens=4')
    assert not h.audit_log_span(wrong,[answer()])['restored']


@pytest.mark.parametrize('raw', [SOLO+SOLO, SOLO[:-1], SOLO+RESTORE,
                                CACHE+CACHE+SOLO, RESTORE.replace('session=0','session=2')+SOLO])
def test_ambiguous_spans_fail_closed(raw):
    assert not h.audit_log_span(raw.encode(), [answer()])['exact']


def test_burst_checks_usage_multiset_without_client_mapping():
    a = h.audit_log_span((SOLO+BATCH).encode(), [answer('b',13),answer('a',12)])
    assert a['exact'] and not a['attributed'] and a['sessions'] == [0,1]
    assert 'restored' not in a
    assert not h.audit_log_span((SOLO+BATCH).encode(), [answer('a',12),answer('b',12)])['exact']


def test_later_same_slot_event_cannot_reach_an_earlier_span(tmp_path):
    first=SOLO.encode();second=(RESTORE+CACHE+SOLO).encode()
    windows=[dict(steps=['a'],log_start=0,log_end=len(first)),
             dict(steps=['b'],log_start=len(first),log_end=len(first+second))]
    rows,_,_,_=h.attribute_spans(windows,[answer('a'),answer('b')],first+second,tmp_path,'arm')
    assert rows[0]['restored'] is False and rows[1]['restored'] is True
    with pytest.raises(RuntimeError,match='outside bracketed'):
        h.attribute_spans(windows,[answer('a'),answer('b')],first+second+BATCH.encode(),tmp_path,'arm')


def test_assistant_fields_including_empty_values_are_preserved():
    message=dict(role='assistant',content=None,reasoning_content='',tool_calls=[])
    conv=h.Conversation('m').user('u').reply(message)
    assert conv.render({})[-1] == message


def test_real_server_command_uses_config_filename(tmp_path):
    cfg=dict(exe='/qualified/strata-q36',args=[],vision={})
    path=tmp_path/'config-candidate_on.json'
    path.write_text(json.dumps(cfg))
    arm=h.Arm(tmp_path,'candidate_on',cfg,sys.executable,tmp_path,11501)
    assert arm.command[arm.command.index('--config')+1]==str(path)
    assert json.loads(Path(arm.command[arm.command.index('--config')+1]).read_text())==cfg
    assert arm.command[-1]=='11501'


def test_server_environment_does_not_inherit_production_settings(tmp_path,monkeypatch):
    cfg=dict(exe='/qualified/strata-q36',args=[],vision={})
    (tmp_path/'config-candidate_on.json').write_text(json.dumps(cfg))
    arm=h.Arm(tmp_path,'candidate_on',cfg,sys.executable,tmp_path,11501)
    monkeypatch.setenv('STRATA_API_KEY','never-copy-this')
    monkeypatch.setenv('STRATA_OTHER_SETTING','unreviewed')
    calls=[]
    class Child:pid=123456789
    def popen(*args,**kwargs):calls.append(kwargs);return Child()
    monkeypatch.setattr(h.subprocess,'Popen',popen)
    arm.start()
    arm.stream.close()
    env=calls[0]['env']
    assert 'STRATA_API_KEY' not in env and 'STRATA_OTHER_SETTING' not in env
    assert env['STRATA_MMQ_KQ']=='1' and env['TMPDIR']==str(tmp_path)
    assert env['PYTHONDONTWRITEBYTECODE']=='1'


def test_arm_stop_does_not_implicitly_reap_before_group_cleanup(tmp_path,monkeypatch):
    cfg=dict(exe='/qualified/strata-q36',args=[],vision={})
    (tmp_path/'config-candidate_on.json').write_text(json.dumps(cfg))
    arm=h.Arm(tmp_path,'candidate_on',cfg,sys.executable,tmp_path,11501)
    calls=[]
    class Child:
        pid=123456789
        returncode=None
        def terminate(self):raise AssertionError('Popen.terminate may implicitly reap')
        def wait(self,timeout):calls.append('wait');self.returncode=0;return 0
    arm.child=Child();arm.stream=(tmp_path/'log').open('w')
    monkeypatch.setattr(h,'worker_status',lambda child:0)
    monkeypatch.setattr(h,'group_probe',lambda child:'empty' if 'wait' in calls else 'present')
    monkeypatch.setattr(h.os,'kill',lambda pid,sig:calls.append('term'))
    monkeypatch.setattr(h,'group_kill',lambda child:calls.append('killpg'))
    result=arm.stop(time.monotonic()+1)
    assert result['owned_group_empty'] and calls==['term','killpg','wait']


def test_arm_stop_never_signals_an_already_reaped_leader(tmp_path,monkeypatch):
    cfg=dict(exe='/qualified/strata-q36',args=[],vision={})
    (tmp_path/'config-candidate_on.json').write_text(json.dumps(cfg))
    arm=h.Arm(tmp_path,'candidate_on',cfg,sys.executable,tmp_path,11501)
    class Child:pid=123456789;returncode=0
    arm.child=Child();arm.stream=(tmp_path/'log').open('w')
    def forbidden(*a):raise AssertionError('signal after reaping')
    monkeypatch.setattr(h.os,'kill',forbidden)
    monkeypatch.setattr(h,'group_kill',forbidden)
    monkeypatch.setattr(h,'group_probe',lambda child:'present')
    result=arm.stop(time.monotonic()+1)
    assert result['worker_reaped'] and not result['owned_group_empty']


def test_after_image_geometry_uses_real_saved_callbacks():
    # Same boundaries/length/LCP as the already-qualified public Orin fixture.
    a=[7]*1088+[248056]*1024+[9]*1596
    b=a.copy();b[3671]=10
    c=a.copy();c[3671]=11
    audit=g.audit_ids([a,b,c],1088,1024)
    assert max(audit['rewrites'][0]['available'])==3584
    assert audit['rewrites'][0]['lcp']==3671
    assert 2112 not in audit['initial_retained_checkpoints']


def test_short_missing_legal_checkpoint_cannot_be_a_positive_case():
    a=[7]*20+[248056]*1024+[9]*16
    b=a+[5];b[1056]=8
    with pytest.raises(ValueError,match='no retained legal'):
        g.audit_ids([a,b,b],20,1024)


def test_cpu_preparation_never_launches_encoder_or_model():
    tree=ast.parse(Path(g.__file__).read_text())
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('Popen','run','system','execv')
                   for n in ast.walk(tree))


def test_load_rejects_changed_body_and_unaudited_input(tmp_path):
    path=tmp_path/'geometry.json';path.write_text('{}')
    with pytest.raises(ValueError,match='CPU actual-GGUF'):
        g.load(path,tmp_path/'model',tmp_path/'engine',tmp_path/'config',tmp_path)


def test_leader_missing_live_group_is_never_declared_empty(tmp_path,monkeypatch):
    (tmp_path/'owned-groups.jsonl').write_text(json.dumps(dict(pid=123456789,starttime='1',arm='a'))+'\n')
    probes=[]
    def probe(pid,sig):probes.append((pid,sig))
    monkeypatch.setattr(h.os,'killpg',probe)
    result=h.reap_recorded_groups(tmp_path,time.monotonic()+.1)
    assert probes==[(123456789,0)]
    assert not result[0]['group_empty'] and result[0]['operator_cgroup_required']
