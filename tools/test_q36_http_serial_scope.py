"""Focused serial qualification: loopback stand-ins only, no GPU or services."""
import json
from pathlib import Path
import pytest
from tools import q36_image_http_smoke as h
from tools.test_q36_image_http_smoke import run_smoke


def test_scope_excludes_admission_preemption():
    steps=h.geometry_selftest_steps()+h.plan('marker')
    assert len(steps)==10
    assert not any(s['step'].startswith('interleave_four/') for s in steps)
    assert len({s['step'] for s in steps})==10
    assert 'concurrency=NOT_TESTED' in h.PASS_MARKER


def test_serial_three_arm_http_with_all_request_bodies(tmp_path):
    run,watch,report,arms,out=run_smoke(tmp_path,'normal')
    assert run.returncode==0,run.stdout+run.stderr
    assert watch['passed'] and watch['owned_group_empty'] and watch['server_groups_empty']
    assert report['scope']=='serial_image_http_10_requests_per_arm'
    assert report['concurrency']=='NOT_TESTED' and not report['required_failures']
    assert len(report['steps'])==10 and set(arms)==set(h.ARMS)
    for arm,rep in arms.items():
        assert len(rep['reuse'])==10
        assert all(r['attributed'] and r['group'] is None for r in rep['reuse'])
        checks={c['name']:c for c in rep['checks']}
        assert all(c['ok'] for c in checks.values() if c['required'])
        assert checks['serial_scope_complete']['ok'] and checks['serial_spans_only']['ok']
        assert len(list(out.glob(arm+'-*-request.json')))==10
        assert len(list(out.glob(arm+'-*-response.json')))==10
        assert len(rep['span_audits'])==10
        for span in rep['span_audits']:
            assert len(span['steps'])==1 and span['audit']['exact'] and span['audit']['attributed']
    assert {c['name']:c for c in arms['candidate_on']['checks']}['qualified_after_image_restore']['ok']
    assert not report['divergence']['baseline_vs_candidate_off']


@pytest.mark.parametrize('mode',['plan','preflight'])
def test_modes_check_same_serial_geometry_before_start(tmp_path,mode):
    run,watch,report,_,out=run_smoke(tmp_path,'normal',plan_only=mode=='plan',preflight=mode=='preflight')
    assert run.returncode==0,run.stdout+run.stderr
    assert watch['passed'] and len(report['steps'])==10
    assert not list(out.glob('*-engine.log'))
    assert 'IMAGE_HTTP_SERIAL_SMOKE PASS' not in run.stdout


@pytest.mark.parametrize('scenario',['no_restore','cross_talk','missing_record','engine_error'])
def test_serial_faults_still_refused(tmp_path,scenario):
    run,watch,report,_,_=run_smoke(tmp_path,scenario)
    assert run.returncode==1,(run.stdout,run.stderr)
    assert not watch['passed'] and ' PASS ' not in run.stdout
    assert report['status'] in ('FAIL','FAIL_required_checks')


def test_preemption_is_not_hidden_by_deduplicating():
    # Actual Orin shape: one partial solo attempt, then a resumed admission.
    raw=(b'strata serve: session 0: prompt 49 tokens = 0 reused + 49 read in 223 ms '
         b'(220.1 tok/s), 2 generated in 52 ms (38.5 tok/s), drafts accepted 0 of 2 (cancel)\n'
         b'strata batch: slot 0 takes a prompt of 51 tokens = 50 reused + 1 read in 0 ms (5555.5 tok/s)\n')
    assert not h.audit_log_span(raw,[dict(usage=dict(prompt_tokens=49))])['exact']
