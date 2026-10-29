"""Regression tests for conservative bracketed log-span auditing."""
from tools.q36_image_http_smoke import audit_log_span

SOLO = b'strata serve: session 1: prompt 10 tokens = 0 reused + 10 read in 2 ms (5.0 tok/s), 1 generated in 1 ms (1.0 tok/s), drafts accepted 0 of 0 (stop)\n'
BATCH = b'strata batch: slot 1 takes a prompt of 11 tokens = 8 reused + 3 read in 3 ms (1.0 tok/s)\n'
RESTORE = b'KVREPLAY session=1 generation=2 epoch=2 tokens=8 reason=prefix_checkpoint\n'
KVCACHE = b'KVCACHE session=1 generation=2 epoch=2 state=active tokens=11 bytes=100 reserved_tokens=11 result=checkpoint replay=0\n'

def answer(n): return {'usage': {'prompt_tokens': n}}

def test_serial_solo_and_batch_formats():
    for raw, n in ((SOLO, 10), (BATCH, 11)):
        audit = audit_log_span(raw, [answer(n)])
        assert audit['attributed'] and audit['summaries'][0]['prompt'] == n
        assert audit['restored'] is False and audit['result'] is None

def test_same_session_later_restore_is_not_attached_to_earlier_span():
    earlier = audit_log_span(SOLO, [answer(10)])
    later = audit_log_span(RESTORE + KVCACHE + BATCH, [answer(11)])
    assert earlier['restored'] is False and earlier['result'] is None
    assert not any(e['kind'] == 'restored' for e in earlier['events'])
    assert any(e['kind'] == 'restored' for e in later['events'])
    assert later['restored'] is True and later['result']=='checkpoint'

def test_ambiguous_or_incomplete_span_never_attributes():
    for raw, replies in ((SOLO+BATCH, [answer(10)]), (SOLO, [answer(11)]),
                         (SOLO[:-1], [answer(10)]), (SOLO, [{}])):
        assert audit_log_span(raw, replies)['attributed'] is False

def test_burst_only_no_individual_attribution():
    audit = audit_log_span(SOLO+BATCH, [answer(10), answer(11)])
    assert audit['attributed'] is False and audit['reason'] == 'concurrent_burst'
    assert audit['sessions'] == [1] and len(audit['summaries']) == 2
    assert 'restored' not in audit and 'result' not in audit
