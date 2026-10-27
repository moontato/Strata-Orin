#!/usr/bin/env python3
"""Diagnostic-only baseline export inside this isolated worktree; no service ops.
Adds ONLY the same native vision test entry point/cancellation hook; leaves the
baseline's rejection of preallocated vision, parsing, math and allocation intact.
"""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE='a9837161edcbce84be14d6993bcf428ef96b9ac1'
ENGINE='src/qwen35/q36_engine.cu';SHARED='src/qwen35/q36_vision_regression.inc'

def replace(text,old,new):
    if text.count(old)!=1:raise ValueError('baseline diagnostic anchor not unique: '+old[:90])
    return text.replace(old,new)

def overlay(text):
    text=replace(text,'// Developer-only focused GPU/model gate, not a benchmark or an HTTP feature.',
        '#include "q36_vision_regression.inc"\n\n// Developer-only focused GPU/model gate, not a benchmark or an HTTP feature.')
    text=replace(text,'    int read_windows = 1;','    int read_windows = 1;\n    std::function<void(int64_t)> vision_test_on_chunk; // developer test only')
    text=replace(text,'            // Only fresh/restored text-prefill lineage has a fully seeded MTP',
        '            if(geni && vision_test_on_chunk) vision_test_on_chunk(done);\n            // Only fresh/restored text-prefill lineage has a fully seeded MTP')
    text=replace(text,'emb_file, checkpoint_test, sampled_mtp_test, test_state_dump;',
        'emb_file, checkpoint_test, sampled_mtp_test, test_state_dump, vision_test;')
    text=replace(text,'        else if (a == "--kv-preallocate") kv_preallocate = true;',
        '        else if (a == "--kv-preallocate") kv_preallocate = true;\n        else if (a == "--kv-vision-test") vision_test=next();')
    text=replace(text,'    if (!test_state_dump.empty() && checkpoint_test.empty() && sampled_mtp_test.empty())',
        '    if (!test_state_dump.empty() && checkpoint_test.empty() && sampled_mtp_test.empty() && vision_test.empty())')
    text=replace(text,'    if (!checkpoint_test.empty() || !sampled_mtp_test.empty()) {',
        '    if (!vision_test.empty() || !checkpoint_test.empty() || !sampled_mtp_test.empty()) {')
    text=replace(text,'        const int result=!checkpoint_test.empty() ? checkpoint_regression(E,checkpoint_test,test_state_dump)',
        '        const int result=!vision_test.empty() ? vision_preallocation_regression(E,vision_test,test_state_dump)\n                        : !checkpoint_test.empty() ? checkpoint_regression(E,checkpoint_test,test_state_dump)')
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if out.parent!=ROOT or not out.name.startswith('build-') or out.exists():p.error('need a NEW build-* export directly inside the vision test worktree')
    old=subprocess.check_output(['git','show',BASE+':'+ENGINE],cwd=ROOT).decode();new=overlay(old)
    data=subprocess.check_output(['git','archive',BASE],cwd=ROOT);out.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for item in tar.getmembers():
            if not (out/item.name).resolve().is_relative_to(out) or item.islnk() or item.issym():raise ValueError('unsafe archive path')
        tar.extractall(out)
    (out/ENGINE).write_text(new);(out/SHARED).write_bytes((ROOT/SHARED).read_bytes())
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+ENGINE,tofile='b/'+ENGINE))
    patch+=''.join(difflib.unified_diff([], (ROOT/SHARED).read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+SHARED))
    (out/'baseline-vision-diagnostic.patch').write_text(patch)
    (out/'BASELINE-VISION.json').write_text(json.dumps(dict(base=BASE,note='diagnostic-only baseline, no production deployment',
        engine_sha256=hashlib.sha256(new.encode()).hexdigest(),shared_sha256=hashlib.sha256((ROOT/SHARED).read_bytes()).hexdigest()),indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
