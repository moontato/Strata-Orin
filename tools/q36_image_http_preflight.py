#!/usr/bin/env python3
"""Prepare/audit actual HTTP bodies on Orin CPU. Does not operate services."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from datetime import datetime, timezone

STAGE = Path('/mnt/ssd/tmp/strata-checkpoints')
TRIAL = Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')
QUALIFIED_SHA = 'eb5ef5de8f7f36de9bd25206d59ec2161a4fb5548e3f74246483a344f8544c3f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-state', type=Path, default=STAGE / 'image-checkpoint-fixture-state.json')
    p.add_argument('--live-config', type=Path, default=Path('/mnt/ssd/strata-service/strata.json'))
    a = p.parse_args()
    state = json.loads(a.fixture_state.read_text())
    assert Path(state['trial']).resolve() == TRIAL
    assert subprocess.check_output(['git', '-C', str(TRIAL), 'rev-parse', 'HEAD'], text=True).strip() == state['commit']
    for name, expected in state['protected'].items():
        assert sha(TRIAL / name) == expected, 'protected file changed: ' + name
    candidate = Path(state['build']) / 'strata-q36'
    assert sha(candidate) == QUALIFIED_SHA == state['engine_sha256']
    live = json.loads(a.live_config.read_text())
    base_cwd = Path(live.get('cwd') or a.live_config.parent)
    def absolute(path):
        x = Path(path)
        return str((base_cwd / x).resolve()) if not x.is_absolute() else str(x.resolve())
    baseline = Path(absolute(live['exe']))
    assert baseline.is_file() and os.access(baseline, os.X_OK)
    os.environ['STRATA_IMAGE_HTTP_ROOT'] = str(TRIAL)
    os.environ['STRATA_IMAGE_HTTP_SOURCE_COMMIT'] = state['commit']
    os.environ['STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT'] = str(TRIAL)
    sys.path.insert(0, str(TRIAL))
    from q36_image_http_smoke import vision_of
    from q36_image_http_geometry import prepare
    vision = vision_of(live)
    for key in ('exe', 'mmproj', 'model'):
        vision[key] = absolute(vision[key])
    assert vision['model'] == str(Path(state['model']).resolve()), 'encoder model differs'
    safe = dict(vision=vision, tokenizer=absolute(live['tokenizer']),
                model_name=live.get('model_name') or 'qwen3.6-35b-a3b')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
    out = STAGE / ('http-smoke-cpu-' + stamp)
    out.mkdir()
    config = out / 'source-config.json'
    config.write_text(json.dumps(safe, indent=2) + '\n')
    audit = prepare(a.fixture_state, out / 'geometry', config, baseline)
    script = Path(__file__).with_name('q36_image_http_smoke.py')
    command = [sys.executable, str(script), '--engine', str(candidate),
               '--expect-engine-sha256', QUALIFIED_SHA, '--baseline-engine', str(baseline),
               '--model', state['model'], '--source-config', str(config), '--server-root', str(TRIAL),
               '--geometry', str(out / 'geometry/geometry.json'), '--window-seconds', '120',
               '--cleanup-seconds', '10']
    env = dict(os.environ, STRATA_IMAGE_HTTP_REVIEW_WINDOW='1')
    for mode in ('plan-only', 'preflight'):
        result = subprocess.run(command + ['--out', str(out / mode), '--' + mode], env=env,
                                capture_output=True, text=True, timeout=140)
        (out / (mode + '.log')).write_text(result.stdout + result.stderr)
        print(result.stdout, end='', flush=True)
        if result.returncode:
            raise RuntimeError(mode + ' failed; see ' + str(out / (mode + '.log')))
    record = dict(state, geometry=str(out / 'geometry/geometry.json'),
                  source_config=str(config), baseline_engine=str(baseline), baseline_sha256=sha(baseline),
                  cpu_results=str(out), live_config_sha256=sha(a.live_config),
                  scripts=str(script.parent), geometry_sha256=sha(out / 'geometry/geometry.json'))
    (out / 'state.json').write_text(json.dumps(record, indent=2) + '\n')
    (STAGE / 'image-http-smoke-state.json').write_text(json.dumps(record, indent=2) + '\n')
    for row in audit['audit']['rewrites']:
        print('HTTP_GEOMETRY turn={turn} image_end={image_end} LCP={lcp} legal_saved={available}'.format(**row))
    print('IMAGE_HTTP_ACTUAL_GGUF_CPU_PREFLIGHT PASS model_not_started=1 services_untouched=1')
    print('Results:', out)


if __name__ == '__main__':
    main()
