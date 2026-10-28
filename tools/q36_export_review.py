#!/usr/bin/env python3
"""Regenerate a portable review from THIS clone, no previous /workspace/tmp needed.
Exports tracked Git blobs/commits ONLY. Never collects homes, credentials, models,
venvs or build directories. Requires full ancestry (not a shallow clone).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE='765068db267f8cd8f846f8ef45c9cc94f90075d5'
PREREQUISITE='4130bdfdb43bdb273412cc3530660bd19ec14602'
BRANCH='feature/q36-image-checkpoints'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if git('status','--porcelain','--untracked-files=all').strip():p.error('commit intended changes first; never export a dirty tree')
 if git('symbolic-ref','--short','HEAD').decode().strip()!=BRANCH:p.error('export ONLY the intended feature branch')
 git('merge-base','--is-ancestor',BASE,'HEAD')
 dest=a.out.resolve();dest.mkdir(parents=True,exist_ok=True);name='q36-image-checkpoints-review';directory=dest/name
 archive=dest/(name+'.tar.gz');sha=dest/(name+'.tar.gz.sha256')
 if directory.exists() or archive.exists() or sha.exists():p.error('choose a NEW output directory; preserve prior artifacts')
 directory.mkdir();(directory/'commits').mkdir();(directory/'source').mkdir()
 head=git('rev-parse','HEAD').decode().strip()
 (directory/'q36-image-checkpoints.patch').write_bytes(git('format-patch','--stdout','--full-index','--binary',BASE+'..HEAD'))
 subprocess.run(['git','format-patch','--full-index','--binary','--output-directory',str(directory/'commits'),BASE+'..HEAD'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 bundle=directory/'q36-image-checkpoints-incremental.bundle'
 subprocess.run(['git','bundle','create',str(bundle),PREREQUISITE+'..refs/heads/'+BRANCH],cwd=ROOT,check=True)
 proof=subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,capture_output=True)
 (directory/'bundle-verify.txt').write_bytes(proof.stdout+proof.stderr)
 (directory/'COMMITS.txt').write_bytes(git('log','--format=commit %H%nparent %P%ntree %T%nsubject %s%n',PREREQUISITE+'..HEAD'))
 names=git('diff','--name-only',BASE,'HEAD').decode().splitlines()
 # Include every tracked changed/new file, even force-added historical text logs;
 # never walk ignored logs or the filesystem looking for private/generated data.
 for name_ in names:
  path=Path(name_)
  if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe Git path')
  mode=git('ls-tree','HEAD','--',name_).decode().split()[0] if git('ls-tree','HEAD','--',name_).strip() else None
  if mode is None:continue # deletions remain represented in patch
  if mode not in ('100644','100755'):raise ValueError('non-regular export: '+name_)
  target=directory/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(git('show','HEAD:'+name_))
 (directory/'ORIN_HANDOFF.md').write_bytes(git('show','HEAD:docs/Q36_IMAGE_CHECKPOINTS.md'))
 (directory/'MIGRATION_HANDOFF.md').write_bytes(git('show','HEAD:docs/Q36_MULTIMODAL_CACHE_HANDOFF.md'))
 (directory/'README.md').write_text('# Image checkpoints portable review\n\nTip '+head+'; feature delta starts at '+BASE+'.\n\nBundle prerequisite '+PREREQUISITE+'; it includes original diagnostic objects,\nimplementation37911119 and subsequent portable-support/handoff commits.\nDo not replay diagnostics onto the actual Orin follow-up. See ORIN_HANDOFF.md\nand MIGRATION_HANDOFF.md. Source/essential instructions/evidence are on origin;\nthis archive is optional transport, NOT a deployment or CUDA qualification.\n')
 files=sorted(x for x in directory.rglob('*') if x.is_file())
 (directory/'SHA256SUMS').write_text(''.join(digest(x.read_bytes())+'  '+str(x.relative_to(directory))+'\n' for x in files))
 with tarfile.open(archive,'w:gz') as t:t.add(directory,arcname=directory.name)
 sha.write_text(digest(archive.read_bytes())+'  '+archive.name+'\n')
 # Verify stored archive member contents, not just the original directory.
 with tarfile.open(archive) as t:
  manifest=t.extractfile(name+'/SHA256SUMS').read().decode()
  for line in manifest.splitlines():
   expected,relative=line.split('  ',1);assert digest(t.extractfile(name+'/'+relative).read())==expected,relative
 print(json.dumps(dict(head=head,archive=str(archive),sha256=digest(archive.read_bytes()),files=len(files),archive_verified=True),indent=2))
if __name__=='__main__':main()
