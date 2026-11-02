"""Prepare local supplied archives without modifying the originals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

from mmlib import io_data as io


def main() -> int:
    candidates = list(io.ROOT.parent.glob('*/附件'))
    if len(candidates) != 1:
        raise RuntimeError('Expected one sibling problem/attachment directory')
    archives = candidates[0]
    manifest = []
    for p in sorted(archives.iterdir()):
        if p.suffix.lower() not in ('.zip', '.rar'):
            continue
        prefix = p.name[:3]
        target = io.RAW / (p.stem if p.suffix == '.zip' else '（3）冰川高程变化')
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()
        count = 0
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as ar:
                for member in ar.infolist():
                    dest = (root / member.filename).resolve()
                    if not dest.is_relative_to(root):
                        raise RuntimeError('Archive path escapes target')
                    if member.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif not dest.exists() or dest.stat().st_size != member.file_size:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with ar.open(member) as src, dest.open('wb') as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
                    count += 1
        else:
            import libarchive
            with libarchive.file_reader(str(p)) as ar:
                for member in ar:
                    dest = (root / member.pathname).resolve()
                    if not dest.is_relative_to(root):
                        raise RuntimeError('Archive path escapes target')
                    if member.isdir:
                        dest.mkdir(parents=True, exist_ok=True)
                    elif member.isfile:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists() or dest.stat().st_size != member.size:
                            with dest.open('wb') as out:
                                for block in member.get_blocks():
                                    out.write(block)
                    count += 1
        manifest.append({'archive': p.name, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest(),
                         'bytes': p.stat().st_size, 'members': count, 'target': str(target.relative_to(io.ROOT))})
        print(prefix, count, flush=True)
    io.write_json(io.ROOT / '临时产物/原始附件清单.json', {'archives': manifest, 'python': sys.version})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
