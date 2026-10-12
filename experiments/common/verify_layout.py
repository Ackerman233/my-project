"""Read-only verification of the 2026-10-08 source relocation and snapshots."""
import hashlib
import json
from pathlib import Path
import zipfile


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    count = 0
    for package in ('experiments', 'dynamic_skill'):
        data = json.loads((root / package / 'migration_20261008.json').read_text())
        assert sha(root / data['archive']) == data['archive_sha256']
        with zipfile.ZipFile(root / data['archive']) as archive:
            for item in data['modules'].values():
                assert hashlib.sha256(archive.read(item['old_path'])).hexdigest() == item['original_sha256']
                assert sha(root / item['path']) == item['migrated_sha256'], item['path']
                assert not (root / item['old_path']).exists(), item['old_path']
                count += 1
    # Frozen records are checked against THEIR snapshots, not relocated live files.
    checked = 0
    for config in (root / 'results').rglob('config.json'):
        if 'source_snapshot' in config.parts:
            continue
        cfg = json.loads(config.read_text())
        for path, expected in cfg.get('source_hashes', {}).items():
            snapshot = config.parent / 'source_snapshot' / path
            if snapshot.is_file():
                assert sha(snapshot) == expected, str(snapshot)
                checked += 1
    return {'migrated_modules': count, 'verified_snapshot_files': checked,
            'original_archive_intact': True, 'paid_requests': 0}


if __name__ == '__main__':
    print(json.dumps(verify()))
