"""One-time responsibility-based migration, with byte-exact source archive."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'dynamic_skill'
GROUPS = {
    'execution': 'checker interrupter skill_executor',
    'knowledge': 'skill skill_induction skill_updater failure_analyzer',
    'learning': 'evolution',
    'controllers': 'control_only_evolution research_controller value_termination paper_control',
    'alfworld': '''alfworld_rollout alfworld_recovery acquisition_check alfworld_open_guard
        space_subskills space_boundary observable_search_operator micro_loop_protocol react_metrics''',
    'environments': 'puzzles',
    'evidence': 'queue_revision queue_evidence tail_filter',
    'infrastructure': 'campaign_budget thinking_policy',
}


def main():
    manifest = BASE / 'migration_20261008.json'
    if manifest.exists():
        raise SystemExit('Already migrated; do not repeat.')
    mapping = {name: folder for folder, names in GROUPS.items() for name in names.split()}
    files = sorted(p for p in BASE.glob('*.py') if p.name != '__init__.py')
    assert {p.stem for p in files} == set(mapping)
    archive = ROOT / 'experiments/_archive/20261008_dynamic_skill_sources.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as out:
        for source in BASE.glob('*.py'):
            out.write(source, str(source.relative_to(ROOT)))
    records = {}
    for source in files:
        dest = BASE / mapping[source.stem] / source.name
        dest.parent.mkdir(exist_ok=True)
        assert not dest.exists()
        original = source.read_bytes()
        text = original.decode()
        for name, folder in mapping.items():
            text = text.replace('from .' + name + ' import',
                'from dynamic_skill.' + folder + '.' + name + ' import')
        source.rename(dest)
        dest.write_text(text)
        (dest.parent / '__init__.py').touch(exist_ok=True)
        records['dynamic_skill.' + source.stem] = {
            'module': '.'.join(dest.relative_to(ROOT).with_suffix('').parts),
            'old_path': str(source.relative_to(ROOT)), 'path': str(dest.relative_to(ROOT)),
            'original_sha256': hashlib.sha256(original).hexdigest(),
            'migrated_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({'modules': records, 'archive': str(archive.relative_to(ROOT)),
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2) + '\n')
    print(f'Moved {len(records)} runtime modules with explicit cross-layer imports.')


if __name__ == '__main__':
    main()
