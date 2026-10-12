"""One-time, reversible experiment layout migration. No API calls or result edits."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments'
GROUPS = {
    'alfworld/react': 'react',
    'alfworld/static_interrupt': 'fixed_chunk',
    'alfworld/dynamic_interrupt': '''dynamic_interrupt interrupt_recovery_ablation prefix_recovery_trial
        queue_revision_transfer_trial queue_revision_opportunity_trial queue_revision_unknown_trial
        resume_queue_revision_unknown unknown_independent_validation verify_queue_revision
        audit_interrupt_recovery_v3 audit_interrupt_tails audit_prefix_recovery_results
        audit_queue_revision_transfer audit_selective_tail audit_independent_validation95
        audit_search_coverage audit_discarded_tails''',
    'alfworld/skill_evolution': '''dynamic_evolution micro_evolution_loop audit_micro_evolution
        prepare_control_only control_parameter_trial audit_control_parameter''',
    'alfworld/operators': '''space_boundary_trial space_subskill_trial space_subskill_v2_trial
        compare_subskill_pair subskill_exposure_trial subskill_fixed_trial''',
    'alfworld/comparisons': '''qwen_alfworld_pair alfworld_staged_recovery alfworld_final_remaining
        audit_staged_recovery paper_method_trial paper_method_validation paper_method_confirmation
        audit_paper_methods equal_operator_attribution audit_equal_operator_attribution''',
    'alfworld/protocols': '''protocol_diagnostic_trial protocol_diagnostic_v4_trial
        protocol_diagnostic_v5_trial protocol_replay audit_protocol_diagnostic''',
    'alfworld/common': '_common alfworld_campaign audit_alfworld_campaign',
    'puzzles/common': 'prepare_puzzles prepare_curriculum qwen_model_pair',
    'puzzles/dynamic_interrupt': '''controller_research proxy_controller_research
        resume_controller_research audit_controller_research audit_tail_filter''',
    'puzzles/skill_evolution': '''prepare_evolution prepare_small_evolution_run
        skill_evolution audit_small_evolution_run''',
    'puzzles/protocols': '''cold_start_gate prepare_cold_start_gate audit_cold_start_gate
        paper_cold_start_gate prepare_paper_cold_start_gate audit_paper_cold_start
        brief_protocol_pair prepare_brief_protocol_pair audit_brief_protocol_pair
        recovered_brief_protocol_pair resume_brief_protocol_pair generation_diagnostic
        gpt_recovery_diagnostic gpt_rollout_batch''',
    'common': 'puzzle_pilot campaign_runner bounded_research95 verify_interrupt_recovery_offline',
}


def main():
    manifest = BASE / 'migration_20261008.json'
    if manifest.exists():
        raise SystemExit('Already migrated; use the manifest/archive, do not repeat.')
    mapping = {name: folder for folder, names in GROUPS.items() for name in names.split()}
    originals = sorted(p for p in BASE.glob('*.py') if p.name != '__init__.py')
    assert {p.stem for p in originals} == set(mapping), (
        'Unclassified/missing modules', {p.stem for p in originals} ^ set(mapping))
    archive = BASE / '_archive' / '20261008_original_sources.zip'
    archive.parent.mkdir(exist_ok=True)
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as out:
        for p in sorted(BASE.glob('*.py')) + sorted(BASE.glob('*.md')):
            out.write(p, str(p.relative_to(ROOT)))
    records = {}
    for source in originals:
        dest = BASE / mapping[source.stem] / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        content = source.read_bytes()
        text = content.decode().replace('Path(__file__).resolve().parents[1]',
            'next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())')
        source.rename(dest)
        dest.write_text(text)
        records['experiments.' + source.stem] = {
            'module': '.'.join(dest.relative_to(ROOT).with_suffix('').parts),
            'old_path': str(source.relative_to(ROOT)), 'path': str(dest.relative_to(ROOT)),
            'original_sha256': hashlib.sha256(content).hexdigest(),
            'migrated_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
        parent = dest.parent
        while parent != BASE:
            (parent / '__init__.py').touch(exist_ok=True)
            parent = parent.parent
    docs = BASE / 'docs'
    docs.mkdir(exist_ok=True)
    for name in ('PUZZLE_PILOT.md', 'SKILL_EVOLUTION.md'):
        (BASE / name).rename(docs / name)
    manifest.write_text(json.dumps({'modules': records,
        'archive': str(archive.relative_to(ROOT)),
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'note': 'Immutable historical results/snapshots untouched. Old Python imports supported by experiments.__init__. Historical frozen runs must use their recorded source snapshots or the archived original sources, not silently claim current live-source equivalence.'}, ensure_ascii=False, indent=2) + '\n')
    print(f'Moved {len(records)} modules; byte-exact originals archived; results untouched.')


if __name__ == '__main__':
    main()
