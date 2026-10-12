"""Current-location inspect repair; same four tasks, separate frozen experiment."""
from pathlib import Path
import argparse
import shutil
from experiments import protocol_diagnostic_trial as base
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import save


def configure():
    base.ROOT = Path('results/protocol_diagnostic_v4_20261007')
    base.AUTH = 'protocol_diagnostic_v4_20261007_authorization'
    base.configure()


def prepare():
    configure()
    cfg = read(Path('results/protocol_diagnostic_20261007/config.json'))
    cfg['protocol'] = 'subskills-current-location-v4-protocol-diagnostic'
    sources = sorted(set(cfg['source_hashes']) | {
        'experiments/protocol_diagnostic_v4_trial.py', 'tests/test_subskill_current_location.py'})
    cfg['source_hashes'] = {p: digest(Path(p)) for p in sources}
    cfg['note'] += ' Same four tasks reused after observed current-location inspect precondition failure; this is an explicitly adaptive development retry, not independent evaluation.'
    if (base.ROOT / 'config.json').exists():
        assert read(base.ROOT / 'config.json') == cfg
    else:
        save(base.ROOT / 'config.json', cfg)
        for path in sources:
            dst = base.ROOT / 'source_snapshot' / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    print('PREPARED', config_hash(cfg))


def advance():
    configure()
    base.advance()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'authorize', 'summary'])
    args = parser.parse_args()
    configure()
    {'prepare': prepare, 'authorize': base.authorize, 'summary': base.trial.summary}[args.action]()
