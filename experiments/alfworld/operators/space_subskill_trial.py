"""Reuse guarded trial machinery with independent immutable namespace and authorization."""
from pathlib import Path
from experiments import space_boundary_trial as trial

def configure():
    trial.ROOT=Path('results/space_subskills_20261003');trial.AUTH='space_subskills_20261003_authorization'
    trial.ARMS={'H_subskills':dict(macro_mode=True,discovery_interrupt=False),
                'I_subskills_discovery':dict(macro_mode=True,discovery_interrupt=True)}
    trial.SOURCES=trial.SOURCES+['dynamic_skill/space_subskills.py','experiments/space_subskill_trial.py']

def advance():
    configure();trial.advance()

if __name__=='__main__':
    import sys
    configure()
    if sys.argv[1:]==['--prepare']:trial.prepare()
    elif sys.argv[1:]==['--authorize']:trial.authorize()
    elif sys.argv[1:]==['--summary']:trial.summary()
    else:raise SystemExit('Use --prepare, --authorize or --summary; paid execution through campaign_runner')
