"""Separate results for repaired compositional subskill protocol, never overwrite v1."""
from pathlib import Path
from experiments.space_subskill_trial import trial,configure as v1configure
def configure():
    v1configure()
    trial.ROOT=Path('results/space_subskills_v2_20261003');trial.AUTH='space_subskills_v2_20261003_authorization'
    trial.SOURCES=trial.SOURCES+['experiments/space_subskill_v2_trial.py']
def advance():
    configure();trial.advance()
if __name__=='__main__':
    import sys
    configure()
    if sys.argv[1:]==['--prepare']:trial.prepare()
    elif sys.argv[1:]==['--authorize']:trial.authorize()
    elif sys.argv[1:]==['--summary']:trial.summary()
    else:raise SystemExit('Expected --prepare / --authorize / --summary')
