#!/usr/bin/env python3
from experiments._common import run
import sys


if __name__ == "__main__":
    if '--skill-evolution' in sys.argv:
        raise SystemExit('For the full training-free loop use experiments/campaign_runner.py --skill-evolution ...; this legacy entry only runs ALFWorld and exports suggestions.')
    run("dynamic_evolution")
