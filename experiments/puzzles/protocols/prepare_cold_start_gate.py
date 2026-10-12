"""Freeze a six-item gate before any model calls; solutions stay offline."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from experiments.prepare_small_evolution_run import rush, sudoku
from experiments.puzzle_pilot import save
from dynamic_skill.puzzles import make_environment
from dynamic_skill.evolution import digest


def main():
    root=Path('data/cold_start_gate_20260928')
    if root.exists(): raise RuntimeError('gate already frozen')
    tasks=[]; private=[]
    for depth in (1,2,3):
        puzzle,answer=sudoku(2026092830+depth,holes=depth)
        task={'id':f'cold-sudoku-{depth}','benchmark':'sudoku','puzzle':puzzle,'minimum_actions':depth}
        tasks.append(task); private.append({'id':task['id'],'answer':answer})
        board,answer,attempt=rush(2026092840+depth,4+depth,depth)
        task={'id':f'cold-rush-{depth}','benchmark':'rush_hour','board':board,'minimum_actions':depth}
        tasks.append(task); private.append({'id':task['id'],'answer':answer,'attempt':attempt})
    states=[make_environment(t).render() for t in tasks]
    assert len(set(states))==6
    for older in (Path('data/curriculum_20260927/tasks.json'),Path('data/evolution_small_20260928/tasks.json')):
        previous=json.loads(older.read_text())
        assert not set(states)&{make_environment(t).render() for t in previous}
    save(root/'tasks.json',tasks)
    save(root/'validation_private.json',private)
    save(root/'manifest.json',{'tasks_hash':digest(tasks),'count':6,'seeds':'2026092831..33/41..43',
        'gate':'at least 5/6, >=2/3 per benchmark; invalid fraction <=10%; zero format/API/unknown errors',
        'purpose':'new synthetic basic-capability development gate, not held-out long-horizon benchmark'})
    print('Frozen 6 new 1/2/3-action tasks; offline answers never used by runtime.')


if __name__=='__main__':main()
