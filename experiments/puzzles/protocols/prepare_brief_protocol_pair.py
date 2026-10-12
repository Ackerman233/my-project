import json
from pathlib import Path
import sys
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.puzzles import make_environment
from experiments.prepare_small_evolution_run import sudoku, rush
from experiments.puzzle_pilot import save
from experiments.brief_protocol_pair import NAME


def main():
    root=Path('data')/NAME
    if root.exists(): raise RuntimeError('already frozen')
    prior=set()
    for p in Path('data').glob('*/tasks.json'):
        data=json.loads(p.read_text())
        if isinstance(data,list):
            for t in data:
                if isinstance(t,dict) and t.get('benchmark') in ('sudoku','rush_hour'):
                    prior.add(make_environment(t).render())
    tasks=[]; answers=[]
    for depth in (1,2,3):
        puzzle,answer=sudoku(2026093100+depth,holes=depth)
        task={'id':f'brief-sudoku-{depth}','benchmark':'sudoku','puzzle':puzzle,'minimum_actions':depth}
        tasks.append(task); answers.append({'id':task['id'],'answer':answer})
        board,answer,attempt=rush(2026093200+depth,4+depth,depth)
        task={'id':f'brief-rush-{depth}','benchmark':'rush_hour','board':board,'minimum_actions':depth}
        tasks.append(task); answers.append({'id':task['id'],'answer':answer,'attempt':attempt})
    states=[make_environment(t).render() for t in tasks]
    assert len(set(states))==6 and not set(states)&prior
    save(root/'tasks.json',tasks); save(root/'validation_private.json',answers)
    save(root/'manifest.json',{'seeds':'2026093101..03/3201..03','count':6,'prior_states_checked':len(prior),
                             'note':'Freeze before calls; same six tasks for both protocols, offline answers not exposed.'})
    print('Frozen six fresh paired tasks.')

if __name__=='__main__':main()
