"""Exploratory unknown-head prefixes; original frozen rules and prompts unchanged."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
from experiments import queue_revision_transfer_trial as base
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import save
from dynamic_skill.campaign_budget import CampaignBudget

ROOT=Path('results/queue_revision_unknown_20261007')
AUTH='queue_revision_unknown_20261007_authorization'
SOURCE_ROOTS=(Path('results/queue_revision_transfer_20261007'),Path('results/queue_revision_opportunity_20261007'))


def configure():
    base.ROOT,base.AUTH=ROOT,AUTH


def prepare():
    configure()
    assert not (ROOT/'config.json').exists()
    original=read(SOURCE_ROOTS[0]/'config.json')
    assert all(digest(Path(p))==sha for p,sha in original['rule_hashes'].items())
    cases=[]
    for source in SOURCE_ROOTS:
        cfg=read(source/'config.json')
        for task in cfg['tasks']:
            directory=source/'collection'/'predictions'/(cfg['model']+'__'+task['id']+'__fixed_chunk')
            path=directory/'conversation.json'
            if not path.exists():
                continue
            trace=read(path)
            for index,row in enumerate(trace[:-1]):
                tail=row.get('remaining_before_interrupt',[])
                if row['done'] or not tail:
                    continue
                proof=base.AlfworldEvidence(row['env_feedback'],trace[index+1]['admissible_before'],complete=False).executable(tail[0])
                if proof.value is not base.Truth.UNKNOWN:
                    continue
                case={'id':task['id']+'-unknown','source_trace':str(path),'source_sha256':digest(path),
                    'source_initial_sha256':digest(directory/'initial.json'),'game_sha256':task['game_sha256'],
                    'step':row['step'],'tail':tail,'kind':'unknown_head_exploratory','head_verdict':'unknown',
                    'reason':'Frozen original queue review: '+proof.reason}
                env=base.environment(task['gamefile'])
                try:
                    _,_,checkpoint=base.replay_prefix(env,case)
                finally:
                    env.close()
                case['checkpoint'],case['checkpoint_sha256']=checkpoint,config_hash(checkpoint)
                cases.append(case)
                break
    cfg=copy.deepcopy(original)
    cfg.update(protocol='frozen-queue-revision-exploratory-unknown-head-v1',tasks=[],
        source_task_count=sum(len(read(s/'config.json')['tasks']) for s in SOURCE_ROOTS),
        source_configs={str(s/'config.json'):digest(s/'config.json') for s in SOURCE_ROOTS},
        selector='First nonterminal UNKNOWN queued head per newly collected task using next-step public commands; do not filter by the following action validity or episode success. Adaptive exploratory analysis after inspecting the prior audits; not independent validation.',
        intervention=original['intervention'],limits={'batch_usd':3,'max_requests':len(cases)*15,'seconds':3600},
        note='Exploratory unknown-evidence recovery on new-task traces already inspected. Clear and evidence repair both replan here; this cannot test selective skip or dependency retention gain. Same frozen core, adapters, prompt, model and suffix budgets as primary. No additional prefix collection.')
    sources=sorted(set(cfg['source_hashes'])|{'experiments/queue_revision_unknown_trial.py'})
    cfg['source_hashes']={p:digest(Path(p)) for p in sources}
    save(ROOT/'config.json',cfg)
    for p in sources:
        dst=ROOT/'source_snapshot'/p
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst)
    save(ROOT/'cases.json',{'config_sha256':config_hash(cfg),'cases':cases,'omissions':[],
        'job_order':[[case['id'],arm] for index,case in enumerate(cases) for arm in base.ARMS[index%3:]+base.ARMS[:index%3]]})
    print('PREPARED_EXPLORATORY',len(cases),[(c['id'],c['step'],c['tail']) for c in cases])


def authorize():
    configure()
    base.authorize()
    with (base.CAMPAIGN/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=CampaignBudget(base.CAMPAIGN/'budget.json',deadline=0)
        state=budget.read()
        state[AUTH]['cases_sha256']=digest(ROOT/'cases.json')
        budget.write(state)


def advance():
    configure()
    cfg=read(ROOT/'config.json')
    assert all(digest(Path(p))==sha for p,sha in cfg['source_configs'].items())
    base.advance()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','authorize','advance','summary'])
    args=parser.parse_args()
    configure()
    {'prepare':prepare,'authorize':authorize,'advance':advance,'summary':base.summary}[args.action]()
