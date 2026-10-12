from experiments.puzzle_pilot import Settings,run_episode
from experiments.cold_start_gate import gate


def test_base_skill_shared_without_enabling_chunking(tmp_path):
    task={'id':'basic','benchmark':'rush_hour','minimum_actions':1,'board':'......'*2+'AA....'+'......'*3}
    text='A frozen basic skill shared by all arms.'
    prompts=[]
    def planner(prompt):
        prompts.append(prompt)
        return {'response':'<actions>["A+4"]</actions>','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
    for mode in ('react','dynamic_interrupt'):
        row=run_episode(task,mode,Settings(action_only=True),tmp_path,planner,base_skill_text=text)
        assert row['success'] and row['base_skill_sha256']
    assert all(text in prompt for prompt in prompts)
    assert 'exactly ONE' in prompts[0]


def test_gate_requires_both_domains_and_clean_complete_records():
    tasks=[{'id':str(i),'benchmark':'rush_hour' if i<3 else 'sudoku'} for i in range(6)]
    rows=[{'task_id':t['id'],'benchmark':t['benchmark'],'success':True,'invalid_actions':0,
           'environment_steps':3,'model_errors':0,'format_errors':0,'usage_incomplete':False} for t in tasks]
    assert gate(rows,tasks)['passed']
    assert not gate(rows[:-1],tasks)['passed']
    rows[0]['success']=False
    assert gate(rows,tasks)['passed']
    rows[1]['success']=False
    assert not gate(rows,tasks)['passed']
    rows[1]['success']=True;rows[0]['format_errors']=1
    assert not gate(rows,tasks)['passed']
