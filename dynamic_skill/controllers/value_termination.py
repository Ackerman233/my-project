"""Small public-feature branch-value learner; not an Option-Critic reproduction."""
import re
import numpy as np
from dynamic_skill.evidence.queue_evidence import AlfworldEvidence
from dynamic_skill.alfworld.acquisition_check import task_mentions

FEATURES=('bias','tail_length','head_supported','head_false','head_unknown','last_rejected',
          'target_takeable','last_target_pickup','locations_visited','location_revisits','navigation_fraction')


def features(task,observation,admissible,history,tail):
    proof=AlfworldEvidence(observation,admissible,complete=False).executable(tail[0]) if tail else None
    truth=proof.value.value if proof else 'unknown'
    visits=[]
    for row in history:
        match=re.fullmatch(r'go to ([a-z][a-z0-9]* [0-9]+)',row['action'])
        if match and re.search(r'You arrive at '+re.escape(match[1])+r'\.',row['env_feedback']):visits.append(match[1])
    rejected=observation.strip().lower()=='nothing happens.'
    takeable=False
    for a in admissible:
        match=re.fullmatch(r'take ([a-z][a-z0-9]* [0-9]+) from ([a-z][a-z0-9]* [0-9]+)',a)
        if match and task_mentions(task,match[1]):takeable=True
    pickup=re.fullmatch(r'You pick up the ([a-z][a-z0-9]* [0-9]+) from the ([a-z][a-z0-9]* [0-9]+)\.',observation)
    return [1.,len(tail)/4.,float(truth=='true'),float(truth=='false'),float(truth=='unknown'),float(rejected),
        float(takeable),float(bool(pickup and task_mentions(task,pickup[1]))),min(len(set(visits)),30)/30.,
        min(len(visits)-len(set(visits)),10)/10.,sum(a.startswith('go to ') for a in tail)/max(1,len(tail))]


def utility(result,call_cost=.04,action_cost=.002):
    # One cost accounting: API call penalty is already in Q labels, not subtracted again at inference.
    return float(result['success'])-call_cost*result['model_calls']-action_cost*result['suffix_steps']


def fit_matrix(x,y,ridge):
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=1e-8
    return np.linalg.solve(x.T@x+penalty,x.T@y)


def fit(rows):
    assert len(rows)>=4
    x=np.array([r['features'] for r in rows],dtype=float)
    y=np.array([r['utilities'] for r in rows],dtype=float)
    groups=sorted({r['group'] for r in rows})
    trials=[]
    for ridge in (.1,1.,10.):
        errors=[]
        for group in groups:
            mask=np.array([r['group']!=group for r in rows])
            if sum(mask)<2:continue
            beta=fit_matrix(x[mask],y[mask],ridge)
            errors.extend((x[~mask]@beta-y[~mask]).ravel().tolist())
        trials.append({'ridge':ridge,'group_cv_mse':float(np.mean(np.square(errors))) if errors else 1e9})
    selected=min(trials,key=lambda t:(t['group_cv_mse'],-t['ridge']))['ridge']
    beta=fit_matrix(x,y,selected)
    residual=(y-x@beta)[:,1]-(y-x@beta)[:,0]
    return {'feature_schema':FEATURES,'weights':beta.tolist(),'ridge':selected,'cv':trials,
        'training_rows':rows,'residual_delta_std':float(np.std(residual)),
        'min_neighbors':2,'support_radius':.75,'minimum_advantage':.02,
        'objective':{'call_cost':.04,'action_cost':.002,'reward_success':1},
        'scope':'Ridge branch-utility approximation from noisy real prefix pairs; no calibrated confidence or RL gradient guarantee.'}


def predict(model,x):
    assert list(model['feature_schema'])==list(FEATURES)
    array=np.asarray(x);q=array@np.asarray(model['weights'])
    neighbors=[r for r in model['training_rows'] if np.linalg.norm(array-np.asarray(r['features']))<=model['support_radius']]
    groups={r['group'] for r in neighbors}
    margin=max(model['minimum_advantage'],model['residual_delta_std']/max(1,len(groups))**.5)
    supported=len(groups)>=model['min_neighbors']
    return {'q_continue':float(q[0]),'q_replan':float(q[1]),'advantage':float(q[1]-q[0]),
        'support_groups':len(groups),'margin':margin,'supported':supported,
        'terminate':bool(supported and q[1]-q[0]>margin),
        'note':'Margin is empirical conservatism, not a probability or calibrated confidence interval.'}
