"""Finite independently authorized diagnostic. Redacted transport timing, no retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner
from experiments.puzzle_pilot import save

ROOT=Path('results/generation_diagnostic_20260929')
LEDGER=Path('results/gpt54_recovery_20260928/budget.json')
AUTH='generation_diagnostic_20260929'


def causes(error):
    result=[]; seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error)); result.append(type(error).__name__)
        error=error.__cause__ or error.__context__
    return result


def preflight():
    import httpx
    start=time.monotonic(); out={'proxy_variables_present':[k for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY') if os.environ.get(k)]}
    try:
        addresses=socket.getaddrinfo('api.inferera.com',443,type=socket.SOCK_STREAM)
        out['dns']={'seconds':round(time.monotonic()-start,3),'address_count':len(addresses)}
    except Exception as e: out['dns']={'error_chain':causes(e)}
    start=time.monotonic()
    try:
        r=httpx.get('https://api.inferera.com/v1/models',
                    headers={'Authorization':'Bearer '+os.environ['TARGET_AZURE_OPENAI_API_KEY']},
                    timeout=httpx.Timeout(20,connect=10))
        out['models']={'http_status':r.status_code,'seconds':round(time.monotonic()-start,3)}
        if r.status_code==200:
            try: out['models']['gpt54_listed']=any(v.get('id')=='gpt-5.4' for v in r.json().get('data',[]))
            except ValueError: out['models']['valid_json']=False
    except Exception as e: out['models']={'seconds':round(time.monotonic()-start,3),'error_chain':causes(e)}
    save(ROOT/'preflight.json',out); print(json.dumps(out))


def run(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('step',choices=['preflight','tiny','replay','tiny2']); a=p.parse_args(argv)
    ROOT.mkdir(exist_ok=True,parents=True)
    if a.step=='preflight': preflight(); return
    import httpx
    from openai import OpenAI
    with (LEDGER.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b=CampaignBudget(LEDGER,deadline=0); d=b.read()
        if any(r['status']=='pending' for r in d['requests']): raise RuntimeError('pending request')
        if d['policy_violation'] or time.time()<d['cooldown_until']: raise RuntimeError('policy/cooldown guard')
        unknown=sum(r['status']=='unknown_usage' for r in d['requests'])
        if AUTH not in d:
            if a.step!='tiny': raise RuntimeError('start with tiny diagnostic')
            d[AUTH]={'at_unix':time.time(),'initial_requests':len(d['requests']),'initial_unknown':unknown,
                     'old_deadline':d['deadline_unix'],'old_max_requests':d['max_requests'],
                     'old_unknown_cap':d['max_unknown_requests'],'deadline_unix':time.time()+1800,
                     'reason':'User explicitly authorized standalone generation diagnosis and continuation; max 3 diagnostic calls, cumulative $30 unchanged.'}
            d['deadline_unix']=d[AUTH]['deadline_unix']; d['max_requests']=len(d['requests'])+3
            d['max_unknown_requests']=unknown+1; b.write(d)
        auth=d[AUTH]
        if time.time()>=auth['deadline_unix']: raise RuntimeError('diagnostic deadline expired')
        if unknown>auth['initial_unknown']: raise RuntimeError('new unknown usage; stop diagnostic')
        if len(d['requests'])-auth['initial_requests']>=3: raise RuntimeError('diagnostic request cap')
        out=ROOT/(a.step+'.json')
        if out.exists(): raise RuntimeError('already attempted; no silent retry')
        probe=json.loads((ROOT/'preflight.json').read_text())
        if probe.get('models',{}).get('http_status')!=200: raise RuntimeError('free connectivity check has not passed')
        if a.step!='tiny':
            prior=json.loads((ROOT/'tiny.json').read_text())
            if prior['status']!='ok': raise RuntimeError('minimal generation failed')
        if a.step in ('replay','tiny2'):
            old=json.loads(Path('results/controller_research_20260928/dev/react/cr-dev-rush-0/calls.json').read_text())[0]
            messages=[{'role':'system','content':old['system_prompt']},{'role':'user','content':old['prompt']}]
            maximum=256
        else:
            messages=[{'role':'user','content':'Reply with exactly OK.'}]; maximum=32
        original_sdk=a.step=='tiny2'
        rec={'step':a.step,'status':'prepared','model':'gpt-5.4','reasoning_effort':'none',
             'max_completion_tokens':maximum,'temperature':0,'connect_timeout_s':10,'read_timeout_s':90,
             'max_retries':0,'messages':messages,'trace':[],'started_unix':time.time(),
             'transport':'original_sdk_defaults' if original_sdk else 'traced_httpx'}
        if original_sdk: rec.update(connect_timeout_s=45,read_timeout_s=45)
        start=time.monotonic()
        def trace(name,info):
            event={'event':name,'seconds':round(time.monotonic()-start,4)}
            if 'exception' in info: event['error_chain']=causes(info['exception'])
            rec['trace'].append(event)
        class TracedTransport(httpx.HTTPTransport):
            def handle_request(self,request):
                request.extensions['trace']=trace
                return super().handle_request(request)
        options=({'timeout':45} if original_sdk else
                 {'http_client':httpx.Client(transport=TracedTransport(retries=0)),
                  'timeout':httpx.Timeout(90,connect=10,write=15,pool=10)})
        client=OpenAI(base_url='https://api.inferera.com/v1',api_key=os.environ['TARGET_AZURE_OPENAI_API_KEY'],
                      max_retries=0,**options)
        def invoke(prompt):
            response=client.chat.completions.create(model='gpt-5.4',messages=messages,reasoning_effort='none',
                                                   temperature=0,max_completion_tokens=maximum)
            return {'response':response.choices[0].message.content,'usage':response.usage.model_dump() if response.usage else None,
                    'response_id':response.id,'provider_model':response.model,
                    'finish_reason':response.choices[0].finish_reason,
                    'reasoning_content':getattr(response.choices[0].message,'reasoning_content',None)}
        planner=BudgetPlanner(invoke,b,str(out)); planner.before_call(); save(out,rec)
        try:
            rec.update(planner('')); rec['status']='ok'
        except Exception as e:
            rec.update(status='error',error_chain=causes(e),http_status=getattr(e,'status_code',None))
            body=getattr(e,'body',None)
            if isinstance(body,dict):
                error=body.get('error',body)
                if isinstance(error,dict): rec['provider_error']={k:error.get(k) for k in ('type','code','param')}
        finally: client.close()
        rec['seconds']=round(time.monotonic()-start,3)
        d=b.read(); rec['ledger_status']=d['requests'][-1]['status']; rec['guarded_charge_usd']=d['requests'][-1]['budget_charge_usd']
        rec['cumulative_guarded_usd']=b.effective_spend(d); save(out,rec)
        print(json.dumps({k:v for k,v in rec.items() if k not in ('messages','reasoning_content')}))

if __name__=='__main__': run(sys.argv[1:])
