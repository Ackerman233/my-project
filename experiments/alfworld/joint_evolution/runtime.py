"""Shared real ALFWorld evaluator and serialized, fully charged model transport."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from dynamic_skill.controllers.control_only_evolution import run_control_episode
from dynamic_skill.alfworld.micro_loop_protocol import SYSTEM
from experiments.common.puzzle_pilot import ApiPlanner, Settings, save
from .engine import CLEAN


def evaluate_episode(env, task, genome, settings, planner, output, cost=0.0):
    """Both demo and paid runs execute this exact real queue/controller path."""
    policy = genome.policy
    result = run_control_episode(env, task, settings, output, planner, genome.skill_text,
        policy.executable(), dynamic=True, history_evidence=policy.history_evidence,
        state_check=policy.state_check, discovery_interrupt=policy.discovery_interrupt,
        macro_mode=False)
    folder = output / 'predictions' / result['id']
    trace = json.loads((folder / 'conversation.json').read_text()) if (folder / 'conversation.json').exists() else []
    result.update(genome_hash=genome.hash, joint_policy=asdict(policy),
        cost_usd=cost, public_task=env.tasks[0], trace=trace,
        actual_tail_interruptions=sum(bool(r.get('actual_discarded_actions')) and
            not r['done'] and bool(r.get('interruption_reason')) for r in trace))
    save(folder / 'result.json', result)
    return result


class LiveTransport:
    """The caller holds the shared campaign lock for the entire run.

    Actor AND proposer calls use this ledger. Unknown usage is never released.
    A conservative per-request byte bound raises the reservation when a long
    skill/proposal could exceed the normal $0.50 reserve.
    """
    def __init__(self, root, budget, settings, *, connection_retry=False, candidate_scopes=None, network_route="environment", training_cache=None, branch_cache=None, footer_contract=False, pooled_transport=False, citation_mode="per_edit"):
        self.root, self.budget, self.settings = root, budget, settings
        self.connection_retry = connection_retry
        self.candidate_scopes = candidate_scopes
        self.network_route, self.training_cache = network_route, training_cache or {}
        self.branch_cache = branch_cache or {}
        self.footer_contract = footer_contract
        self.pooled_transport = pooled_transport
        self.citation_mode = citation_mode
        self._shared_http = self._shared_api = None
        self._connection_check_count = 0

    def request(self, prompt, *, system, max_tokens, label):
        try:
            return self._request_once(prompt, system=system, max_tokens=max_tokens, label=label)
        except CampaignHalt as exc:
            state = self.budget.read()
            last = state['requests'][-1] if state['requests'] else {}
            if not (self.connection_retry and exc.status == 'api_error' and
                    last.get('error_type') in ('APIConnectionError', 'APITimeoutError') and
                    state['consecutive_errors'] < 2):
                raise
            # One predeclared transport retry of the exact prompt. The failed
            # request remains charged; no trajectory/task/result replacement.
            save(self.root / 'transport' / f"{last['id']:06d}.retry.json",
                 {'failed_request_id': last['id'], 'label': label, 'maximum_retries': 1,
                  'same_prompt': True, 'unknown_reservation_retained': True})
            return self._request_once(prompt, system=system, max_tokens=max_tokens, label=label)

    def _request_once(self, prompt, *, system, max_tokens, label):
        if self.pooled_transport:
            self.warm_connection(label)
        state = self.budget.read()
        upper = ((len(prompt.encode()) + len(system.encode()) + 2048) * state['input_usd_per_million'] +
                 max_tokens * state['output_usd_per_million']) / 1e6
        index = self.budget.reserve(label, max(state['reservation_usd'], upper))
        path = self.root / 'transport' / f'{index + 1:06d}.json'
        record = {'label': label, 'request_index': index, 'status': 'pending',
                  'system': system, 'prompt': prompt, 'max_tokens': max_tokens}
        save(path, record)
        api = None
        try:
            api = ApiPlanner(replace(self.settings, max_tokens=max_tokens), system_prompt=system)
            if self.pooled_transport:
                api.client.close()
                api.client = self._shared_api
            elif self.network_route == 'direct':
                import httpx
                from openai import OpenAI
                api.client.close()
                api.client = OpenAI(base_url=self.settings.base_url,
                    api_key=os.environ['TARGET_AZURE_OPENAI_API_KEY'], max_retries=0,
                    timeout=self.settings.request_timeout, http_client=httpx.Client(trust_env=False))
            reply = api(prompt)
        except Exception as exc:
            self.budget.settle(index, error=exc)
            chain=[]
            cause=exc
            while cause is not None and len(chain)<8:
                chain.append(type(cause).__name__)
                cause=cause.__cause__
            record.update(status='api_error', error_type=type(exc).__name__, error_chain_types=chain)
            save(path, record)
            raise CampaignHalt('api_error') from None
        else:
            self.budget.settle(index, response=reply)
            record.update(reply, status='returned')
            save(path, record)
            latest = self.budget.read()
            row = latest['requests'][index]
            if latest['policy_violation']:
                raise CampaignHalt('thinking_policy_stop')
            if row['status'] != 'returned_usage':
                raise CampaignHalt('unknown_cost_stop')
            return {**reply, 'cost_usd': row['budget_charge_usd']}
        finally:
            if api is not None and not self.pooled_transport:
                api.close()

    def warm_connection(self, label):
        """Probe without an API key before reserving a generation, then reuse TLS.

        Fixed maximum three connection probes. These are HEAD requests to the
        same configured service's models route, never generation requests.
        A failed probe cannot create an unknown paid reservation.
        """
        if self._shared_http is None:
            import httpx
            from openai import OpenAI
            timeout=httpx.Timeout(self.settings.request_timeout,connect=10)
            self._shared_http=httpx.Client(trust_env=self.network_route!='direct',timeout=timeout,
                limits=httpx.Limits(max_connections=1,max_keepalive_connections=1,keepalive_expiry=120))
            self._shared_api=OpenAI(base_url=self.settings.base_url,
                api_key=os.environ['TARGET_AZURE_OPENAI_API_KEY'],max_retries=0,timeout=timeout,
                http_client=self._shared_http)
        for attempt in range(3):
            started=time.time()
            row={'label':label,'attempt':attempt+1,'method':'HEAD','api_key_supplied':False,'paid_requests':0}
            try:
                response=self._shared_http.head(self.settings.base_url.rstrip('/')+'/models',timeout=10)
                row['status']=response.status_code
            except Exception as exc:
                row['error_type']=type(exc).__name__
            row['elapsed_seconds']=time.time()-started
            self._connection_check_count+=1
            save(self.root/'connection_checks'/f'{self._connection_check_count:06d}.json',row)
            if row.get('status') in (200,204,301,302,401,403,404,405):
                return
        raise CampaignHalt('network_preflight_failed')

    def close(self):
        if self._shared_api is not None:
            self._shared_api.close()
            self._shared_api=self._shared_http=None

    def actor(self, label):
        owner = self
        class Actor:
            cost = 0.0

            def before_call(self):
                data = owner.budget.read()
                if time.time() >= data['deadline_unix']:
                    raise CampaignHalt('deadline_stop')
                if len(data['requests']) >= data['max_requests']:
                    raise CampaignHalt('request_cap_stop')

            def __call__(self, prompt):
                result = owner.request(prompt, system=SYSTEM, max_tokens=owner.settings.max_tokens, label=label)
                self.cost += result['cost_usd']
                return result
        return Actor()

    def proposer(self, max_tokens):
        """Legacy one-call proposer, retained only for compatibility tests."""
        def propose(prompt, directory):
            return self.request(prompt, system='Propose bounded joint skill/policy updates. Return JSON only.',
                                max_tokens=max_tokens, label=str(directory) + '/optimizer')
        return propose

    def candidate_builder(self, max_tokens):
        from .skillopt_core import SkillOptCore
        if self.candidate_scopes is None:
            return SkillOptCore(self.request, max_tokens=max_tokens)
        owner = self
        class ScopedBuilder:
            pipeline = SkillOptCore.pipeline
            def __call__(self, parent, evidence, directory):
                scope = owner.candidate_scopes[int(directory.name)]
                return SkillOptCore(owner.request, max_tokens=max_tokens, scope=scope,
                                    require_branch_evidence=True, footer_contract=owner.footer_contract, citation_mode=owner.citation_mode)(parent, evidence, directory)
        return ScopedBuilder()

    @staticmethod
    def environment(gamefile):
        from skillopt.envs.alfworld.rollout import build_alfworld_env
        os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
        split = {'train': 'train', 'valid_seen': 'eval_in_distribution',
                 'valid_unseen': 'eval_out_of_distribution'}[Path(gamefile).parent.parent.parent.name]
        return build_alfworld_env(env_num=1, eval_dataset=split, seed=42,
                                 is_train=split == 'train', specific_gamefiles=[str(gamefile)])

    def evidence_enricher(self, rows, evidence, genome, directory):
        from .branch_feedback import enrich
        return enrich(self, rows, evidence, genome, directory)

    def evaluator(self, tasks, genome, directory):
        from skillopt.envs.alfworld.rollout import build_alfworld_env
        os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
        rows = []
        for task in tasks:
            from .resume import reuse_training
            cached = reuse_training(self.training_cache, task, genome, directory)
            if cached is not None:
                rows.append(cached)
                continue
            path = Path(task['gamefile'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != task['game_sha256']:
                raise ValueError('Gamefile changed since preparation')
            split = {'train': 'train', 'valid_seen': 'eval_in_distribution',
                     'valid_unseen': 'eval_out_of_distribution'}[task['dataset_split']]
            env = build_alfworld_env(env_num=1, eval_dataset=split, seed=42,
                                    is_train=split == 'train', specific_gamefiles=[str(path)])
            actor = self.actor(str(directory / task['id'] / 'actor'))
            try:
                row = evaluate_episode(env, task, genome, self.settings, actor, directory)
            finally:
                env.close()
            label = str(directory / task['id'] / 'actor')
            row['cost_usd'] = sum(r['budget_charge_usd'] for r in self.budget.read()['requests'] if r['label'] == label)
            save(directory / 'predictions' / row['id'] / 'result.json', row)
            rows.append(row)
            if row['status'] not in CLEAN | {'format_error'}:
                raise CampaignHalt(row['status'])
        return rows
