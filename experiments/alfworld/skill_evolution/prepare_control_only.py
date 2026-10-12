"""仅离线准备单参数邻域；没有API客户端，不生成模型成绩。"""
from pathlib import Path
import json
from dataclasses import asdict
from dynamic_skill.control_only_evolution import ControlPolicy,neighbors,fingerprint,accept_proposal
from experiments.puzzle_pilot import save

def main():
    root=Path('results/micro_evolution_qwen36_20260929')
    cfg=json.loads((root/'config.json').read_text())
    out=root/'control_only_v2'
    if out.exists():raise RuntimeError('已准备，禁止覆盖')
    skill=cfg['initial_skill'];baseline=ControlPolicy()
    candidates=[]
    for i,policy in enumerate(neighbors(baseline)):
        proposal={'task_skill_sha256':fingerprint(skill),'control':asdict(policy)}
        checked=accept_proposal(proposal,baseline,skill)
        candidates.append({'id':f'control-{i:02d}',**proposal,'changed_fields':checked['changed_fields'],'status':checked['status']})
    save(out/'manifest.json',{'protocol':'control-only-v2','task_skill':skill,'task_skill_sha256':fingerprint(skill),
         'candidates':candidates,'api_calls':0,'selected_candidate':None,
         '说明':'9个离线参数配置（含初始）；仅schema与实现测试，不代表模型进化或性能结果。任务文本冻结，不修改上一轮结果。',
         'evaluation_rule':'仅开发轨迹筛选；单独验证前不保留。原测试题已看过，不再用作未来独立测试。未来比较须同时重跑同一V2基线。'})
    print(json.dumps({'candidates':len(candidates),'api_calls':0,'output':str(out)}))

if __name__=='__main__':main()
