# Puzzle 基础 skill 冷启动

创建日期：2026-09-28。参照官方 ALFWorld initial.md 的 Overview / 子任务 / General Principles / Common Mistakes 结构，但遵守当前关闭思考、action-only 协议，不复制 ALFWorld 的think输出格式。

- `rush_hour/initial.md`：坐标、车辆方向、位移语义、目标车直接出场、阻挡依赖和失败后重算。
- `sudoku/initial.md`：行列宫候选集、显性/隐性单数、逐步填数、只撤销自己的可编辑项。

这些是人工初始化的通用任务方法，不是从本轮题目答案提炼的规则。它们不决定 chunk 大小或中断阈值；后续 ReAct 和动态方法必须共享同一份基础文本，再比较控制层。

接入：`experiments.puzzle_pilot.run_episode(..., base_skill_text=...)`，任意模式可用。未显式传入时默认仍为空，不改变历史实验。每次调用和结果记录base_skill_sha256；完整文本包含在实际prompt以及本次config中。

冷启动门槛命令：`python experiments/campaign_runner.py --cold-start-gate`。该专用入口读取本目录两份initial.md；6道新固定合成题，各环境最少1/2/3动作，单动作ReAct，每题最多6调用/6动作。GPT-5.4，reasoning_effort=none，温度0，输出上限128，SDK无重试。

门槛提前固定为：完成至少5/6，每个环境至少2/3，非法动作比例不超过10%，且无API/格式/usage缺失错误。不是无skill对照，不能据此单独估计冷启动的因果收益；也不代表长程能力。若以后修改skill，保留版本并使用新的确认题集。

当前状态：94 passed / 5 skipped；题目离线可解性已验证，但真实模型门槛尚未运行。2026-09-28共享GPT账本累计预算占用$4.85855 / $5，剩余低于每请求$0.50预留，在线入口在API请求前停止，本任务新增费用$0。不得重置或换账本隐瞒原费用；需要用户明确新的预算上限后才能测试。

题目：`data/cold_start_gate_20260928/tasks.json`。`validation_private.json`只供离线验证，不由在线入口读取。题目固定后未看模型结果、未调低通过标准。API参数参考OpenAI Docs：https://developers.openai.com/api/docs/models/gpt-5.4 。
