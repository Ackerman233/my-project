# 自动 skill 进化 MVP

实现日期：2026-09-28。这是训练-free 的提示文档和声明式控制规则搜索，不是 RL、权重微调或成功率结论。

## 闭环

1. 初始 skill 在 train 题上运行真实环境，保存成功和失败轨迹。
2. 同一模型读取 train 证据，输出候选 chunk 指导、恢复指导、最大 chunk 长度、中断信号阈值。必须引用至少一个成功、一个失败样本；API 错误不作为策略失败证据。
3. 使用官方 `skillopt.optimizer.skill.apply_patch` 生成候选 skill 文档；没有修改官方 SkillOpt。反思 schema、搜索循环及成对 gate 是本项目实现，并非原样运行 SkillOpt 官方完整训练器。
4. 候选和当前 skill 都重新在 validation 环境运行。任一原本成功题退化则拒绝；成功数提高可接受；成功数不变时，只允许共同成功题调用数下降，且动作数、非法动作不增加。全失败但更省调用不接受。
5. 接受则替换当前 skill，下一代重新生成训练轨迹；拒绝则保留原版本，同时保留候选和负结果。
6. 进化结束冻结 skill；test 仅运行一次三臂比较：ReAct 单动作、初始动态 skill、进化后动态 skill。模型、题目、温度、动作/调用预算一致。test 结果永不用于本次候选生成。

chunk 是模型一次输出的真实 action 数组，队列内不再次调用模型。skill 指导条件性分组和停止边界，不是照抄成功轨迹。中断规则目前的搜索空间是 invalid_action、action_loop、stalled 三种已有 checker 信号的连续阈值 1..3；运行时真实执行，达到阈值后清空剩余动作并重规划。不是任意 Python 策略，也没有预测隐藏状态、solver 或已学会最佳切分的宣称。terminal 与硬预算不能被候选关闭。

## 先做零费用验证

在远程 `/opt/data/private/src/my-project`：

```bash
.venv/bin/python -m pytest -q tests/test_evolution.py
```

完整集成用脚本模型故意构造成功、失败和改进来验证数据流；它**不是在线模型分数**。单测包含真实环境执行、队列中断、候选解析、数据分离及 gate。

## 准备数据

目前完整自动闭环支持 Rush Hour / Sudoku，不支持 ALFWorld 自动进化；ALFWorld 原有动态执行和建议导出仍保留。不要把 puzzle 进化出的规则未经验证当作 ALFWorld skill。

先准备统一 task JSON：Rush Hour 包含 id、benchmark=rush_hour、board（36字符）、minimum_actions；Sudoku 包含 id、benchmark=sudoku、puzzle（9×9数字）、minimum_actions。minimum_actions 仅记录，不传给模型。用真实任务 ID 替换下列示例：

```bash
.venv/bin/python experiments/prepare_evolution.py \
  --tasks data/new_tasks.json \
  --train train_01 train_02 \
  --validation val_01 val_02 \
  --test test_01 test_02 \
  --out data/evolution_splits.json
```

拒绝重复 ID、重命名的重复初始状态、跨 split 的相同 family_id。检查不能自动识别车辆重命名/数独置换后的等价题：数据制作者必须标 family_id 并排除以前调 prompt 用过的题。旧 curriculum 可用于代码开发，但不要重新标成真正未见测试集。至少有成功和失败 train 样本才能产生候选；缺少一类会记录原因并不进化，不制造虚假样本。

## 正式运行（会收费，本次未执行）

须先获得有效的付费授权账本，API key 通过已有安全方式加载至 TARGET_AZURE_OPENAI_API_KEY。下面命令**不会创建、清零或扩展账本**。原 GPT 账本已达 7 请求上限，不能直接用于新实验；不能换路径偷偷重开预算。

```bash
.venv/bin/python experiments/campaign_runner.py --skill-evolution \
  --splits data/evolution_splits.json \
  --out results/evolution_run_01 \
  --budget-json results/AUTHORIZED_CAMPAIGN/budget.json \
  --model gpt-5.4 --generations 2 --max-steps 48 --max-calls 48 \
  --preflight
```

移除 `--preflight` 才实际调用。preflight 不发送请求、不修改账本；锁文件可能被创建。模型也支持 qwen3.5-35b-a3b。GPT 使用 reasoning_effort=none，Qwen 使用 enable_thinking=false；actor action-only，反思 JSON-only。接口沿用 https://api.inferera.com/v1，45秒超时、SDK不重试、全程串行并持有该 campaign 锁。

训练、验证、反思、测试所有请求都走同一个 BudgetPlanner。未知 usage 保留预留并立即停，返回思考也停，已有冷却和截止时间不绕过；prompt 太长可能超过单次预留时在付费前停止。训练证据采样首尾/失败窗口、最多8步，保留原 step 索引及截断标记；完整轨迹仍落盘。预算不足不生成“完成比较”结论。账本费率是保守估算，不是中转商账单担保。

## 看哪些文件

- `manifest.json`：固定分组、协议、数据指纹。
- `generation_*/train/`：每代真实轨迹；`train_evidence.json`：提供给反思器的训练证据。
- `proposal_prompt.json` / `proposal_response.json`：完整反思输入输出（不存 key）。
- `candidate.json` / `candidate.md`：待验证的可执行规则和 skill 文档。
- `validation_incumbent/`、`validation_candidate/`、`gate.json`：成对验证和接受/拒绝原因。
- `history.json`、`best_skill.json`、`best_skill.md`：进化历史与最终冻结版本。
- `test/*/predictions/*/model_calls.json`、`conversation.json`：新生成的测试调用及轨迹，带 skill hash。
- `comparison.json` / `comparison_summary.json`：三臂成功、调用数、动作数、非法动作、中断、多动作 chunk、tokens、延迟及估算费用。
- `budget_snapshot.json`：包含训练/反思/验证/测试的总账；不要仅比较 test 费用就声称总成本下降。
- `run_status.json`：completed 才是完成实验；stopped 是部分记录。接纳 skill 是小验证集上的候选选择，不是统计显著性或泛化保证。

先跑小规模闭环，不追逐单个样本。之后可以加固定 chunk、仅学 chunk、仅学 interrupt 消融及重复种子，并单独验证成功率与调用节省。

API 参数核对：OpenAI Docs https://developers.openai.com/api/docs/models/gpt-5.4 。官方支持不等于中转商实现已独立认证。
