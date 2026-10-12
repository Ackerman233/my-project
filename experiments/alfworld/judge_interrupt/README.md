# 动作队列的模型检查器

在现有 ControlExecutor 的检查阶段接入一个独立调用的 judge。检查结果在执行器清空
队列之前生效，沿用原始动作、终止逻辑和中断次数上限。代码在
`dynamic_skill/controllers/judge.py`；实验入口为本目录的 `cli.py`。

## 角色与费用

| 角色 | 输入 | 输出 | 调用时机 |
| --- | --- | --- | --- |
| actor | 技能、当前观察、历史、可用动作 | 动作队列 | 队列为空时 |
| 规则检查 | 最新反馈、当前队列、公开可用动作 | 是否有检查信号 | 每个动作之后 |
| judge | 任务、最新反馈、剩余队列、最多 8 步公开历史、当前可用动作 | continue / interrupt / defer + 一句证据 | 有信号且剩余队列非空时 |

judge 可以用独立模型；`judge_model` 默认为 actor 的同一模型，独立 system prompt、独立
请求，不共享隐含对话。两种候选角色是 `brief` 和 `dependency`，后者强调失败动作与队尾
之间是否存在依赖。提示词在准备阶段冻结；本轮不调用反思优化器改提示词。

规则门控信号：原规则异常、**下一个动作**当前不可用、新出现的公开 take 动作。
后续动作不能要求现在就可用，因为前面的动作可能建立它的前提。新 take 信号是
ALFWorld 的机会适配，不代表已经验证跨环境通用性。

judge 不读取隐藏状态、游戏答案、分支终局标签；终止和空队列不调用 judge。
defer、解析失败、截断、检查次数用尽时回退原规则，付费检查仍计入成本。
网络和预算错误停止批次，不伪装成免费回退。

**付费 v1 与当前 v2 要分开看。** 下述 30 回合付费实验使用 v1：可用动作列表排除队首时，
仍询问 judge，允许它继续。实际 7 次继续判断全部放行了随后被拒绝的动作。
当前 `prepare` 改为 `gated-local-judge-v2-hard-evidence`：ALFWorld 接口声明动作列表完整，
**三组都**先用零调用规则拦截不可用的队首，judge 不能推翻这个事实。
只有下一条动作受此约束，后续动作可能由队列前面的动作建立条件。
通用组件的 `authoritative_admissibility` 默认关闭，不能把其他环境的不完整动作提示当作证伪依据。
v2 已做离线测试与 7 个真实前缀重放；尚未做新任务上的完整付费对照，不能继承 v1 的成功率。

“同模型、短提示词”只可能减少单次输入/输出 token，不能预先称为低成本。
评价用 actor + judge 的调用、美元和延迟之和；保留角色明细。独立更便宜模型也要
实际比较判断质量与累计费用，不能仅按模型名称判断。

## 2026-10-10 冻结实验

- 原规则、brief judge、dependency judge，10 个新场景配对，逐题轮换组别顺序。
- 三组保持同一技能、actor 提示词、模型和原始操作器。每题总共 8 次模型调用、40 步；
  judge 最多 4 次、每次输出最多 128 token，并为后续 actor 保留至少一次调用。
- actor 三组统一最多输出 1024 token，避免上一轮 512 token 截断干扰；不直接把本轮
  成功率与上一轮不同任务、不同输出长度的 5/10 比较。
- 既有配对筛选门槛使用**总调用与总费用**。不能丢失原来成功的题，并须新增成功题
  或满足共同成功题上的效率条件；judge 格式错误阻止候选通过。
- 只有筛选通过的最佳角色进入另 10 题确认。无候选通过则跳过确认；本轮没有最终测试集。
- 这是固定角色提示词的先导对照，不是已学习的 judge 或完整 SkillOpt 优化。
  本轮全任务差异不能直接当成某次中断的因果收益；需要另做同一前缀的分支恢复。

本轮结果：规则 2/10、brief 2/10、dependency 3/10；均未通过筛选，无接受更新。
两个真实改判前缀另做了继续/中断对照，也未显示成功或调用/费用净收益。
详细数据和后续修复见[实验报告](../../../研究进度与实验报告/模型Judge低成本检查验证20261010.md)。

累计用户额度：200 美元。单批最多 48 美元，沿用原账本、保留全部历史费用及未知用量预留。
结果、配置、源码快照和请求记录位于 `results/alfworld/judge_interrupt/`。

运行（需在仓库根目录，沿用受保护的既有密钥和网络配置）：

```bash
.venv/bin/python -m experiments.alfworld.judge_interrupt.cli prepare --root results/alfworld/judge_interrupt/RUN_ID
.venv/bin/python -m experiments.alfworld.judge_interrupt.cli authorize --root results/alfworld/judge_interrupt/RUN_ID
.venv/bin/python -m experiments.alfworld.judge_interrupt.cli run --root results/alfworld/judge_interrupt/RUN_ID
```

已有结果目录不能原地重跑；配置冻结后源码变化会阻止请求。历史运行保留其源码快照。
