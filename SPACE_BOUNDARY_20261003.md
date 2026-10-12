# SPACE 对照与边界诊断（2026-10-03）

## 最终摘要：本轮有代码机制进展，没有可宣称的性能 gain

| 边界提示实验 | 成功/两题 | 总调用 | 多动作调用比例 | 平均计划长度 | 真实截断 |
| --- | --- | ---: | ---: | ---: | ---: |
| E 原提示 | 2/2 | 11 | 9.1% | 1.18 | 0 |
| F 边界提示 | 1/2 | 27 | 3.7% | 1.04 | 0 |
| G 边界提示＋发现中断 | 2/2 | 22 | 9.1% | 1.14 | 0 |

更长的提示没有解决单动作倾向，不设为默认。G 的中断从未触发，不能将成功归因于它。三组温度 0 也不代表中转模型完全确定；小样本单次比较不能做稳定因果结论。

随后独立做了 2 题 × 2 组子技能接口试验（有/无 discovery interrupt），借鉴 SPACE 的 skill-augmented 侧，而非其部署时已蒸馏策略：

- dynamic_skill/space_subskills.py 提供人工参数化 @inspect、@scan。模型仍选择参数；程序展开原始动作，每个动作真实执行并计入步数。model_native_actions/subskill_expansion 保留来源，不冒充模型自己输出多条 primitive。
- 第一版（space_subskills_20261003）：H 两个格式失败；I 一个格式失败、一个成功。模型确实选择并执行了多动作子技能，但数组组合规则不稳定。
- 第二版（space_subskills_v2_20261003）：允许组合扫描项、对台面 inspect 只执行导航；H/I 都是灯下查看成功、取放因缺失 actions 标签格式失败。H 成功题 3 次调用/4 步，I 成功题 2 次调用/3 步；真实截断仍为 0，不能归因于中断改善。首版与第二版分开保存，不能挑最好结果合并成功率。
- 最终仅离线增加严格的缺失 actions 标签正规化：仅接受完整 thought 后紧跟 JSON 数组，不接受多余文字、不猜动作、不触发付费重试；保留原始响应并记录 protocol_normalized。这个最后解析修复还没有新的在线成绩。

最终 172 passed、5 skipped；累计本轮付费 75 次，均有 usage、无悬挂请求/接口错误。保守账本增加 $4.353300，累计 $48.857600，累计上限 $85；不是中转站实账对账。运行已结束、锁已释放。没有把新增上限全部花完。

所有新开关默认关闭，旧路径未自动升级。新接口目前适合继续机制诊断，不能作为已经胜过原方法的正式算法。下一步应先验证最后的协议适配，再在独立开发任务上测程序子技能的多动作率、成功率和成本；只有出现机制暴露后才评价 interrupt。不要直接开展全量 benchmark。

## 论文与当前实现的区别

来源：用户提供的《Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents》，arXiv:2609.02042v1。

- 第 3–5 页：每次输出 1..K 个 primitive actions，顺序执行，非法动作可提前终止。成功轨迹被归纳成 composite skill 与 subskills；一个子技能对应一个局部 chunk。
- 第 5–6、17 页：混合 primitive rollout 与 skill-augmented rollout，把技能执行展开为具有边界的动作监督，并用 on-policy RL 与 off-policy distillation 训练。不是只改提示词。
- 第 6、11 页：no-thinking 模式，最大 chunk=6；冷启动程序技能提供初始样例。当前仍保留 max_actions=4，避免同时改变长度上限。
- 第 19–21 页：primitive prompt 允许多动作并提供样例；skill induction 要求局部例程、参数化、保留运行观察，不发明缺失信息。附录提示有显式推理措辞，但论文实验声明 no-thinking；本项目保持关闭原生思考，只输出简短 thought 与 JSON actions。

当前没有可训练本地 policy，本轮没有 SPACE 的 RL、离线蒸馏或自动诱导程序库，只借鉴“明确局部子技能边界”的推理时脚手架。因此不能声称复现 SPACE 或获得其训练增益。

## 原单动作倾向与改动

上一轮 59 次调用中，53 次输出单动作、6 次输出多动作。原提示虽然允许 1–4 个动作，但只有单命令格式示例，且要求 Stop before unknown information；后者可能把可预测状态转移也解释成必须等待观察。此为可检验解释，不是已经证实的唯一原因。

dynamic_skill/space_boundary.py 新提示区分：

1. 可合并的取物→导航→放置局部例程（对象编号已知，目标已知可接近）。
2. 去未知容器→打开后读取内容，是合理边界；不能预先猜取物编号。
3. 多个已知地点可形成短搜索队列；单动作仍被允许，不补 no-op、不强制最小长度。

新增 discovery interrupt：只支持简单 put/examine 任务目标语法。执行 go/open/examine 后读取当步公开 admissible commands，若出现精确目标类型和编号的 take 命令，且剩余队列全是搜索动作、下一步将离开目标所在地，则清空搜索队尾并通知下一次规划。无额外模型调用、无自动取物、无隐藏状态。它是机会发现，不是失败，单独记录 discovery_pause，不加入未解决失败列表。未知、错误对象、空队列、已计划取物都不触发。

旧模式默认关闭，全部改动以可选参数接入；没有扩大到 skill 自进化。修改旧共享执行文件会使旧冻结配置的源码校验失效，应使用新实验版本，不能静默继续旧配置。

## 离线验证与实验设计

162 passed，5 skipped。新测试验证默认关闭、真实队尾丢弃、精确对象类型、当前公开证据、已含取物队列不误截断；构造测试不是模型效果。

E_old：原提示＋历史增强＋原对象状态检查。
F_prompt：E＋新边界提示。
G_discovery：F＋目标发现中断。

统一 qwen3.6-35b-a3b，enable_thinking=false，temperature=0，chunk≤4，384 输出 token，每题 20 调用/50 环境步/210 秒，recovery_steps=0。每组相同两个已观察开发题，非独立验证；相对原 12 调用协议提高了上限，但本轮三组完全一致，不能直接混合前后成功率。

新独立目录 results/space_boundary_20261003/，保留所有历史实验。累计上限按用户授权 $85，本批额外上限 $12、120 次请求、1 小时；未知 usage 保留预留，出错即停止新增调用，不自动重试。调用仅通过 campaign_runner.py → BudgetPlanner/TrialBudget/CampaignBudget，共享锁和账本。

复现检查：

```bash
cd /opt/data/private/src/my-project
.venv/bin/python -m experiments.verify_interrupt_recovery_offline
.venv/bin/python -m experiments.space_boundary_trial --summary
```

冻结入口为 `.venv/bin/python experiments/campaign_runner.py --space-boundary-next`；不要重复授权或覆盖结果。观察多动作比例、实际执行长度、非法动作、真实截断、发现中断、成功率及费用，不能仅按队列变长宣称成功。
