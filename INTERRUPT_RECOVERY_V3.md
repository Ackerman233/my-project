# Interrupt / recovery 可验证闭环（2026-09-30）

## 2026-10-02 授权运行结果（以下覆盖旧文中“尚未运行”的状态）

用户明确授权启动后执行冻结的 2 题 × A/B/C/D。59 次新增请求均返回 usage，无接口错误、无悬挂请求；运行锁已释放。7 个 episode 完整结束，D 第二题在 2 次调用后被预算保护中止，不能作为任务失败或完整配对样本。

| 组 | 第一题：取放 | 第二题：灯下查看 | 总调用 | 环境步 | 真实截断 |
| --- | --- | --- | ---: | ---: | ---: |
| A | 12 调用上限，未完成 | 成功 | 14 | 18 | 0 |
| B | 成功 | 成功 | 14 | 14 | 0 |
| C | 12 调用上限，未完成 | 成功 | 17 | 18 | 0 |
| D | 12 调用上限，未完成 | 预算中止，结果缺失 | 14 | 15 | 0 |

C 第一题第 11 步 go to drawer 4 收到 Nothing happens；当时没有待丢弃队尾，因此 anomaly_replans=1、interrupt_count=0。下一次规划（第 12 次调用）确实注入历史证据，输出仍为 thought/actions，但未在剩余调用额度内完成任务，恢复保持 unverified。D 共执行 2 次适用状态检查，没有 semantic pause。B 虽然 2/2 成功，却没有一次真实截断，不能把成功归因于 interrupt。模型多数只输出单动作，chunk 机制暴露不足；此轮不能证明方法有效或无效。

账本保守新增费用 $2.857075，累计 $44.504300（非中转站实账确认）。下一次请求预留 $0.50 将超过累计 $45 上限，因而正常停止；没有降低预留、提高预算或重试。结果与完整 prompts/轨迹在 results/interrupt_recovery_v3/online/，汇总 summary.json。汇总中的 D success_rate=0 是原始“成功/尝试”字段，不应拿它作完整任务成功率比较；必须结合 budget_stop 状态。

本次未调整算法或 prompt。下一步若继续，优先审查这些自然轨迹为何很少形成可截断队列，并单独设计机制暴露诊断；不要根据这两个样本直接宣称 gain，也不要未经新授权扩大付费实验。

## 本次范围与安全边界

在现有 dirty 工作区和 dynamic-interrupt-prototype 分支上增量修改，没有重置、覆盖用户未提交改动或修改官方 SkillOpt。部署前逐文件核对快照 SHA；修改前版本保存在 results/interrupt_recovery_v3_code_backup/。本轮付费调用为 0，不启动 benchmark 或 skill 自进化。

## 行为变化

- dynamic_skill/alfworld_recovery.py：复用历史证据整理逻辑，新增 thought_actions 协议；旧 actions_only 默认协议保留。历史记录注明来源、步号、年龄，不保证仍是当前状态。
- dynamic_skill/control_only_evolution.py：control-only 入口新增默认关闭的 history_evidence、state_check；history_evidence 与 recovery_steps 独立。关闭时保持原 prompt 行为。分别记录通用异常提示、真正注入历史证据、实际单步恢复。
- dynamic_skill/acquisition_check.py：只检查任务提到的对象类型对应的具体对象编号，支持 take 的精确取物反馈、inventory 的完整持有列表。confirmed/contradicted/unknown 分开，不将 unknown 当成明确失败或成功。
- experiments/alfworld_campaign.py：提供执行前公开观察与合法动作、规划上下文、动作队列审计日志挂钩。
- experiments/interrupt_recovery_ablation.py 和 campaign_runner.py：准备统一配置 A/B/C/D；当前没有付费授权，运行入口会拒绝未经新授权的付费执行。

## 最小状态检查定义

取物前需要位于来源、对象可接近且取物允许；日志保存当步公开 admissible commands 与上一观察，不能从隐藏状态推断前提。取物后期待 holding(具体对象)。完全匹配“对象编号 + 来源”的 pickup 反馈确认取物；反馈对象不同构成矛盾；其他文本为 unknown。来源不同但同一对象确已拿到时，记录来源偏差，但不阻止仅依赖持有该对象的后续动作。

inventory 仅在整个反馈可解析时判断当前持有；明确空背包或完整列表不含目标构成矛盾，未识别文本为 unknown。旧 pickup 记录绝不当作当前 inventory 证据。

只有剩余队列中 move/clean/heat/cool 对同一具体对象有持有前提时，未确认的持有状态才暂停队列重规划。若先遇到 inventory 或重新 take 同一对象，则先允许真实验证/重新获取；不阻止无关导航。未知时可向规划器建议 inventory，但不自动插入；若规划器选择 inventory，作为真实环境动作计入步数。不能据此宣称整个任务完成。

## 日志

conversation.json 中新增 original_action_queue、action_position（从 0 开始）、remaining_before_interrupt、actual_discarded_actions、state_check（expected/precondition/evidence/status/holding_evidence_status/dependent_actions）、semantic_pause、interruption_reason。

规划日志记录 anomaly_received、state_recovery_received、historical_evidence_injected、generic_anomaly_prompt_used、single_step_recovery_active、effective cap。恢复日志分开记录 recovery_counter_ended 与 original_failure_verified_resolved，并保存 failure_resolution 的具体证据。anomaly_replans 包含队尾异常；interrupt_count 只计算实际丢弃非空队尾。

## 完整例子：真实环境，脚本规划，非模型成绩

先执行 go to desk 1。下一队列：

1. take bowl 1 from desk 1 —— 环境确认取到 bowl 1。
2. move bowl 1 to desk 1 —— 放回桌面。
3. inventory —— 环境返回 `You are not carrying anything.`。
4. move bowl 1 to desk 1 —— 仍然要求 holding(bowl 1)。

关闭状态检查：inventory 没有通用错误，继续第 4 个动作，得到非法反馈，再规划。

开启状态检查：inventory 的当步完整反馈明确排除 bowl 1，持有预期为 contradicted；第 4 个动作依赖该预期，因此实际丢弃它，并将矛盾与对象传入下一次规划。脚本下一轮选择 take bowl 1 from desk 1；仅在收到该对象的精确取物确认后，才将对应失败标为 verified_resolved。

两组都包含 1 次真实 inventory、3 次脚本规划；关闭组 6 步、开启组 5 步，付费模型调用均为 0。这只证明代码能改变后续执行，不是成功率或模型效率 gain。

当前证据目录：results/interrupt_recovery_v3/final_verification/；另有 offline replay 审计及独立人工构造的错对象反馈反例。人工反例不声称是环境自然发生的反馈。

## 测试及复现

```bash
cd /opt/data/private/src/my-project
.venv/bin/python -m experiments.verify_interrupt_recovery_offline
.venv/bin/python -m experiments.audit_interrupt_recovery_v3
audit_dir=$(mktemp -d /tmp/interrupt-recovery-v3-XXXXXX)
ALFWORLD_DATA=/opt/data/private/data/alfworld .venv/bin/python -m experiments.audit_interrupt_recovery_v3 --live-environment --out-root "$audit_dir"
.venv/bin/python -m experiments.interrupt_recovery_ablation --prepare --summary
git diff --check
```

最终测试：158 passed，5 skipped（数据条件式集成测试），1 个无害 pytest 模块重写警告。原有 3 项测试依赖已过期的固定实验截止日期；离线 wrapper 仅在测试进程内调整该常量，并禁止真实 HTTP transport，不修改生产截止日期或预算账本。另已真实启动 ALFWorld 完成上述脚本环境验证。

覆盖队列中段/末尾异常、历史开关与单步恢复独立、协议兼容、无通用错误但明确状态矛盾、unknown、不必要中断防护、恢复计数结束但原失败未解决。3 条既有真实轨迹、37 个动作回放无新决策分歧；不是广泛无误中断的证明。

## 待执行对照配置（未运行）

| 组 | 规则中断 | 历史增强 | 状态检查 |
| --- | --- | --- | --- |
| A 固定 chunk | 关 | 关 | 关 |
| B | 开 | 关 | 关 |
| C | 开 | 开 | 关 |
| D | 开 | 开 | 开 |

统一 qwen3.6-35b-a3b、enable_thinking=false、thought/actions、同一 task skill、同一 2 个预先准备但未尝试的 validation 任务、chunk 上限 4、temperature=0、384 输出 token、45 秒请求超时、每题最多 12 调用/50 环境步/210 秒。recovery_steps 全部固定为 0，不同时调参。

记录成功率、调用/token/成本/延迟/环境步、真实截断与重规划、机制触发、已验证和未验证恢复；接口/格式错误按状态独立展示。误中断经 interrupt_review.json 的证据化人工审核，未判断为 null，不冒充 0。未触发机制则比较无结论。

未来获得新的明确付费授权、绑定配置 SHA/截止日期/请求上限并核对共享账本后，单次推进命令如下；本轮没有执行，也没有添加授权：

```bash
.venv/bin/python experiments/campaign_runner.py --interrupt-recovery-v3-next
```

仍经过共享 CampaignBudget/BudgetPlanner、锁、冷却及未知费用保护。当前直接运行应拒绝，不应绕过授权闸门。

## 尚未实现

没有全任务语义验证、温度/洁净/灯状态检查或自动补充观察；没有把 SkillPlan.active_index/completion_signals 冒充实际动作队列的状态机。这里只是具体取物/持有条件契约。通用失败没有相应可观察验证器时仍为 unverified；倒计时结束不等于修复。没有证据表明任务成功率改善，没有新训练或进化实验。
