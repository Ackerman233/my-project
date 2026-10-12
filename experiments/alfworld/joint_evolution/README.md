# 技能与中断策略联合进化

这里实现多轮“执行反馈 → 联合改写技能与中断策略 → 筛选 → 再评估”。
旧的一轮原型和仅控制参数实验保留在 `../skill_evolution/`。

当前已接入并离线验证上游 **Reflect → Aggregate → Rank → Apply** 核心链路。
失败反思可以直接编辑可执行 interrupt 策略，与技能文本一起接受评估。
实际调用的是 `run_error_analyst_minibatch`、`run_success_analyst_minibatch`、
`merge_patches`、`rank_and_select`、`apply_patch_with_report`，没有另写替代算法。
跨轮优化器元记忆（meta-skill）与 slow-update 暂未启用；这与核心链路是否接入分开记录。
已进行真实付费验证，当前没有被接受的联合候选或收益证据；部分批次因连接错误停止。
最新10题筛选中技能候选4/10、初始版本5/10，未接受更新；三个有方向的分支样本均支持现有控制器。详见[160美元阶段报告](../../../研究进度与实验报告/中断学习信号与组件候选验证20261008.md)；[105美元历史报告](../../../研究进度与实验报告/联合进化付费验证20261008.md)保留原始记录。
之前只接入 Apply 的简化版保留为历史测试入口，不再是正式运行默认。

## 每轮怎样运行

1. 当前版本执行训练任务，保存动作、公开反馈、剩余队列、实际丢弃动作、成功与调用成本。
2. 将技能文本与有界策略放进同一个文档。官方 Reflect 分别读取失败/成功训练轨迹；失败反思提出纠错与 interrupt 修改，成功反思提取可复用技能。策略通过完整 JSON 替换补丁修改，每次反思的实际训练输入完整记录。`per_edit`模式强制模型引用；新`batch_provenance`模式贴近原版SkillOpt，引用可缺失，但模型若显式引用则必须合法，缺失字段不代填。输入来源不等于逐编辑因果依据。
3. 官方 Aggregate 按失败优先原则聚合补丁，Rank 选择最多4项，Apply 应用选中的真实编辑。再解析策略并检查父版本、编辑目标和参数范围；允许只改技能、只改策略，或同时改两者；至少一个组件必须改变。空补丁或不合法候选会被拒绝，不强制产生更新。
4. 在选择集上重新运行当前版本及候选。成功任务不能退步；若成功数不增加，必须在共同成功任务上减少调用，且费用、步数、非法动作不增加。两个都失败但更早退出不算收益。
5. 只将通过筛选的最佳候选送入本轮预先划定的确认集，与当前版本配对复评。每轮确认集不同；未通过就保留旧版本，下一轮重新收集训练反馈。
6. 全部轮次结束后冻结最终版本，才运行测试集四组：初始版本、仅换技能、仅换策略、联合版本。技能和策略完全相同的组复用结果，明确记录为同一配置，不重复计为独立样本。测试结果不能回写选择过程。

选择集会反复使用，属于开发数据；确认集参与版本接受，也不等同于最终测试集。
划分按物理场景检查，不能只改任务 ID。准备阶段检查历史结果中的场景暴露；
如显式启用 `allow_previously_exposed_evaluation: true`，只能作为开发实验，不能声称独立评估。
这仍是 ALFWorld 适配，没有跨环境泛化证明。当前四组用于分解技能/策略贡献；
它们不能代替与同权限 ReAct、原版 SkillOpt 的正式对照。

## 文件职责

| 文件 | 内容 |
| --- | --- |
| `schema.py` | 技能/策略版本、上游文本编辑、有界策略与划分检查 |
| `skillopt_core.py` | 官方 Reflect/Aggregate/Rank/Apply 接入、联合文档解析、逐阶段证据与调用审计 |
| `branch_feedback.py` | 从真实训练队尾重放同一前缀，比较继续/中断；只向训练反思提供带不确定性的结果 |
| `engine.py` | 多轮执行、训练证据、筛选、确认与冻结测试 |
| `runtime.py` | ALFWorld 适配、执行及优化请求的共享预算结算 |
| `cli.py` | 离线演示、配置冻结、检查及正式运行入口 |
| `skillopt_demo.py` | 零 API 演示：真实失败进入官方链路，脚本提供模型回复 |
| `demo.py` | 旧单次直接提案演示及共享小环境，仅保留历史测试兼容 |
| `config.example.json` | 正式运行配置模板，场景路径必须替换为真实数据 |

可优化策略：`max_actions` 1–4，`trigger` 为 invalid/anomaly，
`invalid_patience` 1–2，`recovery_steps` 0–3，以及布尔型的
`state_check`、`discovery_interrupt`、`history_evidence`。
这些字段直接影响执行行为；不是仅写进提示的建议。当前演示中的 patience=2
只是测试用候选，不能替代“证伪与依赖”判断，也没有被设为全项目默认规则。

## 运行

从项目根目录使用已安装依赖的 `.venv/bin/python`；不需要切换 conda。

```bash
.venv/bin/python -m experiments.alfworld.joint_evolution.cli demo \
  --out results/alfworld/joint_evolution/my_offline_demo
.venv/bin/python -m experiments.common.verify_interrupt_recovery_offline -q
```

演示跑两轮，使用真实队列/控制执行路径，但模型回复由脚本生成。
第一轮包含失败与成功训练轨迹：原策略因丢弃可执行导航队尾而在1次规划预算内失败；
失败 Reflect 修改策略，成功 Reflect 补充技能，Aggregate 合并为5项编辑，Rank 选择4项，
Apply 后候选在选择/确认任务中完成目标。第二轮没有进一步收益，被拒绝。
四组对照显示这个构造例子的差异来自策略。全部模型回复是脚本，不能称为模型学习成果或 ALFWorld gain。

每个候选最多4次优化请求：失败 Reflect、成功 Reflect、Aggregate、Rank。
空轨迹组会跳过反思；单一补丁组时 Aggregate、编辑池不超限时 Rank 会沿用上游的
无模型调用路径，日志明确记录，不能虚报优化请求。上游声明的3次重试在本适配器中
不生效；调用失败、截断、非法 JSON 或证据引用不合规时，不允许靠上游备用编辑接受候选。

正式配置必须填入真实的 `game.tw-pddl` 路径，`scene_id` 等于它的上两级目录名。
训练、选择、每轮确认、最终测试互不重合；确认折数等于 `generations`。
模板的 URL、模型与共享账本价格口径必须在运行前核对，不要用模板推断当前服务定价。

```bash
.venv/bin/python -m experiments.alfworld.joint_evolution.cli prepare \
  --config /path/to/filled_config.json \
  --out results/alfworld/joint_evolution/my_live_batch
.venv/bin/python -m experiments.alfworld.joint_evolution.cli preflight \
  --root results/alfworld/joint_evolution/my_live_batch
# 以下打开现有预算内的一次有界批次，并执行付费请求：
.venv/bin/python -m experiments.alfworld.joint_evolution.cli authorize \
  --root results/alfworld/joint_evolution/my_live_batch
.venv/bin/python -m experiments.alfworld.joint_evolution.cli run \
  --root results/alfworld/joint_evolution/my_live_batch
```

`prepare`/`preflight` 不请求模型，也不打开账本额度；`authorize` 是沿用项目预算工具的
批次状态转换，不增加用户最新授权的累计 $160 上限。执行与优化模型请求都计入同一共享
账本，并通过同一个运行锁串行化。每次先保守预留，再按 usage 结算；缺失 usage 的预留
不释放。预算耗尽、API/基础设施异常停止运行，默认不重试；160美元批次预先启用最多一次连接/超时原提示重试，保留未知费用，连续两次错误仍停止，不替换任务或重跑挑选结果。
已有运行目录不覆盖；未完成运行明确标记 `stopped`，不能当成完整收益结果。
新批次协议为 `joint-skill-interrupt-live-v3`，冻结上游反思/聚合/排序/编辑源码及提示模板，
请求上限包含每候选最多4次逻辑优化调用、训练恢复分支以及配置启用的一次传输重试。旧v1冻结批次不能隐式改成新链路运行。

## 产物

源码目录不保存运行中间文件。全部放在 `results/alfworld/joint_evolution/<run-id>/`：

```text
config.json, source_snapshot/       冻结配置和实际源码（正式运行）
transport/, accounting.json        请求及总费用（含优化成本）
run/
  manifest.json, status.json
  generations/000/
    execution_feedback/, branch_feedback/, training_evidence.json
    candidates/00/
      optimizer/{inputs.json,initial_joint.md,training_predictions/,calls/}
      optimizer/{reflect_failure.json,reflect_success.json,aggregate.json,rank.json,apply.json,status.json}
      genome.json, skill.md, screen.json, selection/
    selection/incumbent/, confirmation/, decision.json
  best/{genome.json,skill.md,frozen_before_test.json}
  test/{initial/,skill_only/,policy_only/,evolved/,comparison.json}
```

离线演示直接将上述 `run/` 内容放在指定目录，附 `DEMO_ONLY.json`。
真实环境单步适配检查附 `SMOKE_ONLY.json`，也不属于性能评估。
正式研究报告应同时报告最终测试效果、训练/候选/确认/优化的总研发费用、协议失败和真实截断次数。

## “允许联合优化”是什么意思

两个组件都是可学习对象，但不要求每次一起改变。技能决定模型如何规划；中断策略决定队列何时被截断及怎样恢复。比如证据只支持改搜索顺序，就仅改技能；若固定前缀显示丢弃可用队尾有害，就可提出仅改策略的假设，再用新任务筛选和确认。引用一条分支证据不是任意策略参数都有因果收益的证明。

160美元批次各划分10题。每个训练任务最多取一个真实非空队尾，继续和中断分支各限4次调用、16步；双方都失败记为未知，双方成功再看调用、动作和费用。两个候选槽分别探索技能和策略（后者可连带改技能），均可拒绝更新。

连接故障续跑的完整训练/分支结果可按hash复用，包括任务失败结果。评估预留任务仅在可证明从未执行时继承；已看过的评估结果不能用于新协议的独立测试。`test_unchanged=false`时，没有接受更新就保留测试集，不反复购买同一基线。`pooled_transport=true`复用HTTP连接，并在模型计费预留前最多做3次无密钥HEAD连接检查；这些检查不调用生成接口。
