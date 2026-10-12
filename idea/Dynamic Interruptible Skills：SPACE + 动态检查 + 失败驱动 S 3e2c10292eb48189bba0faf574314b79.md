# Dynamic Interruptible Skills：SPACE + 动态检查 + 失败驱动 Skill Evolution

<aside>
🎯

核心想法（2026-09-25 更新）：便宜的检查器负责判断“原计划是否还成立”，昂贵的规划器负责决定“接下来怎么做”；执行经验持续改进两者之间的交接时机。以可执行 macro-skill 减少主模型规划调用，用低价模型根据中间 observation 检查剩余计划的适用性，必要时中断并交还规划控制权；再利用成功、失败及恢复轨迹，修正 skill 的适用条件、检查位置与恢复行为，并通过验证后保留更新。核心研究问题是：能否以较低的监控成本，可靠地执行更长的动作序列，并随着经验积累减少不必要的检查和重规划？

</aside>

## 1. 研究动机

SPACE 已经解决了一个重要问题：从成功轨迹中诱导 programmatic skills，并学习合适的 chunk boundary，以减少长程任务中的 LLM decision rounds。

但“学会在哪里切 chunk”仍存在一个问题：**chunk 在开始执行时可能正确，但执行过程中环境状态会改变；若 Agent 仍继续执行整个 macro，就可能产生 silent error / over-commitment。**

> **Research Question：When should an agent commit, continue, or interrupt a macro-action?**
> 

相关旧笔记：[长程agent问题 挑战 ](https://app.notion.com/p/agent-3ddc10292eb4807383e4dcd1bee58df5?pvs=21) · [Training-Free 长程 Agent 优化：Research Gap 初步整理](https://app.notion.com/p/Training-Free-Agent-Research-Gap-3ddc10292eb481f99a85d3ad9ce8a4d4?pvs=21) · [On Training LLMs for Long-Horizon Tasks：长程定义与解决方法 ](https://app.notion.com/p/On-Training-LLMs-for-Long-Horizon-Tasks-3cfc10292eb481aeb1ecc518478f5b24?pvs=21)

## 2. 统一框架：Skill Lifecycle

- 成功轨迹：发现可复用 action chunk / macro-skill，并提供哪些情况下可以继续执行的证据，用于校准过严的 guard、减少误中断。
- 执行中 observation：低价 checker 判断当前计划是否仍成立，决定继续执行、补充观察，或交还主规划器。
- 失败与恢复轨迹：定位原计划何时失效、checker 是否漏检或中断过迟，以及主规划器如何恢复，为后续 skill 更新提供证据。
- Skill update + validation：根据成功、失败及恢复证据，更新适用条件、检查位置、动作分支和 recovery；验证后保留更新，持续改进 checker 与主规划器的交接时机。

整体闭环：成功轨迹诱导 skill → 执行 macro → 低价 checker 验证原计划 → 继续 / 补充观察 / 中断 → 主规划器恢复 → 收集成功、失败与恢复轨迹 → 更新 skill 与检查条件 → 验证 → 再次复用。

## 3. Skill 不只是 Action Sequence

原始 skill 可看成 $S=(a_t,a_{t+1},...,a_{t+k})$。

更完整的定义：$S={Precondition, ActionProgram, Invariants, Postcondition, Recovery}$。

### Skill Body：做什么

由成功轨迹切出可复用 action subsequence，例如：go countertop → take apple → go fridge → open fridge → put apple in fridge。

### Skill Guard：什么时候还能继续

检查时机：首版可在每个 primitive action 后进行低价 observation check，后续根据轨迹证据优化检查位置和频率。检查决策为 continue / interrupt / need-observation；目标是在不反复调用主规划器的情况下，及时识别剩余计划的失效。

Checker 的职责：判断“原计划是否还成立”，重点检查当前阶段的前置条件、中间 invariant、实际结果与预期结果的一致性，以及剩余动作是否仍适用于当前目标。输入包括任务目标、skill 执行位置、必要的局部历史、最新 observation、剩余动作与待验证条件；输出为 continue / interrupt / need-observation，并指出证据或被违反的条件。没有观察到成功不等于失败，证据不足时应补充观察。低价模型主要承担局部验证，完整重规划交给主模型。

### Skill Recovery：失效以后怎么办

Planner 的职责：决定“接下来怎么做”。当 checker 发现原计划失效时，暂停剩余 macro，把失败条件、相关 observation 与当前进度交给主规划器；由其补充探索、生成恢复动作、修订计划或选择另一项 skill。恢复完成后重新进入低价检查与 macro 执行流程。

## 4. ALFWorld 直观例子

Macro：put apple in fridge。

- Precondition：apple is accessible。
- Invariant 1：after TAKE(apple), inventory contains apple。
- Invariant 2：after OPEN(fridge), fridge is open。
- Postcondition：apple is in fridge。
- Fallback：若 apple 未拿到或 fridge 无法打开，则 interrupt / replan。

若执行 TAKE(apple) 后环境返回 “You cannot see any apple here.”，open-loop macro 可能仍继续去 fridge；动态机制则会检测 invariant violated，立即 interrupt，并回到 primitive reasoning / replanning，避免错误继续传播。

## 5. 成功、失败与恢复轨迹如何用于 Skill Evolution

Skill evolution 的对象包括动作程序、适用条件、检查位置与恢复行为，重点学习“什么时候继续信任 skill，什么时候交还规划控制权”。成功轨迹用于识别合理的继续执行条件、校准误中断；失败轨迹用于识别遗漏条件和中断过迟；恢复轨迹用于提炼有效的补救步骤。整条轨迹的成败不能直接作为其中每一步的正确性标签，需要结合局部证据与回放验证。

失败后诊断：

- skill body 是否错误？
- precondition 是否太宽？
- invariant 是否缺失？
- interrupt 是否太迟？
- postcondition 是否不完整？
- 是否应该加入 fallback / recovery？

更新示例：漏检或中断过迟 → 补充 guard、提前检查点；成功执行被无谓打断 → 放宽过严条件；同一位置反复需要重规划 → 缩短 chunk、加入分支或 recovery；检查长期没有提供有效信息 → 在验证支持下减少检查。候选更新需要同时验证失败修复、既有成功保持与新增成本。

> 核心闭环：checker 提供“哪里需要交还控制权”的定位信息，skill evolution 改善未来的检查条件与执行方式，使两者共同减少错误传播和不必要的主模型介入。
> 

## 6. 借鉴 SkillOpt 的更新方式

SkillOpt 更适合作为 optimizer backbone，而不是主创新点：**collect rollout → diagnose → bounded skill edit → validation → accept / reject**。

形式上：$S_{i+1}=Edit(S_i,T_fail,T_succ)$，只有在 held-out validation tasks 上改善才接受更新。

需要注意：SPACE 中 programmatic skill 更多像 training-time teacher；若做 training-free 版本，则应让 skill 成为部署时真正被调用的外部 executable skill，才能和 SkillOpt-style evolution 自然连接。

## 7. 三层贡献结构

### 主贡献：Dynamic Skill Execution / Mid-Skill Interrupt

研究 continue vs. interrupt vs. replan，解决 over-commitment、silent failure、environment drift 和 chunk 执行期间状态变化。

### 第二贡献：基于成功／失败证据的 Guard 与交接时机进化

利用成功、失败与恢复轨迹共同优化 Precondition / Invariant / Postcondition / Recovery 及检查位置，兼顾漏检与误中断。该机制与运行时 checker 共同服务于同一目标：以较低的监控成本判断何时继续执行、何时交还规划控制权。

一句话：**learn when a skill should no longer be trusted**。

### 第三贡献：Validation-Gated Skill Optimization

借鉴 SkillOpt：生成候选 edit，在验证任务上测试，改善才保留，避免 skill 越改越差。

## 8. Horizon Reduction 仍然成立

动态检查不意味着退化回每步都调用主 LLM。可以保持：**50 environment actions ≈ 10 macro decisions + 50 cheap observation checks**。

环境反馈仍逐步获得，但 checker 可以是 rule / symbolic guard / small model / lightweight LLM，因此仍能减少高层 LLM decision rounds。

核心 trade-off：$Efficiency <-> Robustness$。chunk 越长越省决策，但越容易因状态变化出错；dynamic interrupt 的目标是在两者之间取得更优平衡。

## 9. Training-Free 首版实验路线

优先做不训练主模型的版本：**Successful trajectories → LLM segmentation → Macro skills → Generate guards/invariants → Execute → Observation checker → Interrupt/Replan → Collect failure/recovery trajectory → SkillOpt-style update → Validation gate**。

首版可用 ALFWorld / ScienceWorld + GPT/Qwen API + 外部 skill library，不更新 policy model 参数。先验证现象，再考虑是否把 interrupt policy distill / train 到模型里。

## 10. 建议实验指标

- Success Rate
- Decision Rounds / Effective Horizon
- Primitive Actions
- Interrupt Precision / Recall
- False Interrupt Rate
- Failure Detection Delay
- Recovery Success Rate
- Skill Reuse Rate
- Validation Acceptance Rate
- Token / API Cost

关键 ablation：**ReAct → fixed chunk → dynamic interrupt → dynamic interrupt + failure evolution → dynamic interrupt + failure evolution + validation gate**。

## 11. 一句话 Research Question

<aside>
💡

Can a low-cost checker decide whether an ongoing plan remains valid, hand control back to an expensive planner only when needed, and use successful, failed, and recovery trajectories to continually improve these handoff decisions? 中文：能否通过低成本检查判断原计划是否仍成立，并利用执行经验持续改进检查器与规划器之间的交接时机？

</aside>

## 12. 当前定位

方向不再只是 trajectory → macro skill → horizon reduction，而是：**trajectory → executable macro skill → runtime monitoring → dynamic interrupt → recovery → failure-driven guard evolution**。

关键词：**Long-Horizon Agent / Action Chunking / Temporal Abstraction / Executable Skills / Dynamic Interrupt / Skill Evolution / Training-Free**。

<aside>
📌

当前判断（2026-09-25 更新）：主线是低成本检查与技能进化共同支持可靠的长序列执行。便宜的检查器判断“原计划是否还成立”，昂贵的规划器决定“接下来怎么做”，执行经验持续改进二者的交接时机。待验证的关键假设是：局部验证能由低价模型可靠完成，且边界进化能在保持或提高成功率的同时减少误中断、检查开销和主模型调用。

</aside>