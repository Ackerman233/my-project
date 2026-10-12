# 实验目录

按数据集、再按方法组织。新代码使用完整模块路径，例：
`python -m experiments.alfworld.joint_evolution.cli --help`。

| 目录 | 内容 |
| --- | --- |
| `alfworld/react/` | 原版ReAct入口 |
| `alfworld/static_interrupt/` | 固定动作块入口；这里“静态”指固定边界，不是额外学习的中断器 |
| `alfworld/dynamic_interrupt/` | 动态中断、固定前缀恢复、证伪/依赖验证及对应审计 |
| `alfworld/joint_evolution/` | **新主线：执行反馈→联合改写技能和策略→筛选→独立复评** |
| `alfworld/judge_interrupt/` | 规则门控后调用独立 judge；同模型不同角色对照，actor 与 judge 共用调用预算 |
| `alfworld/skill_evolution/` | 早期一轮更新及仅控制参数进化实验 |
| `alfworld/operators/` | 搜索、拾取、SPACE子技能及消融 |
| `alfworld/comparisons/` | 模型/工作流/同操作器公平对照 |
| `alfworld/protocols/` | 输出协议诊断与重放 |
| `alfworld/common/` | ALFWorld共享入口、历史campaign |
| `puzzles/common/` | Rush Hour/Sudoku数据准备；二者过去共用实验协议 |
| `puzzles/dynamic_interrupt/` | 拼图控制器与队尾过滤 |
| `puzzles/skill_evolution/` | 拼图技能进化 |
| `puzzles/protocols/` | 拼图冷启动、显式短推理与协议对照 |
| `common/` | 共享模型客户端、预算保护和离线测试入口 |
| `docs/` | 旧协议说明，按历史文档保留 |
| `_archive/` | 整理前源码压缩归档，非运行输出 |

## 兼容与历史证据

79个平铺模块已移动。`migration_20261008.json`记录旧名、新路径及前后哈希。
旧Python导入及`python -m experiments.react`等模块入口保留兼容；旧的
`python experiments/react.py`文件路径命令改用模块入口。共享模块导入指向同一对象。

历史付费运行的`results/`、预算、配置、源码快照都保留原位。旧配置记录的
文件路径不再是当前目录布局；严格重现旧运行应使用该运行的`source_snapshot/`
或`_archive/20261008_original_sources.zip`，不能将路径迁移伪装成原源码哈希未变。
可以用`python -m experiments.common.verify_layout`核验迁移、归档与旧源码快照。

新运行统一写入`results/<数据集>/<方法>/<日期或run-id>/`，例如
`results/alfworld/joint_evolution/20261008_demo/`。源码目录不存候选、轨迹、日志或临时输出。
旧结果不盲目搬移，以保留账本标签、历史引用和审计链。

完整离线检查：
`python -m experiments.common.verify_interrupt_recovery_offline -q`。
