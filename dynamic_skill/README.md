# 运行模块按职责分层

| 目录 | 职责 |
| --- | --- |
| `execution/` | 通用动作队列、观察检查、继续/中断/终止决策 |
| `knowledge/` | SkillPlan/SkillStep、任务分解、失败归纳、技能更新建议 |
| `controllers/` | 可执行控制策略、成本价值终止、研究控制器 |
| `evidence/` | 证伪/未知/支持、依赖修订及不同环境的证据适配 |
| `alfworld/` | ALFWorld执行适配、获取状态检查、恢复、搜索/拾取子技能、输出协议 |
| `environments/` | Rush Hour与Sudoku本地可执行环境 |
| `learning/` | 历史通用/拼图技能进化算法 |
| `infrastructure/` | 预算与请求保护、原生thinking策略、旧导入兼容层 |

运行组件负责“怎样执行”，实验目录负责“如何划分数据、生成候选与比较结果”。
新的联合进化流程位于`experiments/alfworld/joint_evolution/`，共用这里的执行器和控制策略，
不再复制第二份队列实现。旧的一轮/控制参数实验放在`experiments/alfworld/skill_evolution/`。

`controllers/judge.py` 提供可选的模型检查器：在原执行器提交中断前检查公开状态与队尾依赖，
支持同模型不同角色或独立 judge 模型；对应实验放在 `experiments/alfworld/judge_interrupt/`。

保留旧的`from dynamic_skill.skill_executor import ChunkExecutor`导入；新代码使用
`from dynamic_skill.execution.skill_executor import ChunkExecutor`。两者指向同一模块/类。
`migration_20261008.json`记录27个文件迁移及哈希，原始源码归档在
`experiments/_archive/20261008_dynamic_skill_sources.zip`。历史结果和源码快照未改。
