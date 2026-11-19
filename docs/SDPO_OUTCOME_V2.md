# SDPO 执行后结果反馈：小规模验证

## 协议

- 分支：`ex/1111-outcome-feedback`。新增可选 `relspatial-outcome-v2`，保留旧 `relspatial-action-v1` 与旧实验归档。任务仍为 `PutOnPlateInScene25Main-v3`，学生和 EMA 教师均由同一 OpenVLA warm-up 权重初始化。
- 学生只看执行前图像与指令，按温度 `1.0` 采样 7 个原始动作 token；环境执行该动作。脚本立即比较对应环境的执行前后状态，记录夹爪／物体或物体／目标的中心距离变化、物体移动、抓持建立／丢失、任务当步成功及终止状态。缺失状态写 `unknown`；距离缩短不代表动作正确。文本没有预设动作建议或 `aligned/opposed` 标签。
- 环境步完成后，才将反馈和原始动作一起写入内存 buffer。采满一批后，学生按原动作 token 计算 logits；EMA 教师读取同图、同指令和结果文本，在相同原动作 token 上计算 logits。教师停止梯度，损失仍为 `KL(student || teacher)`；不把教师新生成动作当标签。
- 基础配置：`NUM_ENVS=2`、`EPISODE_LEN=80`、每批 `160` 条；`buffer_minibatch=2`、`alg_gradient_accum=80`、`alg_ppo_epoch=1`，整批累计梯度后学生更新一次，EMA 更新一次。更新频率参数独立于反馈版本。smoke 使用 `seed=0`、`160` 个环境步、末尾 train/test 各 `2` 集评测；样本太少，不用于判断反馈效果。
- 当前任务的初始化函数不支持单个并行环境部分重置。某个环境提前结束时，先记录所有环境本步真实执行结果，再整体重置两环境；未结束同伴标记 `rollout_interrupted=true`。buffer 保存逐决策指令并替换下一决策观测，避免将上一 episode 的状态或指令误配到新动作。末步结束后由下一批统一重置。

## 验证

复现命令：`WANDB_MODE=offline bash SimplerEnv/scripts/run_sdpo_outcome_v2_smoke.sh`。默认在线 W&B；2026-11-17 本机和 robot 访问 `https://api.wandb.ai` 均返回 HTTP `403`，故当前 smoke 使用离线 W&B，保留运行文件等待同步，不声称云端可见。

`SimplerEnv/tests/test_sdpo_primitives.py` 的 21 个测试函数在 `rlvla_env` 中直接调用全部通过；环境未安装 `pytest`，并非 `pytest` 命令的通过记录。覆盖结果文本、未知状态、并行环境动作匹配、回合重置、逐决策指令对齐、原动作 token 重评及每累计两小批更新一次的模拟测试。`py_compile` 与 `git diff --check` 通过。

2026-11-17 本机 4090 完成一批真实 smoke：W&B 离线 run ID `6r6147yg`，原件在 `experiments/sdpo-outcome-v2/outcome-v2-sdpo-correct-seed0-160steps/wandb/offline-run-20261117_161051-6r6147yg/`。反馈归档 `glob/feedback/episode_000000.jsonl` 有 `160` 条、`160` 个唯一的环境／回合／决策键，跨 `8` 段 episode；每条记录均有 7 个原动作 token、同一键的执行前后状态，不含旧建议和 `aligned/opposed`。归档中有 `3` 个任务成功且终止的动作、`3` 次同伴提前结束引发的中断；剩余 episode 在采样批次末截断。抓持建立／丢失各 `9`／`6` 次。这些是训练采样统计，不是独立成功率。

W&B 记录 `train/optimizer_steps=1`、`feedback/effective_ratio=1.0`、`train/teacher_student_kl=3.880`，rollout `184.1 s`、训练更新 `64.2 s`、峰值 CUDA 已分配 `33.73 GiB`。单次更新后的冻结评测在固定 train 两集为 `2/2`，固定 test 两集为 `0/2`；该样本量不能判断方法优劣，更不能替代此前 v1 的 30 集协议。checkpoint 的 `glob/steps_0000/sdpo_protocol.json` 记录反馈版本、原 token 对齐、reverse-KL、EMA 与 `2 × 80` 累计设置。

早期 smoke 暴露并修正了两个回合边界问题：本任务不支持部分重置，以及内部 episode seed 在重置后可能重复。最终版改为成功后完整重置并以重置代数区分归档回合；早期无效产物已删除，仅保留上面的最终 run。已尝试 `wandb sync`，但 API 仍返回 HTTP `403`，且无 `.synced` 标记；命令退出码 `0` 不代表同步成功。云端同步待 API 恢复后重试：`wandb sync experiments/sdpo-outcome-v2/outcome-v2-sdpo-correct-seed0-160steps/wandb/offline-run-20261117_161051-6r6147yg`。

## v1 同规模训练与冻结评测

2026-11-17 在本机 4090 完成 outcome-v2 正确反馈 seed0/1 训练。每个 seed 2400 环境步、2 个并行环境、15 批，每批 160 条；`buffer_minibatch=2`、`alg_gradient_accum=1`、`alg_ppo_epoch=1`，与 v1 的训练更新频率一致，不沿用 smoke 的每批仅更新一次配置。训练命令：`bash SimplerEnv/scripts/run_sdpo_outcome_v2_comparison.sh`。每个 seed 保存 800、1600、2400 步 checkpoint，运行原件位于 `experiments/sdpo-outcome-v2/outcome-v2-sdpo-correct-seed{0,1}-2400steps/`。W&B run ID 分别为 `5hduzoye`、`39pfcf71`，均为离线产物，未核验云端可见。

对六个冻结学生 checkpoint，使用与 v1 一致的 test 集 episode ID `0..29`、评测 seed0、6 并行环境、episode 上限 80、动作温度 0.6，分别评测 30 集。评测命令：`bash SimplerEnv/scripts/run_sdpo_outcome_v2_eval.sh`。每组的 5 批、30 个逐集 ID 均校验完整；结果不是训练 rollout 成功率。

| 训练 seed | 800 步 | 1600 步 | 2400 步 |
| --- | ---: | ---: | ---: |
| 0 | 0/30 成功，0/30 曾抓持 | 0/30，0/30 | 0/30，0/30 |
| 1 | 0/30 成功，0/30 曾抓持 | 0/30，0/30 | 0/30，0/30 |

最终合计 0/60 成功、0/60 曾抓持。相同冻结协议下，v1 正确反馈的两个 seed 在三个 checkpoint 也都是 0/30 成功；warm-up 为 14/30 成功、23/30 曾抓持。当前结果不支持 outcome-v2 改进任务成功率，且失败已出现在抓持前；不能据此断言结果反馈机制普遍无效。两版本均只有两个训练 seed，且在线训练与冻结评测不同。六组逐集原件在 `experiments/sdpo-outcome-v2/evaluation/`；训练与评测均已结束。
