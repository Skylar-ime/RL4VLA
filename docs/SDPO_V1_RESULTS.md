# RelSpatial 动作建议反馈 v1：2400 步实验结果

## 协议

- 分支：`ex/1111-v1`；训练作业启动时实现 commit 为 `3f15e74`，其后 `17770c4` 只修正调度脚本的远端命令依赖，未更改反馈或损失的 Python 实现。旧版 `ex/1111` 反馈保持原样；本实验用 `--sdpo-feedback-version relspatial-action-v1` 单独启用动作建议与采样动作判断。教师读取决策时刻特权反馈，学生只读取正常图像和指令。动作判断是几何启发式，不是执行后的成功标签，也不是 ROAD-VLA 的 advantage-logit 方法。
- 任务：`PutOnPlateInScene25Main-v3`；warm-up：`gen-robot/openvla-7b-rlvla-warmup` revision `735fd5d1f4472c4c9f3f998f0e89aa641af5ad03`。四组均从同一 warm-up 初始化：PPO、旧版 RelSpatial SDPO、正确 v1 SDPO、打乱 v1 SDPO。
- 训练：seed `0,1`；各 `2400` environment steps，`NUM_ENVS=2`、`EPISODE_LEN=80`，共 15 轮，每轮 160 步。`buffer_minibatch=2`、`alg_gradient_accum=1`、`alg_ppo_epoch=1`。SDPO 使用 EMA 教师、student-to-teacher reverse-KL，不叠加 PPO policy loss。
- 评测：各组在 800、1600、2400 步的冻结 checkpoint，及 warm-up；统一任务 `test` 集 episode ID `0..29`、评测 seed `0`、`NUM_ENVS=6`、`EPISODE_LEN=80`、动作采样温度 `0.6`。每个 checkpoint 独立评测 30 集，不从训练中复用评测回合。六环境评测与旧版两环境六集评测不是同一随机采样协议，不能直接合并统计。
- 计算：robot 单张 H800 训练，本机 4090 渐进评测，同时最多使用两张 GPU。robot 仓库位于 `/public/home/s2250018/project/RL4VLA`，与本机同一 Git 分支同步。

## 成功率

表内每个数字表示该训练 seed 的 30 个固定 test episode 中成功的数量；不同 checkpoint 重复使用同一组 episode ID，不应把三个时间点当作独立样本相加。warm-up 在此协议下为 `14/30` 成功、`23/30` 曾抓持。

| 训练组 | seed0：800 / 1600 / 2400 步 | seed1：800 / 1600 / 2400 步 | 2400 步合计 |
| --- | --- | --- | --- |
| PPO | 18/30 · 14/30 · 4/30 | 5/30 · 2/30 · 0/30 | 4/60 |
| 旧版 RelSpatial SDPO | 0/30 · 0/30 · 1/30 | 1/30 · 0/30 · 0/30 | 1/60 |
| 正确 v1 SDPO | 0/30 · 0/30 · 0/30 | 0/30 · 0/30 · 0/30 | 0/60 |
| 打乱 v1 SDPO | 0/30 · 0/30 · 0/30 | 0/30 · 0/30 · 1/30 | 1/60 |

正确 v1 在两个 seed、全部三个 checkpoint 上均未成功，且没有超过打乱反馈对照。旧版 RelSpatial 也未显示稳定收益。PPO seed0 在 800 步达到 `18/30`，但在 2400 步降为 `4/30`；seed1 从 `5/30` 降为 `0/30`。因此，“延长训练即可使当前反馈学习成功”不受本次结果支持；也不能由这两个 seed 推断 SDPO 方法在所有设置中无效。评测保留随机动作采样，结论仅针对上述固定协议。

正确 v1 seed0 的首份反馈归档有 160 条决策记录，其中启发式判断 `aligned=47`、`not aligned=86`、`opposed=27`、`unknown=0`。这只证明反馈生成器在真实 rollout 中输出了可判定文本，不证明判断与真实最优动作一致，也不证明教师会据此改善动作分布。

按每个训练 run 的前五轮和末五轮各 800 条决策记录重新统计，正确 v1 的启发式 `aligned` 数量在 seed0 从 `265/800` 降至 `82/800`，seed1 从 `252/800` 降至 `55/800`；`opposed` 数量分别从 `126/800` 升至 `315/800`、从 `128/800` 升至 `481/800`。实时抓持状态为真的记录分别从 `77/800`、`140/800` 降至末段两个 seed 均为 `0/800`。这些是训练时策略与启发式的关系，不是独立评测成功率；趋势与最终评测退化一致，但不能单凭它判定退化原因。

现有反馈归档可按 `episode_id` 和连续 `decision_index` 对齐相邻决策时刻的模拟器状态，审计单步距离代理：未抓持时为夹爪到操作物体距离，已抓持时为操作物体到目标容器距离；距离减少超过 `0.001 m` 记为改善。两个 seed 各有 `2370` 对连续状态。未抓持记录中，seed0 的 `aligned` 为 `395/455` 改善，`opposed` 为 `17/664`；seed1 分别为 `363/430` 和 `142/867`。这表明动作方向启发式与下一决策时刻的局部几何运动有明显关联，但不能证明它是完整任务的正确策略、教师已学会使用反馈或学生能获得成功率收益。已抓持记录很少，不据此判断抓持后阶段。复算命令：`python SimplerEnv/scripts/audit_sdpo_v1_feedback_transitions.py`。

同一对齐记录中，未抓持到下一决策时刻新建立抓持的次数很少：`aligned` 在 seed0/1 分别为 `2/455`、`8/430`；`not aligned` 为 `6/1174`、`13/933`；`opposed` 为 `3/664`、`4/867`。这些稀疏事件不足以证明“距离改善”会稳定转化为抓持或完整任务成功；不能把方向标签直接当成真实动作价值标签。

进一步检查接触阶段：v1 的“闭合夹爪”建议在 seed0 的全部 `2400` 条记录中出现 `0` 次，在 seed1 只出现 `34` 次；其中 `aligned` 有 `4` 次，下一决策时刻抓持建立 `3` 次，`not aligned` 有 `30` 次且无新抓持。两 seed 的连续状态对中分别发生 `11`、`25` 次新抓持，绝大多数没有伴随“闭合夹爪”建议。生成器目前仅在未抓持且夹爪到操作物体**中心**小于 `0.03 m` 时给出闭合建议；这可能漏掉有效接触状态，但当前数据不足以确认阈值是失败原因。动作方向标签与距离变化使用相近几何信息，距离审计也不是完全独立的真值。后续应先审计物体表面距离和夹爪接触，而不是凭单步距离结果直接扩大训练。

对这 `36` 次新抓持回溯决策前状态，seed0 的物体中心距夹爪中位数为 `0.0390 m`，`11` 次中 `0` 次低于 `0.03 m`；seed1 中位数 `0.0407 m`，`25` 次中仅 `3` 次低于阈值。这直接证明现行**中心距离**门槛覆盖不了多数实际抓持建立时刻，但不证明改大门槛就能提升策略，亦不能判断接触面距离或动作中的具体闭合时刻。复算脚本同上。

与此同时，正确 v1 的每轮平均 teacher-student KL 在 seed0 从前五轮 `1.350` 降至末五轮 `0.247`，seed1 从 `1.292` 降至 `0.229`。KL 下降与抓持、测试成功率退化同时发生，验证了仅凭蒸馏损失下降不能判定学生策略改善。静态代码核查确认冻结评测只加载学生 adapter；`alg_name=ppo` 不会创建或使用 EMA 教师，因此评测中的失败不能归因于误用教师策略。

## 训练成本

下表为两个训练 seed 的平均值，覆盖每个 run 的 15 轮采样与更新，不含模型加载、checkpoint 保存及附带的两集在线评估。显存为 H800 上 `torch.cuda.max_memory_allocated` 的轮内最大值。

| 训练组 | 采样＋更新约耗时 | 采样耗时 | 更新耗时 | 峰值 CUDA 已分配 |
| --- | ---: | ---: | ---: | ---: |
| PPO | 1300 s | 1105 s | 203 s | 19.32 GiB |
| 旧版 RelSpatial SDPO | 1473 s | 1134 s | 334 s | 33.85 GiB |
| 正确 v1 SDPO | 1414 s | 1071 s | 337 s | 33.91 GiB |
| 打乱 v1 SDPO | 1405 s | 1064 s | 335 s | 33.90 GiB |

正确 v1 相对 PPO 的报告轮时约增加 `8.8%`，更新耗时约增加 `66%`，CUDA 已分配峰值约增加 `75%`，但没有观察到成功率收益。训练日志中 PPO seed1 的第 9 轮存在计时不自洽：总轮时比采样与更新耗时之和短 `35.8 s`；故上述总轮时只作近似成本比较，不能解释为精确壁钟开销。PPO 与 SDPO 的 loss 定义不同，其数值不能直接比较。

## 产物与限制

- robot Slurm 作业 `16633` 最终状态为 `FAILED 2:0`，末尾报 `run_sdpo_v1_comparison.sh: error reading input file`。报错发生在最后一个 run 的 W&B 结束提示之后；具体原因未证实。独立检查确认 8 个训练 run、24 个 checkpoint、每个 SDPO run 的 15 份反馈归档（每份 160 条）全部存在，checkpoint 的反馈版本和 reverse-KL 协议字段正确。现版续跑脚本检测到 8 个 run 均已完成，不会重复训练。不得把该 Slurm 作业写成 `COMPLETED`。
- 本地 4090 的 warm-up 加 24 个冻结 checkpoint 共 25 组评测全部完成；每组有 30 个 episode ID 及逐集结果。`SimplerEnv/scripts/summarize_sdpo_v1_eval.py --strict` 校验配置、批次数和逐集 ID；`SimplerEnv/scripts/summarize_sdpo_v1_training.py` 汇总 Slurm 日志。训练原件保留在 robot `experiments/sdpo-v1/`；本地评测保留在 `experiments/sdpo-v1/evaluation/`。复制来的临时 checkpoint 在评测后已清理。
- W&B 训练与评测均以 offline run 完整保存。2026-11-13 再次调用官方 `api.wandb.ai` 验证凭据仍返回 `AuthenticationError`；本机通过代理、绕过代理以及 robot 直接访问该端点均得到 HTTP `403`。**没有证据证明这些 run 已上传云端**。网络／认证修复后应逐 run 同步并通过 W&B API 核验，不依赖 CLI 的 `.synced` 标记。
- 8 个训练 offline run（不含大型 checkpoint）已复制到本机 `experiments/sdpo-v1/training_offline/`，含 90 份 SDPO 反馈归档；8 个 `run-*.wandb` 文件与 robot 原件逐文件 SHA-256 一致。25 个评测 offline run 原本就在本机 `experiments/sdpo-v1/evaluation/`。完整 checkpoint 仍只保留在 robot，避免填满本机磁盘。
- 当前观察是机制与反馈效果的负面证据；不建议直接追加大规模 seed 或 OOD 实验。下一轮应先固定评测协议，检查教师在正确反馈下的动作分布、启发式建议的实际有效率及训练学习率／EMA 稳定性，再做单因素消融。
- 现有训练反馈 JSONL 保存了状态、建议和解码后的平移动作，可审计连续状态的单步几何代理，但没有持久化同一决策的正常观测图像、原始 action token IDs 及行为 log-prob；这些只在在线 rollout buffer 中存在。因而**不能**从当前归档精确重算历史决策的学生／EMA 教师分布。分布诊断须先采集新的、时间对齐的只读校准样本，不得声称重放了旧 rollout。

训练 W&B run ID：PPO seed0/1 为 `fwgvbh6z`、`fd07lbp8`；旧版 RelSpatial SDPO 为 `vpeb8ep1`、`d05g0msp`；正确 v1 SDPO 为 `0b7ctuqb`、`kfds90ff`；打乱 v1 SDPO 为 `vemil9or`、`ktbfankm`。这些 ID 当前只对应离线产物，不应称为云端链接。

服务恢复且 W&B API 能正常返回 viewer 后，可从本机同步这 33 个离线 run；同步后仍须用 API 或项目页面逐一核验 run ID 和状态：

```bash
conda run -n rlvla_env wandb sync --include-synced \
  --entity wqx349349-tongji-university --project RLVLA \
  experiments/sdpo-v1/training_offline/v1-*/wandb/offline-run-* \
  experiments/sdpo-v1/evaluation/*/wandb/offline-run-*
```

## 已执行的只读诊断

获批方案原定本机一张 4090、最多 200 个环境交互步，从 warm-up 和 v1 seed0 最终学生各采 100 个决策，并在固定图像和动作前缀上重算四个 v1 checkpoint 的学生与 EMA 教师分布。2026-11-13 已执行；不追加训练，也不修改模型参数。首次 warm-up 采集在第 20 至 38 个交互之间因 episode 提前结束而中止，未存档；为保持总交互数不超过 200，正式归档改为两个来源各 80 个决策（总交互数最多 198）。采集脚本现能在 episode 结束后重置，并记录实际请求的 test episode ID。数据保存在 `experiments/sdpo-v1/diagnostics/20261113-*`（被 `.gitignore` 排除）；包含压缩图像、指令、原始 action token IDs、行为 log-prob、决策时刻双指接触力、物体世界坐标包围盒尺寸、夹爪命令、状态与反馈、执行后状态及 episode 边界。临时 checkpoint 副本与重复试算文件已清理。

80 个 warm-up 决策来自 6 段 episode：33 个决策有任一手指接触、22 个双指接触、3 个下一步新抓持、0 个“闭合夹爪”建议；33 个接触决策的物体中心到夹爪距离均大于 0.03 m（范围 0.042–0.089 m）。80 个 v1 seed0 最终学生决策来自 2 段 episode：0 个接触、0 个新抓持、0 个“闭合夹爪”建议；只有 3/80 个采样动作获几何启发式 `aligned`，warm-up 为 57/80。包围盒数值是世界坐标轴上的尺寸，不是最小／最大坐标；这些小样本不足以校准新的接触或闭合阈值，也不能单独估计任务成功率。

评分使用相同的 160 张图像和原始动作前缀；学生不读反馈，EMA 教师分别读取正确反馈、同来源固定随机错配反馈和无反馈。下表每格为 `student-to-teacher reverse-KL / 教师期望平移动作沿启发式目标方向的投影（mm）`；投影在评估温度 0.6 下由 256 个动作 token 的分布计算，**不是实际任务成功率**。

| checkpoint | 图像来源 | 正确反馈 | 错配反馈 | 无反馈 |
| --- | --- | ---: | ---: | ---: |
| seed0 800 | warm-up | 0.918 / -0.73 | 0.908 / -1.04 | 2.709 / 7.18 |
| seed0 800 | v1 seed0 最终学生 | 1.013 / -2.49 | 1.025 / -2.64 | 2.043 / -1.11 |
| seed0 2400 | warm-up | 0.568 / -3.06 | 0.572 / -2.96 | 2.348 / 3.99 |
| seed0 2400 | v1 seed0 最终学生 | 0.383 / -2.92 | 0.387 / -2.91 | 1.825 / -0.90 |
| seed1 800 | warm-up | 0.834 / -1.88 | 0.844 / -2.08 | 2.391 / 7.47 |
| seed1 800 | v1 seed0 最终学生 | 0.779 / -2.48 | 0.783 / -2.68 | 1.909 / -0.79 |
| seed1 2400 | warm-up | 0.477 / -6.36 | 0.479 / -6.22 | 2.630 / -2.09 |
| seed1 2400 | v1 seed0 最终学生 | 0.353 / -1.63 | 0.363 / -1.60 | 2.121 / -4.89 |

四个 checkpoint 的正确反馈与错配反馈分布指标近似；在该校准集上，8 个 checkpoint／图像来源组合中有 7 个的无反馈教师方向投影高于正确反馈教师。较小的正确反馈 KL 仅表明学生较接近有反馈教师，不证明教师动作更合理。v1 学生 80 个决策的建议均为“向下移动”，因此错配对照在动作语义上区分度有限。当前没有充分证据支持直接扩大训练规模或仅调整学习率／EMA；下一步应先改进接触阶段与闭合建议的判定，并在独立 episode 上验证反馈条件教师确实改善行为。此步骤需另行确认预算。
