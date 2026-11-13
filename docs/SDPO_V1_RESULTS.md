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

训练 W&B run ID：PPO seed0/1 为 `fwgvbh6z`、`fd07lbp8`；旧版 RelSpatial SDPO 为 `vpeb8ep1`、`d05g0msp`；正确 v1 SDPO 为 `0b7ctuqb`、`kfds90ff`；打乱 v1 SDPO 为 `vemil9or`、`ktbfankm`。这些 ID 当前只对应离线产物，不应称为云端链接。

服务恢复且 W&B API 能正常返回 viewer 后，可从本机同步这 33 个离线 run；同步后仍须用 API 或项目页面逐一核验 run ID 和状态：

```bash
conda run -n rlvla_env wandb sync --include-synced \
  --entity wqx349349-tongji-university --project RLVLA \
  experiments/sdpo-v1/training_offline/v1-*/wandb/offline-run-* \
  experiments/sdpo-v1/evaluation/*/wandb/offline-run-*
```
