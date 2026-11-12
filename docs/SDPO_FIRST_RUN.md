# SDPO VLA 首次试跑

## 当前实现

- 仓库：`https://github.com/Skylar-ime/RL4VLA.git`
- 分支：`ex/1111`
- 当前实现 commit：以 `git rev-parse HEAD` 记录实际运行版本；warm-up 模型 revision：`735fd5d1f4472c4c9f3f998f0e89aa641af5ad03`。
- 试跑环境关键版本：Python `3.10`、PyTorch `2.2.0+cu121`、Transformers `4.40.1`、PEFT `0.11.1`。
- 初始化：RL4VLA 官方 warm-up checkpoint `gen-robot/openvla-7b-rlvla-warmup`；未加载 LoRA，动作分布为 `256` 个 bin，动作 token 为原 rollout token IDs。
- 损失：学生正常 prompt 到 EMA 教师反馈 prompt 的 reverse KL；教师输出停止梯度。
- SDPO 参考与本任务首版差异：本机 `/home/skylar/@lab/SDPO/verl/trainer/config/sdpo.yaml` 及 `verl/trainer/config/actor/actor.yaml` 默认 `alpha=0.5`（JSD）、top-100 加 tail、EMA 更新率 `0.05`；本任务明确选择 `alpha=1` 的 student-to-teacher reverse-KL，因 OpenVLA 动作支持仅 `256` bins 而对完整支持集计算精确分布，不做 top-K 重归一化；EMA 更新率 `0.01`，不叠加 PPO policy loss、advantage 权重或 rollout IS 修正。采样 temperature `1.0`，每个 rollout 立即更新一轮；该轮含多个 minibatch，早先 minibatch 的 rollout 对后续 minibatch 会有轻微批内 off-policy 偏差。以上是显式首版选择，不声称复现参考仓库默认配置；协议保存于 SDPO checkpoint 的 `sdpo_protocol.json`。
- 关键先验与研究定位：ROAD-VLA 论文（[arXiv:2606.25800](https://arxiv.org/abs/2606.25800)，v1，2026-06-24）在 VR-UnseenTable 三 seed ablation 报告 PPO `87.2 ± 3.6%`、ROAD-VLA (RelSpatial PI) `4.68 ± 0.0%`、Plan+RelSpatial `4.68 ± 0.0%`、MCTS PI `75.8 ± 2.0%`。作者指出文本提示无法稳定提供 action-level supervision。故本计划 RelSpatial 文本教师是高风险研究假设，不能预设有效；小样本 smoke 的 `0/2` 不等于验证该结论。该论文方法也不同：其完整法以 advantage 扰动构造 action-space proximal teacher，并用 teacher-to-student forward KL；此处按任务要求使用 EMA 反馈文本教师与 student-to-teacher reverse-KL，结果不是 ROAD-VLA 复现或直接公平比较。先按计划完成有预算的小规模消融，再依据独立成功率决定是否继续，不把 KL 下降当成功。
- 反馈：决策时刻 RelSpatial 状态文本。记录原始位置、关系向量、抓持字段、时间戳和文本到 `wandb` 的 `glob/feedback/episode_*.jsonl`。
- 坐标约定：SAPIEN 世界坐标；`x<0` left、`x>0` right，`y<0` front、`y>0` back，`z<0` below、`z>0` above；方向死区 `0.01 m`，按最大绝对轴决定文本方向。

## 试跑命令

在 `rlvla_env` 环境、仓库根目录执行：

```bash
conda activate rlvla_env
cd /home/skylar/@lab/RL4VLA
PYTHONPATH=SimplerEnv:ManiSkill:openvla \
MS_ASSET_DIR=/home/skylar/@lab/hf-cache/maniskill \
python SimplerEnv/simpler_env/train_ms3_ppo.py \
  --alg-name sdpo \
  --env-id PutOnPlateInScene25Main-v3 \
  --vla-path gen-robot/openvla-7b-rlvla-warmup \
  --vla-load-path '' \
  --num-envs 1 \
  --episode-len 8 \
  --steps-max 8 \
  --buffer-minibatch 1 \
  --alg-gradient-accum 1 \
  --wandb true \
  --name sdpo-smoke
```

反馈控制：

- `--sdpo-feedback-mode correct`：正确反馈，默认。
- `--sdpo-feedback-mode shuffled`：批内循环错配反馈。
- `--sdpo-feedback-mode none`：无反馈 EMA 蒸馏。
- SDPO checkpoint 同目录写入 `sdpo_protocol.json`，记录 divergence、EMA、action support、反馈模式及 PPO/advantage 开关。
- WandB 默认在线记录；若服务端返回认证或网络错误，可设置 `WANDB_MODE=offline`，仍保留本地 run、配置和指标归档。

## 多 seed 对照入口

`SimplerEnv/scripts/run_sdpo_comparison.sh` 固定同一初始化和任务，依次运行 PPO、correct SDPO、shuffled SDPO、none SDPO。默认是 3 个 seed、每个 8 步的短跑；正式实验前显式设置 `EPISODE_LEN`、`STEPS_MAX` 和 `WANDB_MODE`，避免误启动长任务。

批量 runner 默认关闭视频保存（`SAVE_RENDER=false`），避免每个 7B checkpoint 生成多 GB 视频；需要视频时显式设置 `SAVE_RENDER=true`。

已完成 seed 0、每组 160 环境步的一轮延长 pilot（详见下文）。用户已确认下一轮预算：四个训练组各跑 seeds `0,1,2`、每组 `STEPS_MAX=480`、共 12 个训练 run；固定 `EPISODE_LEN=80`、`NUM_ENVS=2`，即每个 run 三轮、每轮 160 环境步。`INTERVAL_EVAL=1` 在 160/320/480 步评估固定 train/test episode `0,1`，作工程学习曲线；最终每个训练 seed checkpoint 独立评估 6 个固定 test episodes，warm-up 评估同一 6 集，共 78 个最终评估 episode。`INTERVAL_SAVE=40` 只保存终点；12 个 checkpoint 预计约 `12 GiB`，训练与评估均关闭视频。

远端方案：`ssh robot` 是登录节点，仓库同步到 `/public/home/s2250018/project/RL4VLA`，GPU 通过 `/opt/gridview/slurm/bin/sbatch` 投递至 `h800` 分区。项目同时最多占用 2 张训练 GPU：本机矩阵占 1 张时，远端 smoke 只申请 1 张 H800；备用正式矩阵须在本机训练结束后，以单个 Slurm 作业申请 2 张 H800，在作业内并行跑两个 seed，第三个 seed 等前两者结束后复用 GPU，不使用 job array 或额外 GPU 申请。`smoke` 模式执行一次真实 SDPO 更新并重载 checkpoint 评估，自动删除其专属临时 run/checkpoint 目录，只保留 Slurm 日志；`matrix` 的离线 W&B run 保存在 `experiments/<job-id>/seed<N>`。远端尚无 W&B 登录凭据，run 先用 `WANDB_MODE=offline` 保留完整数据，再复制回本机用已验证凭据同步至官方 W&B 项目。

本机已有 480 步三 seed 矩阵运行中；必须先核对完成的 run，不得再用远端 `matrix` 重复消耗同一预算。远端 Slurm 日志保存在仓库外的 `/public/home/s2250018/project/rlvla-sdpo-<job-id>.out`；`experiments/` 为 Git 忽略的运行产物。robot 无法直连 Hugging Face，模型按固定 revision 从 `https://hf-mirror.com` 下载，完整下载后逐分片与本机 warm-up snapshot 的 SHA-256 校验一致再做 smoke。

仓库迁移到 `project/RL4VLA` 后，robot 的 editable 包须从新路径重装：`python -m pip install --no-deps -e ManiSkill -e SimplerEnv -e openvla`。环境使用 `tensorflow-metadata==1.17.0`、`protobuf==4.21.12`；较新的 metadata 与 protobuf 4.25.9 组合在导入时缺少 `google.protobuf.runtime_version`。在 `PYTHONNOUSERSITE=1` 下已核对 `mani_skill`、`simpler_env`、`prismatic` 均从新仓库路径导入。

在 `ssh robot` 登录节点执行，smoke 完成并核验后再投递矩阵：

```bash
cd /public/home/s2250018/project/RL4VLA
/opt/gridview/slurm/bin/sbatch SimplerEnv/scripts/run_sdpo_comparison_h800.sbatch smoke
/opt/gridview/slurm/bin/squeue -u "$USER"
```

```bash
cd /public/home/s2250018/project/RL4VLA
/opt/gridview/slurm/bin/sbatch --gres=gpu:2 SimplerEnv/scripts/run_sdpo_comparison_h800.sbatch matrix
```

```bash
SEEDS="0 1 2" NUM_ENVS=2 EPISODE_LEN=80 STEPS_MAX=480 \
INTERVAL_EVAL=1 EVAL_EPISODES=2 INTERVAL_SAVE=40 SAVE_RENDER=false WANDB_MODE=offline \
  bash SimplerEnv/scripts/run_sdpo_comparison.sh
```

```bash
SEEDS="0 1 2" NUM_ENVS=2 EPISODE_LEN=80 STEPS_MAX=4000 WANDB_MODE=online \
  bash SimplerEnv/scripts/run_sdpo_comparison.sh
```

## Warm-up 评估基线

可用 `--eval-only` 对未后训练的 warm-up OpenVLA 执行固定测试 episode 评估，不采样训练 rollout、不更新参数。评估 episode ID 与策略随机数种子固定，便于各方法复用相同测试实例；结果写入 W&B run 的 `files/eval_only.jsonl`，含每集和每批指标。

```bash
WANDB_MODE=online MS_ASSET_DIR=/home/skylar/@lab/hf-cache/maniskill \
PYTHONPATH=SimplerEnv:ManiSkill:openvla \
conda run -n rlvla_env python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
  --alg-name ppo --env-id PutOnPlateInScene25Main-v3 \
  --vla-path /home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03 \
  --vla-load-path '' --seed 0 --num-envs 2 --episode-len 80 \
  --eval-only --eval-episodes 6 --eval-obj-set test \
  --name warmup-baseline-seed0-eval6
```

多 checkpoint 或 ID/OOD 环境评估可使用 `SimplerEnv/scripts/run_eval_matrix.sh`。输入 TSV 每行一个 `label<TAB>checkpoint`，示例见 `docs/sdpo_eval_matrix.tsv.example`；`ENV_IDS` 可传空格分隔的多个已在当前源码核实过的环境 ID。该入口只评估冻结 checkpoint，不更新参数。

本轮 480 步矩阵结束后，从 W&B 本地 run 的配置生成最终评估清单。生成器要求四组各三 seed、同一任务与预算、完整终点 checkpoint 和 SDPO 协议，缺项或重复即报错；清单首行为不后训练 warm-up。下列命令固定 test episode `0..5`，每个 checkpoint 六集，共 `13 × 6 = 78` 集。务必等本机矩阵进程结束后执行，避免共用 4090 显存；评估 run 保持 `WANDB_MODE=online`。

```bash
conda run -n rlvla_env python SimplerEnv/scripts/build_sdpo_final_eval_manifest.py \
  --wandb-dir wandb > /tmp/rlvla-final-eval.tsv
CHECKPOINT_MANIFEST=/tmp/rlvla-final-eval.tsv \
EVAL_EPISODES=6 NUM_ENVS=2 EPISODE_LEN=80 SEED=0 EVAL_OBJ_SET=test \
WANDB_MODE=online bash SimplerEnv/scripts/run_eval_matrix.sh
```

评估入口 smoke 已通过：PPO seed0 checkpoint、`PutOnPlateInScene25Main-v3`、2 个 test episode，W&B run `wandb/offline-run-20261112_032258-zl20amqe`，success `1/2`。

同一冻结 PPO seed0 checkpoint 的 OOD smoke：`PutOnPlateInScene25VisionImage-v1` success `2/2`（W&B `wandb/offline-run-20261112_032920-8yxyjxq5`）；`PutOnPlateInScene25PositionChangeTo-v1` success `0/2`（W&B `wandb/offline-run-20261112_033145-yrb0c0ed`）。每个环境仅 2 个 episode，只证明环境入口和评估协议可运行，不代表泛化结论。

组合辨别 OOD smoke：`PutOnPlateInScene25Instruct-v1` success `2/2`（W&B `wandb/offline-run-20261112_033635-yuzgo7ql`）。样本极少，仅作协议连通性记录。

## 已验证

- `compileall`：通过。
- reverse-KL smoke：通过，梯度仅到学生。
- 模型级同权重/同上下文检查：新初始化的 SDPO 学生与 EMA 教师均在 `eval()` 模式，对同一正常观测、同一原始动作 token 做 teacher forcing，反馈均为空；两分支 action logits 形状均为 `[2,7,256]`，最大绝对差 `0`，reverse-KL `0`。本地 W&B run `wandb/offline-run-20261112_062920-vtk34eoa`。这验证无反馈时的分布/位置对齐，不证明反馈质量。
- 模型级更新检查：同配置 8-step SDPO 闭环前后，比较全部 878 个可训练非 value-head tensor；学生参数变化 L1 总和 `16560.87`，EMA 教师变化 `344.35`，二者终点差 `16229.82`，且教师参数均 `requires_grad=False`。W&B 本地 run `wandb/offline-run-20261112_063221-ypcmqin0`；checkpoint 和反馈归档已保存。仅验证更新方向与 EMA 非零变化，不代表效果收益。
- rollout buffer feedback smoke：通过，含 `buffer-minibatch=-1` 全量采样路径。
- 真实 GPU smoke：通过。`PutOnPlateInScene25Main-v3` 单环境单步完成 rollout、反馈归档、SDPO 更新、EMA、评估和 checkpoint 保存；WandB 本地 run 为 `wandb/offline-run-20261111_231030-gouf9xje`，训练 loss `4.898`、有效 token `7`。
- checkpoint 恢复 smoke：通过。以 `glob/steps_0000` 重载时日志确认 `SDPO teacher load: restored EMA state` 与 optimizer 恢复。
- 首轮控制 smoke：PPO run `wandb/offline-run-20261111_232751-rkm0bfjf` 等仅作工程记录；对应 SDPO 记录因旧 action token 对齐缺陷作废。shuffled 使用 `num-envs=2` 的错配逻辑本身已验证，但训练损失不能作方法结果。
- 8 步短程对照：旧 correct SDPO run `wandb/offline-run-20261111_234727-xpt1cfj5` 与 PPO run `wandb/offline-run-20261111_235243-6p5nhm1d`；SDPO 记录因 token 对齐缺陷作废，PPO 记录保留为工程基线。
- 历史多 seed runner 记录：这些 SDPO run 使用旧的 decode/re-tokenize 路径，后来发现不保持原 action token IDs；因此只能作为失败工程记录，不能作为 SDPO 方法结果。对应 W&B 历史保留，不用于结论。
- 旧 PPO 短跑为 seed `0` `wandb/offline-run-20261112_010051-svov1bhf`、seed `1` `...012321-xcbqtdaz`、seed `2` `...014423-nsp9nu6w`；这些 run 也受后述 joint log-prob 广播错误影响，仅作工程记录。
- 旧 `SEEDS=1` 记录同时存在 `NUM_ENVS=1` shuffled 退化和 action token 对齐缺陷；保留 W&B 历史，但不作为实验结论。
- runner 与单次训练均启用 W&B。runner 默认 `WANDB_MODE=online`；当前环境在线认证返回 HTTP 403，故上述 run 通过 `WANDB_MODE=offline` 保存在本地 W&B 目录。2026-11-12 对 30 个短程 run 执行 `python -m wandb sync` 时，CLI 虽打印 `done`，但 `wandb/debug-cli.skylar.log` 记录每次 `upsert_run` 均为 HTTP 403；云端上传**未验证成功**。CLI 还错误生成 `.synced` 标记；凭据/网络修复后重试须加 `--include-synced`，并用 W&B API 或项目页面核验，不能仅依赖 CLI 退出码或 `.synced` 文件。
- Warm-up 固定测试集基线：6 个独立 episode（episode ID `0..5`，每批 2 个）success `0/6`、抓持 `4/6`。W&B run `wandb/offline-run-20261112_021531-fke1hpqs`；逐批指标在 `glob/eval_only.jsonl`。此为有限测试集基线，非统计显著结论。
- 旧 correct SDPO checkpoint 评估 run `wandb/offline-run-20261112_024330-ml6drwkc`、旧 shuffled checkpoint 评估 run `wandb/offline-run-20261112_030731-hhom6kc2` 均受 action token 对齐缺陷影响，不能作为 SDPO 结果。
- 旧 seed `0` PPO checkpoint 测试 success `2/6`、抓持 `6/6`；W&B run `wandb/offline-run-20261112_025659-skfiyjg2`。checkpoint 训练受 joint log-prob 广播错误影响，不作为公平 PPO 基线。
- 训练期评估现固定 episode ID，并用独立种子采样动作，随后恢复训练 RNG，避免训练过程消耗不同随机数导致各组评估不一致。
- 训练 W&B 指标现包含 `time/rollout_seconds`、`time/update_seconds`、`time/eval_*_seconds`、`feedback/effective_ratio`；SDPO 更新另记录 `time/student_forward_seconds`、`time/teacher_forward_seconds`、`time/update_seconds`，用于拆分环境采样、教师评分和参数更新成本。
- 新指标 GPU smoke 已完成：W&B run `wandb/offline-run-20261112_034227-fq3j7953`，完成 8 environment steps、反馈归档、SDPO 更新、EMA、checkpoint 和 ID/OOD 评估；对应 checkpoint 位于该 run 的 `glob/steps_0000`。
- 修正 action token 对齐：训练前缀现在直接拼接 rollout 原始 token IDs，不再执行 decode/re-tokenize；实测完整 action-bin 词表 decode/re-tokenize round-trip 不成立（`256` 个 token 变为 `257` 个），故旧路径不满足原序列保持要求。修复后的真实 8-step SDPO smoke：W&B `wandb/offline-run-20261112_040926-ne1cou7z`，`feedback/effective_ratio=1.0`、`train/loss=3.6186`、学生前向 `0.3651 s`、教师前向 `0.2449 s`、更新 `0.2413 s`，checkpoint 与协议文件已保存。

## 未完成门槛

- `pytest` 未安装；11 个 SDPO primitive 测试函数已在 `rlvla_env` 中直接执行通过，需在有 pytest 的环境再跑测试框架。
- 上述均为工程闭环 smoke，不是成功率结论；每组仅 1 个 rollout episode，正式比较仍需固定步数、多 seed 和独立评估 episode。
- gripper-open 从 WidowX `left_finger` / `right_finger` 当前关节位置读取，平均位置 `>0.026 m` 判为 open；字段仍支持异常情况下显式 unknown。
- 修复后已完成三 seed 的 8-step 短程矩阵；尚需在确认训练预算后增加训练步数和独立评估 episode。
- 紧凑 EMA checkpoint smoke：修复后 SDPO checkpoint 目录约 `1.1 GB`（旧实现约 `15 GB`，因重复保存冻结 base teacher）；恢复日志确认 `SDPO teacher load: restored EMA state`、optimizer 恢复。W&B run `wandb/offline-run-20261112_042912-ajbcmztx`，恢复评估 run `wandb/offline-run-20261112_043206-izyhhz44`。
- 修复 raw action-token 对齐后的正式短程矩阵已完成：`SEEDS="0 1 2"`、`NUM_ENVS=2`、`EPISODE_LEN=4`、`STEPS_MAX=8`、`SAVE_RENDER=false`，共 12 个本地 W&B run。PPO run：seed0 `wandb/offline-run-20261112_043438-tlo6v5c5`、seed1 `wandb/offline-run-20261112_044318-w7bxs0uw`、seed2 `wandb/offline-run-20261112_045151-69kcb8as`；correct SDPO：seed0 `wandb/offline-run-20261112_043644-jdehu819`、seed1 `wandb/offline-run-20261112_044523-po0i3j3e`、seed2 `wandb/offline-run-20261112_045354-pe0zmezo`；shuffled SDPO：seed0 `wandb/offline-run-20261112_043852-1reaud5l`、seed1 `wandb/offline-run-20261112_044733-hdz55eow`、seed2 `wandb/offline-run-20261112_045607-rw6o7bsb`；none SDPO：seed0 `wandb/offline-run-20261112_044104-yyr81arh`、seed1 `wandb/offline-run-20261112_044942-fqj6s2hy`、seed2 `wandb/offline-run-20261112_045815-m6e6u99i`。这组仅用于验证矩阵、W&B 归档、checkpoint 保存和反馈模式可重复运行，不作为成功率或方法优越性结论；当前 W&B 在线认证仍返回 HTTP 403，恢复凭据后可逐目录执行 `wandb sync`。
- 对这 12 个新 checkpoint 的固定测试集短评估已完成：`PutOnPlateInScene25Main-v3`、episode ID `0,1`、`EPISODE_LEN=8`、`NUM_ENVS=2`、评估 seed `0`。PPO、correct、shuffled、none 四组的三个训练 seed 均为 success `0/2`、grasp `0/2`；逐 run 记录在 `wandb/offline-run-20261112_051418-w7en3dtd` 至 `wandb/offline-run-20261112_053536-8fxtto9z` 的 `glob/eval_only.jsonl`。此前尝试 `EPISODE_LEN=80`、6 episode 的首个 run `wandb/offline-run-20261112_050619-v6ai4iwu` 未完成且无结果，已停止。8 步评估不足以判定任务成功率，不能据此推断反馈是否有效；正式预算待确认。
- 修复 PPO joint log-prob 维度：采样和 `evaluate_action` 都返回每条动作序列一个 log-prob（`[B,1]`），旧 buffer 却存为 `[B,7]`，隐式广播导致 policy loss 对 7 个相同 ratio 求和。已将 buffer 存储改为 `[T,N,1]`，并加入打乱 minibatch 后观测、action token、反馈和旧 log-prob 的对齐测试。此前所有 PPO 训练 run（包括上述 12 组短矩阵中的 PPO 三组）不能作为修复后的 PPO 对照；SDPO 分支不使用旧 log-prob，记录仍可作 SDPO 工程短跑。
- 修复后的 PPO 三 seed 同预算 8-step smoke 均完成并保存 checkpoint：seed0 `wandb/offline-run-20261112_054850-v9fwcdga`、seed1 `wandb/offline-run-20261112_055117-hiej1cxs`、seed2 `wandb/offline-run-20261112_055322-7brwiou7`。同一 `PutOnPlateInScene25Main-v3`、固定 test episode ID `0,1`、`EPISODE_LEN=8` 评估分别为 `wandb/offline-run-20261112_055541-f1lvkk38`、`wandb/offline-run-20261112_055728-3rj230j6`、`wandb/offline-run-20261112_055915-c575yq7x`，均 success `0/2`、grasp `0/2`。这只确认修复后 PPO 闭环及同协议短评估可运行，不是效果比较。
- 多物体反馈抽查：任务源码使用每槽位 `select_carrot_ids`/`select_plate_ids`；本地 `wandb/offline-run-20261112_062239-6ti6jm99/glob/feedback/episode_000000.jsonl` 的 episode 0/1 分别记录 `hamburger`/`bread`，均不是固定取第一个对象。方向文本已从不自然的 `below of`/`front of` 改为 `below`/`in front of` 等规范英文；六个方向有单元测试，新 checkpoint 协议标记 `feedback_text_version=relspatial-v2-grammatical-directions`。上述旧 SDPO 短跑仍使用旧措辞，不能与未来 v2 训练作等价反馈协议比较。
- v2 反馈真实 GPU smoke：`wandb/offline-run-20261112_063929-ic858jxo` 完成 8 环境步、SDPO 更新、EMA、评估、反馈归档和 checkpoint；协议文件含 `feedback_text_version=relspatial-v2-grammatical-directions`。8 条反馈记录已解析核对，文本使用 `below the gripper`、`in front of plate` 等措辞，无旧的 `below of`/`front of`。仍只是工程闭环，不是效果结论。
- 为避免对照组成本混淆，PPO/GRPO 训练不再生成或归档教师特权反馈；SDPO 不再额外计算未用于蒸馏的 bootstrap value。修正后 seed0 同配置 8-step GPU smoke：PPO `wandb/offline-run-20261112_061417-uwlle4jp`，反馈有效比例 `0`、rollout `10.35 s`、update `3.32 s`；correct SDPO `wandb/offline-run-20261112_061622-f8toi9oq`，反馈有效比例 `1`、rollout `10.53 s`、update `4.87 s`，其中学生前向平均 `0.357 s`、教师前向平均 `0.244 s`。只代表短程单 seed 工程测量；正式预算仍待确认。
- 训练 W&B 增加统一的 `train/policy_entropy`（256-bin 动作分布，每 token 平均）以及各 CUDA 设备每轮 rollout+update 的 `gpu/cuda_*_peak_allocated_gib` 和 `gpu/cuda_*_peak_reserved_gib`；峰值在评估前读取。8-step GPU smoke：PPO `wandb/offline-run-20261112_062035-50avd3ij`，熵 `1.365`、CUDA 0 峰值已分配 `19.31 GiB`；correct SDPO `wandb/offline-run-20261112_062239-6ti6jm99`，熵 `1.413`、CUDA 0 峰值已分配 `33.83 GiB`。单 seed 测量仅供容量估计，不作为最终成本或收益结论。

## 2026-11-12 延长 pilot 与 W&B 同步

- 在 commit `5cbc0fd89febc88769e88177ba13e5721eacac34` 上运行一组 160 environment steps 的 seed 0 pilot：`NUM_ENVS=2`、`EPISODE_LEN=80`、每组一个 rollout episode；四组均使用同一 warm-up revision、任务和步数预算，`INTERVAL_EVAL=1`、`INTERVAL_SAVE=40`、关闭渲染。它是单 seed 工程 pilot，不是正式效果比较。
- W&B 官方 API 已通过本机凭据认证。Viewer username 是 `wqx349349`，实际 workspace entity 是 `wqx349349-tongji-university`；先前使用 username 作 entity 被服务端拒绝。四个离线 run 已显式同步到 `wqx349349-tongji-university/RLVLA`，并通过 API 确认状态均为 `finished`：PPO `h49tar1x`、correct SDPO `lxttagsy`、shuffled SDPO `zvpra5sl`、none SDPO `0h9i5dw1`。云端链接分别为 `https://wandb.ai/wqx349349-tongji-university/RLVLA/runs/h49tar1x`、`https://wandb.ai/wqx349349-tongji-university/RLVLA/runs/lxttagsy`、`https://wandb.ai/wqx349349-tongji-university/RLVLA/runs/zvpra5sl`、`https://wandb.ai/wqx349349-tongji-university/RLVLA/runs/0h9i5dw1`。
- 每个训练 rollout 结束后，对 train 和 test/OOD 各用 2 个固定 episode 评估（`NUM_ENVS=2`；不是配置中的 6，因为训练路径每次 eval 调用只评估当前两个并行环境）。W&B summary `eval/success` / `eval/success_ood`：PPO `1.0 / 0.0`，correct SDPO `1.0 / 0.5`，shuffled SDPO `1.0 / 0.0`，none SDPO `1.0 / 0.5`。每格仅 2 个 episode，不能解释为稳定成功率或 SDPO 优势；需要独立、多 episode、多 seed 复验。
- 单轮训练数据：PPO `210.92 s`，correct SDPO `247.31 s`、loss `2.5786`，shuffled SDPO `242.32 s`、loss `2.6067`，none SDPO `243.11 s`、loss `0.0764`。这些数只覆盖这一配置的一轮 rollout/update，不是显著性或最终成本结论。四组 checkpoint 和 SDPO feedback JSONL 均保留于各自本地 `wandb/offline-run-*/glob/`。
- 之前 30 个历史 run 曾同步失败，不能因本次 4 个 run 成功就视作已补传；如需云端完整历史，须筛选有效 run、用正确 workspace entity 加 `--include-synced` 重试，并逐一经 API 核验。旧 `.synced` 标记不作为成功证据。

## 2026-11-12 六集冻结 checkpoint 对照

- 为避免训练期仅评估 episode `0,1` 的小样本限制，对 warm-up、PPO 和三种 SDPO seed 0 / 160-step checkpoint 统一运行冻结策略 test 评估：`PutOnPlateInScene25Main-v3`、`EPISODE_LEN=80`、seed `0`、episode ID `0..5`，每批 2 个环境。各组同一 checkpoint 和测试 episode；无参数更新。结果按本地 `glob/eval_only.jsonl` 的三批统计等权合并。
- Success / grasp：warm-up `0/6` / `4/6`；PPO `1/6` / `4/6`；correct SDPO `4/6` / `6/6`；shuffled SDPO `1/6` / `5/6`；none SDPO `2/6` / `6/6`。单 seed、六个 episode 只是探索性差异，不能推断泛化或反馈因果收益；尤其 none 与 correct 同样抓持 `6/6`，还需至少三训练种子和更多独立测试 episode。
- W&B 全部已核实 `finished`，实体 `wqx349349-tongji-university`、项目 `RLVLA`：warm-up `fke1hpqs`；PPO `09n7h64z`；correct `d2drnr18`；shuffled `bqku833v`；none `7fqyjjfl`。原始本地 `glob/eval_only.jsonl` 保留每批统计；新评估 run 还在 W&B 文件中附每集指标，便于按 episode ID 配对分析。
- 发现 eval-only 旧逻辑的 W&B summary 只保留最后一批指标。现新增 `baseline/overall_*` 六集均值；并依据已归档 JSONL 回填上述五个 run 的云端 summary。验证代码通过 `py_compile`。之后运行 eval-only 会同时记录批次曲线和全评估集均值。
- 同一问题也存在训练期：旧训练循环每次只评估一个并行 batch，未使用 `eval_episodes=6`。`eval_over_episodes` 现按配置评估并聚合固定 episode；`NUM_ENVS=2` 时每次 train/test checkpoint 评估各 6 集，`NUM_ENVS>=6` 时至少评估一个完整并行 batch。新增 mock 测试验证分批 episode IDs 和均值，不启动 GPU。
- W&B SDK 不会自动上传 run 目录中新写入的 JSONL，故 eval-only 现显式 `wandb.save(..., policy="now")`。在线 2-episode artifact smoke run `7b45gele` 已由 API 核验 `finished`，云端 `eval_only.jsonl` 大小 `307 B`，summary 含总体均值；完整六集归档已落本地，逐集上传路径由该 smoke 验证。

## 2026-11-12 当前版本六集评估复核

- 在上述四个 seed 0 / 160-step checkpoint 与 warm-up 上，按相同 `PutOnPlateInScene25Main-v3` test episode `0..5`、冻结策略重新评估；结果仍是 warm-up `0/6`、PPO `1/6`、correct SDPO `4/6`、shuffled SDPO `1/6`、none SDPO `2/6`。W&B API 核实五个 run 均为 `finished`，且每个云端 run 都有 `eval_only.jsonl`（921 B）：warm-up `bknc4sip`、PPO `0lgozlos`、correct `tfkgli85`、shuffled `g8qvuqqy`、none `ie45uvxx`。
- 这是同一训练 seed、每组 6 个固定测试 episode 的探索性结果；当前不能判定反馈带来的因果改进。pytest 模块缺失（`python -m pytest` 报 `No module named pytest`），但测试文件中的 12 个 `test_*` 函数已在 `rlvla_env` 中直接执行并全部通过。
- 后续三 seed、每组 480 environment steps 的对照矩阵尚未运行。按计划先等待训练预算确认；不据此启动长时间训练。
