# SDPO VLA 首次试跑

## 当前实现

- 仓库：`https://github.com/Skylar-ime/RL4VLA.git`
- 分支：`ex/1111`
- 当前实现 commit：以 `git rev-parse HEAD` 记录实际运行版本；warm-up 模型 revision：`735fd5d1f4472c4c9f3f998f0e89aa641af5ad03`。
- 试跑环境关键版本：Python `3.10`、PyTorch `2.2.0+cu121`、Transformers `4.40.1`、PEFT `0.11.1`。
- 初始化：RL4VLA 官方 warm-up checkpoint `gen-robot/openvla-7b-rlvla-warmup`；未加载 LoRA，动作分布为 `256` 个 bin，动作 token 为原 rollout token IDs。
- 损失：学生正常 prompt 到 EMA 教师反馈 prompt 的 reverse KL；教师输出停止梯度。
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

```bash
SEEDS="0 1 2" NUM_ENVS=2 EPISODE_LEN=80 STEPS_MAX=4000 WANDB_MODE=online \
  bash SimplerEnv/scripts/run_sdpo_comparison.sh
```

## Warm-up 评估基线

可用 `--eval-only` 对未后训练的 warm-up OpenVLA 执行固定测试 episode 评估，不采样训练 rollout、不更新参数。评估 episode ID 与策略随机数种子固定，便于各方法复用相同测试实例；结果写入 W&B，并保存到 `glob/eval_only.jsonl`。

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

评估入口 smoke 已通过：PPO seed0 checkpoint、`PutOnPlateInScene25Main-v3`、2 个 test episode，W&B run `wandb/offline-run-20261112_032258-zl20amqe`，success `1/2`。

同一冻结 PPO seed0 checkpoint 的 OOD smoke：`PutOnPlateInScene25VisionImage-v1` success `2/2`（W&B `wandb/offline-run-20261112_032920-8yxyjxq5`）；`PutOnPlateInScene25PositionChangeTo-v1` success `0/2`（W&B `wandb/offline-run-20261112_033145-yrb0c0ed`）。每个环境仅 2 个 episode，只证明环境入口和评估协议可运行，不代表泛化结论。

组合辨别 OOD smoke：`PutOnPlateInScene25Instruct-v1` success `2/2`（W&B `wandb/offline-run-20261112_033635-yuzgo7ql`）。样本极少，仅作协议连通性记录。

## 已验证

- `compileall`：通过。
- reverse-KL smoke：通过，梯度仅到学生。
- rollout buffer feedback smoke：通过，含 `buffer-minibatch=-1` 全量采样路径。
- 真实 GPU smoke：通过。`PutOnPlateInScene25Main-v3` 单环境单步完成 rollout、反馈归档、SDPO 更新、EMA、评估和 checkpoint 保存；WandB 本地 run 为 `wandb/offline-run-20261111_231030-gouf9xje`，训练 loss `4.898`、有效 token `7`。
- checkpoint 恢复 smoke：通过。以 `glob/steps_0000` 重载时日志确认 `SDPO teacher load: restored EMA state` 与 optimizer 恢复。
- 首轮控制 smoke：PPO run `wandb/offline-run-20261111_232751-rkm0bfjf` 等 PPO 记录仍可作工程参考；对应 SDPO 记录因旧 action token 对齐缺陷作废。shuffled 使用 `num-envs=2` 的错配逻辑本身已验证，但训练损失不能作方法结果。
- 8 步短程对照：旧 correct SDPO run `wandb/offline-run-20261111_234727-xpt1cfj5` 与 PPO run `wandb/offline-run-20261111_235243-6p5nhm1d`；SDPO 记录因 token 对齐缺陷作废，PPO 记录保留为工程基线。
- 历史多 seed runner 记录：这些 SDPO run 使用旧的 decode/re-tokenize 路径，后来发现不保持原 action token IDs；因此只能作为失败工程记录，不能作为 SDPO 方法结果。对应 W&B 历史保留，不用于结论。
- PPO 部分仍可作为有效短跑记录：`NUM_ENVS=2`、每组总计 `8` steps 的 PPO run 为 seed `0` `wandb/offline-run-20261112_010051-svov1bhf`、seed `1` `...012321-xcbqtdaz`、seed `2` `...014423-nsp9nu6w`；其 SDPO 对照因 token 对齐缺陷全部作废。
- 旧 `SEEDS=1` 记录同时存在 `NUM_ENVS=1` shuffled 退化和 action token 对齐缺陷；保留 W&B 历史，但不作为实验结论。
- runner 与单次训练均启用 W&B。runner 默认 `WANDB_MODE=online`；当前环境在线认证返回 HTTP 403，故上述 run 通过 `WANDB_MODE=offline` 保存在本地 W&B 目录，网络/凭据恢复后可用 `wandb sync <run目录>` 上传。
- Warm-up 固定测试集基线：6 个独立 episode（episode ID `0..5`，每批 2 个）success `0/6`、抓持 `4/6`。W&B run `wandb/offline-run-20261112_021531-fke1hpqs`；逐批指标在 `glob/eval_only.jsonl`。此为有限测试集基线，非统计显著结论。
- 旧 correct SDPO checkpoint 评估 run `wandb/offline-run-20261112_024330-ml6drwkc`、旧 shuffled checkpoint 评估 run `wandb/offline-run-20261112_030731-hhom6kc2` 均受 action token 对齐缺陷影响，不能作为 SDPO 结果。
- 同协议 seed `0` PPO checkpoint 测试 success `2/6`、抓持 `6/6`；W&B run `wandb/offline-run-20261112_025659-skfiyjg2`。该 PPO 结果仍可作为有限基线。
- 训练期评估现固定 episode ID，并用独立种子采样动作，随后恢复训练 RNG，避免训练过程消耗不同随机数导致各组评估不一致。
- 训练 W&B 指标现包含 `time/rollout_seconds`、`time/update_seconds`、`time/eval_*_seconds`、`feedback/effective_ratio`；SDPO 更新另记录 `time/student_forward_seconds`、`time/teacher_forward_seconds`、`time/update_seconds`，用于拆分环境采样、教师评分和参数更新成本。
- 新指标 GPU smoke 已完成：W&B run `wandb/offline-run-20261112_034227-fq3j7953`，完成 8 environment steps、反馈归档、SDPO 更新、EMA、checkpoint 和 ID/OOD 评估；对应 checkpoint 位于该 run 的 `glob/steps_0000`。
- 修正 action token 对齐：训练前缀现在直接拼接 rollout 原始 token IDs，不再执行 decode/re-tokenize；实测完整 action-bin 词表 decode/re-tokenize round-trip 不成立（`256` 个 token 变为 `257` 个），故旧路径不满足原序列保持要求。修复后的真实 8-step SDPO smoke：W&B `wandb/offline-run-20261112_040926-ne1cou7z`，`feedback/effective_ratio=1.0`、`train/loss=3.6186`、学生前向 `0.3651 s`、教师前向 `0.2449 s`、更新 `0.2413 s`，checkpoint 与协议文件已保存。

## 未完成门槛

- `pytest` 未安装；5 个 SDPO primitive 测试函数已在 `rlvla_env` 中直接执行通过，需在有 pytest 的环境再跑测试框架。
- 上述均为工程闭环 smoke，不是成功率结论；每组仅 1 个 rollout episode，正式比较仍需固定步数、多 seed 和独立评估 episode。
- gripper-open 从 WidowX `left_finger` / `right_finger` 当前关节位置读取，平均位置 `>0.026 m` 判为 open；字段仍支持异常情况下显式 unknown。
- 修复后仅完成 1 次 8-step SDPO 闭环 smoke；正式三 seed 对照必须重新运行，尚需更长训练和更多独立评估 episode。
