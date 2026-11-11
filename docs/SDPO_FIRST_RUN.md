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
  --wandb false \
  --name sdpo-smoke
```

反馈控制：

- `--sdpo-feedback-mode correct`：正确反馈，默认。
- `--sdpo-feedback-mode shuffled`：批内循环错配反馈。
- `--sdpo-feedback-mode none`：无反馈 EMA 蒸馏。
- SDPO checkpoint 同目录写入 `sdpo_protocol.json`，记录 divergence、EMA、action support、反馈模式及 PPO/advantage 开关。
- WandB 默认在线记录；若服务端返回认证或网络错误，可设置 `WANDB_MODE=offline`，仍保留本地 run、配置和指标归档。

## 已验证

- `compileall`：通过。
- reverse-KL smoke：通过，梯度仅到学生。
- rollout buffer feedback smoke：通过，含 `buffer-minibatch=-1` 全量采样路径。

## 未完成门槛

- 尚未在本机完成真实 GPU OpenVLA rollout/update/checkpoint 恢复；warm-up 权重需先完整下载。
- pytest 未安装；测试文件存在但尚未由 pytest 执行。
- gripper-open 从 WidowX `left_finger` / `right_finger` 当前关节位置读取，平均位置 `>0.026 m` 判为 open；字段仍支持异常情况下显式 unknown。
- 需要实际试跑确认 `PutOnPlateInScene25Main-v3`、warm-up checkpoint、显存和模型复制开销。
