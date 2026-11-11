# SDPO VLA 首次试跑

## 当前实现

- 仓库：`https://github.com/Skylar-ime/RL4VLA.git`
- 分支：`ex/1111`
- 基座：原版离散 OpenVLA action head，动作分布为 `256` 个 bin，动作 token 为原 rollout token IDs。
- 损失：学生正常 prompt 到 EMA 教师反馈 prompt 的 reverse KL；教师输出停止梯度。
- 反馈：决策时刻 RelSpatial 状态文本。记录原始位置、关系向量、抓持字段、时间戳和文本到 `wandb` 的 `glob/feedback/episode_*.jsonl`。

## 试跑命令

在 `rlvla_env` 环境、仓库根目录执行：

```bash
conda activate rlvla_env
cd /home/skylar/@lab/RL4VLA
PYTHONPATH=SimplerEnv:ManiSkill:openvla \
python SimplerEnv/simpler_env/train_ms3_ppo.py \
  --alg-name sdpo \
  --env-id PutOnPlateInScene25Main-v3 \
  --vla-path openvla/openvla-7b \
  --vla-load-path /path/to/warmup-lora \
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

## 已验证

- `compileall`：通过。
- reverse-KL smoke：通过，梯度仅到学生。
- rollout buffer feedback smoke：通过，含 `buffer-minibatch=-1` 全量采样路径。

## 未完成门槛

- 尚未在本机完成真实 GPU OpenVLA rollout/update/checkpoint 恢复。
- pytest 未安装；测试文件存在但尚未由 pytest 执行。
- 当前 gripper-open 字段在环境没有稳定公开状态时写为显式 unknown。
- 需要实际试跑确认 `PutOnPlateInScene25Main-v3`、warm-up checkpoint、显存和模型复制开销。
