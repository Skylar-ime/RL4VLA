# 多模态纠错反馈 v2

分支 `ex/v2`。`multimodal-correction-v2` 是新增可选反馈版本；旧 `relspatial-action-v1` 与 `relspatial-outcome-v2` 不覆盖。任务仍为 `PutOnPlateInScene25Main-v3`。本版本已完成真实 Qwen2.5-VL 的 8 步机制 smoke 和 30 集冻结学生评测；目前**没有证明任务成功率改善**。

## 单决策流程

1. 学生只读取执行前图像和任务指令，采样原始 7 个 OpenVLA 动作 token；仿真环境执行由这些 token 解码的动作。
2. 在同一环境、同一决策键下采集执行前后 RGB 图像、状态、距离变化、抓持变化、任务成功和终止标志。将任务指令、解码后的实际控制命令、两张图像和状态变化发送给外部多模态模型。缺失状态保持 `unknown`；模型不得将距离缩短当作完整成功证据。
3. 外部模型返回 JSON：观察结果、离散任务阶段、下一子任务、是否重做／调整顺序、目标修正、机器人 root 坐标系下的末端方向、轨迹修正、夹爪操作、回看原决策的纠错和不确定性。当前开发版进一步将回看纠错拆成**原目标、原方向、原夹爪、原顺序**四个字段，避免一个笼统句子遗漏操作维度。响应必须回报请求的模型 ID；所有字段必须非空，无证据时写 `unknown`。下一子任务、回看纠错及至少一个低层修正必须有具体内容；全为 `unknown` 的空泛反馈被拒绝。root 坐标系三维数值向量与 `-z` 等写法可规范化为带符号的轴描述；原始值保留归档。执行后未抓持时，不接受先去目标容器的子任务或明显背离源物体的方向。源物体中心距离超过 `0.08 m` 且两指均未检测到明显接触时，若模型过早写成“抓持／闭合”，将相关阶段、子任务、轨迹、夹爪与回看纠错字段改为“先靠近／保持打开”，并同时归档模型原文和逐字段修正原因；这不是最优动作标签。格式或逻辑错误最多重问 2 次，仍错误则中止，不悄悄退回纯规则反馈。
4. 纠错文本仅交给 EMA 自教师。学生只读执行前图像和原任务指令；教师在同一执行前图像、原指令和纠错文本上重新评分学生**原动作 token**。停止教师梯度，以 `KL(student || teacher)` 更新学生，每条反馈做一次优化器更新，随后对教师做一次 EMA 更新。外部模型不生成可执行动作 token，不直接控制机器人，也不提供动作监督标签。
5. 反馈和原 token、执行结果一起归档。每批另存执行前后 JPEG 图像 tar，JSONL 记录成员名与 SHA-256，便于审计多模态输出；不把图片 base64 塞进 JSONL。评测／部署只加载学生，不调用外部模型。

并行两环境的同一时间步先都执行，然后分别生成反馈；每条反馈独立触发学生更新及教师 EMA 更新。与以前“先采满 160 条、再集中更新”不同；不能直接将两种更新日程当作同一实验。`alg_gradient_accum=1`、`alg_ppo_epoch=1`、`sdpo_feedback_mode=correct` 在此模式强制要求。对照实验须另行固定共同预算及反馈质量门槛。

当前开发版 JSON 仍是扁平字段，但教师输入文本按 **High-level plan／Low-level guidance／Original-decision correction** 分段；新增 `retrospective_target_adjustment`、`retrospective_direction_root`、`retrospective_gripper_operation`、`retrospective_order_adjustment`，归档与 checkpoint 标注 `correction_schema=high-low-retrospective-v2`。请求中同时给出执行后的目标偏移和**原决策时**目标偏移，均明确转换至 robot root 坐标；原方向如果不需调整，也必须重述带符号的轴，而非只写 `keep`。对于执行前后均可确认未抓持、且原目标偏移已知的靠近样本，四个回看字段不允许 `unknown`，否则要求模型重答；其他阶段保留诚实表达不确定性的能力。方向字段校验同轴符号冲突；未执行的闭合不能写成“闭合失败”。旧 `ymlefep0` 归档和 checkpoint **没有**这四个字段或新标记；真实 Qwen 已在两条旧归档上做离线复核，但尚未用新版协议训练，不能套用旧评测结果。

为在不新增环境交互的条件下复核真实模型，提供 `SimplerEnv/scripts/recheck_sdpo_multimodal_v2_archive.py`：读取指定旧归档行、校验两张 JPEG 的 SHA-256、还原原动作与状态，再向外部模型请求新格式反馈；原图 JPEG 字节不重新编码。`--dry-run` 只做归档校验，不请求模型；旧 run 的第 6、8 行已通过 dry-run。模拟接口端到端测试也已覆盖归档还原、请求、新格式校验与原图哈希保持。生成模式需显式给 `--endpoint`、`--output`，密钥只从 `EXTERNAL_FEEDBACK_API_KEY` 环境变量读取，不写 CLI。

2026-11-19 用 robot h800 分区的 **CPU-only** 作业 `17502` 启动真实 Qwen2.5-VL-7B-Instruct 服务，仅对旧归档第 6、8 行离线推理，未启动仿真、训练或评测。原脚本执行到第 6 行时，模型连续 3 次返回相同 JSON：`retrospective_correction`、`target_adjustment`、`trajectory_correction` 等关键字段为 `unknown`；严格校验拒绝，原脚本因而未写结果。为同时检查第 8 行，再用只读审计调用逐行保存模型原文、校验结果及成功时的完整记录，保留在 `experiments/sdpo-multimodal-v2-root-real-smoke/diagnostics/ymlefep0-rows6-8-qwen-v2-raw-audit.jsonl`。审计重跑第 6 行仍是 3 次相同的空泛答复，均被拒绝；第 8 行一次通过，正确识别未抓持、夹爪原命令为 `OPEN`、下一步继续靠近面包、root 方向 `negative z`，不再错误归因为“闭合失败”。但第 8 行的目标、轨迹和回看纠错几乎只写 `Keep`，缺少明确位置或顺序调整；当前校验器接受了这类形式合格但信息量弱的反馈。第 6、8 行原图 JPEG 均按 SHA-256 校验，未重新编码。**新版协议尚未达到可靠反馈门槛，不应直接扩训**；旧 checkpoint 和评测结果保持不变。

## 外部模型接口与资源

通过 OpenAI-compatible `POST /v1/chat/completions` 接入，默认模型 ID `Qwen/Qwen2.5-VL-7B-Instruct`。服务需支持同一消息中的两张 `image_url` JPEG data URL；脚本只传入 `EXTERNAL_FEEDBACK_URL` 的 `/v1` 基址。仓库提供 `SimplerEnv/scripts/serve_qwen_feedback.py`，在独立环境中运行 Qwen2.5-VL 推理；服务默认只绑定 `127.0.0.1`，跨节点时通过 SSH 隧道访问。服务要求 PyTorch、`transformers==4.51.3`、`qwen-vl-utils` 与 Pillow，不修改 VLA 训练环境。CPU 可用于小规模机制 smoke，规模训练应使用 GPU。例如在可访问 GPU 的节点：

```bash
CUDA_VISIBLE_DEVICES=0 python SimplerEnv/scripts/serve_qwen_feedback.py \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct --port 8000
EXTERNAL_FEEDBACK_URL=http://127.0.0.1:8000/v1 WANDB_MODE=offline bash SimplerEnv/scripts/run_sdpo_multimodal_v2_smoke.sh
```

以上为同机两卡示意。若 Qwen 服务在 robot H800、VLA 在本机 4090，需建立 SSH 转发到计算节点后在本机设置 `EXTERNAL_FEEDBACK_URL`；不要将两者强行放在同一张 4090 上。训练最多使用两张 GPU。跨节点开放端口时启用服务端 `QWEN_SERVER_API_KEY` 与客户端 `EXTERNAL_FEEDBACK_API_KEY`，**不要**写入 CLI、W&B 配置或 Git。当前 W&B 云端 API 返回 `403`；默认离线记录，恢复后再同步并核验。`wandb sync` 曾显示 `done`，但 debug 日志报 `upsert_run` 失败且 API 查询也返回 `403`，所以该 run **不能视为已同步**；误生成的 `.synced` 标记已删除。

运行前应先检查服务能对两张图像稳定返回完整 JSON，再做 8 环境步 smoke，核对 8 条反馈与 8 次学生／EMA 更新、原 token 对齐、输出归档和评测不读取反馈。smoke 后只清理无关临时文件，保留有效审计产物。通过质量门槛前不直接启动 2400 步或更多 seed。

2026-11-18 已用**本地模拟接口**跑通 8 环境步 smoke：归档输出 8 条反馈，训练日志 `train/optimizer_steps=8`；模拟 run 和临时日志已清理。随后在 robot 下载真实 `Qwen/Qwen2.5-VL-7B-Instruct`，用 h800 分区的 **CPU-only** 作业 `17343` 运行模型服务，本机 4090 训练。早期输出把未抓持物体直接送往盘子，或在物体中心距夹爪约 `0.15 m`、无接触时建议闭合夹爪；无效 run 已清理。加入阶段、接触、方向校验和保守修正后，完成真实模型 8 环境步 smoke：W&B 离线 run ID `spmmwmlv`，归档 8 条唯一决策、16 张前后 JPEG 且 SHA-256 一致，`train/optimizer_steps=8`，学生与 EMA 各更新 8 次，checkpoint 记录逐反馈更新协议。该 run **当时归档**显示 7/8 条反馈被规则修正、共 12 个字段；事后发现闭合检测把英语 `close enough`（距离不够近）错认作闭合夹爪指令。修复检测器并对同一归档**离线重算**后，只有 2/8 条原始反馈需要修正、共 7 个字段；这不改变已训练 checkpoint，不能把离线重算当作新实验。剩余两条原始反馈在没有接触时仍建议提前闭合，并把 `+1` 误写成闭合。复查还发现该 run 曾把世界方向当作控制器 root 方向；新代码已改为用 root 四元数转换，但旧归档没有 root 姿态，**不能用来验证新坐标协议**。模型纠错质量与新坐标协议均需重新验证。采样中的成功与新抓持均为 0，样本量不具效果判断力；教师读到的部分文本也经过规则修正，不应据此宣称机制有效或扩大训练。最终原件在 `experiments/sdpo-multimodal-v2/multimodal-v2-sdpo-seed0-8steps-smoke/`，含日志、反馈 JSONL、图像 tar、checkpoint 与离线 W&B；无效中间 run、本地重复权重和临时密钥已清理。作业 `17343` 于 smoke 后主动取消。CPU 服务只用于机制验证，不适合规模训练。

修正 root 坐标协议后，另用本地模拟接口跑通 2 环境步集成 smoke：2 条唯一决策、4 张 JPEG 哈希一致、2 次学生／EMA 更新；归档同时记录决策前 root 四元数和 `ee_direction_frame=robot_root_at_decision`，checkpoint 记录 `external_feedback_schema=robot-root-direction-v2`。本任务的 root 四元数接近 `(0, 0, 0, 1)`，世界方向不可直接当作 root 方向。该次仅验证代码与归档链路，模拟反馈与 checkpoint 均已清理，**不替代真实 Qwen 的新协议 smoke 或效果评测**。

随后以真实 Qwen2.5-VL-7B-Instruct 重新完成新版协议的 8 环境步 smoke。本机 4090 训练、robot 的 h800 分区 CPU-only 作业 `17367` 推理，未申请训练 GPU。W&B 离线 run `ymlefep0`；8 条唯一反馈、16 张 JPEG 的哈希全部匹配，`train/optimizer_steps=8`，归档均标注 `robot-root-direction-v2` 且记录决策前 root 四元数。模型 8 次请求均一次成功；**当时运行的旧护栏**只改写 1 条反馈的 1 个字段。但模型在另 2 条记录仍于无抓持、无接触且物体中心距夹爪分别约 `0.119 m` 和 `0.103 m` 时建议闭合夹爪，未触发当时 `>0.12 m` 护栏。这说明低规则改写率**不等于**反馈可靠。8 步采样无成功、无新抓持；现有证据不能证明学生动作改善。审计原件保留在 `experiments/sdpo-multimodal-v2-root-real-smoke/multimodal-v2-sdpo-seed0-8steps-smoke/`；服务作业已取消，隧道与临时密钥、服务脚本、重复日志已清理。下一步需审计接触与抓持时序，并用冻结学生固定评测比较 warm-up 与新 checkpoint，达标前不扩训。

对归档中的 warm-up 决策离线核对：未抓持时 6 次闭合命令的执行前物体中心距离全部不超过 `0.066 m`；3 次下一步新抓持的距离为 `0.058–0.073 m`。这仅是 80 条决策、6 段 episode 的有限观察，不能视为通用抓持阈值。当前保守护栏据此把“无抓持、无明显接触、中心距离 `>0.08 m`”视作继续靠近；**闭合可以建立接触**，不要求先有接触才允许闭合。同时修复 `without closing` 被误判为闭合建议，以及模型单词 `CLOSE`、`did not close properly` 漏判的问题。重新审计 `ymlefep0` 的原始模型文本，新护栏会改写 2/8 条反馈中的 6 个字段，正是两条过早闭合建议；已训练 checkpoint 仍使用原护栏和原文本，离线复算不是新实验。

逐条人工复核还发现 `ymlefep0` 第 8 条：学生解码命令保持夹爪打开，模型却把未抓持归因于“夹爪没有正确闭合”，混淆了**未尝试闭合**与**闭合失败**。现有请求 prompt 明确禁止这种归因；若响应仍这样写，校验器会拒绝并要求模型重答，而非直接给教师。该修正晚于 `ymlefep0` 训练，同样不能追溯改变旧 checkpoint。

`ymlefep0` 的 8 步学生 checkpoint 已完成冻结评测：`PutOnPlateInScene25Main-v3`、`test` 物体集、seed 0、30 集（6 环境×5 批）、每集最多 80 步，评测期间无外部模型输入。逐集成功 **11/30**，`is_src_obj_grasped` **26/30**，`consecutive_grasp` **24/30**。同一评测协议的未训练 warm-up 为成功 **14/30**、`is_src_obj_grasped` **23/30**、`consecutive_grasp` **20/30**；配对成功记录中仅 `ymlefep0` 成功 4 集，仅 warm-up 成功 7 集。两组均属单一 seed、30 集；8 步 smoke 的结果**没有证明学生任务成功率改善**，也不能据此定量归因于反馈质量或否定更长训练。`ymlefep0` 使用旧 `0.12 m` 护栏及错误归因文本，不能代表后续修正后的协议。原始逐集记录在 `experiments/sdpo-multimodal-v2-root-real-smoke/evaluation/ymlefep0-steps8/` 和 `experiments/sdpo-v1/evaluation/v1-warmup/`，两者均为离线 W&B。`wandb sync --entity wqx349349 --project RLVLA` 对本次 run 返回 `403 Forbidden`，即使 CLI 退出码为 0，云端同步仍未证实。

另用 `SimplerEnv/scripts/diagnose_sdpo_multimodal_v2_offline.py` 对上述 8 条训练归档执行**冻结教师离线反事实检查**：恢复 `ymlefep0` 的 EMA 教师与学生，逐条使用执行前 JPEG、原始 7 个动作 token 和机器人 root 方向，比较原反馈、与原反馈等 token 数的中性文本、无反馈。每种条件均计算沿原动作前缀的教师方向期望投影，以及从 `Out:` 自由贪心生成 7 个 token 的实际平移投影；这两种量不能互换。原反馈／等长中性／无反馈的平均前缀投影分别为 **+10.05／+14.37／+13.53 mm**，自由贪心投影分别为 **+11.42／+12.67／+13.23 mm**。8 条原反馈自由动作均朝源物体投影为正，但相较中性文本仅 5/8 条投影增加；前缀投影仅 2/8 条增加。教师确实响应文本，**这批反馈却没有显示稳定增益**。样本全处于靠近阶段、仅 2 段并行 rollout、且是训练所用图像；不能推断独立评测成功率或其他子任务效果。原始 24 行评分保留在 `experiments/sdpo-multimodal-v2-root-real-smoke/diagnostics/ymlefep0-feedback-sensitivity.jsonl`，无新环境交互或训练。

## 冻结 warm-up 教师的四条件反事实检查（2026-11-19）

脚本 `SimplerEnv/scripts/diagnose_sdpo_v2_warmup_feedback.py` 从已有 warm-up 诊断归档 `experiments/sdpo-v1/diagnostics/20261113-warmup/` 选 12 条决策（靠近、抓持、搬运各 4 条），固定**同一个未经过 SDPO 训练的 warm-up 教师**、执行前图像、任务 `put nonstop can on plate` 和原 7 个动作 token，仅替换教师输入文本。`correct` 和 `wrong` 是按归档状态、root 方向、距离、接触与抓持状态人工构造的受控反馈／故意反向反馈，**不是 Qwen 原始输出，也不是最优动作真值**；`neutral_length` 与 `correct` 等 tokenizer 长度，`none` 无反馈。方向文本为 robot-root 轴，抓持阶段只选源物体中心距夹爪不超过 `0.08 m` 的样本；搬运阶段要求归档显示已抓持。抽查执行前图像与状态一致。所有 12 条来自同一物体 `nonstop can`、少数相邻轨迹，不能视作独立任务分布。

每条反馈分别评分同一原动作前缀的 7 个条件 token 分布，计算 `KL(correct || alternative)`、按条件分布期望平移沿目标方向投影、条件夹爪闭合概率；另各做一次自由贪心生成和一次固定随机种子的自由采样，并投影其解码平移。原始逐条结果在 `experiments/sdpo-multimodal-v2-root-real-smoke/diagnostics/warmup-teacher-controlled-feedback-12.jsonl`。下表投影单位为 mm，按阶段各 4 条取均值；`P(close)` 来自夹爪 token 条件分布。

| 阶段 | 文本 | 条件投影 | 贪心投影 | 单次采样投影 | P(close) |
| --- | --- | ---: | ---: | ---: | ---: |
| 靠近 | correct / wrong / neutral / none | +13.27 / +11.63 / +8.09 / +8.71 | +9.19 / +9.48 / +9.47 / +11.72 | +7.90 / +8.91 / +5.81 / +3.22 | 约 0 / 约 0 / 约 0 / 约 0 |
| 抓持 | correct / wrong / neutral / none | +1.66 / +0.71 / +0.28 / -1.17 | -3.05 / +4.53 / +0.06 / -1.99 | -0.88 / -0.18 / +2.94 / -0.62 | 0.400 / 0.401 / 0.344 / 0.971 |
| 搬运 | correct / wrong / neutral / none | +2.62 / +0.45 / +5.79 / +17.15 | +8.05 / +7.59 / +9.26 / +13.89 | +16.30 / +5.63 / +10.37 / +16.98 | 0.996 / 0.964 / 0.993 / 1.000 |

`correct` 与 `wrong` 的平均 7-token 条件 KL 在靠近、抓持、搬运分别为 **0.826、0.780、1.334 nat/token**，说明同一冻结教师的动作分布确实受文本影响。但**影响未稳定朝正确方向**：靠近阶段 `correct` 相对 `wrong` 的条件投影只在 3/4 条上升，贪心投影仅 1/4 条严格上升；抓持阶段 `correct` 的平均闭合概率与 `wrong` 几乎相同，且低于 `none`；搬运阶段 `none` 的条件投影高于 `correct`。`wrong` 与 `correct` 的 token 长度未强制一致，KL 同时可能含文本长度效应；等长中性组专门控制了这一点。单次随机采样方差大，不作为稳定性证据。条件分布沿原动作 token 前缀评分，不等于自由生成的边缘动作分布；不能把这些离线投影解释为环境成功率。结论：**“teacher 会受反馈影响”成立，“正确反馈能稳定改善 teacher 动作”尚不成立，因此不能只扩大训练。**先改进反馈质量与抓持／搬运阶段的判别，再做更大、多物体、独立轨迹的冻结反事实检查和小规模训练评测。

## 限制

- 多模态纠错是外部模型的**建议**，不是最优动作真值。末端方向使用控制器的机器人 root 坐标系；状态坐标先由世界系经归档的 robot root 四元数转换。图像相机、世界、root 方向不能混用。无执行后夹爪接触或轨迹真值时，应保留不确定性。
- 当前接口每次反馈调用一次模型服务，没有缓存或异步批处理；耗时会明显高于 v1。失败时不会更新该条反馈对应的教师，但已经完成的前序更新不会自动回滚。
- 需要独立验证模型对图像、方向与操作顺序的敏感性，并用冻结学生固定 episode 评测。仅凭训练 loss 下降或模型文字看起来合理，不能认定机制有效。
