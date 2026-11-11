import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from simpler_env.feedback import build_rel_spatial_feedback


class SimlerWrapper:
    def __init__(self, all_args, unnorm_state, extra_seed=0):
        self.args = all_args
        self.unnorm_state = unnorm_state

        self.num_envs = self.args.num_envs
        robot_control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

        env_config = dict(
            id=self.args.env_id,
            num_envs=self.args.num_envs,
            obs_mode="rgb+segmentation",
            control_mode=robot_control_mode,
            sim_backend=getattr(self.args, "sim_backend", "gpu"),
            sim_config={
                "sim_freq": 500,
                "control_freq": 5,
            },
            max_episode_steps=self.args.episode_len,
            sensor_configs={"shader_pack": "default"},
        )
        self.env: BaseEnv = gym.make(**env_config)
        self.env.reset(seed=[self.args.seed * 1000 + i + extra_seed for i in range(self.args.num_envs)])

        # variables
        self.reward_old = torch.zeros(self.args.num_envs, 1, dtype=torch.float32, device=self.env.unwrapped.device)
        self.episode_ids = [None] * self.num_envs
        self.decision_index = [0] * self.num_envs

        # constants
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def get_reward(self, info):
        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(info["success"].device)  # [B, 1]

        reward += info["is_src_obj_grasped"].reshape(-1, 1) * 0.1
        reward += info["consecutive_grasp"].reshape(-1, 1) * 0.1
        reward += (info["success"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)) * 1.0

        # diff
        if self.reward_old.device != reward.device:
            self.reward_old = self.reward_old.to(reward.device)
        reward_diff = reward - self.reward_old
        self.reward_old = reward

        return reward_diff

    def _process_action(self, raw_actions: torch.Tensor) -> torch.Tensor:
        action_scale = 1.0

        # Extract predicted action tokens and translate into (normalized) continuous actions
        pact_token = raw_actions.cpu().numpy()  # [B, dim]
        dact = 32000 - pact_token  # [B, dim]
        dact = np.clip(dact - 1, a_min=0, a_max=254)  # [B, dim]
        normalized_actions = np.asarray([self.bin_centers[da] for da in dact])  # [B, dim]

        # Unnormalize actions
        action_norm_stats = self.unnorm_state
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))  # [dim]
        mask = np.asarray(mask).reshape(1, -1)  # [1, dim]
        action_high = np.array(action_norm_stats["q99"]).reshape(1, -1)  # [1, dim]
        action_low = np.array(action_norm_stats["q01"]).reshape(1, -1)  # [1, dim]
        raw_action_np = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        raw_action = {
            "world_vector": raw_action_np[:, :3],
            "rotation_delta": raw_action_np[:, 3:6],
            "open_gripper": raw_action_np[:, 6:7],  # range [0, 1]; 1 = open; 0 = close
        }
        action = {}
        action["world_vector"] = raw_action["world_vector"] * action_scale  # [B, 3]
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0  # [B, 1]

        # origin euler
        action["rot_axangle"] = raw_action["rotation_delta"]

        action = {k: torch.tensor(v) for k, v in action.items()}  # to float32 ?

        action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)

        # to tpdv
        action = action.to(raw_actions.device)

        return action

    def reset(self, obj_set: str, same_init: bool = False):
        options = {}
        options["obj_set"] = obj_set
        if same_init:
            options["episode_id"] = torch.randint(1000000000, (1,)).expand(self.num_envs).to(self.env.device)  # [B]

        obs, info = self.env.reset(options=options)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.env.unwrapped.get_language_instruction()

        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32).to(obs_image.device)  # [B, 1]
        episode_seed = getattr(self.env.unwrapped, "_episode_seed", [None] * self.num_envs)
        self.episode_ids = [int(x) for x in episode_seed]
        self.decision_index = [0] * self.num_envs

        return obs_image, instruction, info

    def capture_feedback(self):
        """Capture privileged state before action execution, aligned to current observation."""
        task = self.env.unwrapped
        extra = getattr(task, "extra_stats", {})
        episode_stats = getattr(task, "episode_stats", {})
        def selected_names(ids_key, names_key, database_key, fallback_key):
            ids = getattr(task, ids_key, None)
            names = getattr(task, names_key, None)
            database = getattr(task, database_key, None)
            if ids is not None and names is not None and database is not None:
                ids = ids.detach().cpu().tolist() if hasattr(ids, "detach") else list(ids)
                return [database[names[int(index)]]["name"] for index in ids]
            fallback = getattr(task, fallback_key, None)
            return fallback if isinstance(fallback, (list, tuple)) else [fallback] * self.num_envs

        source_names = selected_names("select_carrot_ids", "carrot_names", "model_db_carrot", "source_obj_name")
        target_names = selected_names("select_plate_ids", "plate_names", "model_db_plate", "target_obj_name")
        gripper_open = None
        robot = getattr(getattr(task, "agent", None), "robot", None)
        if robot is not None:
            active_joints = getattr(robot, "active_joints", [])
            joint_names = [joint.name for joint in active_joints]
            gripper_indices = [joint_names.index(name) for name in ("left_finger", "right_finger") if name in joint_names]
            if gripper_indices:
                gripper_width = robot.get_qpos()[:, gripper_indices].mean(dim=1)
                gripper_open = gripper_width > 0.026

        def at(values, index):
            if values is None:
                return None
            try:
                return values[index]
            except (IndexError, KeyError, TypeError):
                return values

        records = []
        for index in range(self.num_envs):
            record = build_rel_spatial_feedback(
                episode_id=self.episode_ids[index], decision_index=self.decision_index[index],
                source_name=source_names[index], target_name=target_names[index],
                source_position=at(extra.get("extra_pos_carrot"), index),
                target_position=at(extra.get("extra_pos_plate"), index),
                gripper_position=at(extra.get("extra_pos_gripper"), index),
                source_grasped=at(episode_stats.get("is_src_obj_grasped"), index),
                gripper_open=at(gripper_open, index),
            )
            self.decision_index[index] += 1
            records.append(record)
        return records

    def step(self, raw_action):
        action = self._process_action(raw_action)

        obs, _reward, terminated, truncated, info = self.env.step(action)
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        done = torch.logical_or(terminated, truncated).reshape(-1, 1)  # [B, 1]

        # calculate reward
        reward = self.get_reward(info)

        # process episode info
        if done.any():
            info["episode"] = {}
            for k in ["is_src_obj_grasped", "consecutive_grasp", "success"]:
                v = [info[k][idx].item() for idx in range(self.num_envs)]
                info["episode"][k] = v

        return obs_image, reward, done, info
