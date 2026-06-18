from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from fish_floating_grasp_env import FloatingGraspConfig, MujocoFishFloatingGraspEnv


def maybe_set(cfg: FloatingGraspConfig, key: str, value) -> None:
    if value is not None:
        setattr(cfg, key, value)


class VecNormalizeSaveCallback(BaseCallback):
    def __init__(self, save_path: Path, save_freq: int = 50_000) -> None:
        super().__init__()
        self.save_path = save_path
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.n_calls % max(1, self.save_freq) == 0 and self.model.get_env() is not None:
            vec_env = self.model.get_env()
            if isinstance(vec_env, VecNormalize):
                vec_env.save(str(self.save_path))
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, update_every: int = 2048, eval_env=None, start_progress: float = 0.0) -> None:
        super().__init__()
        self.total_timesteps = max(1, int(total_timesteps))
        self.update_every = max(1, int(update_every))
        self.eval_env = eval_env
        self.start_progress = float(np.clip(start_progress, 0.0, 1.0))

    def _current_progress(self) -> float:
        local = float(self.num_timesteps) / float(self.total_timesteps)
        return min(1.0, self.start_progress + (1.0 - self.start_progress) * local)

    def _on_training_start(self) -> None:
        if self.training_env is not None:
            self.training_env.env_method("set_training_progress", self.start_progress)

    def _on_step(self) -> bool:
        if self.training_env is None:
            return True
        if self.n_calls % self.update_every == 0 or self.num_timesteps <= self.training_env.num_envs:
            progress = self._current_progress()
            self.training_env.env_method("set_training_progress", progress)
            if self.eval_env is not None:
                self.eval_env.env_method("set_training_progress", progress)
        return True


class DiagnosticsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._keys = (
            "is_success",
            "ever_entered_success_region",
            "max_hold_fraction",
            "dropped_from_success_region",
            "drift_fail",
        )
        self._buffer = {key: [] for key in self._keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for info, done in zip(infos, dones):
            if not done:
                continue
            for key in self._keys:
                if key in info:
                    self._buffer[key].append(float(info[key]))
        return True

    def _on_rollout_end(self) -> None:
        metric_names = {
            "is_success": "rollout/final_success_rate",
            "ever_entered_success_region": "rollout/reach_success_region_rate",
            "max_hold_fraction": "rollout/max_hold_fraction",
            "dropped_from_success_region": "rollout/drop_from_region_rate",
            "drift_fail": "rollout/drift_fail_rate",
        }
        for key, values in self._buffer.items():
            if values:
                self.logger.record(metric_names[key], float(np.mean(values)))
                values.clear()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(xml_path: str, rank: int, seed: int, cfg: FloatingGraspConfig):
    def _init():
        env = MujocoFishFloatingGraspEnv(xml_path=xml_path, config=cfg)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def cfg_to_jsonable(cfg: FloatingGraspConfig) -> dict:
    raw = asdict(cfg)
    out = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练方案 A：尾部 4 关节同时做 base 稳定 + 尾部定位")
    parser.add_argument("--xml", type=str, default="phy3.02_schemeA_grasp.xml")
    parser.add_argument(
        "--stage-preset",
        choices=["none", "stage1_reach_hold", "stage2_final_hold", "stage2_final_hold_slow_assist", "stage2_final_hold_conversion"],
        default="none",
        help="两阶段训练预设：stage1_reach_hold 用于短 hold 到达；stage2_final_hold/stage2_final_hold_slow_assist/stage2_final_hold_conversion 用于最终 hold 微调",
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="checkpoints/floating_grasp_recurrentppo")
    parser.add_argument("--log-dir", type=str, default="logs/tensorboard")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--algo", choices=["recurrentppo", "ppo", "sac"], default="recurrentppo")
    parser.add_argument("--resume", type=str, default=None, help="已有模型 zip 路径")
    parser.add_argument("--vecnorm", type=str, default=None, help="已有 VecNormalize pkl 路径")
    parser.add_argument("--assist-mode", choices=["off", "constant", "decay"], default="off")
    parser.add_argument("--assist-max-strength", type=float, default=0.35)
    parser.add_argument("--assist-decay-end", type=float, default=0.35)
    parser.add_argument("--assist-episode-randomization", action="store_true")
    parser.add_argument("--assist-random-zero-prob", type=float, default=0.30)
    parser.add_argument("--assist-random-min-scale", type=float, default=0.20)
    parser.add_argument("--assist-random-max-scale", type=float, default=1.00)
    parser.add_argument("--curriculum-max-target-scale", type=float, default=1.0)
    parser.add_argument("--success-distance-easy", type=float, default=None)
    parser.add_argument("--success-distance-final", type=float, default=None)
    parser.add_argument("--hold-steps-easy", type=int, default=None)
    parser.add_argument("--hold-steps-final", type=int, default=None)
    parser.add_argument("--success-ee-speed", type=float, default=None)
    parser.add_argument("--near-radius", type=float, default=None)
    parser.add_argument("--dense-reward-weight", type=float, default=None)
    parser.add_argument("--progress-reward-weight", type=float, default=None)
    parser.add_argument("--hold-reward-weight", type=float, default=None)
    parser.add_argument("--precision-reward-weight", type=float, default=None)
    parser.add_argument("--near-target-speed-penalty-weight", type=float, default=None)
    parser.add_argument("--near-target-base-pos-penalty-weight", type=float, default=None)
    parser.add_argument("--near-target-base-rot-penalty-weight", type=float, default=None)
    parser.add_argument("--success-region-drop-penalty", type=float, default=None)
    parser.add_argument("--success-bonus", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--constant-lr", action="store_true", help="使用固定学习率而非 linear_schedule（短段微调推荐）")
    parser.add_argument("--resume-progress", type=float, default=0.0,
                        help="resume 时的起始 training_progress（0~1），避免每次从 0 重置课程和辅助力")
    parser.add_argument("--n-steps", type=int, default=None, help="每个 rollout 的步数；默认 recurrent=512, ppo=1024")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--buffer-size", type=int, default=300_000, help="SAC replay buffer 大小")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="SAC 开始训练前的随机探索步数")
    parser.add_argument("--tau", type=float, default=0.005, help="SAC soft update 系数")
    parser.add_argument("--frame-stack", type=int, default=1, help="环境内部观测堆叠帧数（默认 1 不堆叠）")
    parser.add_argument("--progress-bar", action="store_true", help="显示训练进度条")
    return parser.parse_args()


def apply_stage_preset(cfg: FloatingGraspConfig, args: argparse.Namespace) -> None:
    if args.stage_preset == "none":
        return

    if args.stage_preset == "stage1_reach_hold":
        cfg.assist_mode = "constant"
        cfg.assist_max_strength = max(float(args.assist_max_strength), 0.35)
        cfg.assist_episode_randomization = False
        cfg.target_random_base_xyz_final = cfg.target_random_base_xyz_final * min(
            float(args.curriculum_max_target_scale), 0.75
        )
        cfg.success_distance_easy = 0.14
        cfg.success_distance_final = 0.09
        cfg.hold_steps_required_easy = 2
        cfg.hold_steps_required_final = 3
        cfg.success_ee_speed = 0.12
        cfg.near_radius = 0.12
        cfg.dense_reward_weight = 2.0
        cfg.progress_reward_weight = 12.0
        cfg.hold_reward_weight = 8.0
        cfg.precision_reward_weight = 4.0
        cfg.near_target_speed_penalty_weight = 0.03
        cfg.success_region_drop_penalty = 1.0
        cfg.success_bonus = 100.0
        return

    if args.stage_preset == "stage2_final_hold":
        cfg.assist_mode = "decay"
        cfg.assist_max_strength = float(args.assist_max_strength)
        cfg.assist_decay_end = float(args.assist_decay_end)
        cfg.assist_episode_randomization = True
        cfg.assist_random_zero_prob = max(float(args.assist_random_zero_prob), 0.40)
        cfg.assist_random_min_scale = min(float(args.assist_random_min_scale), 0.15)
        cfg.assist_random_max_scale = float(args.assist_random_max_scale)
        cfg.success_distance_easy = 0.10
        cfg.success_distance_final = 0.05
        cfg.hold_steps_required_easy = 6
        cfg.hold_steps_required_final = 20
        cfg.success_ee_speed = 0.07
        cfg.near_radius = 0.08
        cfg.dense_reward_weight = 1.5
        cfg.progress_reward_weight = 5.0
        cfg.hold_reward_weight = 14.0
        cfg.precision_reward_weight = 6.0
        cfg.near_target_speed_penalty_weight = 0.15
        cfg.success_region_drop_penalty = 4.0
        cfg.success_bonus = 220.0
        return

    if args.stage_preset == "stage2_final_hold_slow_assist":
        cfg.assist_mode = "decay"
        cfg.assist_max_strength = float(args.assist_max_strength)
        cfg.assist_decay_end = max(float(args.assist_decay_end), 0.97)
        cfg.assist_episode_randomization = True
        cfg.assist_random_zero_prob = min(float(args.assist_random_zero_prob), 0.25)
        cfg.assist_random_min_scale = max(float(args.assist_random_min_scale), 0.30)
        cfg.assist_random_max_scale = float(args.assist_random_max_scale)
        cfg.success_distance_easy = 0.10
        cfg.success_distance_final = 0.05
        cfg.hold_steps_required_easy = 6
        cfg.hold_steps_required_final = 20
        cfg.success_ee_speed = 0.07
        cfg.near_radius = 0.08
        cfg.dense_reward_weight = 1.5
        cfg.progress_reward_weight = 5.0
        cfg.hold_reward_weight = 14.0
        cfg.precision_reward_weight = 6.0
        cfg.near_target_speed_penalty_weight = 0.15
        cfg.success_region_drop_penalty = 4.0
        cfg.success_bonus = 220.0
        return

    if args.stage_preset == "stage2_final_hold_conversion":
        cfg.assist_mode = "decay"
        cfg.assist_max_strength = float(args.assist_max_strength)
        cfg.assist_decay_end = float(args.assist_decay_end)
        cfg.assist_episode_randomization = True
        cfg.assist_random_zero_prob = max(float(args.assist_random_zero_prob), 0.40)
        cfg.assist_random_min_scale = min(float(args.assist_random_min_scale), 0.15)
        cfg.assist_random_max_scale = float(args.assist_random_max_scale)
        cfg.success_distance_easy = 0.10
        cfg.success_distance_final = 0.05
        cfg.hold_steps_required_easy = 6
        cfg.hold_steps_required_final = 20
        cfg.success_ee_speed = 0.07
        cfg.near_radius = 0.12
        cfg.dense_reward_weight = 1.2
        cfg.progress_reward_weight = 3.5
        cfg.hold_reward_weight = 18.0
        cfg.hold_reward_progress_power = 1.8
        cfg.precision_reward_weight = 7.5
        cfg.near_hold_decay_steps = 1  # 在 near 区缓慢衰减（原 2）
        cfg.drop_hold_penalty_steps = 1  # 飘出成功区时缓慢清零（原 4），避免短暂振荡就清零
        cfg.near_target_speed_penalty_weight = 0.25
        cfg.near_target_base_pos_penalty_weight = 3.0
        cfg.near_target_base_rot_penalty_weight = 1.0
        cfg.success_region_drop_penalty = 6.0
        cfg.success_bonus = 260.0


def build_config(args: argparse.Namespace) -> FloatingGraspConfig:
    cfg = FloatingGraspConfig()
    cfg.assist_mode = args.assist_mode
    cfg.assist_max_strength = float(args.assist_max_strength)
    cfg.assist_decay_end = float(args.assist_decay_end)
    cfg.assist_episode_randomization = bool(args.assist_episode_randomization)
    cfg.assist_random_zero_prob = float(args.assist_random_zero_prob)
    cfg.assist_random_min_scale = float(args.assist_random_min_scale)
    cfg.assist_random_max_scale = float(args.assist_random_max_scale)
    cfg.target_random_base_xyz_final = cfg.target_random_base_xyz_final * float(args.curriculum_max_target_scale)
    cfg.obs_stack_size = int(args.frame_stack)
    apply_stage_preset(cfg, args)
    maybe_set(cfg, "success_distance_easy", args.success_distance_easy)
    maybe_set(cfg, "success_distance_final", args.success_distance_final)
    maybe_set(cfg, "hold_steps_required_easy", args.hold_steps_easy)
    maybe_set(cfg, "hold_steps_required_final", args.hold_steps_final)
    maybe_set(cfg, "success_ee_speed", args.success_ee_speed)
    maybe_set(cfg, "near_radius", args.near_radius)
    maybe_set(cfg, "dense_reward_weight", args.dense_reward_weight)
    maybe_set(cfg, "progress_reward_weight", args.progress_reward_weight)
    maybe_set(cfg, "hold_reward_weight", args.hold_reward_weight)
    maybe_set(cfg, "precision_reward_weight", args.precision_reward_weight)
    maybe_set(cfg, "near_target_speed_penalty_weight", args.near_target_speed_penalty_weight)
    maybe_set(cfg, "near_target_base_pos_penalty_weight", args.near_target_base_pos_penalty_weight)
    maybe_set(cfg, "near_target_base_rot_penalty_weight", args.near_target_base_rot_penalty_weight)
    maybe_set(cfg, "success_region_drop_penalty", args.success_region_drop_penalty)
    maybe_set(cfg, "success_bonus", args.success_bonus)
    return cfg


def main() -> None:
    args = parse_args()
    save_base = Path(args.save)
    save_base.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)

    env_fns = [make_env(args.xml, rank=i, seed=args.seed, cfg=cfg) for i in range(args.n_envs)]
    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)

    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = True
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env_raw = DummyVecEnv([make_env(args.xml, rank=10_000, seed=args.seed, cfg=cfg)])
    if args.vecnorm:
        eval_env = VecNormalize.load(args.vecnorm, eval_env_raw)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    eval_env.env_method("set_training_progress", 1.0)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    sac_policy_kwargs = dict(net_arch=[256, 256])
    recurrent_policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    if args.resume:
        if args.algo == "recurrentppo":
            model = RecurrentPPO.load(args.resume, env=vec_env, device=args.device)
        elif args.algo == "sac":
            model = SAC.load(args.resume, env=vec_env, device=args.device)
        else:
            model = PPO.load(args.resume, env=vec_env, device=args.device)
        # 覆盖学习率（RecurrentPPO.load 会恢复 checkpoint 里的调度，需要手动覆盖）
        new_lr = args.learning_rate if args.constant_lr else linear_schedule(args.learning_rate)
        model.learning_rate = new_lr
        # SB3 内部缓存了 lr_schedule 和 clip_range_schedule callable，必须一并覆盖
        try:
            from stable_baselines3.common.utils import FloatSchedule, ConstantSchedule
            make_schedule = ConstantSchedule if args.constant_lr else FloatSchedule
            model.lr_schedule = make_schedule(args.learning_rate)
            model.clip_range = make_schedule(0.2)
        except ImportError:
            from stable_baselines3.common.utils import get_schedule_fn
            model.lr_schedule = get_schedule_fn(new_lr)
            model.clip_range = get_schedule_fn(0.2)
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = args.learning_rate
    else:
        lr = args.learning_rate if args.constant_lr else linear_schedule(args.learning_rate)
        clip_range = 0.2 if args.constant_lr else linear_schedule(0.2)
        if args.algo == "recurrentppo":
            model = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=vec_env,
                learning_rate=lr,
                n_steps=args.n_steps or 512,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=clip_range,
                verbose=1,
                tensorboard_log=args.log_dir,
                seed=args.seed,
                device=args.device,
            )
        elif args.algo == "sac":
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=0.99,
                ent_coef="auto",
                policy_kwargs=sac_policy_kwargs,
                verbose=1,
                tensorboard_log=args.log_dir,
                seed=args.seed,
                device=args.device,
            )
        else:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=lr,
                n_steps=args.n_steps or 1024,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=clip_range,
                ent_coef=0.002,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=args.log_dir,
                seed=args.seed,
                device=args.device,
            )

    callbacks = CallbackList(
        [
            ProgressCallback(total_timesteps=args.timesteps, update_every=max(256 // max(args.n_envs, 1), 1), eval_env=eval_env, start_progress=args.resume_progress),
            DiagnosticsCallback(),
            CheckpointCallback(
                save_freq=max(10_000 // max(args.n_envs, 1), 1),
                save_path=str(save_base.parent / f"{save_base.name}_ckpts"),
                name_prefix=save_base.name,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_base.parent / f"{save_base.name}_best"),
                log_path=str(save_base.parent / f"{save_base.name}_eval"),
                eval_freq=max(20_000 // max(args.n_envs, 1), 1),
                n_eval_episodes=8,
                deterministic=True,
                render=False,
            ),
            VecNormalizeSaveCallback(
                save_path=save_base.with_name(save_base.name + "_vecnormalize.pkl"),
                save_freq=max(50_000 // max(args.n_envs, 1), 1),
            ),
        ]
    )

    config_path = save_base.with_name(save_base.name + "_config.json")
    config_path.write_text(
        json.dumps(
            {
                "xml": args.xml,
                "timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "algo": args.algo,
                "stage_preset": args.stage_preset,
                "cfg": cfg_to_jsonable(cfg),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=args.progress_bar)
    model.save(str(save_base))
    vec_env.save(str(save_base.with_name(save_base.name + "_vecnormalize.pkl")))

    eval_env.close()
    vec_env.close()
    print(f"[OK] 模型已保存到: {save_base}.zip")


if __name__ == "__main__":
    main()
