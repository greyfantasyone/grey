from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fish_floating_grasp_env import FloatingGraspConfig, MujocoFishFloatingGraspEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放训练好的方案 A / 过渡课程版鱼尾抓取策略")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", choices=["recurrentppo", "ppo"], default=None)
    parser.add_argument("--assist-mode", choices=["off", "constant", "decay"], default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def load_saved_config(model_path: Path, config_arg: str | None) -> dict | None:
    config_path = None
    if config_arg:
        config_path = Path(config_arg)
    else:
        candidate = model_path.with_name(model_path.stem + "_config.json")
        if candidate.exists():
            config_path = candidate
    if config_path and config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return None


def build_cfg(saved_cfg: dict | None, assist_override: str | None) -> FloatingGraspConfig:
    cfg = FloatingGraspConfig()
    if saved_cfg:
        for key, value in saved_cfg.get("cfg", {}).items():
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                if isinstance(current, np.ndarray):
                    setattr(cfg, key, np.asarray(value, dtype=np.float64))
                else:
                    setattr(cfg, key, value)
    if assist_override is not None:
        cfg.assist_mode = assist_override
    return cfg


def set_default_camera(viewer: mujoco.viewer.Handle) -> None:
    cam = viewer.cam
    cam.azimuth = 130.0
    cam.elevation = -20.0
    cam.distance = 1.35
    cam.lookat[:] = np.array([-0.20, 0.00, 0.03], dtype=np.float64)


def sync_realtime(viewer: mujoco.viewer.Handle, dt: float, tic: float) -> float:
    toc = time.perf_counter()
    elapsed = toc - tic
    if elapsed < dt:
        time.sleep(dt - elapsed)
    viewer.sync()
    return time.perf_counter()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    saved = load_saved_config(model_path, args.config)

    xml_path = args.xml or (saved.get("xml") if saved else None) or "phy3.02_schemeA_grasp.xml"
    algo = args.algo or (saved.get("algo") if saved else None) or "recurrentppo"
    cfg = build_cfg(saved, args.assist_mode)

    raw_env = MujocoFishFloatingGraspEnv(xml_path=xml_path, config=cfg)
    raw_env.set_training_progress(1.0)
    vec_env = DummyVecEnv([lambda: raw_env])

    vecnorm_path = Path(args.vecnorm) if args.vecnorm else model_path.with_name(model_path.stem + "_vecnormalize.pkl")
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    if algo == "recurrentppo":
        model = RecurrentPPO.load(str(model_path), env=vec_env)
    else:
        model = PPO.load(str(model_path), env=vec_env)

    obs = vec_env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        set_default_camera(viewer)
        tic = time.perf_counter()
        finished_episodes = 0

        while viewer.is_running() and finished_episodes < args.episodes:
            if algo == "recurrentppo":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=args.deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, rewards, dones, infos = vec_env.step(action)
            tic = sync_realtime(viewer, raw_env.dt, tic)
            episode_starts = dones

            if dones[0]:
                info = infos[0]
                outcome = "SUCCESS" if info.get("is_success", False) else "FAIL/TRUNC"
                print(
                    f"episode={finished_episodes:03d} outcome={outcome} "
                    f"distance={info.get('distance', float('nan')):.4f} "
                    f"base_err={info.get('anchor_pos_err', float('nan')):.4f} "
                    f"base_rot={info.get('anchor_rot_err_deg', float('nan')):.2f}deg "
                    f"hold={info.get('hold_counter', -1)}/{info.get('hold_required', -1)} "
                    f"assist={info.get('assist_strength', float('nan')):.3f} reward={rewards[0]:.3f}"
                )
                finished_episodes += 1
                obs = vec_env.reset()
                if algo == "recurrentppo":
                    lstm_states = None
                    episode_starts = np.ones((1,), dtype=bool)

    vec_env.close()
    raw_env.close()


if __name__ == "__main__":
    main()
