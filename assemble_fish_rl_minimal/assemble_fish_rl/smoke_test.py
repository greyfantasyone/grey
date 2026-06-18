from __future__ import annotations

import argparse
import math

import numpy as np

from fish_floating_grasp_env import FloatingGraspConfig, MujocoFishFloatingGraspEnv


def run_case(xml: str, assist_mode: str, progress: float, seed: int) -> None:
    cfg = FloatingGraspConfig(assist_mode=assist_mode)
    env = MujocoFishFloatingGraspEnv(xml_path=xml, config=cfg)
    env.set_training_progress(progress)
    obs, info = env.reset(seed=seed)
    print(
        f"[reset] assist={assist_mode} progress={progress:.2f} obs_shape={obs.shape} "
        f"dist={info['distance']:.4f} base_err={info['anchor_pos_err']:.4f}"
    )

    total_reward = 0.0
    for step in range(40):
        if step % 2 == 0:
            action = np.array([
                0.5 * math.sin(0.12 * step),
                0.8 * math.sin(0.07 * step + 0.3),
                -0.9 * math.sin(0.11 * step + 0.5),
                0.6 * math.sin(0.09 * step + 1.2),
            ], dtype=np.float32)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(
                f"[done] step={step:03d} reward_sum={total_reward:.3f} "
                f"success={info['is_success']} drift_fail={info.get('drift_fail', False)} "
                f"dist={info['distance']:.4f} base_err={info['anchor_pos_err']:.4f}"
            )
            break
    else:
        print(
            f"[ok] 40 steps finished reward_sum={total_reward:.3f} "
            f"dist={info['distance']:.4f} base_err={info['anchor_pos_err']:.4f}"
        )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对方案 A 环境做最小自检")
    parser.add_argument("--xml", type=str, default="phy3.02_schemeA_grasp.xml")
    args = parser.parse_args()
    run_case(args.xml, assist_mode="off", progress=1.0, seed=0)
    run_case(args.xml, assist_mode="decay", progress=0.1, seed=1)
    print("[OK] smoke test finished")
