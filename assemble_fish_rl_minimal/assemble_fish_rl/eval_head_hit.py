"""加载训练好的 head_hit checkpoint，跑 N 个 episodes 评估真实 success rate。

意义：训练时 eval_freq 用的是 32 ep，统计误差较大（25% ±8%）。这里用 128 ep
缩小误差到 ~4%，并按 deterministic / stochastic 两种模式分别评估。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fish_head_hit_env import MujocoFishHeadHitEnv


def make_env(xml: str, progress: float):
    env = MujocoFishHeadHitEnv(xml_path=xml)
    env.set_training_progress(progress)
    return env


def evaluate(model_path: str, vecnorm_path: str, xml: str, progress: float,
             n_episodes: int, deterministic: bool, seed: int) -> dict:
    raw = DummyVecEnv([lambda: make_env(xml, progress)])
    vec = VecNormalize.load(vecnorm_path, raw)
    vec.training = False
    vec.norm_reward = False
    model = RecurrentPPO.load(model_path, env=vec)
    results = []
    obs = vec.reset()
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    while len(results) < n_episodes:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_starts, deterministic=deterministic
        )
        obs, _, dones, infos = vec.step(action)
        ep_starts = dones
        if dones[0]:
            results.append(infos[0])
            lstm_states = None
            ep_starts = np.ones((1,), dtype=bool)

    success = np.array([r["is_success"] for r in results], dtype=np.float64)
    vx_after = np.array([r["max_ball_vx_after_hit"] for r in results], dtype=np.float64)
    head_hit = np.array([r["head_hit_occurred"] for r in results], dtype=np.float64)
    body_hit = np.array([r["body_hit_count"] for r in results], dtype=np.float64)
    ball_x = np.array([r["ball_x"] for r in results], dtype=np.float64)

    n = len(results)
    se_success = float(np.std(success) / np.sqrt(n))
    return {
        "n": n,
        "deterministic": deterministic,
        "success_rate": float(np.mean(success)),
        "success_rate_se": se_success,
        "vx_after_mean": float(np.mean(vx_after)),
        "vx_after_std": float(np.std(vx_after)),
        "vx_after_p90": float(np.percentile(vx_after, 90)),
        "head_hit_rate": float(np.mean(head_hit)),
        "body_hit_mean": float(np.mean(body_hit)),
        "ball_x_end_mean": float(np.mean(ball_x)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, required=True)
    parser.add_argument("--xml", type=str, default="phy3.02_head_hit.xml")
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print(f"[eval] model={Path(args.model).name}  episodes={args.episodes}  progress={args.progress}")
    print(f"[ref] random baseline (50 ep): success_rate=0.08  vx_after≈0.015\n")

    for det in (True, False):
        r = evaluate(args.model, args.vecnorm, args.xml, args.progress,
                     args.episodes, deterministic=det, seed=args.seed)
        mode = "DET" if det else "STO"
        print(
            f"[{mode}] n={r['n']:>3d}  "
            f"success={r['success_rate']:.3f}±{r['success_rate_se']:.3f}  "
            f"vx_after_mean={r['vx_after_mean']:.4f}  p90={r['vx_after_p90']:.4f}  "
            f"head_hit={r['head_hit_rate']:.2f}  body_hit_avg={r['body_hit_mean']:.1f}  "
            f"ball_x_end={r['ball_x_end_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
