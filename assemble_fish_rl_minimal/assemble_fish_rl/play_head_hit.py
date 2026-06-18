"""加载 head_hit checkpoint，跑 N episodes，渲染视频 + 记录关节/球状态时间序列。

输出：
    - videos/<tag>_ep{i}.mp4 — 每个 episode 的 side-view 视频
    - logs/<tag>_traces.npz — 每个 episode 的 (action, ball_pos, ball_vel, joint_q, head_pos) 时间序列

意义：可视化让我们能定性看到 RL 学到的 timing；时间序列让我们能定量分析
        击球瞬间前后的关节动作模式（论文图所需）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fish_head_hit_env import MujocoFishHeadHitEnv


def make_env(xml: str, progress: float, render: bool):
    env = MujocoFishHeadHitEnv(xml_path=xml, render_mode="rgb_array" if render else None,
                                rgb_width=640, rgb_height=480)
    env.set_training_progress(progress)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vecnorm", type=str, required=True)
    parser.add_argument("--xml", type=str, default="phy3.02_head_hit.xml")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--tag", type=str, default="ppo_100k")
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--out-dir", type=str, default="videos")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # load model with a render-disabled vec env (vec env doesn't render; we
    # render via env._env directly after each step)
    raw_env = make_env(args.xml, args.progress, render=True)
    vec_raw = DummyVecEnv([lambda: raw_env])
    vec = VecNormalize.load(args.vecnorm, vec_raw)
    vec.training = False
    vec.norm_reward = False
    model = RecurrentPPO.load(args.model, env=vec)

    # access raw env to call render()
    inner_env: MujocoFishHeadHitEnv = vec.venv.envs[0]

    results_summary = []
    all_traces = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        obs = vec.reset()
        # vec.reset() doesn't accept seed; we rely on env's np_random being seeded
        # via reset before. To be reproducible, seed the inner env manually:
        np.random.seed(seed)
        obs_raw, info = inner_env.reset(seed=seed)
        # need to re-feed normalised obs (vec just reset above; bypass it):
        obs = vec.normalize_obs(obs_raw)[None, :]  # shape (1, obs_dim)

        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)
        frames = []
        trace = {
            "action": [],
            "ball_pos": [],
            "ball_vel": [],
            "joint_q": [],
            "head_pos": [],
            "ball_head_dist": [],
            "reward": [],
            "contact_kind": [],
        }
        last_info = info
        for t in range(inner_env.cfg.max_steps):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=args.deterministic
            )
            # step raw env directly so we get unnormalized info + render
            obs_raw, r, term, trunc, info = inner_env.step(action[0])
            obs = vec.normalize_obs(obs_raw)[None, :]
            ep_starts = np.array([term or trunc], dtype=bool)
            # render frame
            frame = inner_env.render()
            if frame is not None:
                frames.append(frame)
            # log trace
            trace["action"].append(action[0].copy())
            trace["ball_pos"].append([info["ball_x"], info["ball_y"], info["ball_z"]])
            trace["ball_vel"].append([info["ball_vx"], info["ball_vy"], info["ball_vz"]])
            trace["joint_q"].append(inner_env.data.qpos[inner_env.joint_qpos_indices].copy())
            trace["head_pos"].append(inner_env._get_head_pos_world())
            trace["ball_head_dist"].append(info["ball_head_dist"])
            trace["reward"].append(r)
            trace["contact_kind"].append(info["last_contact_kind"])
            last_info = info
            if term or trunc:
                break

        # save video
        video_path = out_dir / f"{args.tag}_ep{ep}_seed{seed}.mp4"
        if frames:
            imageio.mimsave(str(video_path), frames, fps=20)

        # save trace
        trace_np = {k: np.asarray(v) for k, v in trace.items() if k != "contact_kind"}
        trace_np["contact_kind"] = np.array(trace["contact_kind"], dtype="U8")

        is_success = last_info["is_success"]
        max_vx = last_info["max_ball_vx_after_hit"]
        n_steps = len(trace["action"])
        end_reason = last_info["terminated_reason"] or ("truncated" if trunc else "term")
        print(f"[ep {ep}] seed={seed} steps={n_steps} success={is_success} "
              f"max_vx_after={max_vx:.4f} reason={end_reason} "
              f"video={video_path.name}")
        results_summary.append({
            "ep": ep,
            "seed": seed,
            "steps": n_steps,
            "success": is_success,
            "max_vx_after": max_vx,
            "end_reason": end_reason,
        })
        all_traces.append(trace_np)

    # save aggregated traces
    traces_path = log_dir / f"{args.tag}_traces.npz"
    flat = {}
    for i, tr in enumerate(all_traces):
        for k, v in tr.items():
            flat[f"ep{i}_{k}"] = v
    np.savez_compressed(str(traces_path), **flat)
    print(f"\n[summary] {sum(r['success'] for r in results_summary)}/{len(results_summary)} success | "
          f"max_vx: {max(r['max_vx_after'] for r in results_summary):.4f} best | "
          f"traces saved to {traces_path}")


if __name__ == "__main__":
    main()
