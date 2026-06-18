"""启发式控制器：在调参后的 env 上比较 zero / random / burst 三种策略。

burst 策略：球-头距离很近时，所有尾部关节做大幅快速抖动（"瞬时抽搐"），
试图在接触瞬间增加 base 朝 +x 的反推力。

意义：在投入 PPO 训练前，先确认 (a) 至少有一种简单策略能比 zero 得更高分
说明 reward 信号有梯度；(b) 是否能偶尔触发成功 / 接近成功。
"""
from __future__ import annotations

import argparse
import numpy as np
from fish_head_hit_env import MujocoFishHeadHitEnv, HeadHitConfig


def policy_zero(t: int, info: dict) -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def policy_random(t: int, info: dict, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)


def policy_burst(t: int, info: dict) -> np.ndarray:
    """球-头距离 < 8cm 时启动尾部抖动。"""
    dist = info.get("ball_head_dist", 1.0)
    head_hit = info.get("head_hit_occurred", False)
    if dist < 0.08 or head_hit:
        # 高频抖动：4 个 joint 都用最大幅度 + 高频
        phase = t * 1.2  # 快速反转
        return np.array([
            0.9 * np.sin(phase),
            0.9 * np.sin(phase + 1.0),
            -1.0 * np.sin(phase + 2.0),
            1.0 * np.sin(phase + 3.0),
        ], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


def run(policy_name: str, n_episodes: int, progress: float, seed: int) -> dict:
    env = MujocoFishHeadHitEnv()
    env.set_training_progress(progress)
    rng = np.random.default_rng(seed)
    rewards, head_hits, body_hits, successes, max_vx_list, ball_x_end = [], [], [], [], [], []
    end_reasons = {}
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_r = 0.0
        for t in range(env.cfg.max_steps):
            if policy_name == "zero":
                a = policy_zero(t, info)
            elif policy_name == "random":
                a = policy_random(t, info, rng)
            elif policy_name == "burst":
                a = policy_burst(t, info)
            else:
                raise ValueError(policy_name)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            if term or trunc:
                break
        rewards.append(ep_r)
        head_hits.append(int(info["head_hit_occurred"]))
        body_hits.append(int(info["body_hit_count"]))
        successes.append(int(info["is_success"]))
        max_vx_list.append(float(info["max_ball_vx_after_hit"]))
        ball_x_end.append(float(info["ball_x"]))
        reason = info["terminated_reason"] or ("truncated" if trunc else "term")
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
    env.close()
    return {
        "policy": policy_name,
        "n": n_episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "head_hit_rate": float(np.mean(head_hits)),
        "body_hit_count_mean": float(np.mean(body_hits)),
        "success_rate": float(np.mean(successes)),
        "max_vx_after_mean": float(np.mean(max_vx_list)),
        "max_vx_after_best": float(np.max(max_vx_list)),
        "ball_x_end_mean": float(np.mean(ball_x_end)),
        "end_reasons": end_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"\n=== Heuristic eval | episodes={args.episodes} progress={args.progress} ===\n")
    for name in ["zero", "random", "burst"]:
        r = run(name, args.episodes, args.progress, args.seed)
        print(f"[{r['policy']:<7s}] R={r['reward_mean']:>7.1f}±{r['reward_std']:>5.1f}  "
              f"head_hit={r['head_hit_rate']:.2f}  body_hit_avg={r['body_hit_count_mean']:.1f}  "
              f"success={r['success_rate']:.2f}  "
              f"vx_after_mean={r['max_vx_after_mean']:.3f}  best={r['max_vx_after_best']:.3f}  "
              f"ball_x_end={r['ball_x_end_mean']:.3f}")
        print(f"          end_reasons={r['end_reasons']}")
    print()


if __name__ == "__main__":
    main()
