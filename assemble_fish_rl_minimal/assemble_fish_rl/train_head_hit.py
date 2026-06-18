"""鱼头拍球任务的简化训练脚本。

设计原则（贴近原 train_recurrent_ppo_floating_grasp.py 但简化）：
    - SubprocVecEnv 并行（默认 8 envs，留余地给系统）
    - VecNormalize 归一化 obs（关键：48 维原始 obs 量纲差异大）
    - RecurrentPPO + MlpLstmPolicy（沿用原项目算法栈）
    - CPU 训练（小网络 + mujoco 仿真瓶颈，GPU 加速有限）
    - 自定义 SuccessRateCallback：每 N 步在独立 eval env 跑 16 ep，记录 success_rate
      （reward 主导项是接触/对准，看 reward 看不出 RL 真的学到 success）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from fish_head_hit_env import HeadHitConfig, MujocoFishHeadHitEnv


def make_env_fn(xml_path: str, progress: float, seed: int):
    def _fn():
        env = MujocoFishHeadHitEnv(xml_path=xml_path)
        env.set_training_progress(progress)
        env.reset(seed=seed)
        return env
    return _fn


class SuccessRateCallback(BaseCallback):
    """每 eval_freq 步在独立 vec env 上跑 n_eval_episodes，统计 success_rate / mean_R / mean_max_vx。"""

    def __init__(
        self,
        eval_env: VecNormalize,
        n_eval_episodes: int = 16,
        eval_freq: int = 10000,
        log_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.history: list[dict] = []
        self.best_success_rate = -1.0
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        results = []
        obs = self.eval_env.reset()
        lstm_states = None
        ep_starts = np.ones((1,), dtype=bool)
        while len(results) < self.n_eval_episodes:
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True
            )
            obs, _, dones, infos = self.eval_env.step(action)
            ep_starts = dones
            if dones[0]:
                results.append(infos[0])
                lstm_states = None
                ep_starts = np.ones((1,), dtype=bool)
        success_rate = float(np.mean([r["is_success"] for r in results]))
        mean_max_vx = float(np.mean([r["max_ball_vx_after_hit"] for r in results]))
        head_hit_rate = float(np.mean([r["head_hit_occurred"] for r in results]))
        body_hit_mean = float(np.mean([r["body_hit_count"] for r in results]))
        ball_x_end = float(np.mean([r["ball_x"] for r in results]))
        entry = {
            "timesteps": int(self.num_timesteps),
            "success_rate": success_rate,
            "mean_max_vx_after": mean_max_vx,
            "head_hit_rate": head_hit_rate,
            "body_hit_mean": body_hit_mean,
            "ball_x_end_mean": ball_x_end,
        }
        self.history.append(entry)
        if self.verbose > 0:
            print(
                f"[eval @ {self.num_timesteps:>7d}] "
                f"success={success_rate:.3f}  vx_after_mean={mean_max_vx:.4f}  "
                f"head_hit={head_hit_rate:.2f}  body_hit_avg={body_hit_mean:.1f}  "
                f"ball_x_end={ball_x_end:.3f}"
            )
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
        if self.log_path:
            Path(self.log_path).write_text(json.dumps(self.history, indent=2))
        # log to sb3 logger so it shows in progress bar
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_max_vx_after", mean_max_vx)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="phy3.02_head_hit.xml")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--save", type=str, default="checkpoints/head_hit_v1")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="every N agent steps (per env), run eval")
    parser.add_argument("--n-eval-episodes", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={args.device}  n_envs={args.n_envs}  timesteps={args.timesteps}  "
          f"progress={args.progress}")
    print(f"[setup] torch.cuda={torch.cuda.is_available()}  cpu_count={torch.get_num_threads()}")

    # 训练环境（多进程）
    env_fns = [make_env_fn(args.xml, args.progress, args.seed + i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 独立 eval 环境（单进程，deterministic 评估）
    eval_env_raw = DummyVecEnv([make_env_fn(args.xml, args.progress, args.seed + 9999)])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False
    # 在每个 eval 之前会被 sync 到训练 env 的统计量

    class SyncStatsCallback(BaseCallback):
        """每 eval 前把 train env 的 obs_rms 同步到 eval env。"""
        def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
            super().__init__()
            self.train_env = train_env
            self.eval_env = eval_env
        def _on_step(self) -> bool:
            return True

    success_cb = SuccessRateCallback(
        eval_env=eval_env,
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        log_path=str(save_path.parent / f"{save_path.name}_eval_log.json"),
        verbose=1,
    )

    # 把 eval env 的 obs_rms 在每次 eval 前从 train env 同步
    class SyncRMSCallback(BaseCallback):
        def __init__(self, train_env, eval_env, eval_freq):
            super().__init__()
            self.train_env = train_env
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self._last = 0
        def _on_step(self) -> bool:
            if self.num_timesteps - self._last >= self.eval_freq:
                self._last = self.num_timesteps
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms
            return True

    sync_cb = SyncRMSCallback(vec_env, eval_env, args.eval_freq)

    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=args.device,
        verbose=0,
        seed=args.seed,
    )

    t0 = time.time()
    print(f"[train] start | total_params={sum(p.numel() for p in model.policy.parameters()):,}")
    model.learn(total_timesteps=args.timesteps, callback=[sync_cb, success_cb], progress_bar=False)
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # 保存
    model.save(str(save_path) + ".zip")
    vec_env.save(str(save_path) + "_vecnormalize.pkl")
    # 把 cfg + eval history 也存下来便于追溯
    cfg_dump = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in vars(HeadHitConfig()).items()
    }
    Path(str(save_path) + "_config.json").write_text(json.dumps({
        "args": vars(args),
        "cfg": cfg_dump,
        "eval_history": success_cb.history,
        "best_success_rate": success_cb.best_success_rate,
        "elapsed_sec": elapsed,
    }, indent=2, default=str))
    print(f"[save] model={save_path}.zip  vecnorm={save_path}_vecnormalize.pkl")
    print(f"[final] best_success_rate={success_cb.best_success_rate:.3f}")


if __name__ == "__main__":
    main()
