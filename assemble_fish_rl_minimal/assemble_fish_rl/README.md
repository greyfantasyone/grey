# 机器鱼尾部抓取 RL 训练

自由漂浮机器鱼（free-floating base），4 个尾部关节（j1-j4），目标：控制尾部末端执行器（EE）到达并保持在目标点附近。

- 目标点在鱼体坐标系下标称位置：`[-0.46, 0.00, 0.00]`（沿体轴向后 0.46 m）
- 随机扰动范围（final）：±[4.5, 10, 8] cm
- 不是任意位置跟踪，只在尾部可达工作空间内
- 必须使用 `phy3.02_schemeA_grasp_ellipsoid.xml`（含隐藏椭球流体几何体，尾-base 耦合强 13x；默认 XML 耦合极弱，训练无意义）

---

## 环境配置

### 依赖

```
Python        3.11
torch         2.11.0
mujoco        3.6.0
gymnasium     1.2.3
stable-baselines3  2.8.0
sb3-contrib   2.8.0
numpy         2.4.4
tqdm          4.67.3
rich          15.0.0
```

### 安装

```bash
conda create -n fish-rl python=3.11
conda activate fish-rl
pip install torch==2.11.0
pip install mujoco==3.6.0 gymnasium==1.2.3
pip install stable-baselines3==2.8.0 sb3-contrib==2.8.0
pip install numpy tqdm rich
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `fish_floating_grasp_env.py` | Gymnasium 环境核心实现 |
| `train_recurrent_ppo_floating_grasp.py` | 训练脚本（RecurrentPPO / PPO / SAC） |
| `play_recurrent_ppo_floating_grasp.py` | 可视化回放 |
| `smoke_test.py` | 环境最小自检 |
| `inspect_tail_coupling.py` | 检查尾部关节对 base 漂移的影响 |
| `phy3.02_schemeA_grasp_ellipsoid.xml` | 当前使用的 MuJoCo 模型 |

---

## 当前最优 Checkpoint

```
checkpoints/stage2new_final_hold_conversion_cont19_ckpts/
    stage2new_final_hold_conversion_cont19_40000_steps.zip
checkpoints/stage2new_final_hold_conversion_cont19_vecnormalize.pkl
```

手动评估结果（64 episodes，`training_progress=1.0`，无辅助力）：

| 指标 | 值 |
|------|----|
| success_rate | **85.9%** |
| distance (mean) | 0.051 m |
| max_hold_fraction | 0.929 |
| drift_fail_rate | 3.1% |

成功标准：EE 距离 < 0.05 m 且 EE 速度 < 0.07 m/s，连续保持 20 步。

---

## 快速开始

### 回放

```bash
M=checkpoints/stage2new_final_hold_conversion_cont19_ckpts/stage2new_final_hold_conversion_cont19_40000_steps.zip
V=checkpoints/stage2new_final_hold_conversion_cont19_vecnormalize.pkl
conda run -n fish-rl python play_recurrent_ppo_floating_grasp.py \
  --model $M --xml phy3.02_schemeA_grasp_ellipsoid.xml --vecnorm $V \
  --episodes 20 --deterministic
```

### 环境自检

```bash
conda run -n fish-rl python smoke_test.py
```

---

## 训练方法

### 环境设计

**成功判据（EE-only）**：只要求 EE 到位并保持，base 漂移仅作为终止条件（漂移 > 0.35 m 或旋转 > 60° 时终止并给负奖励）。早期版本同时要求 base 稳定，导致几乎不可能成功。

**课程学习**：`training_progress` 从 0→1 线性控制成功判据从 easy 到 final：

| | easy | final |
|--|------|-------|
| success_distance | 0.10 m | 0.05 m |
| hold_steps | 6 | 20 |
| target_random_range | ±[1.5, 4, 4] cm | ±[4.5, 10, 8] cm |

**辅助力（assist）**：训练早期给 base 一个弱 PD 保持器，随 `training_progress` 衰减到 0。`assist_mode=decay`，`assist_max_strength=0.5`，`assist_decay_end=0.9`。同时开启 episode 级随机化（40% 概率 assist=0），防止 policy 依赖辅助力。

**hold_counter 机制**：
- EE 在成功区内（dist < success_distance 且 ee_speed < 0.07）：counter +1
- EE 在 near 区内（dist < near_radius=0.12 m）但不在成功区：counter -1（`near_hold_decay_steps=1`）
- EE 在 near 区外：counter -1（`drop_hold_penalty_steps=1`）
- counter 达到 20 即成功

`drop_hold_penalty_steps` 设为 1（而非更大值）是关键：EE 短暂振荡飘出时 counter 缓慢衰减，避免一次振荡就清零。

### 奖励设计（stage2_final_hold_conversion）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| dense_reward | 1.2 | exp(-7·dist)，鼓励靠近 |
| progress_reward | 3.5 | 每步距离改善量 |
| hold_reward | 18.0 | 在 near 区内，随 hold_counter 进度增大 |
| precision_reward | 7.5 | exp(-18·dist)·exp(-8·ee_speed) |
| success_bonus | 260.0 | 达成 20 步 hold 时一次性奖励 |
| success_region_drop_penalty | -6.0 | 进入成功区后飘出时惩罚 |
| near_target_speed_penalty | -0.25·ee_speed² | 在 near 区内晃动惩罚 |
| near_target_base_pos_penalty | -3.0·anchor_err² | 在 near 区内 base 漂移惩罚 |
| drift_termination_penalty | -25.0 | base 漂移过大终止时惩罚 |

### 两阶段训练流程

**Stage 1**（`--stage-preset stage1_reach_hold`）

目标：先学会 reach + 短暂 hold，不追求 final 条件。

```bash
conda run -n fish-rl python train_recurrent_ppo_floating_grasp.py \
  --xml phy3.02_schemeA_grasp_ellipsoid.xml \
  --algo recurrentppo \
  --stage-preset stage1_reach_hold \
  --assist-max-strength 0.50 \
  --timesteps 800000 \
  --n-envs 8 \
  --save checkpoints/stage1_reach_hold
```

**Stage 2**（`--stage-preset stage2_final_hold_conversion`）

目标：从 Stage 1 出发，学会 final hold（20 步，无辅助力）。

```bash
S1=checkpoints/stage1_reach_hold.zip
S1V=checkpoints/stage1_reach_hold_vecnormalize.pkl

conda run -n fish-rl python train_recurrent_ppo_floating_grasp.py \
  --xml phy3.02_schemeA_grasp_ellipsoid.xml \
  --algo recurrentppo \
  --stage-preset stage2_final_hold_conversion \
  --assist-max-strength 0.50 \
  --assist-decay-end 0.90 \
  --assist-random-zero-prob 0.40 \
  --assist-random-min-scale 0.15 \
  --assist-random-max-scale 1.00 \
  --learning-rate 5e-5 \
  --constant-lr \
  --resume-progress 1.0 \
  --timesteps 40000 \
  --n-envs 8 \
  --resume $S1 --vecnorm $S1V \
  --save checkpoints/stage2_contXX \
  --progress-bar
```

### 短段微调策略

Stage 2 采用"短段挑战式微调"：每次只训 40k 步，结束后做 64 episodes 手动评估，只有优于当前 best 才晋级为新 best。

**必须使用的选项**：

| 选项 | 原因 |
|------|------|
| `--constant-lr` | `linear_schedule` 在 40k 内会衰减到接近 0，导致 policy 退步 |
| `--resume-progress 1.0` | 避免每次 resume 把课程和辅助力重置到 easy 状态 |
| `--learning-rate 5e-5` | 比默认 3e-4 保守，防止过度更新 |

**手动评估脚本**：

```python
# 在项目目录下运行
import numpy as np, json
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from fish_floating_grasp_env import MujocoFishFloatingGraspEnv, FloatingGraspConfig

def evaluate(model_path, vecnorm_path, n_episodes=64):
    saved = json.loads(open(model_path.replace('.zip', '_config.json')).read())
    cfg = FloatingGraspConfig()
    for k, v in saved.get('cfg', {}).items():
        if hasattr(cfg, k):
            cur = getattr(cfg, k)
            setattr(cfg, k, np.asarray(v, dtype=np.float64) if isinstance(cur, np.ndarray) else v)
    env = MujocoFishFloatingGraspEnv(xml_path='phy3.02_schemeA_grasp_ellipsoid.xml', config=cfg)
    env.set_training_progress(1.0)
    vec_env = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
    vec_env.training = False; vec_env.norm_reward = False
    model = RecurrentPPO.load(model_path, env=vec_env)
    results, obs = [], vec_env.reset()
    lstm_states, ep_starts = None, np.ones((1,), dtype=bool)
    while len(results) < n_episodes:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        ep_starts = dones
        if dones[0]:
            results.append(infos[0]); lstm_states = None; ep_starts = np.ones((1,), dtype=bool)
    print(f"success={np.mean([r['is_success'] for r in results]):.3f} "
          f"dist={np.mean([r['distance'] for r in results]):.4f} "
          f"max_hold={np.mean([r['max_hold_fraction'] for r in results]):.3f} "
          f"drift={np.mean([r.get('drift_fail', False) for r in results]):.3f}")
```

---

## 已知问题与修复

训练脚本中修复了以下 bug，升级 SB3 版本时需注意：

1. **`--learning-rate` 在 resume 时未生效**：`RecurrentPPO.load()` 恢复 checkpoint 里的 lr 调度，需手动覆盖 `model.lr_schedule`（见 `train_recurrent_ppo_floating_grasp.py` resume 分支）
2. **`clip_range` 随 `linear_schedule` 衰减**：resume 时 `clip_range` 从 0.2 衰减到 ~0.016，critic 更新受限，`--constant-lr` 时同步固定为 0.2
3. **`eval_env` 固定在 `progress=0.5`**：EvalCallback 选出的 best model 对应中等难度而非 final 条件，已改为 `progress=1.0`
4. **`ProgressCallback` 每次 resume 从 0 重置**：辅助力和课程难度每段都重置到 easy，已加 `--resume-progress` 参数

---

## 待解决

- **Sim-to-real**：流体参数需系统辨识（ETH 方法：2 条真实轨迹标定），当前仅在仿真中验证
- **base 漂移**：3.1% 的 episode 因 base 漂移过大终止，实际部署时需评估真实流体下的漂移幅度
