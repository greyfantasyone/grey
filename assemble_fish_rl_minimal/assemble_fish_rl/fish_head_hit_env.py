"""鱼头拍球环境（Phase 1 子任务 1）。

任务：一个低速漂来的球在鱼头前方出现，机器鱼控制 4 节尾部关节，主动用鱼头把球
"打回去"（球 x 方向速度翻号且足够大）。

与 fish_floating_grasp_env.py 的关系：
    - 鱼模型、流体、动作空间、课程框架、obs 归一化思路全部沿用
    - 不复用 MujocoFishFloatingGraspEnv（任务语义差太多），但复用其中的 helper
      函数（四元数、smoothstep 等）
    - 使用新 XML phy3.02_head_hit.xml（target_body 替换为带 freejoint 的实心球，
      head_proxy 移到鱼头 mesh 表面）

设计哲学：
    - 球的"反弹能量"来自鱼主动撞击，不靠被动 restitution
    - 球初始场景贴近鱼头（8~12 cm）+ 低速（0.3~0.5 m/s），匹配水下物理：
      水阻让球自由飞行只能走几 cm
    - 只有 head_proxy 附近接触才算"有效击球"；其他部位接触给小惩罚
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from fish_floating_grasp_env import (
    JOINT_NAMES,
    JOINT_ACTUATOR_NAMES,
    MAGNET_ACTUATOR_NAMES,
    lerp,
    quat_from_euler_xyz_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_to_rotmat_wxyz,
    rotate_vec_wxyz,
    safe_exp_neg,
    smoothstep01,
)


@dataclass
class HeadHitConfig:
    # --- 仿真与动作（保持与原 env 一致） ---
    frame_skip: int = 10
    max_steps: int = 120              # 球漂到鱼头约 25-50 步，给 120 步足够
    settle_steps: int = 4
    action_delta_scale: float = 0.10
    action_smoothing: float = 0.70

    # --- 鱼 base 初始姿态扰动（较小，鱼大致在原点朝 +x） ---
    base_init_pos_noise: np.ndarray = field(
        default_factory=lambda: np.array([0.010, 0.010, 0.008], dtype=np.float64)
    )
    base_init_rpy_noise: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([3.0, 3.0, 5.0], dtype=np.float64))
    )
    init_joint_noise: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([4.0, 6.0, 8.0, 8.0], dtype=np.float64))
    )
    init_joint_vel_std: float = 0.03

    # --- 球初始场景 ---
    # 球距 head_proxy 多远出发（base 坐标系下 +x 方向）
    # easy: 球贴鱼头出发，立刻就接触，强化 contact reward 信号
    # final: 球离鱼头 10 cm，需要 RL 等几十步看球漂过来再 burst
    ball_init_x_offset_easy: float = 0.08
    ball_init_x_offset_final: float = 0.12
    ball_init_yz_noise_easy: float = 0.010
    ball_init_yz_noise_final: float = 0.030
    # 球初速度（沿 -x，朝鱼飞来）
    ball_init_vx_mean_easy: float = -0.30
    ball_init_vx_mean_final: float = -0.50
    ball_init_vx_noise: float = 0.05
    ball_init_vy_noise_final: float = 0.04
    ball_init_vz_noise_final: float = 0.02

    # --- 成功 / 接触判定 ---
    head_contact_radius: float = 0.05    # 接触点距 head_proxy 这么近才算 "head hit"
    hit_success_ball_vx: float = 0.020   # 球被打回后 vx > 此值即成功（物理可达：被动基线 ≈0.013-0.016，要求高 25-50%）
    head_alignment_gate_dist: float = 0.05  # 球-头距离 > 此值才给 alignment 奖励（避免接触后白送分）

    # --- 失败判定（球出界/鱼漂移） ---
    fail_ball_x_min: float = -0.15
    fail_ball_yz_abs: float = 0.30
    fail_base_pos: float = 0.30
    fail_base_rot_deg: float = 60.0

    # --- 奖励权重（按"贴近物理现实"范式重新设计） ---
    # 不奖励"鱼游过去"，重奖"接触发生 + 接触后球速反向且大"
    approach_reward_weight: float = 0.3        # 降低：避免廉价奖励 dominate
    approach_reward_scale: float = 6.0
    approach_progress_weight: float = 1.5      # 降低：鱼基本动不了，不强求每步缩距
    head_alignment_weight: float = 1.0         # 提高：姿态对准是核心可控变量
    contact_head_bonus: float = 15.0           # 提高：首次头接触是关键事件
    contact_body_penalty: float = -3.0
    ball_vx_reward_weight: float = 25.0        # 大幅提高:接触后球 vx 是核心成功信号
    success_bonus: float = 200.0
    fail_penalty: float = -30.0
    action_penalty_weight: float = 0.01
    action_rate_penalty_weight: float = 0.20
    joint_vel_penalty_weight: float = 0.001
    base_drift_penalty_weight: float = 0.05

    # --- 观测历史堆叠（无 LSTM 算法时可加） ---
    obs_stack_size: int = 1


class MujocoFishHeadHitEnv(gym.Env[np.ndarray, np.ndarray]):
    """鱼头拍球任务环境。

    动作空间：4 维连续 [-1,1]，对应 4 个尾部关节的 position target delta（与原 env 一致）。
    观测空间：48 维（鱼状态 + 球状态 + 击球进度），可用 obs_stack_size 多帧堆叠。

    成功：球被鱼头打回，且 ball.x > success_ball_x 时一次性奖励并终止。
    失败：球穿过鱼到 -x 方向或偏离过远，或鱼漂移失控。
    """

    metadata = {"render_modes": [None, "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "phy3.02_head_hit.xml",
        config: HeadHitConfig | None = None,
        render_mode: str | None = None,
        rgb_width: int = 640,
        rgb_height: int = 480,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML 文件不存在: {self.xml_path}")
        self.cfg = config or HeadHitConfig()
        self.render_mode = render_mode
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer: mujoco.Renderer | None = None

        # body / site / joint / actuator ids
        self.base_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.ball_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "ball_body")
        self.ball_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ball_joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        self.head_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "head_proxy")
        self.ball_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "ball_site")

        self.joint_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        ]
        self.joint_actuator_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_ACTUATOR_NAMES
        ]
        self.magnet_actuator_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MAGNET_ACTUATOR_NAMES
        ]
        self.joint_qpos_indices = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_dof_indices = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_ctrl_min = self.model.actuator_ctrlrange[self.joint_actuator_ids, 0].astype(np.float64)
        self.joint_ctrl_max = self.model.actuator_ctrlrange[self.joint_actuator_ids, 1].astype(np.float64)
        # ball freejoint qpos/qvel addresses
        self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_adr = int(self.model.jnt_dofadr[self.ball_joint_id])

        # state caches
        self.training_progress = 0.0
        self.ctrl_joint_targets = np.zeros(4, dtype=np.float64)
        self.prev_action = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.prev_ball_head_dist = 0.0
        self.head_hit_occurred = False         # 鱼头是否已击到球
        self.body_hit_count = 0                # 球碰到非头部位次数（累计）
        self.max_ball_vx_after_hit = 0.0       # 击球后球 vx 最大值
        self.last_contact_kind = "none"        # 'head' | 'body' | 'none'

        # observation space
        obs_dim = 48
        self._obs_stack_size = max(1, self.cfg.obs_stack_size)
        self._obs_buf = np.zeros((self._obs_stack_size, obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        full_obs_dim = obs_dim * self._obs_stack_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(full_obs_dim,), dtype=np.float32
        )
        self.np_random = np.random.default_rng()

    # ------------------ helpers ------------------

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"XML 中找不到对象: {name}")
        return obj_id

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.cfg.frame_skip)

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def _curriculum_alpha(self) -> float:
        return smoothstep01(self.training_progress)

    def _push_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_stack_size > 1:
            self._obs_buf[:-1] = self._obs_buf[1:]
            self._obs_buf[-1] = obs
            return self._obs_buf.flatten()
        return obs

    def _get_base_rot_world(self) -> np.ndarray:
        return self.data.xmat[self.base_body_id].reshape(3, 3).copy()

    def _get_base_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, vel, 0
        )
        return vel[3:].copy(), vel[:3].copy()  # linvel, angvel

    def _get_ball_pos_world(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    def _get_ball_vel_world(self) -> np.ndarray:
        return self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3].copy()

    def _get_head_pos_world(self) -> np.ndarray:
        return self.data.site_xpos[self.head_site_id].copy()

    # ------------------ ball init ------------------

    def _sample_ball_init(self) -> tuple[np.ndarray, np.ndarray]:
        a = self._curriculum_alpha()
        x_off = float(lerp(self.cfg.ball_init_x_offset_easy, self.cfg.ball_init_x_offset_final, a))
        yz_noise = float(lerp(self.cfg.ball_init_yz_noise_easy, self.cfg.ball_init_yz_noise_final, a))
        vx_mean = float(lerp(self.cfg.ball_init_vx_mean_easy, self.cfg.ball_init_vx_mean_final, a))
        vy_noise = a * self.cfg.ball_init_vy_noise_final
        vz_noise = a * self.cfg.ball_init_vz_noise_final

        # 在鱼头世界坐标系下采样，再加到 head_proxy world position 上
        head_world = self._get_head_pos_world()
        head_x = head_world[0]
        ball_x = head_x + x_off  # 鱼头前方 x_off
        ball_y = head_world[1] + float(self.np_random.uniform(-yz_noise, yz_noise))
        ball_z = head_world[2] + float(self.np_random.uniform(-yz_noise, yz_noise))
        pos = np.array([ball_x, ball_y, ball_z], dtype=np.float64)

        vx = vx_mean + float(self.np_random.uniform(-self.cfg.ball_init_vx_noise, self.cfg.ball_init_vx_noise))
        vy = float(self.np_random.uniform(-vy_noise, vy_noise))
        vz = float(self.np_random.uniform(-vz_noise, vz_noise))
        vel = np.array([vx, vy, vz], dtype=np.float64)
        return pos, vel

    def _set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 3] = pos
        self.data.qpos[self.ball_qpos_adr + 3:self.ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3] = vel
        self.data.qvel[self.ball_dof_adr + 3:self.ball_dof_adr + 6] = 0.0

    # ------------------ contact analysis ------------------

    def _analyze_ball_contact(self) -> str:
        """返回 'head' / 'body' / 'none'。

        head 判定：接触点距 head_proxy 世界位置 < head_contact_radius。
        body 判定：球与 base_link mesh 有接触但接触点离 head_proxy 较远。
        """
        if self.data.ncon == 0:
            return "none"
        head_pos = self._get_head_pos_world()
        kind = "none"
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 != self.ball_geom_id and c.geom2 != self.ball_geom_id:
                continue
            # 该接触涉及球
            d = float(np.linalg.norm(c.pos - head_pos))
            if d < self.cfg.head_contact_radius:
                return "head"  # 一旦确认头接触，直接返回（优先级最高）
            kind = "body"
        return kind

    # ------------------ obs ------------------

    def _get_obs(self) -> np.ndarray:
        base_rot = self._get_base_rot_world()
        world_to_base = base_rot.T
        base_pos = self.data.xpos[self.base_body_id]
        linvel_world, angvel_world = self._get_base_velocity_world()

        head_world = self._get_head_pos_world()
        ball_world = self._get_ball_pos_world()
        ball_vel_world = self._get_ball_vel_world()

        joint_q = self.data.qpos[self.joint_qpos_indices] / 1.57
        joint_qd = np.tanh(0.25 * self.data.qvel[self.joint_dof_indices])

        ball_from_head_base = world_to_base @ (ball_world - head_world)
        ball_vel_base = world_to_base @ ball_vel_world
        ball_from_base_base = world_to_base @ (ball_world - base_pos)
        head_from_base_base = world_to_base @ (head_world - base_pos)
        linvel_base = world_to_base @ linvel_world
        angvel_base = world_to_base @ angvel_world

        # 标志位 / 进度
        head_hit_flag = np.array([1.0 if self.head_hit_occurred else 0.0], dtype=np.float64)
        body_hit_frac = np.array([min(1.0, self.body_hit_count / 5.0)], dtype=np.float64)
        progress_frac = np.array([self.training_progress], dtype=np.float64)
        # base 在世界 x 方向位移（鱼是否前冲）—— 给一个绝对线索
        base_x_world = np.array([float(base_pos[0])], dtype=np.float64)

        obs = np.concatenate([
            joint_q,              # 4
            joint_qd,             # 4
            ball_from_head_base,  # 3
            ball_vel_base,        # 3
            ball_from_base_base,  # 3
            head_from_base_base,  # 3
            linvel_base,          # 3
            angvel_base,          # 3
            self.prev_action,     # 4
            head_hit_flag,        # 1
            body_hit_frac,        # 1
            progress_frac,        # 1
            base_x_world,         # 1
        ])  # 共 4+4+3+3+3+3+3+3+4+1+1+1+1 = 34 + 14 = 必须凑到 48
        # 上面是 34 维。补齐到 48 用一些额外特征：
        ball_speed = np.array([float(np.linalg.norm(ball_vel_world))], dtype=np.float64)
        ball_head_dist = np.array([float(np.linalg.norm(ball_world - head_world))], dtype=np.float64)
        base_speed = np.array([float(np.linalg.norm(linvel_world))], dtype=np.float64)
        base_angspeed = np.array([float(np.linalg.norm(angvel_world))], dtype=np.float64)
        last_contact_oh = np.zeros(3, dtype=np.float64)
        last_contact_oh[{"none": 0, "head": 1, "body": 2}[self.last_contact_kind]] = 1.0
        max_vx_after = np.array([self.max_ball_vx_after_hit], dtype=np.float64)
        ball_x_world = np.array([float(ball_world[0])], dtype=np.float64)
        step_frac = np.array([self.step_count / max(1, self.cfg.max_steps)], dtype=np.float64)
        obs_extra = np.concatenate([
            ball_speed, ball_head_dist, base_speed, base_angspeed,
            last_contact_oh, max_vx_after, ball_x_world, step_frac,
        ])  # 1+1+1+1+3+1+1+1 = 10
        obs_full = np.concatenate([obs, obs_extra])  # 34 + 10 = 44 — 还差 4
        # 再补 4 个：球到鱼头单位方向向量（base 坐标）+ ball z (world)
        dist = max(1e-6, float(np.linalg.norm(ball_from_head_base)))
        head_to_ball_dir = ball_from_head_base / dist  # 3
        ball_z_world = np.array([float(ball_world[2])], dtype=np.float64)
        obs_full = np.concatenate([obs_full, head_to_ball_dir, ball_z_world])  # +4 = 48
        assert obs_full.shape[0] == 48, f"obs dim mismatch: {obs_full.shape[0]}"
        return obs_full.astype(np.float32)

    # ------------------ info ------------------

    def _build_info(self, *, is_success: bool, terminated_reason: str = "") -> dict[str, Any]:
        ball_world = self._get_ball_pos_world()
        ball_vel = self._get_ball_vel_world()
        head_world = self._get_head_pos_world()
        return {
            "ball_x": float(ball_world[0]),
            "ball_y": float(ball_world[1]),
            "ball_z": float(ball_world[2]),
            "ball_vx": float(ball_vel[0]),
            "ball_vy": float(ball_vel[1]),
            "ball_vz": float(ball_vel[2]),
            "ball_speed": float(np.linalg.norm(ball_vel)),
            "ball_head_dist": float(np.linalg.norm(ball_world - head_world)),
            "head_hit_occurred": bool(self.head_hit_occurred),
            "body_hit_count": int(self.body_hit_count),
            "max_ball_vx_after_hit": float(self.max_ball_vx_after_hit),
            "last_contact_kind": self.last_contact_kind,
            "is_success": bool(is_success),
            "terminated_reason": terminated_reason,
            "training_progress": self.training_progress,
        }

    # ------------------ reset / step ------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action[:] = 0.0
        self.head_hit_occurred = False
        self.body_hit_count = 0
        self.max_ball_vx_after_hit = 0.0
        self.last_contact_kind = "none"

        # base 初始姿态（在原点附近小扰动，朝 +x）
        init_pos = self.np_random.uniform(
            -self.cfg.base_init_pos_noise, self.cfg.base_init_pos_noise
        )
        init_rpy = self.np_random.uniform(
            -self.cfg.base_init_rpy_noise, self.cfg.base_init_rpy_noise
        )
        init_quat = quat_normalize_wxyz(
            quat_from_euler_xyz_wxyz(float(init_rpy[0]), float(init_rpy[1]), float(init_rpy[2]))
        )
        self.data.qpos[:3] = init_pos
        self.data.qpos[3:7] = init_quat
        joint_init = self.np_random.uniform(-self.cfg.init_joint_noise, self.cfg.init_joint_noise)
        self.data.qpos[self.joint_qpos_indices] = joint_init
        self.data.qvel[:] = 0.0
        self.data.qvel[self.joint_dof_indices] = self.np_random.normal(
            0.0, self.cfg.init_joint_vel_std, size=4
        )

        # 把所有 actuator 设回 0（包括 magnet）
        self.ctrl_joint_targets = joint_init.copy()
        self.data.ctrl[self.magnet_actuator_ids] = 0.0
        self.data.ctrl[self.joint_actuator_ids] = self.ctrl_joint_targets

        mujoco.mj_forward(self.model, self.data)

        # 球初始化（必须在 mj_forward 之后，因为要拿 head_proxy 世界坐标）
        ball_pos, ball_vel = self._sample_ball_init()
        self._set_ball_state(ball_pos, ball_vel)
        mujoco.mj_forward(self.model, self.data)

        # settle 几步让接触力进入稳态
        for _ in range(self.cfg.settle_steps):
            mujoco.mj_step(self.model, self.data)
            # 但球在 settle 期间不应该自由减速太多 → 让球速度强制保持
            self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3] = ball_vel

        self.prev_ball_head_dist = float(
            np.linalg.norm(self._get_ball_pos_world() - self._get_head_pos_world())
        )
        self._obs_buf[:] = 0.0
        obs = self._push_obs(self._get_obs())
        info = self._build_info(is_success=False)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        prev_ctrl = self.ctrl_joint_targets.copy()
        proposed = np.clip(
            self.ctrl_joint_targets + self.cfg.action_delta_scale * action,
            self.joint_ctrl_min,
            self.joint_ctrl_max,
        )
        self.ctrl_joint_targets = (
            self.cfg.action_smoothing * self.ctrl_joint_targets
            + (1.0 - self.cfg.action_smoothing) * proposed
        )

        contact_kind_this_step = "none"
        for _ in range(self.cfg.frame_skip):
            self.data.ctrl[self.magnet_actuator_ids] = 0.0
            self.data.ctrl[self.joint_actuator_ids] = self.ctrl_joint_targets
            mujoco.mj_step(self.model, self.data)
            # 在 substep 中检测接触（一次 frame_skip 内可能多个接触瞬间）
            kind = self._analyze_ball_contact()
            if kind == "head":
                contact_kind_this_step = "head"
            elif kind == "body" and contact_kind_this_step != "head":
                contact_kind_this_step = "body"

        self.step_count += 1
        self.prev_action = action.copy()

        ball_pos = self._get_ball_pos_world()
        ball_vel = self._get_ball_vel_world()
        head_pos = self._get_head_pos_world()
        base_pos = self.data.xpos[self.base_body_id]
        linvel_world, angvel_world = self._get_base_velocity_world()

        ball_head_dist = float(np.linalg.norm(ball_pos - head_pos))
        approach_improvement = self.prev_ball_head_dist - ball_head_dist
        self.prev_ball_head_dist = ball_head_dist

        # 接触事件处理
        contact_head_bonus = 0.0
        contact_body_penalty = 0.0
        if contact_kind_this_step == "head":
            if not self.head_hit_occurred:
                contact_head_bonus = self.cfg.contact_head_bonus
            self.head_hit_occurred = True
        elif contact_kind_this_step == "body":
            self.body_hit_count += 1
            contact_body_penalty = self.cfg.contact_body_penalty
        self.last_contact_kind = contact_kind_this_step

        if self.head_hit_occurred:
            self.max_ball_vx_after_hit = max(self.max_ball_vx_after_hit, float(ball_vel[0]))

        # --- 奖励分量 ---
        approach_reward = self.cfg.approach_reward_weight * safe_exp_neg(
            self.cfg.approach_reward_scale, ball_head_dist
        )
        approach_progress = self.cfg.approach_progress_weight * approach_improvement

        # 鱼头朝向（base 的 +x 单位向量在世界）与"鱼头→球"方向的余弦相似度
        # 只在球还远的时候给（gate by ball_head_dist），避免接触后白送对准分
        base_rot = self._get_base_rot_world()
        head_forward_world = base_rot @ np.array([1.0, 0.0, 0.0])
        to_ball = ball_pos - head_pos
        to_ball_norm = float(np.linalg.norm(to_ball))
        if to_ball_norm > self.cfg.head_alignment_gate_dist:
            cos_align = float(np.dot(head_forward_world, to_ball / to_ball_norm))
            head_alignment_reward = self.cfg.head_alignment_weight * max(0.0, cos_align)
        else:
            head_alignment_reward = 0.0

        # 击中后只要球 vx>0 就奖励
        ball_vx_reward = 0.0
        if self.head_hit_occurred:
            ball_vx_reward = self.cfg.ball_vx_reward_weight * max(0.0, float(ball_vel[0]))

        action_penalty = -self.cfg.action_penalty_weight * float(np.mean(np.square(action)))
        action_rate_penalty = -self.cfg.action_rate_penalty_weight * float(
            np.mean(np.square(self.ctrl_joint_targets - prev_ctrl))
        )
        joint_vel_penalty = -self.cfg.joint_vel_penalty_weight * float(
            np.mean(np.square(self.data.qvel[self.joint_dof_indices]))
        )
        base_drift_penalty = -self.cfg.base_drift_penalty_weight * float(np.linalg.norm(base_pos))

        reward = (
            approach_reward
            + approach_progress
            + head_alignment_reward
            + contact_head_bonus
            + contact_body_penalty
            + ball_vx_reward
            + action_penalty
            + action_rate_penalty
            + joint_vel_penalty
            + base_drift_penalty
        )

        # --- 终止判定 ---
        # 成功：鱼头已接触 + 击球后最大球速 > 阈值（物理可达水平）
        is_success = (
            self.head_hit_occurred
            and self.max_ball_vx_after_hit > self.cfg.hit_success_ball_vx
        )
        ball_escaped = (
            float(ball_pos[0]) < self.cfg.fail_ball_x_min
            or abs(float(ball_pos[1])) > self.cfg.fail_ball_yz_abs
            or abs(float(ball_pos[2])) > self.cfg.fail_ball_yz_abs
        )
        base_drift_fail = float(np.linalg.norm(base_pos)) > self.cfg.fail_base_pos
        # 简易自转判定：base_rot 与单位矩阵的偏差（用对角线 trace）
        trace = float(np.clip((np.trace(base_rot) - 1.0) / 2.0, -1.0, 1.0))
        rot_angle = float(np.arccos(trace))
        base_rot_fail = rot_angle > np.deg2rad(self.cfg.fail_base_rot_deg)
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

        terminated_reason = ""
        if is_success:
            reward += self.cfg.success_bonus
            terminated_reason = "success"
            terminated = True
        elif ball_escaped:
            reward += self.cfg.fail_penalty
            terminated_reason = "ball_escaped"
            terminated = True
        elif base_drift_fail or base_rot_fail:
            reward += self.cfg.fail_penalty
            terminated_reason = "base_drift" if base_drift_fail else "base_rot"
            terminated = True
        elif not finite:
            terminated_reason = "nan"
            terminated = True
        else:
            terminated = False
        truncated = self.step_count >= self.cfg.max_steps

        info = self._build_info(is_success=is_success, terminated_reason=terminated_reason)
        info.update({
            "reward_breakdown": {
                "approach": approach_reward,
                "approach_progress": approach_progress,
                "head_alignment": head_alignment_reward,
                "contact_head_bonus": contact_head_bonus,
                "contact_body_penalty": contact_body_penalty,
                "ball_vx_reward": ball_vx_reward,
                "action_penalty": action_penalty,
                "action_rate_penalty": action_rate_penalty,
                "joint_vel_penalty": joint_vel_penalty,
                "base_drift_penalty": base_drift_penalty,
            }
        })
        obs = self._push_obs(self._get_obs())
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model, height=self.rgb_height, width=self.rgb_width
            )
        self.renderer.update_scene(self.data, camera="grasp_side")
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


__all__ = ["HeadHitConfig", "MujocoFishHeadHitEnv"]
