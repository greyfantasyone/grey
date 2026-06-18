from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
JOINT_ACTUATOR_NAMES = ("j1", "j2", "j3", "j4")
MAGNET_ACTUATOR_NAMES = ("forward", "left", "right")


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_from_euler_xyz_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return quat_normalize_wxyz(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_vec_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat_wxyz(q) @ np.asarray(v, dtype=np.float64)


def orientation_error_world(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(r_cur[:, 0], r_des[:, 0])
        + np.cross(r_cur[:, 1], r_des[:, 1])
        + np.cross(r_cur[:, 2], r_des[:, 2])
    )


def safe_exp_neg(scale: float, x: float) -> float:
    return float(math.exp(-float(scale) * float(max(0.0, x))))


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: np.ndarray | float, b: np.ndarray | float, t: float):
    return (1.0 - t) * a + t * b


@dataclass
class FloatingGraspConfig:
    # 仿真与动作
    frame_skip: int = 10
    max_steps: int = 360
    settle_steps: int = 6
    action_delta_scale: float = 0.08
    action_smoothing: float = 0.72

    # stage0 已完成后，base 理想操作位附近的锚点（仅作奖励/可选弱辅助，不直接锁定）
    base_anchor_world: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    base_anchor_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    base_anchor_pos_noise: np.ndarray = field(
        default_factory=lambda: np.array([0.010, 0.010, 0.008], dtype=np.float64)
    )
    base_anchor_rpy_noise: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([2.0, 2.0, 4.0], dtype=np.float64))
    )
    init_pos_perturb: np.ndarray = field(
        default_factory=lambda: np.array([0.010, 0.010, 0.008], dtype=np.float64)
    )
    init_rpy_perturb: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([3.0, 3.0, 5.0], dtype=np.float64))
    )
    init_joint_noise: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([6.0, 8.0, 10.0, 10.0], dtype=np.float64))
    )
    init_joint_vel_std: float = 0.04

    # 目标在 base 坐标系中的相对期望位置与随机范围。
    target_nominal_base: np.ndarray = field(
        default_factory=lambda: np.array([-0.46, 0.00, 0.00], dtype=np.float64)
    )
    target_random_base_xyz_easy: np.ndarray = field(
        default_factory=lambda: np.array([0.015, 0.040, 0.040], dtype=np.float64)
    )
    target_random_base_xyz_final: np.ndarray = field(
        default_factory=lambda: np.array([0.045, 0.100, 0.080], dtype=np.float64)
    )

    # 可选弱站位保持器：为过渡课程学习准备。assist_mode='off' 时完全不用。
    assist_mode: str = "off"  # off | constant | decay
    assist_max_strength: float = 0.35
    assist_decay_end: float = 0.35
    assist_pos_kp: float = 60.0
    assist_pos_kd: float = 10.0
    assist_rot_kp: float = 18.0
    assist_rot_kd: float = 3.0
    assist_force_limit: float = 20.0
    assist_torque_limit: float = 6.0
    assist_episode_randomization: bool = False
    assist_random_zero_prob: float = 0.30
    assist_random_min_scale: float = 0.20
    assist_random_max_scale: float = 1.00

    # MuJoCo 原生流体模型与小海流随机化。默认开，但按课程逐渐加大。
    enable_wind: bool = True
    wind_base_max: float = 0.05
    wind_wave_amp_max: float = 0.03
    wind_wave_freq_range_hz: tuple[float, float] = (0.05, 0.30)
    wind_noise_std: float = 0.002
    disturbance_start_progress: float = 0.30
    randomize_fluid: bool = True
    density_scale_range: tuple[float, float] = (0.90, 1.10)
    viscosity_scale_range: tuple[float, float] = (0.70, 1.40)

    # 课程学习：前期放松成功判据，后期收紧。
    success_distance_easy: float = 0.150
    success_distance_final: float = 0.050
    success_ee_speed: float = 0.080
    success_base_speed_easy: float = 0.15
    success_base_speed_final: float = 0.06
    success_base_angspeed_easy: float = 0.80
    success_base_angspeed_final: float = 0.35
    success_anchor_pos_easy: float = 0.18
    success_anchor_pos_final: float = 0.07
    success_anchor_rot_easy_deg: float = 35.0
    success_anchor_rot_final_deg: float = 15.0
    hold_steps_required_easy: int = 4
    hold_steps_required_final: int = 20
    near_radius: float = 0.15

    fail_anchor_pos: float = 0.35
    fail_anchor_rot_deg: float = 60.0

    # 奖励
    dense_reward_weight: float = 3.0
    dense_reward_scale: float = 7.0
    progress_reward_weight: float = 12.0
    hold_reward_weight: float = 8.0
    hold_speed_scale: float = 9.0
    hold_reward_gate_scale: float = 1.20
    hold_reward_gate_margin: float = 0.015
    hold_reward_progress_power: float = 1.0
    near_hold_decay_steps: int = 1
    drop_hold_penalty_steps: int = 2
    precision_reward_weight: float = 4.0
    precision_dist_scale: float = 18.0
    precision_speed_scale: float = 8.0

    base_pos_reward_weight: float = 0.3
    base_pos_reward_scale: float = 10.0
    base_rot_reward_weight: float = 0.2
    base_rot_reward_scale: float = 4.0
    anchor_progress_reward_weight: float = 1.0
    rot_progress_reward_weight: float = 0.3

    base_speed_penalty_weight: float = 0.01
    base_angspeed_penalty_weight: float = 0.005
    near_target_speed_penalty_weight: float = 0.08
    near_target_base_pos_penalty_weight: float = 0.0
    near_target_base_rot_penalty_weight: float = 0.0
    action_penalty_weight: float = 0.02
    action_rate_penalty_weight: float = 0.50
    joint_vel_penalty_weight: float = 0.001
    success_region_drop_penalty: float = 2.5

    success_bonus: float = 160.0
    drift_termination_penalty: float = -25.0

    # 观测历史堆叠（为无 LSTM 的算法提供时序信息）
    obs_stack_size: int = 1

    rest_joint_qpos: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))


class MujocoFishFloatingGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """方案 A 环境：只用尾部 4 个关节，同时兼顾 base 稳定与尾部定位。

    - assist_mode='off'：纯方案 A，无任何外部 base 保持器。
    - assist_mode='decay'：训练早期给一个弱保持器，并随训练进度衰减到 0。
    - assist_mode='constant'：始终保留一个弱保持器，主要用于 debug 或对照实验。
    """

    metadata = {"render_modes": [None, "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path = "phy3.02_schemeA_grasp.xml",
        config: FloatingGraspConfig | None = None,
        render_mode: str | None = None,
        rgb_width: int = 640,
        rgb_height: int = 480,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML 文件不存在: {self.xml_path}")

        self.cfg = config or FloatingGraspConfig()
        if self.cfg.assist_mode not in {"off", "constant", "decay"}:
            raise ValueError(f"assist_mode 非法: {self.cfg.assist_mode}")
        self.render_mode = render_mode
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer: mujoco.Renderer | None = None

        self.base_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.target_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "target_body")
        self.ee_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.target_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self.joint_ids = [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.joint_actuator_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_ACTUATOR_NAMES
        ]
        self.magnet_actuator_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MAGNET_ACTUATOR_NAMES
        ]

        self.joint_qpos_indices = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_dof_indices = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_ctrl_min = self.model.actuator_ctrlrange[self.joint_actuator_ids, 0].astype(np.float64)
        self.joint_ctrl_max = self.model.actuator_ctrlrange[self.joint_actuator_ids, 1].astype(np.float64)

        self._base_density = float(self.model.opt.density)
        self._base_viscosity = float(self.model.opt.viscosity)

        self.training_progress = 0.0
        self.anchor_pos_world = self.cfg.base_anchor_world.copy()
        self.anchor_quat_wxyz = quat_normalize_wxyz(self.cfg.base_anchor_quat_wxyz)
        self.anchor_rot_world = quat_to_rotmat_wxyz(self.anchor_quat_wxyz)
        self.target_pos_world = np.zeros(3, dtype=np.float64)
        self.ctrl_joint_targets = self.cfg.rest_joint_qpos.copy()
        self.prev_action = np.zeros(4, dtype=np.float64)

        self._wind_base = np.zeros(3, dtype=np.float64)
        self._wind_amp = np.zeros(3, dtype=np.float64)
        self._wind_w = 0.0
        self._wind_phase = np.zeros(3, dtype=np.float64)
        self._assist_strength_cache = 0.0
        self._assist_episode_scale = 1.0
        self._disturbance_scale_cache = 0.0

        self.prev_distance = 0.0
        self.prev_anchor_pos_err = 0.0
        self.prev_anchor_rot_err = 0.0
        self.hold_counter = 0
        self.max_hold_counter = 0
        self.step_count = 0
        self._was_in_success_region = False
        self._ever_entered_success_region = False

        obs_dim = 36
        self._obs_stack_size = max(1, self.cfg.obs_stack_size)
        self._obs_buf = np.zeros((self._obs_stack_size, obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        full_obs_dim = obs_dim * self._obs_stack_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(full_obs_dim,), dtype=np.float32)
        self.np_random = np.random.default_rng()

    def _push_obs(self, obs: np.ndarray) -> np.ndarray:
        """将单帧 obs 推入历史 buffer，返回堆叠后的观测。"""
        if self._obs_stack_size > 1:
            self._obs_buf[:-1] = self._obs_buf[1:]
            self._obs_buf[-1] = obs
            return self._obs_buf.flatten()
        return obs

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.cfg.frame_skip)

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def _require_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"XML 中找不到对象: {name}")
        return obj_id

    def get_joint_qpos(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_indices].copy()

    def get_joint_qvel(self) -> np.ndarray:
        return self.data.qvel[self.joint_dof_indices].copy()

    def get_base_rot_world(self) -> np.ndarray:
        return self.data.xmat[self.base_body_id].reshape(3, 3).copy()

    def get_base_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            vel,
            0,
        )
        angvel_world = vel[:3].copy()
        linvel_world = vel[3:].copy()
        return linvel_world, angvel_world

    def get_ee_pos_world(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_target_pos_world(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site_id].copy()

    def get_ee_jacobian_pos(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp

    def get_ee_vel_world(self) -> np.ndarray:
        jacp = self.get_ee_jacobian_pos()
        return jacp @ self.data.qvel

    def _sample_uniform_vec(self, half_range: np.ndarray) -> np.ndarray:
        return self.np_random.uniform(low=-half_range, high=half_range, size=half_range.shape)

    def _curriculum_alpha(self) -> float:
        return smoothstep01(self.training_progress)

    def _disturbance_scale(self) -> float:
        if self.training_progress <= self.cfg.disturbance_start_progress:
            return 0.0
        raw = (self.training_progress - self.cfg.disturbance_start_progress) / max(
            1e-6, 1.0 - self.cfg.disturbance_start_progress
        )
        return smoothstep01(raw)

    def _assist_strength(self) -> float:
        if self.cfg.assist_mode == "off":
            return 0.0
        base = float(np.clip(self.cfg.assist_max_strength, 0.0, 1.0))
        if self.cfg.assist_mode == "constant":
            return base * self._assist_episode_scale
        if self.cfg.assist_decay_end <= 1e-6:
            return 0.0
        phase = 1.0 - self.training_progress / self.cfg.assist_decay_end
        return base * max(0.0, phase) * self._assist_episode_scale

    def _sample_assist_episode_scale(self) -> float:
        if self.cfg.assist_mode == "off" or not self.cfg.assist_episode_randomization:
            return 1.0
        if self.np_random.random() < float(np.clip(self.cfg.assist_random_zero_prob, 0.0, 1.0)):
            return 0.0
        lo = float(np.clip(self.cfg.assist_random_min_scale, 0.0, 1.0))
        hi = float(np.clip(self.cfg.assist_random_max_scale, lo, 1.0))
        return float(self.np_random.uniform(lo, hi))

    def _scheduled_target_random_range(self) -> np.ndarray:
        a = self._curriculum_alpha()
        return lerp(self.cfg.target_random_base_xyz_easy, self.cfg.target_random_base_xyz_final, a)

    def _success_distance(self) -> float:
        return float(lerp(self.cfg.success_distance_easy, self.cfg.success_distance_final, self._curriculum_alpha()))

    def _success_base_speed(self) -> float:
        return float(lerp(self.cfg.success_base_speed_easy, self.cfg.success_base_speed_final, self._curriculum_alpha()))

    def _success_base_angspeed(self) -> float:
        return float(
            lerp(self.cfg.success_base_angspeed_easy, self.cfg.success_base_angspeed_final, self._curriculum_alpha())
        )

    def _success_anchor_pos(self) -> float:
        return float(lerp(self.cfg.success_anchor_pos_easy, self.cfg.success_anchor_pos_final, self._curriculum_alpha()))

    def _success_anchor_rot(self) -> float:
        deg = float(
            lerp(
                self.cfg.success_anchor_rot_easy_deg,
                self.cfg.success_anchor_rot_final_deg,
                self._curriculum_alpha(),
            )
        )
        return float(np.deg2rad(deg))

    def _hold_steps_required(self) -> int:
        return int(
            round(
                float(
                    lerp(
                        float(self.cfg.hold_steps_required_easy),
                        float(self.cfg.hold_steps_required_final),
                        self._curriculum_alpha(),
                    )
                )
            )
        )

    def _sample_anchor_pose(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        if "base_anchor_world" in options:
            self.anchor_pos_world = np.asarray(options["base_anchor_world"], dtype=np.float64).copy()
        else:
            self.anchor_pos_world = self.cfg.base_anchor_world + self._sample_uniform_vec(self.cfg.base_anchor_pos_noise)

        if "base_anchor_quat_wxyz" in options:
            self.anchor_quat_wxyz = quat_normalize_wxyz(np.asarray(options["base_anchor_quat_wxyz"], dtype=np.float64))
        else:
            d_rpy = self._sample_uniform_vec(self.cfg.base_anchor_rpy_noise)
            dq = quat_from_euler_xyz_wxyz(float(d_rpy[0]), float(d_rpy[1]), float(d_rpy[2]))
            self.anchor_quat_wxyz = quat_mul_wxyz(self.cfg.base_anchor_quat_wxyz, dq)
            self.anchor_quat_wxyz = quat_normalize_wxyz(self.anchor_quat_wxyz)
        self.anchor_rot_world = quat_to_rotmat_wxyz(self.anchor_quat_wxyz)

    def _sample_target_world(self, options: dict[str, Any] | None = None) -> np.ndarray:
        options = options or {}
        if "target_world" in options:
            return np.asarray(options["target_world"], dtype=np.float64).copy()
        if "target_base" in options:
            target_base = np.asarray(options["target_base"], dtype=np.float64).copy()
        else:
            target_base = self.cfg.target_nominal_base + self._sample_uniform_vec(self._scheduled_target_random_range())
        return self.anchor_pos_world + rotate_vec_wxyz(self.anchor_quat_wxyz, target_base)

    def _set_target_position(self, target_world: np.ndarray) -> None:
        self.model.body_pos[self.target_body_id] = np.asarray(target_world, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _set_joint_targets(self, joint_targets: np.ndarray) -> None:
        self.ctrl_joint_targets = np.clip(joint_targets, self.joint_ctrl_min, self.joint_ctrl_max)
        self.data.ctrl[self.magnet_actuator_ids] = 0.0
        self.data.ctrl[self.joint_actuator_ids] = self.ctrl_joint_targets

    def _apply_training_assist(self) -> None:
        self.data.xfrc_applied[:] = 0.0
        assist = self._assist_strength_cache
        if assist <= 1e-8:
            return

        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.get_base_rot_world()
        linvel_world, angvel_world = self.get_base_velocity_world()
        pos_err = self.anchor_pos_world - base_pos
        rot_err = orientation_error_world(self.anchor_rot_world, base_rot)

        force = assist * (
            self.cfg.assist_pos_kp * pos_err - self.cfg.assist_pos_kd * linvel_world
        )
        torque = assist * (
            self.cfg.assist_rot_kp * rot_err - self.cfg.assist_rot_kd * angvel_world
        )
        force = np.clip(force, -self.cfg.assist_force_limit, self.cfg.assist_force_limit)
        torque = np.clip(torque, -self.cfg.assist_torque_limit, self.cfg.assist_torque_limit)
        self.data.xfrc_applied[self.base_body_id, :3] = force
        self.data.xfrc_applied[self.base_body_id, 3:] = torque

    def _randomize_fluid(self) -> None:
        self.model.opt.density = self._base_density
        self.model.opt.viscosity = self._base_viscosity
        if not self.cfg.randomize_fluid:
            return
        s = self._disturbance_scale_cache
        if s <= 1e-8:
            return
        ds_sample = self.np_random.uniform(*self.cfg.density_scale_range)
        vs_sample = self.np_random.uniform(*self.cfg.viscosity_scale_range)
        ds = 1.0 + s * (ds_sample - 1.0)
        vs = 1.0 + s * (vs_sample - 1.0)
        self.model.opt.density = self._base_density * ds
        self.model.opt.viscosity = self._base_viscosity * vs

    def _randomize_wind_params(self) -> None:
        self._wind_base[:] = 0.0
        self._wind_amp[:] = 0.0
        self._wind_w = 0.0
        self._wind_phase[:] = 0.0
        self.model.opt.wind[:] = 0.0
        if not self.cfg.enable_wind:
            return
        s = self._disturbance_scale_cache
        if s <= 1e-8:
            return
        self._wind_base = self.np_random.uniform(-self.cfg.wind_base_max, self.cfg.wind_base_max, size=3) * s
        self._wind_amp = self.np_random.uniform(0.0, self.cfg.wind_wave_amp_max, size=3) * s
        f_hz = self.np_random.uniform(*self.cfg.wind_wave_freq_range_hz)
        self._wind_w = 2.0 * math.pi * float(f_hz)
        self._wind_phase = self.np_random.uniform(0.0, 2.0 * math.pi, size=3)
        self._update_wind()

    def _update_wind(self) -> None:
        if not self.cfg.enable_wind or self._disturbance_scale_cache <= 1e-8:
            self.model.opt.wind[:] = 0.0
            return
        t = float(self.data.time)
        wind = self._wind_base + self._wind_amp * np.sin(self._wind_w * t + self._wind_phase)
        if self.cfg.wind_noise_std > 0.0:
            wind += self.np_random.normal(0.0, self.cfg.wind_noise_std, size=3) * self._disturbance_scale_cache
        self.model.opt.wind[:] = wind

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self.get_target_pos_world() - self.get_ee_pos_world()))

    def _base_anchor_errors(self) -> tuple[float, float, np.ndarray, np.ndarray]:
        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.get_base_rot_world()
        pos_err_world = self.anchor_pos_world - base_pos
        rot_err_world = orientation_error_world(self.anchor_rot_world, base_rot)
        return (
            float(np.linalg.norm(pos_err_world)),
            float(np.linalg.norm(rot_err_world)),
            pos_err_world,
            rot_err_world,
        )

    def _is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    def _get_obs(self) -> np.ndarray:
        base_rot = self.get_base_rot_world()
        world_to_base = base_rot.T
        base_pos = self.data.xpos[self.base_body_id]
        linvel_world, angvel_world = self.get_base_velocity_world()
        target_world = self.get_target_pos_world()
        ee_world = self.get_ee_pos_world()
        ee_vel_world = self.get_ee_vel_world()
        _, _, anchor_err_world, anchor_rot_err_world = self._base_anchor_errors()

        joint_q = self.get_joint_qpos() / 1.57
        joint_qd = np.tanh(0.25 * self.get_joint_qvel())

        ee_to_target_base = world_to_base @ (target_world - ee_world)
        ee_vel_base = world_to_base @ ee_vel_world
        target_from_base = world_to_base @ (target_world - base_pos)
        anchor_err_base = world_to_base @ anchor_err_world
        anchor_rot_err_base = world_to_base @ anchor_rot_err_world
        linvel_base = world_to_base @ linvel_world
        angvel_base = world_to_base @ angvel_world
        hold_fraction = np.array([self.hold_counter / max(1, self._hold_steps_required())], dtype=np.float64)
        assist_fraction = np.array([self._assist_strength_cache], dtype=np.float64)
        disturbance_fraction = np.array([self._disturbance_scale_cache], dtype=np.float64)

        obs = np.concatenate(
            [
                joint_q,
                joint_qd,
                ee_to_target_base,
                ee_vel_base,
                target_from_base,
                anchor_err_base,
                anchor_rot_err_base,
                linvel_base,
                angvel_base,
                self.prev_action,
                hold_fraction,
                assist_fraction,
                disturbance_fraction,
            ]
        )
        return obs.astype(np.float32)

    def _build_info(self, *, is_success: bool) -> dict[str, Any]:
        dist = self._distance_to_target()
        ee_speed = float(np.linalg.norm(self.get_ee_vel_world()))
        linvel_world, angvel_world = self.get_base_velocity_world()
        anchor_pos_err, anchor_rot_err, _, _ = self._base_anchor_errors()
        return {
            "distance": dist,
            "ee_speed": ee_speed,
            "base_speed": float(np.linalg.norm(linvel_world)),
            "base_angspeed": float(np.linalg.norm(angvel_world)),
            "anchor_pos_err": anchor_pos_err,
            "anchor_rot_err_rad": anchor_rot_err,
            "anchor_rot_err_deg": float(np.rad2deg(anchor_rot_err)),
            "hold_counter": int(self.hold_counter),
            "hold_required": int(self._hold_steps_required()),
            "hold_fraction": float(self.hold_counter / max(1, self._hold_steps_required())),
            "max_hold_counter": int(self.max_hold_counter),
            "max_hold_fraction": float(self.max_hold_counter / max(1, self._hold_steps_required())),
            "assist_strength": float(self._assist_strength_cache),
            "assist_episode_scale": float(self._assist_episode_scale),
            "disturbance_scale": float(self._disturbance_scale_cache),
            "is_success": bool(is_success),
            "in_success_region": bool(self._was_in_success_region),
            "ever_entered_success_region": bool(self._ever_entered_success_region),
            "target_world": self.get_target_pos_world().copy(),
            "ee_world": self.get_ee_pos_world().copy(),
        }

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
        self.hold_counter = 0
        self.max_hold_counter = 0
        self._was_in_success_region = False
        self._ever_entered_success_region = False
        self.prev_action[:] = 0.0

        self._assist_episode_scale = self._sample_assist_episode_scale()
        self._assist_strength_cache = self._assist_strength()
        self._disturbance_scale_cache = self._disturbance_scale()

        self._sample_anchor_pose(options)
        self.target_pos_world = self._sample_target_world(options)
        self._randomize_fluid()
        self._randomize_wind_params()

        init_pos = self.anchor_pos_world + self._sample_uniform_vec(self.cfg.init_pos_perturb)
        init_rpy = self._sample_uniform_vec(self.cfg.init_rpy_perturb)
        init_quat = quat_mul_wxyz(
            self.anchor_quat_wxyz,
            quat_from_euler_xyz_wxyz(float(init_rpy[0]), float(init_rpy[1]), float(init_rpy[2])),
        )
        init_quat = quat_normalize_wxyz(init_quat)

        self.data.qpos[:3] = init_pos
        self.data.qpos[3:7] = init_quat
        joint_init = self.cfg.rest_joint_qpos + self._sample_uniform_vec(self.cfg.init_joint_noise)
        self.data.qpos[self.joint_qpos_indices] = joint_init
        self.data.qvel[:] = 0.0
        self.data.qvel[self.joint_dof_indices] = self.np_random.normal(0.0, self.cfg.init_joint_vel_std, size=4)

        self._set_target_position(self.target_pos_world)
        self._set_joint_targets(joint_init)
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.cfg.settle_steps):
            self._apply_training_assist()
            self._set_joint_targets(self.ctrl_joint_targets)
            self._update_wind()
            mujoco.mj_step(self.model, self.data)

        self.prev_distance = self._distance_to_target()
        self.prev_anchor_pos_err, self.prev_anchor_rot_err, _, _ = self._base_anchor_errors()
        self._obs_buf[:] = 0.0
        obs = self._push_obs(self._get_obs())
        info = self._build_info(is_success=False)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

        prev_ctrl = self.ctrl_joint_targets.copy()
        proposed_ctrl = np.clip(
            self.ctrl_joint_targets + self.cfg.action_delta_scale * action,
            self.joint_ctrl_min,
            self.joint_ctrl_max,
        )
        self.ctrl_joint_targets = (
            self.cfg.action_smoothing * self.ctrl_joint_targets
            + (1.0 - self.cfg.action_smoothing) * proposed_ctrl
        )
        self.ctrl_joint_targets = np.clip(self.ctrl_joint_targets, self.joint_ctrl_min, self.joint_ctrl_max)

        for _ in range(self.cfg.frame_skip):
            self._update_wind()
            self._apply_training_assist()
            self._set_joint_targets(self.ctrl_joint_targets)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.prev_action = action.copy()

        dist = self._distance_to_target()
        ee_speed = float(np.linalg.norm(self.get_ee_vel_world()))
        linvel_world, angvel_world = self.get_base_velocity_world()
        base_speed = float(np.linalg.norm(linvel_world))
        base_angspeed = float(np.linalg.norm(angvel_world))
        anchor_pos_err, anchor_rot_err, _, _ = self._base_anchor_errors()

        improvement = self.prev_distance - dist
        anchor_pos_improvement = self.prev_anchor_pos_err - anchor_pos_err
        anchor_rot_improvement = self.prev_anchor_rot_err - anchor_rot_err
        self.prev_distance = dist
        self.prev_anchor_pos_err = anchor_pos_err
        self.prev_anchor_rot_err = anchor_rot_err

        success_distance = self._success_distance()
        success_base_speed = self._success_base_speed()
        success_base_angspeed = self._success_base_angspeed()
        success_anchor_pos = self._success_anchor_pos()
        success_anchor_rot = self._success_anchor_rot()
        hold_reward_dist = min(
            self.cfg.near_radius,
            success_distance * self.cfg.hold_reward_gate_scale + self.cfg.hold_reward_gate_margin,
        )

        in_success_region = (
            dist < success_distance
            and ee_speed < self.cfg.success_ee_speed
        )
        near_success_region = dist < hold_reward_dist
        dropped_from_success_region = self._was_in_success_region and (not in_success_region)
        if in_success_region:
            self.hold_counter += 1
            self._ever_entered_success_region = True
        elif near_success_region:
            self.hold_counter = max(0, self.hold_counter - self.cfg.near_hold_decay_steps)
        else:
            self.hold_counter = max(0, self.hold_counter - self.cfg.drop_hold_penalty_steps)
        self.max_hold_counter = max(self.max_hold_counter, self.hold_counter)
        self._was_in_success_region = in_success_region

        dense_reward = self.cfg.dense_reward_weight * safe_exp_neg(self.cfg.dense_reward_scale, dist)
        progress_reward = self.cfg.progress_reward_weight * improvement
        hold_reward = 0.0
        if near_success_region:
            proximity = max(0.0, 1.0 - dist / max(hold_reward_dist, 1e-6))
            hold_progress = self.hold_counter / max(1, self._hold_steps_required())
            hold_reward = (
                self.cfg.hold_reward_weight
                * proximity
                * (0.10 + 0.90 * (hold_progress ** self.cfg.hold_reward_progress_power))
                * math.exp(-self.cfg.hold_speed_scale * ee_speed)
            )
        precision_reward = (
            self.cfg.precision_reward_weight
            * math.exp(-self.cfg.precision_dist_scale * dist)
            * math.exp(-self.cfg.precision_speed_scale * ee_speed)
        )

        base_pos_reward = self.cfg.base_pos_reward_weight * safe_exp_neg(
            self.cfg.base_pos_reward_scale, anchor_pos_err
        )
        base_rot_reward = self.cfg.base_rot_reward_weight * safe_exp_neg(
            self.cfg.base_rot_reward_scale, anchor_rot_err
        )
        anchor_progress_reward = self.cfg.anchor_progress_reward_weight * anchor_pos_improvement
        rot_progress_reward = self.cfg.rot_progress_reward_weight * anchor_rot_improvement

        action_penalty = -self.cfg.action_penalty_weight * float(np.mean(np.square(action)))
        action_rate_penalty = -self.cfg.action_rate_penalty_weight * float(
            np.mean(np.square(self.ctrl_joint_targets - prev_ctrl))
        )
        joint_vel_penalty = -self.cfg.joint_vel_penalty_weight * float(np.mean(np.square(self.get_joint_qvel())))
        base_speed_penalty = -self.cfg.base_speed_penalty_weight * (base_speed * base_speed)
        base_angspeed_penalty = -self.cfg.base_angspeed_penalty_weight * (base_angspeed * base_angspeed)
        near_target_speed_penalty = 0.0
        near_target_base_pos_penalty = 0.0
        near_target_base_rot_penalty = 0.0
        if near_success_region:
            near_target_speed_penalty = -self.cfg.near_target_speed_penalty_weight * (ee_speed * ee_speed)
            near_target_base_pos_penalty = -self.cfg.near_target_base_pos_penalty_weight * (anchor_pos_err * anchor_pos_err)
            near_target_base_rot_penalty = -self.cfg.near_target_base_rot_penalty_weight * (anchor_rot_err * anchor_rot_err)
        drop_penalty = -self.cfg.success_region_drop_penalty if dropped_from_success_region else 0.0

        reward = (
            dense_reward
            + progress_reward
            + hold_reward
            + precision_reward
            + base_pos_reward
            + base_rot_reward
            + anchor_progress_reward
            + rot_progress_reward
            + action_penalty
            + action_rate_penalty
            + joint_vel_penalty
            + base_speed_penalty
            + base_angspeed_penalty
            + near_target_speed_penalty
            + near_target_base_pos_penalty
            + near_target_base_rot_penalty
            + drop_penalty
        )

        success = self.hold_counter >= self._hold_steps_required()
        drift_fail = (
            anchor_pos_err > self.cfg.fail_anchor_pos
            or anchor_rot_err > float(np.deg2rad(self.cfg.fail_anchor_rot_deg))
        )
        finite = self._is_finite()
        terminated = bool(success or drift_fail or (not finite))
        truncated = bool(self.step_count >= self.cfg.max_steps)
        if success:
            reward += self.cfg.success_bonus
        elif drift_fail:
            reward += self.cfg.drift_termination_penalty

        info = self._build_info(is_success=success)
        info.update(
            {
                "success_distance": success_distance,
                "success_anchor_pos": success_anchor_pos,
                "success_anchor_rot_deg": float(np.rad2deg(success_anchor_rot)),
                "drift_fail": bool(drift_fail),
                "near_success_region": bool(near_success_region),
                "reached_success_region": bool(self._ever_entered_success_region),
                "dropped_from_success_region": bool(dropped_from_success_region),
            }
        )
        obs = self._push_obs(self._get_obs())
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.rgb_height, width=self.rgb_width)
        camera_name = "grasp_side"
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


__all__ = ["FloatingGraspConfig", "MujocoFishFloatingGraspEnv"]
