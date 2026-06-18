from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np


JOINT_ACTUATOR_NAMES = ("j1", "j2", "j3", "j4")
MAGNET_ACTUATOR_NAMES = ("forward", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查尾部关节开环摆动是否会引起 free-floating base 漂移")
    parser.add_argument("--xml", type=str, default="phy3.02_schemeA_grasp.xml")
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    base_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
    act_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)) for n in JOINT_ACTUATOR_NAMES]
    mag_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)) for n in MAGNET_ACTUATOR_NAMES]

    data.ctrl[mag_ids] = 0.0
    mujoco.mj_forward(model, data)
    start = data.xpos[base_id].copy()
    traj = []
    steps = int(args.seconds / model.opt.timestep)

    for t in range(steps):
        tt = t * model.opt.timestep
        data.ctrl[act_ids[0]] = 0.25 * math.sin(2.0 * math.pi * 0.7 * tt)
        data.ctrl[act_ids[1]] = 0.45 * math.sin(2.0 * math.pi * 1.0 * tt + 0.4)
        data.ctrl[act_ids[2]] = -0.65 * math.sin(2.0 * math.pi * 1.4 * tt + 0.8)
        data.ctrl[act_ids[3]] = 0.35 * math.sin(2.0 * math.pi * 1.8 * tt + 1.2)
        mujoco.mj_step(model, data)
        traj.append(data.xpos[base_id].copy())

    traj = np.asarray(traj)
    drift = traj[-1] - start
    print(f"xml={xml_path}")
    print(f"start_base={start}")
    print(f"end_base={traj[-1]}")
    print(f"net_drift={drift} norm={float(np.linalg.norm(drift)):.6f} m")
    print(
        "range_xyz=(%.6f, %.6f, %.6f) m"
        % (float(np.ptp(traj[:, 0])), float(np.ptp(traj[:, 1])), float(np.ptp(traj[:, 2])))
    )
    print("说明：如果这里不是几乎 0，说明尾部关节变化确实会通过水动力/耦合影响 free-floating base。")


if __name__ == "__main__":
    main()
