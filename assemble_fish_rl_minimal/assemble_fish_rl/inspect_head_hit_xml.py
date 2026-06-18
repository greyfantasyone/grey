"""Quick sanity check for phy3.02_head_hit.xml.

Loads model, places ball in front of fish, gives it -x velocity, steps 600 frames
(~1.2s at timestep=0.002), prints ball trajectory + first contact info.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="phy3.02_head_hit.xml")
    parser.add_argument("--ball-x0", type=float, default=0.50, help="ball start x in world (m)")
    parser.add_argument("--ball-vx0", type=float, default=-0.60, help="ball start vx in world (m/s)")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml)))
    data = mujoco.MjData(model)

    ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_body")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    head_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_proxy")
    assert ball_jnt >= 0 and ball_body >= 0 and ball_geom >= 0
    print(f"[ids] ball_jnt={ball_jnt} ball_body={ball_body} ball_geom={ball_geom} "
          f"base_body={base_body} head_site={head_site}")

    # set ball pose and velocity via qpos/qvel of its freejoint
    qpos_adr = model.jnt_qposadr[ball_jnt]
    dof_adr = model.jnt_dofadr[ball_jnt]
    data.qpos[qpos_adr:qpos_adr + 3] = [args.ball_x0, 0.0, 0.0]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_adr:dof_adr + 3] = [args.ball_vx0, 0.0, 0.0]
    data.qvel[dof_adr + 3:dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    head_pos = data.site_xpos[head_site].copy()
    print(f"[init] head_proxy world={head_pos} ball_pos={data.xpos[ball_body]} "
          f"ball_vel={data.qvel[dof_adr:dof_adr + 3]}")

    first_contact_step = None
    first_contact_info = None
    nan_seen = False
    trajectory = []

    for t in range(args.steps):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            nan_seen = True
            print(f"[FAIL] NaN at step {t}")
            break
        pos = data.xpos[ball_body].copy()
        vel = data.qvel[dof_adr:dof_adr + 3].copy()
        trajectory.append((t, pos, vel))

        if first_contact_step is None and data.ncon > 0:
            for i in range(data.ncon):
                c = data.contact[i]
                if c.geom1 == ball_geom or c.geom2 == ball_geom:
                    first_contact_step = t
                    first_contact_info = {
                        "step": t,
                        "geom1": c.geom1,
                        "geom2": c.geom2,
                        "pos_world": c.pos.copy(),
                        "dist_to_head": float(np.linalg.norm(c.pos - head_pos)),
                        "ball_vel_before": trajectory[-2][2] if len(trajectory) >= 2 else None,
                    }
                    break

    if nan_seen:
        return

    final_t, final_pos, final_vel = trajectory[-1]
    print(f"[end] step={final_t} ball_pos={final_pos} ball_vel={final_vel}")

    # crude no-contact decel measurement (frames 0-50 before any contact)
    if first_contact_step is None or first_contact_step > 50:
        vx_5 = trajectory[5][2][0]
        vx_50 = trajectory[50][2][0]
        decel_pct = (1.0 - vx_50 / vx_5) * 100 if abs(vx_5) > 1e-6 else 0.0
        print(f"[fluid] free-flight vx step5={vx_5:.4f} step50={vx_50:.4f} "
              f"(decel {decel_pct:.1f}% over ~90ms)")

    if first_contact_step is None:
        print("[contact] no contact in this run (ball may have stopped or missed)")
    else:
        ci = first_contact_info
        print(f"[contact] step={ci['step']} pos_world={ci['pos_world']} "
              f"dist_to_head_proxy={ci['dist_to_head']:.4f} m "
              f"ball_vel_before={ci['ball_vel_before']}")
        # show velocity immediately after contact
        if ci['step'] + 1 < len(trajectory):
            vel_after = trajectory[ci['step'] + 1][2]
            print(f"[contact] ball_vel_after_1step={vel_after}")

    print("[OK] head_hit XML sanity finished")


if __name__ == "__main__":
    main()
