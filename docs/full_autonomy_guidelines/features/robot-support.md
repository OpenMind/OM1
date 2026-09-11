---
title: Robot & Simulation Support
description: "Which autonomy features run on which robot, and where simulation differs from hardware."
icon: table-cells
---

Not every autonomy feature runs on every robot, and a few behave differently in the [Cloud Simulator](../../simulators/cloud-isaac-sim.md) than on physical hardware. This page is the single source of truth for those differences; the feature guides link here rather than repeating the caveats.

Your integration doesn't have to hard-code any of this — the robot tells you what it can do. `GET /status` reports `robot_type`, `use_sim`, `slam_3d_supported`, and `slam_3d_color_supported`, so you can check a capability at runtime and adapt to whatever robot you're talking to.

The columns below use the portal's short names — **Go2** is the Unitree Go2, **M20 Pro** is the Deep Robotics M20 Pro (its `robot_type` is `m20`).

## Support matrix

| Feature | Go2 (hardware) | Go2 (sim) | M20 Pro |
|---|:---:|:---:|:---:|
| 2D mapping & navigation | ✅ | ✅ | ✅ |
| 3D mapping — SLAM / Color / Full | ❌ | ✅ | ✅ |
| 3D map navigation | ❌ | ✅ | ✅ |
| Patrol (+ scheduling) | ✅ | ✅ | ✅ |
| Auto-charging | ✅ | ✅ | ✅ |

## The key differences

- **3D on Go2 is simulation-only.** A real Go2 supports **2D SLAM only** — its 3D modes (SLAM, Color, and Full) work in the simulator but not on hardware, where the portal simply doesn't offer them. The **M20 Pro** supports 3D everywhere, in the sim and on hardware.
- **Auto-charging works everywhere.** It's available on the Go2 and the M20 Pro, and behaves the same in the simulator as on hardware — so you can rehearse dock-and-resume in the sim.
- **2D mapping, navigation, and patrol are not sim-gated.** They run the same in both places.

> **G1 and Tron:** the Unitree G1 and LimX Tron can be launched in the Cloud Simulator, but the autonomy features above are built around the quadrupeds (Go2, M20 Pro).
