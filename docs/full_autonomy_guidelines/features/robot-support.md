---
title: Robot & Simulation Support
description: "Which autonomy features run on which robot, and where simulation differs from hardware."
icon: table-cells
---

Not every autonomy feature runs on every robot, and a few behave differently in the [Cloud Simulator](../../simulators/cloud-isaac-sim.md) than on physical hardware. This page is the single source of truth for those differences; the feature guides link here rather than repeating the caveats.

The robot reports its own capabilities in `GET /status` — `robot_type`, `use_sim`, `slam_3d_supported`, and `slam_3d_color_supported` — so you can gate behavior at runtime instead of hard-coding it.

## Support matrix

| Feature | Go2 (hardware) | Go2 (sim) | M20 |
|---|:---:|:---:|:---:|
| 2D mapping & navigation | ✅ | ✅ | ✅ |
| 3D mapping — SLAM / Color / Full | ❌ | ✅ | ✅ |
| 3D map navigation | ❌ | ✅ | ✅ |
| Patrol (+ scheduling) | ✅ | ✅ | ✅ |
| Auto-charging | ✅ | ✅ | ✅ |

## The key differences

- **3D on Go2 is simulation-only.** A real Go2 maps and navigates in **2D only** — its 3D SLAM stack (geometry, color, and full) is available in the simulator but not on hardware. This is enforced in the SDK: the 3D modes are offered to `{go2, m20}` when `use_sim` is true, and to `{m20}` otherwise. On a real Go2, `slam_3d_supported` and `slam_3d_color_supported` come back `false`, and the portal disables the 3D SLAM cards. **M20 supports 3D on hardware.**
- **Auto-charging has no sim gate.** It's available on the Go2 and the M20, and it behaves the same in the simulator as on hardware — so you can rehearse dock-and-resume in the sim.
- **2D mapping, navigation, and patrol are not sim-gated.** They run the same in both places.

> **G1 and Tron:** the Unitree G1 and LimX Tron can be launched in the Cloud Simulator, but the autonomy features above are defined for the quadrupeds (Go2, M20). <!-- CONFIRM: which autonomy features (if any) are supported on G1 / Tron -->

<!-- CONFIRM: patrol on M20 — code isn't robot-gated and the M20 auto-charge panel references the patrol schedule, but please confirm M20 patrol is supported/tested on hardware and in sim -->
