---
title: Hybrid Localization
description: "Introducing Hybrid Localization"
icon: arrows-spin
---

## What is Robot Localization?

Every mobile robot must answer one fundamental question before it can do anything useful: **"Where am I?"** The process of determining a robot's position and orientation within a map of its environment is called **localization**. Without it, the robot cannot plan a path to a destination, avoid obstacles along the way, or know when it has arrived. Localization is the invisible foundation that makes autonomous navigation possible.

Think of it like waking up in a hotel room in a city you have visited before. You know what the city looks like (you have a map). You can see the room around you (your sensors are working). But you do not know which hotel you are in, which floor, or which direction you are facing. You need to figure out your position on the map before you can navigate to the breakfast buffet.

## Why is Localization Hard?

Localization sounds straightforward, but in practice it involves several genuinely difficult problems:

| Challenge | Description |
|-----------|-------------|
| **Global localization (cold start)** | The robot is powered on at an unknown location in a known map. It must determine its position from scratch, without any prior information. This is the hardest variant. |
| **Continuous tracking** | Once the robot knows where it is, it must keep track of its position as it moves. Sensors drift over time — wheels slip, gyroscopes accumulate error, laser scans get noisy. |
| **Environmental change** | The real world changes. Furniture gets moved, doors open and close, boxes appear in hallways. The map may not perfectly match what sensors see today. |
| **Symmetric environments** | Long corridors, identical meeting rooms, and repeating architectural features look the same from multiple positions. The laser scanner sees identical geometry at different locations. |
| **Kidnapped robot problem** | The robot is physically picked up and placed somewhere else (or its localization catastrophically fails). It must detect that it is lost and re-localize automatically. |

> No single localization technique handles all of these well. That observation is what led to this hybrid system.

## What This System Does

The hybrid localization system combines three complementary technologies into one architecture:

| Technology | Description |
|------------|-------------|
| **Visual Place Recognition (VPR)** | Uses camera images to estimate which general area of the map the robot is in |
| **Correlative Scan Matching (CSM)** | Uses LiDAR to brute-force search for the exact position, guided by the VPR hint |
| **Nav2 AMCL** | The standard ROS 2 particle filter for smooth, continuous position tracking during navigation |

Together, these three systems handle every failure mode listed above:

- **VPR** breaks symmetry
- **CSM** provides instant global localization
- **AMCL** provides robust continuous tracking

A health monitor watches AMCL's output and triggers automatic recovery when needed. **No human intervention is required at any point.**

## Architecture

The system is implemented as three ROS 2 nodes working together, orchestrated by a state machine in the Hybrid Localization Manager. This node does not publish any TF transforms — Nav2 AMCL is the sole authority for the `map→odom` transform. The hybrid node acts as a quality gate: it finds initial poses, validates them, feeds them to AMCL, and monitors AMCL's output for failures.

### The Three Nodes

| Node | Phase | Description |
|------|-------|-------------|
| **VPR Capture Node** | Mapping | Captures camera snapshots during SLAM, computes visual embeddings, stores `(embedding, pose)` pairs to disk |
| **VPR Query Node** | Navigation | Matches current camera view against the database at 2 Hz, publishes rough location hints |
| **Hybrid Localization Manager** | Navigation | Main orchestrator containing CSM, candidate ranking, AMCL seeding, and health monitoring |

#### Node 1: VPR Capture Node (Mapping Phase)

This node runs while the robot is building a map using SLAM. As the robot explores its environment, the capture node periodically takes camera snapshots, computes a compact visual fingerprint (called an embedding) for each image, and stores it alongside the robot's current map-frame position. The result is a database of `(embedding, pose)` pairs saved to disk alongside the map.

**Capture triggers:**
- Movement > **10 cm** from last capture
- Rotation > **~11°** from last capture

This ensures good spatial coverage without wasting storage on redundant images.

#### Node 2: VPR Query Node (Navigation Phase)

This node runs during autonomous navigation at **2 Hz** (every 0.5 seconds). It:

1. Takes the latest camera image
2. Computes its embedding
3. Compares against every saved embedding in the database
4. If best match exceeds similarity threshold (**0.65**), publishes the corresponding pose as a location hint

The hint has high uncertainty (**2 meter standard deviation**) because VPR provides a neighborhood estimate, not a precise position. The query node also publishes a "ready" signal after its first query attempt.

#### Node 3: Hybrid Localization Manager (Navigation Phase)

This is the main orchestrator. It subscribes to:

| Topic | Purpose |
|-------|---------|
| `/scan` | Laser scanner data for CSM |
| `/map` | Occupancy grid from map_server |
| `/visual_pose_hint` | VPR location hints |
| `/amcl_pose` | AMCL pose output for monitoring |

It contains the correlative scan matcher for global localization, the candidate ranking logic, the AMCL seeding mechanism, and the health monitoring system. Everything is coordinated through a **five-state state machine**.

### The Five States

| State | Description |
|-------|-------------|
| `WAITING_FOR_MAP` | Startup state. Waits for the occupancy grid map from `map_server`. On receipt, preprocesses the map (crop, gradient field, free space extraction) and transitions to `GLOBAL_LOCALIZING`. |
| `GLOBAL_LOCALIZING` | Collects VPR camera hints for up to 5 seconds. Computes robust consensus from multiple hints. Runs VPR-guided correlative scan matching. Ranks up to 5 candidate poses. Selects the best and transitions to `SEEDING_AMCL`. |
| `SEEDING_AMCL` | Publishes the discovered pose to `/initialpose_validated` (which AMCL subscribes to). Re-publishes every 2 seconds until AMCL responds. Transitions to `MONITORING` when AMCL publishes its first pose. Times out after 30 seconds and falls back to `RECOVERING`. |
| `MONITORING` | Steady-state. Watches every AMCL pose update through three checks: pose jump detection, scan match cross-validation, and covariance monitoring. Triggers `RECOVERING` if sustained failures are detected. |
| `RECOVERING` | Triggered when monitoring detects persistent failure. Clears old VPR hints, collects fresh camera data, and re-runs the full global localization pipeline. Functionally identical to `GLOBAL_LOCALIZING` but entered from a failed state. |

### The /initialpose interception pattern

A critical architectural detail: Nav2 AMCL does not subscribe to the standard /initialpose topic in this system. Instead, it subscribes to /initialpose_validated. The hybrid node intercepts all /initialpose messages (including those from RViz’s "2D Pose Estimate" button), evaluates them via scan matching, and only forwards them to /initialpose_validated if they pass quality validation (score above 90%).

This means every pose that reaches AMCL has been validated. A careless RViz click or a malfunctioning external node cannot inject a bad pose and destroy a perfectly good localization. During the early states (before monitoring begins), external poses are forwarded without validation since there is no existing good localization to protect.

## Complete Startup Sequence

To tie everything together, here is the exact sequence of events from power-on to autonomous navigation:

| Time | Event |
|------|-------|
| **t = 0s** | `map_server` publishes the occupancy grid. The hybrid node receives it, crops to the occupied region plus margin, builds the gradient field, extracts free space cells. Transitions to `GLOBAL_LOCALIZING`. Starts the VPR warmup delay (5 seconds). |
| **t = 0.5s** | VPR Query Node matches the current camera view against the database. Publishes the first `/visual_pose_hint` with the saved pose of the best matching image. Publishes `/vpr_query_ready = true`. |
| **t = 1–4s** | VPR continues publishing hints at 2 Hz. The hybrid node accumulates these into a list. Once 3 or more hints are collected and the ready signal has been received, it computes the VPR consensus (median, outlier removal, inlier average). |
| **t = 2–4s** | The hybrid node runs `perform_global_localization`: generates ~160,000 VPR-biased candidate poses, evaluates all of them against the gradient map via Numba, selects the top 5 spatially distinct candidates, refines each via iterative hill-climbing, ranks them with VPR proximity bonus. |
| **t = 3–5s** | Best candidate published to `/initialpose_validated`. The hybrid node transitions to `SEEDING_AMCL`. Starts re-publish timer (every 2 seconds). |
| **t = 3–6s** | Nav2 AMCL receives `/initialpose_validated`. Initializes particle cloud around the given pose. Publishes first `/amcl_pose`. The hybrid node detects this, cancels re-publish timer, transitions to `MONITORING`. Starts 5-second settle timer. |
| **t = 8–11s** | Settle time elapses. AMCL particles have converged. The hybrid node begins active health monitoring (jump detection, scan match validation, covariance checks). The robot is now localized and ready for autonomous navigation. |

> **Total time from power-on to navigation-ready: typically 8–11 seconds**, fully automatic, no human intervention. Compared to Nav2 AMCL alone (which requires a manual 2D Pose Estimate click and 10–30 seconds of wandering), this is a significant operational improvement.

## Comparison with Previous Approaches

The following table summarizes how the hybrid system addresses each localization challenge compared to using AMCL or the correlative scan matcher alone:

| Challenge | AMCL Alone | CSM Alone | Hybrid System |
|-----------|------------|-----------|---------------|
| **Global localization on startup** | Manual RViz click or slow global service (10–30s wandering) | Instant brute-force search (2–3s) | VPR-guided CSM search in under 5 seconds, fully automatic |
| **Continuous tracking** | Excellent: smooth, low-CPU particle filter | Poor: expensive per-scan, prone to jumps | AMCL handles tracking; CSM only runs on-demand |
| **Environmental changes** | Moderate with tuning (`z_rand`, beam skip) | Poor: gradient map is static | AMCL handles moderate changes; recovery handles major changes |
| **Symmetric environments** | Vulnerable: particles converge to wrong location | Vulnerable: identical geometry scores equally | VPR breaks symmetry; jump validation catches errors |
| **Kidnapped robot** | Poor: slow random particle injection | Good: can re-localize instantly | Health monitor detects failure and triggers VPR-guided re-localization |
| **Bad external pose (RViz)** | Accepted blindly: can destroy good localization | N/A | Scan match validation gate: rejected if quality < 90% |

## Hardware Context

This system was designed and tested on the **Unitree Go2** quadruped robot with the following sensor configuration:

| Sensor | Type | Purpose |
|--------|------|---------|
| **Primary LiDAR** | RPLidar S2L (2D, 360°) | Mounted on top. Used for mapping, localization (CSM + AMCL), and obstacle detection in Nav2 costmap. |
| **Secondary LiDAR** | 4D LiDAR | Mounted on nose. Used only for ground obstacle detection via STVL costmap layer. Not used for localization. |
| **Camera** | Any ROS 2 compatible | Used for Visual Place Recognition. Any camera publishing `sensor_msgs/Image` is compatible. |

### Platform-Specific Challenges

The quadruped platform presents unique localization challenges that make the hybrid architecture particularly valuable:

| Challenge | Impact | How Hybrid Helps |
|-----------|--------|------------------|
| Body pitch/roll during gait | Laser scan distortion | Multi-modal validation (LiDAR + camera) |
| Leg odometry noise | Different characteristics than wheel odometry | Covariance monitoring detects drift |
| Sit/stand transitions | Odometry jumps | Pose jump detection triggers recovery |
