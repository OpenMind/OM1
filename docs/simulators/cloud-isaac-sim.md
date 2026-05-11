---
title: Cloud Isaac Sim
description: "Learn how to run cloud Isaac Sim integrated with OM1"
icon: robot
---

# Cloud Isaac Sim Developer Walkthrough

Cloud Isaac Sim enables you to run robot simulations on managed cloud infrastructure, fully integrated with OM1. This guide walks you through launching an instance and connecting your OM1 setup.

## Prerequisites

- OpenMind Portal account on **Builder plan** or higher
- API key (found in your portal)
- OM1 codebase with `uv` environment configured

## Step 1: Launch a Cloud Simulator Instance

1. Log in to the [OpenMind Portal](https://portal.openmind.com)
2. Navigate to **Cloud Simulator** from the sidebar

![ ](../../assets/cloud-isaac-sim-assets/cloud_isaac_sim.png)

### Instance Types

Choose based on your simulation workload:

| Instance Type | vCPUs | RAM | Best For |
|---|---|---|---|
| **Standard** | 8 | 32 GB | Development & testing |
| **Performance** | 16 | 64 GB | Heavy compute, multi-robot scenarios |

### Supported Robots

- Unitree Go2
- Unitree G1
- LimX Tron

### Launch Time

Expect **4-5 minutes** for your instance to fully initialize.

## Step 2: Wait for Instance Finalization

Once you initiate the launch, the system begins setting up your cloud environment.

![ ](../../assets/cloud-isaac-sim-assets/env%20config.png)

![ ](../../assets/cloud-isaac-sim-assets/finalising.png)

The instance is ready when the status changes to **Running**.

## Step 3: Access your Active Cloud Isaac Sim Session

View your running session and logs from the portal dashboard.

![ ](../../assets/cloud-isaac-sim-assets/session.png)

## Step 4: Run OM1 with Cloud Simulator

Two options are available once your instance is running:

### Option A: Cloud VS Code

Access a full VS Code environment running in the cloud with OM1 pre-configured.

![ ](../../assets/cloud-isaac-sim-assets/cloud-vscode.png)

From here, you can:
- Edit and run OM1 code directly in the cloud
- Execute `uv run` commands without local hardware
- Test and debug your robot behaviors

### Option B: Local Environment

Run OM1 on your local machine and connect to the cloud simulator:

1. Copy your **API Key** from the portal
2. Open `config/cloud_sim.json5` in your local OM1 repo
3. Update with your cloud instance details
4. Run the config:

```bash
uv run src/run.py cloud_sim
```

## Cleaning Up

When you're finished with your simulation:

1. Return to the Cloud Simulator dashboard
2. Click **Delete Instance**

![ ](../../assets/cloud-isaac-sim-assets/delete-instance.png)

3. Confirm the deletion — this stops billing and frees cloud resources
