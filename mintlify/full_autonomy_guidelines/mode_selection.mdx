---
title: Switch Modes
description: "Different ways to switch between modes"
---

Once your Go2 is setup to run in full autonomy, you can get started with exploring different modes offered via OM1 and explore the functionalities.
There are multiple ways to do this:
1. Context Aware
2. Time based
3. Input triggered
4. Manual trigger (via portal)

## Context Aware

The system supports context-aware transition rules that enable automatic mode switching based on operational state and task completion for a particular mode.

Once the robot is powered on, it autonomously progresses through predefined operational modes without requiring user commands.

- Upon startup, the robot enters Welcome Mode, where it greets the user and performs facial capture for identification purposes.

- Facial data is processed in compliance with privacy-preserving mechanisms. Face detection and anonymisation takes place on the edge device.

- After successful initialization and user recognition, the robot automatically transitions to SLAM (Simultaneous Localization and Mapping) Mode.

- The robot maps the surrounding environment. Location labels are generated and stored for future navigation tasks.

- Once the SLAM process is completed successfully, the robot transitions to Navigation Mode. The robot can now navigate autonomously within the mapped area.

- Guard Mode is excluded from context aware transitions and must be explicitly activated by the user via voice commands or through the OpenMind portal, as required.

Example config to setup context_aware transition type for transitioning into navigation mode from slam mode.

```
    {
      from_mode: "slam",
      to_mode: "navigation",
      transition_type: "context_aware",
      context_conditions: { exploration_done: true },
      priority: 3,
      cooldown_seconds: 5.0,
    }
```

```
    {
      from_mode: "mode_1",
      to_mode: "mode_2",
      transition_type: "context_aware",
      context_conditions: { owner_identified: true, temperature: 70 },
      priority: 2,
      cooldown_seconds: 5.0,
    }
```

## Time based

Time-based transitions enable the robot to automatically switch operational modes after a predefined period of elapsed time. These transitions are designed to ensure safe and secure operation without requiring continuous user input.

Below is an example of configuring a time-based transition. Once configured, the system automatically transitions from the `from_mode` to the specified `to_mode` after the defined `timeout_seconds` has elapsed (300 seconds in this example).

If an intervention occurs during this interval, such as user interaction or another eligible transition being triggered, the system evaluates all applicable transition rules. The transition associated with the highest priority is selected and executed.

Example config to setup time based transition type for transitioning into guard mode from conversation mode.

```
   {
      "from_mode": "conversation",
      "to_mode": "guard",
      "transition_type": "time_based",
      "trigger_keywords": ["guard", "security", "patrol", "keep watch"],
      "priority": 2,
      "timeout_seconds": 300.0 // Switch to guard mode after 5 minutes of inactivity
    }
```

## Input triggered (Voice Commands)

Step 1: Configure your API key in ~/.bashrc file and start your machine in full autonomy mode.
Step 2: Start talking to your robot dog and ask it to switch to a particular mode.
    For example,
    The robot says, hi I'm your friendly robot dog, how may I help you?
    The user can then request the robot to switch to a particular mode by saying, switch to [desired mode].

- Desired mode: **Welcome**
    trigger_keywords: ["reset", "start over", "welcome mode", "restart", "initialize"]
- Desired mode: **Conversation**
    trigger_keywords: ["talk", "chat", "conversation", "tell me", "ask you", "discuss"]
- Desired mode: **Slam**
    trigger_keywords: ["explore", "map", "navigate", "look around", "slam", "wander"]
- Desired mode: **Navigation**
    trigger_keywords: ["navigate", "navigation", "go to", "take me to", "show me"]
- Desired mode: **Guard**
    trigger_keywords: ["guard", "security", "patrol", "keep watch"]

## Manual Trigger (via Portal)

Step 1: Configure your API key in ~/.bashrc file and start your machine in full autonomy mode.
Step 2: Login to your OM1 portal and head over to Machine Teleops on the left navigation bar.
    ![Machine Teleops](../assets/full-autonomy-assets/machine_teleops.png)
Step 3: Once connected, you’ll see your machine listed as **Online** at the top of the screen.
    ![Online Machines](../assets/full-autonomy-assets/online_machine.png)
Step 4: Scroll down to access the Mode Selection section. From here, choose the mode you want your robot to switch to.
    ![Online Machines](../assets/full-autonomy-assets/portal_mode_selection.png)
Step 5: In SLAM Mode, you can manually guide the robot through its environment to generate a map. As you move, you can label specific areas and have the robot remember them. The resulting map should appear as follows:
    ![Online Machines](../assets/full-autonomy-assets/slam_map.png)
Step 6: Once the map is saved, switch to Navigation Mode to make the robot move autonomously between locations.
    Use the dropdown menu to select a destination.
    ![Online Machines](../assets/full-autonomy-assets/select_location_to_navigate.png)
Step 7: You can also monitor three live camera streams directly from the portal.
    ![Online Machines](../assets/full-autonomy-assets/video_streams.png)

These steps and exploration methods provide a structured approach to understanding and managing OM1’s modes.
