---
title: Modes & Lifecycle
description: "How OM1 organizes behavior into modes and manages their lifecycle."
icon: layer-group
---

## Overview

OM1 structures robot behavior into **modes** — distinct operational states such as social interaction, exploration, patrol, or autonomous navigation — and governs how the system moves between them.

This section covers:

- [**Modes**](modes.md) — what modes are and how each one shapes perception, input handling, and task priorities.
- [**Mode Selection**](mode_selection.md) — the different ways to switch modes: context-aware, time-based, input-triggered, or manual.
- [**Transition Rules**](transition_rules.md) — how and when the robot moves between modes, defined as prioritized rule objects.
- [**Lifecycle**](lifecycle.md) — the activation-to-termination boundaries of a mode, ensuring predictable, safe transitions.
