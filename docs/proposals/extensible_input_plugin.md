# Proposal: Extensible Input Plugin Architecture

## Overview

Currently, OM1 handles input modules in a fixed manner.  
This proposal introduces a **plugin-based architecture** that allows developers to add, remove, or extend input sources without modifying core code.

---

## Goals

- Provide a clear **InputPlugin interface**
- Enable future **robotic input integrations** (keyboard, sensors, cameras)
- Simplify testing and debugging of input modules
- Encourage community contributions with minimal friction

---

## Proposed Architecture


- **InputManager**: central controller
- **InputPlugin**: base class with methods `start()`, `stop()`, `read()`
- **Plugin registration**: via config or discovery
- **Error handling**: plugin failures do not crash core

---

## Benefits

- Easier onboarding for new contributors
- Supports experimentation without breaking the system
- Enables unit testing of individual plugins
- Aligns with OM1’s modular design philosophy

---

## Next Steps (Future Work)

- Implement `InputPlugin` base class
- Convert existing input sources to plugins
- Add plugin lifecycle tests
- Document plugin development guidelines

---

## References

- [OM1 Contribution Guidelines](https://github.com/OpenMind/OM1/blob/main/CONTRIBUTING.md)
- [OM1 Wiki: Bounty Program](https://github.com/OpenMind/OM1/wiki/Bounty-Program)
