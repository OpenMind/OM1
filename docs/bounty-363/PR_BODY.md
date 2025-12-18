
Bounty #363 – Milestone-1: Improve Gazebo simulator (deterministic, setup, CI, docs)

Scope

Reproducible setup, deterministic world/seed, CI smoke test (headless), quickstart.

Changes

scripts/gazebo_setup.sh, scripts/gazebo_run_deterministic.sh

worlds/bounty363.world

.github/workflows/gazebo-smoke.yml

docs/bounty-363/QUICKSTART.md

Demo

Short unlisted video will be added after verification.

How to test

./scripts/gazebo_setup.sh

SEED=12345 ./scripts/gazebo_run_deterministic.sh
