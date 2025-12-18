# Bounty #363 — Milestone 1 Notes

## Scope covered in this PR
- Reproducible setup (scripts + short quickstart)
- Deterministic run (fixed seed)
- Headless smoke test in CI (gzserver)
- Minimal world suited for fast checks

## Files added/updated
- `worlds/bounty363.world`
- `scripts/gazebo_setup.sh`, `scripts/gazebo_run_deterministic.sh`
- `.github/workflows/gazebo-smoke.yml`
- `docs/bounty-363/QUICKSTART.md`

## How to reproduce locally
```bash
./scripts/gazebo_setup.sh
SEED=12345 ./scripts/gazebo_run_deterministic.sh

```

If Gazebo isn’t installed, the script falls back to a container.

## Notes on determinism
CI and local runs use a fixed --seed 12345 to keep results comparable.

## Rationale
Headless gzserver avoids GUI/X11 issues on runners.

Small, focused changes keep review surface minimal.

## Next (future milestones)
Sensor models & lightweight perf checks

Scenario suite + simple metrics for regressions
