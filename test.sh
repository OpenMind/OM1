#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}

if [ -z "$MODE" ]; then
  echo "Usage: ./test.sh [base|new]"
  exit 2
fi

case "$MODE" in
  base)
    # Run existing tests in the repo (as they are at base commit)
    pytest -q || exit $?
    ;;
  new)
    # Run only the new tests we added
    pytest -q tools/asr-eval/tests || exit $?
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 2
    ;;
esac
