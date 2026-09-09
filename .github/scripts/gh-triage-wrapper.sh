#!/usr/bin/env bash
# Installed in place of the real `gh` binary during pr-triage.yml's triage
# step: only lets Gemini's shell tool run an allowlisted set of gh commands
# against the one PR it's currently triaging.
set -euo pipefail

REAL_GH="REAL_GH_PATH_PLACEHOLDER"
ALLOWED_PR_FILE=".pr-triage-allowed-pr"

if [ ! -f "$ALLOWED_PR_FILE" ]; then
  echo "gh-triage-wrapper: $ALLOWED_PR_FILE missing — refusing all gh commands." >&2
  exit 1
fi
TRIAGE_ALLOWED_PR="$(cat "$ALLOWED_PR_FILE")"

LABEL="needs-human-review"

deny() {
  echo "gh-triage-wrapper: blocked command: gh $*" >&2
  exit 1
}

has_flag() {
  local flag="$1"
  shift
  for arg in "$@"; do
    [ "$arg" = "$flag" ] && return 0
  done
  return 1
}

has_flag_value() {
  local flag="$1" expected="$2"
  shift 2
  local prev=""
  for arg in "$@"; do
    if [ "$prev" = "$flag" ] && [ "$arg" = "$expected" ]; then
      return 0
    fi
    prev="$arg"
  done
  return 1
}

sub1="${1:-}"
sub2="${2:-}"
target="${3:-}"

case "$sub1 $sub2" in
  "pr view" | "pr diff")
    [ "$target" = "$TRIAGE_ALLOWED_PR" ] || deny "$@"
    ;;
  "pr review")
    [ "$target" = "$TRIAGE_ALLOWED_PR" ] || deny "$@"
    has_flag "--request-changes" "$@" || deny "$@"
    has_flag "--approve" "$@" && deny "$@"
    has_flag "--merge" "$@" && deny "$@"
    ;;
  "pr edit")
    [ "$target" = "$TRIAGE_ALLOWED_PR" ] || deny "$@"
    has_flag_value "--add-label" "$LABEL" "$@" || deny "$@"
    ;;
  "pr close")
    [ "$target" = "$TRIAGE_ALLOWED_PR" ] || deny "$@"
    ;;
  "pr comment")
    [ "$target" = "$TRIAGE_ALLOWED_PR" ] || deny "$@"
    ;;
  "label create" | "label edit")
    has_flag "$LABEL" "$@" || deny "$@"
    ;;
  *)
    deny "$@"
    ;;
esac

exec "$REAL_GH" "$@"
