#!/usr/bin/env bash
# Validating shim installed in place of the real `gh` binary for the
# duration of the "Triage PRs (Gemini CLI)" step in pr-triage.yml.
#
# Why this exists: that step hands Gemini (running in --approval-mode=yolo,
# i.e. no per-command confirmation) untrusted, attacker-controlled content —
# the title/body/diff of an external contributor's PR — while it holds a gh
# token with repo-wide pull-requests:write/issues:write. A PR body could
# contain text crafted to hijack the model into acting on some OTHER PR or
# issue, not just the one it's meant to be triaging. Telling the model "only
# touch this PR" in the prompt is not an enforceable control by itself, so
# it's enforced here instead: every `gh` invocation the model's shell tool
# makes is checked against an explicit allowlist — right subcommand, right
# flags, and (for anything that reads or writes a specific PR) the one PR
# number this run was launched for — before it's allowed through to the
# real binary. Everything else is rejected.
#
# Deliberately NOT configured via environment variables: this workflow
# already hit one confirmed case (see pr-triage.yml's gh-auth comment) of
# env vars set for this step not reliably reaching subprocesses spawned by
# Gemini CLI's shell tool. So the real gh path is baked into this file as a
# literal string at install time (see the sed substitution in pr-triage.yml
# — REAL_GH_PATH_PLACEHOLDER below is replaced there, never left as-is),
# and the allowed PR number is read from a plain file in the current
# directory instead, the same way this workflow already shares
# .pr-triage-candidates.tsv with Gemini's shell commands.
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
