#!/usr/bin/env bash
#
# bootstrap-go.sh — install a local Go toolchain without root.
#
# Downloads the official Go tarball from https://go.dev/dl and extracts it into
# a user-writable directory (default: ~/.local/go), so `go` is available on
# machines/sandboxes where Go isn't preinstalled and apt/root aren't available.
#
# The Go version defaults to the `go` directive in ../go.mod so the toolchain
# matches what the module requires; override with GO_VERSION=x.y.z.
#
# Usage:
#   scripts/bootstrap-go.sh              # install; prints the PATH line to eval
#   scripts/bootstrap-go.sh --persist    # also append PATH to ~/.bashrc
#   GO_VERSION=1.25.1 scripts/bootstrap-go.sh
#   GOROOT_INSTALL=/opt/go scripts/bootstrap-go.sh
#
# Note: requires network access to go.dev / dl.google.com. In a restricted
# sandbox those hosts must be on the egress allowlist first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${GOROOT_INSTALL:-${HOME}/.local/go}"
PERSIST=0
[ "${1:-}" = "--persist" ] && PERSIST=1

log() { printf '>> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- resolve version ------------------------------------------------------
resolve_version() {
  if [ -n "${GO_VERSION:-}" ]; then
    printf '%s' "${GO_VERSION}"
    return
  fi
  local v
  v="$(awk '/^go[[:space:]]+[0-9]/ {print $2; exit}' "${REPO_ROOT}/go.mod" 2>/dev/null || true)"
  # go.mod may list "1.25" (no patch); pad to x.y.0 for the download filename.
  case "${v}" in
    *.*.*) : ;;
    *.*)   v="${v}.0" ;;
    *)     v="" ;;
  esac
  [ -n "${v}" ] || die "could not determine Go version; set GO_VERSION=x.y.z"
  printf '%s' "${v}"
}

# --- detect platform ------------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Linux)  printf 'linux' ;;
    Darwin) printf 'darwin' ;;
    *)      die "unsupported OS: $(uname -s)" ;;
  esac
}
detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)  printf 'amd64' ;;
    aarch64|arm64) printf 'arm64' ;;
    *)             die "unsupported arch: $(uname -m)" ;;
  esac
}

main() {
  local ver os arch file url tmp sha_expected sha_actual
  ver="$(resolve_version)"
  os="$(detect_os)"
  arch="$(detect_arch)"
  file="go${ver}.${os}-${arch}.tar.gz"
  url="https://go.dev/dl/${file}"

  # Already installed and matching? Skip re-download.
  if [ -x "${INSTALL_DIR}/bin/go" ] && \
     "${INSTALL_DIR}/bin/go" version 2>/dev/null | grep -q "go${ver} "; then
    log "Go ${ver} already installed at ${INSTALL_DIR}"
  else
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT

    log "Downloading ${url}"
    curl -fSL --retry 3 --max-time 300 "${url}" -o "${tmp}/${file}" \
      || die "download failed (is go.dev on the egress allowlist?)"

    # Best-effort checksum verification from the release manifest.
    if command -v sha256sum >/dev/null 2>&1; then
      sha_expected="$(curl -fsSL --max-time 30 \
        "https://go.dev/dl/?mode=json&include=all" 2>/dev/null \
        | tr ',{}' '\n' | grep -A2 "\"${file}\"" | grep -o '"sha256":"[0-9a-f]*"' \
        | head -1 | sed 's/.*:"//;s/"//' || true)"
      if [ -n "${sha_expected}" ]; then
        sha_actual="$(sha256sum "${tmp}/${file}" | awk '{print $1}')"
        [ "${sha_expected}" = "${sha_actual}" ] \
          || die "checksum mismatch for ${file}"
        log "checksum verified"
      else
        log "WARNING: could not fetch checksum; skipping verification"
      fi
    fi

    file "${tmp}/${file}" | grep -qi gzip || die "downloaded file is not a gzip archive"

    log "Installing to ${INSTALL_DIR}"
    rm -rf "${INSTALL_DIR}"
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    tar -C "$(dirname "${INSTALL_DIR}")" -xzf "${tmp}/${file}"
    # The tarball extracts to a top-level "go/" dir; rename if needed.
    if [ "$(basename "${INSTALL_DIR}")" != "go" ]; then
      mv "$(dirname "${INSTALL_DIR}")/go" "${INSTALL_DIR}"
    fi
  fi

  "${INSTALL_DIR}/bin/go" version || die "go did not run after install"

  if [ "${PERSIST}" = "1" ]; then
    if ! grep -q "${INSTALL_DIR}/bin" "${HOME}/.bashrc" 2>/dev/null; then
      printf '\nexport PATH="%s/bin:$PATH"\n' "${INSTALL_DIR}" >> "${HOME}/.bashrc"
      log "appended PATH to ~/.bashrc"
    fi
  fi

  cat >&2 <<EOF

Go ${ver} ready. Add it to PATH for this shell:

    export PATH="${INSTALL_DIR}/bin:\$PATH"

Then, from ${REPO_ROOT}:

    go build ./... && go vet ./... && go test ./plugins/inputs/asr/...
EOF
}

main "$@"
