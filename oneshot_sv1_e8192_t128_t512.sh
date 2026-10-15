#!/usr/bin/env bash
# Compatibility wrapper — real script: scripts/regression/oneshot_sv1_e8192_t128_t512.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/regression/oneshot_sv1_e8192_t128_t512.sh" "$@"
