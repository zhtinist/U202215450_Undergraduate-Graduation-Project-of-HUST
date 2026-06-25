#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for this workspace.
# Default is dry-run. Use --apply to execute deletions.

ROOT="/home/zht/HaotianZhu"
REPRO="$ROOT/log/repro"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

TARGETS=(
  "$REPRO/archive/tmp_logs/__pycache__"
  "$REPRO/archive/tmp_logs/*.log"
  "$REPRO/archive/debug_steps/*.txt"
  "$REPRO/archive/debug_steps/*_exit_code.txt"
)

echo "[cleanup] mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo "[cleanup] root: $ROOT"
echo

for pattern in "${TARGETS[@]}"; do
  shopt -s nullglob
  matches=( $pattern )
  shopt -u nullglob

  if [[ ${#matches[@]} -eq 0 ]]; then
    continue
  fi

  for p in "${matches[@]}"; do
    if [[ $APPLY -eq 1 ]]; then
      rm -rf "$p"
      echo "deleted: $p"
    else
      echo "would delete: $p"
    fi
  done
done

echo
echo "[cleanup] done."
