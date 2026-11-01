#!/usr/bin/env bash
# Read-only scheduler discovery. No allocation, installation, SSH hop or scientific job.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
mkdir -p .hpc/discovery
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output=".hpc/discovery/${stamp}-$$.txt"
{
  date -u
  hostname -f
  uname -sm
  getconf GNU_LIBC_VERSION
  command -v qsub
  echo '=== PBS server ==='
  qstat -Bf
  echo '=== Node summary (live CPU, memory, state and jobs) ==='
  pbsnodes -aSvj
  echo '=== Node details (resource names and placement) ==='
  pbsnodes -av
  echo '=== Queue limits/defaults ==='
  for queue in workq fat normal nvidia; do qstat -Qf "$queue"; done
  echo '=== Current user jobs ==='
  qstat -u "$USER"
  echo '=== Storage capacity and personal quota ==='
  df -h "$PWD"
  if command -v lfs >/dev/null 2>&1; then lfs quota -u "$USER" /home; fi
} > "$output" 2>&1
cat "$output"
printf '\nSaved discovery evidence: %s\n' "$output"
