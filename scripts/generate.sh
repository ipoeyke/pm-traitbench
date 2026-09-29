#!/usr/bin/env bash
# Run the dataset generation end to end, fetch-market through probes.
# Usage: scripts/generate.sh [--config PATH] [--data-dir PATH] [--from STAGE] [--force]
#   --config PATH    YAML config overriding default settings (env: PM_TRAITBENCH_CONFIG)
#   --data-dir PATH  directory for pipeline tables (env: PM_TRAITBENCH_DATA_DIR, default: data)
#   --from STAGE     start at STAGE, skipping the ones before it
#   --force          overwrite existing output tables
# Flags override the environment and a .env file at the repo root; see .env.example.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

stages=(fetch-market sample market engine gate1 plan dialogue validate gate2 probes)

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }

die() {
  echo "error: $1" >&2
  exit 2
}

config="${PM_TRAITBENCH_CONFIG:-}"
data_dir="${PM_TRAITBENCH_DATA_DIR:-data}"
from="${stages[0]}"
force=()
while (($#)); do
  case "$1" in
    --config|--data-dir|--from)
      (($# >= 2)) || die "$1 needs a value"
      case "$1" in
        --config) config="$2" ;;
        --data-dir) data_dir="$2" ;;
        --from) from="$2" ;;
      esac
      shift 2
      ;;
    --force) force=(--force); shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

start=-1
for i in "${!stages[@]}"; do
  [[ "${stages[$i]}" == "$from" ]] && start=$i
done
((start >= 0)) || die "unknown stage: $from (one of: ${stages[*]})"

common=(--data-dir "$data_dir")
[[ -n "$config" ]] && common+=(--config "$config")

for stage in "${stages[@]:start}"; do
  echo "==> $stage"
  if [[ "$stage" == fetch-market ]]; then
    # The raw cache is reused across runs; fetch-market is a no-op without real seeds.
    uv run pm-traitbench "$stage" "${common[@]}"
  else
    uv run pm-traitbench "$stage" "${common[@]}" ${force[@]+"${force[@]}"}
  fi
done

echo "done: dataset written to $data_dir"
