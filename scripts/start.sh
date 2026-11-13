#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
if [[ ! -f frontend/dist/index.html ]]; then
  echo '화면 빌드가 없습니다. 먼저 ./scripts/setup.sh 를 실행하세요.'
  exit 1
fi
exec uv run --frozen quantum-lab serve "$@"
