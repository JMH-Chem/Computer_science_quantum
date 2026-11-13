#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'uv가 필요합니다: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
command -v npm >/dev/null || { echo 'Node.js 24 LTS와 npm이 필요합니다.'; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
uv sync --frozen
npm --prefix frontend ci --cache "$PWD/.cache/npm"
npm --prefix frontend run build
echo '설치 완료. ./scripts/start.sh 로 실행하세요.'
