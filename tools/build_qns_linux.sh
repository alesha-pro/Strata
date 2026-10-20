#!/usr/bin/env bash
# Build this fork's engine and picture reader from source for the NVIDIA card in this PC (Linux), for Mirai's
# Qwen3.8-Flash-Next-S packs (docs/QNS.md). The installers and `strata update` download the upstream engine, which
# cannot read these packs, so build it here.
#   tools/build_qns_linux.sh           -> build/strata, build-vision/bin/strata-vision, .venv
# Needs git, python3 (3.10+) with venv, a C++ compiler and the CUDA toolkit 12.8 or newer (13.x for RTX 50).
set -euo pipefail
cd "$(dirname "$0")/.."

# the newest nvcc on PATH or in /usr/local/cuda*, unless NVCC is set (a distribution nvcc on PATH is often older)
nvcc_ver() { "$1" --version 2>/dev/null | sed -n "s/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p" | awk "{print \$1*100+\$2}"; }
if [ -z "${NVCC:-}" ]; then
    best=0
    for n in $(command -v nvcc || true) /usr/local/cuda*/bin/nvcc; do
        [ -x "$n" ] || continue
        v=$(nvcc_ver "$n"); [ "${v:-0}" -gt "$best" ] && { NVCC=$n; best=$v; }
    done
fi
[ -x "${NVCC:-}" ] || { echo "no nvcc: install the CUDA toolkit 12.8 or newer, or set NVCC=/path/to/nvcc"; exit 1; }
[ "$(nvcc_ver "$NVCC")" -ge 1208 ] || { echo "$NVCC is older than CUDA 12.8: install 12.8 or newer, or set NVCC"; exit 1; }
CUDA_ROOT=$(cd "$(dirname "$NVCC")/.." && pwd)
ARCH=${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
echo "GPU sm_$ARCH, $("$NVCC" --version | tail -1)"

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt -r requirements-build.txt
export PATH="$PWD/.venv/bin:$CUDA_ROOT/bin:$PATH"

cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DSTRATA_BUILD_TESTS=OFF
cmake --build build --target strata -j "$(nproc)"

cmake -G Ninja -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_DIR="$PWD/build/_deps/strata_llamacpp-src" -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$CUDA_ROOT"
cmake --build build-vision --target strata-vision -j "$(nproc)"

echo "built $PWD/build/strata and $PWD/build-vision/bin/strata-vision"
