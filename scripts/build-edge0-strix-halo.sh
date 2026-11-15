#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_dir="$repo_root/edge0-strata"
rocm_root="${EDGE0_ROCM_ROOT:-/opt/rocm/current}"
build_dir="${EDGE0_BUILD_DIR:-$strata_dir/build-hip-gfx1151}"
build_jobs="${EDGE0_BUILD_JOBS:-16}"
uv_cache="${EDGE0_UV_CACHE_DIR:-/tmp/edge0-uv-cache}"

llama_revision="3cf03257f219afbe7334045ff7c6a06ac68c627d"
llama_sha256="c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280"
download_dir="$strata_dir/third_party/downloads"
archive="$download_dir/llama-3cf03257.tar.gz"
llama_dir="$strata_dir/third_party/llama.cpp"

for required in \
  "$rocm_root/lib/llvm/bin/clang++" \
  "$rocm_root/bin/hipconfig" \
  "$strata_dir/tools/hip/gfx1151-hipblaslt-100500.txt"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$download_dir"
if [[ ! -f "$archive" ]]; then
  temporary_archive="$archive.partial"
  curl -fL \
    "https://github.com/ggml-org/llama.cpp/archive/$llama_revision.tar.gz" \
    -o "$temporary_archive"
  printf '%s  %s\n' "$llama_sha256" "$temporary_archive" | sha256sum --check --status
  mv "$temporary_archive" "$archive"
fi
printf '%s  %s\n' "$llama_sha256" "$archive" | sha256sum --check

if [[ ! -f "$llama_dir/CMakeLists.txt" ]]; then
  if [[ -e "$llama_dir" ]]; then
    echo "$llama_dir exists but is incomplete; move it aside and rerun" >&2
    exit 1
  fi
  temporary_dir="$(mktemp -d "$strata_dir/third_party/.llama.cpp.XXXXXX")"
  trap 'rm -rf -- "$temporary_dir"' EXIT
  tar -xzf "$archive" --strip-components=1 -C "$temporary_dir"
  mv "$temporary_dir" "$llama_dir"
  trap - EXIT
fi

HIP_PATH="$rocm_root" ROCM_PATH="$rocm_root" \
cmake -S "$strata_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER="$rocm_root/lib/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$rocm_root" \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR="$llama_dir"

cmake --build "$build_dir" --target strata --parallel "$build_jobs"

mkdir -p "$build_dir/tuning"
cp "$strata_dir/tools/hip/gfx1151-hipblaslt-100500.txt" "$build_dir/tuning/"

if [[ ! -x "$strata_dir/.venv/bin/python" ]]; then
  UV_CACHE_DIR="$uv_cache" uv venv --python 3.12 "$strata_dir/.venv"
fi
UV_CACHE_DIR="$uv_cache" uv pip install \
  --python "$strata_dir/.venv/bin/python" \
  -r "$strata_dir/requirements.txt"

"$build_dir/strata" --version
