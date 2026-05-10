#!/usr/bin/env bash
# Bootstrap the surgical LD_LIBRARY_PATH for runtime-required
# system libraries that nix's devenv profile masks. Two
# concerns are handled here:
#
#  1. JVM (DeepOnto) — openjdk-11's libzip.so depends on
#     libz.so.1.
#  2. CUDA (torch / GRPO P5) — torch's CUDA runtime depends on
#     libcuda.so.1 (the driver shim) and libnvidia-ml.so.1
#     (NVML, used by torch.cuda.is_available()).
#
# Both are present at /usr/lib/x86_64-linux-gnu/ but masked by
# devenv. Pointing LD_LIBRARY_PATH at all of /usr/lib would also
# load a system libc that breaks the nix-built numpy/torch
# wheels (GLIBC version mismatch). The fix: symlink only the
# specific libs into a private dir and point LD_LIBRARY_PATH
# there.
#
# Source this script before invoking any Python that needs
# either JVM or CUDA — the Justfile recipes do so automatically.

set -euo pipefail

JVM_LIBS_DIR="${JVM_LIBS_DIR:-/tmp/jvm-libs}"

# System paths to the masked libraries.
declare -A SURGICAL_LIBS=(
    [libz.so.1]=/usr/lib/x86_64-linux-gnu/libz.so.1
    [libcuda.so.1]=/usr/lib/x86_64-linux-gnu/libcuda.so.1
    [libnvidia-ml.so.1]=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
)

mkdir -p "$JVM_LIBS_DIR"
for libname in "${!SURGICAL_LIBS[@]}"; do
    src="${SURGICAL_LIBS[$libname]}"
    dst="$JVM_LIBS_DIR/$libname"
    if [ ! -e "$src" ]; then
        # libnvidia-ml.so.1 is optional — log and skip rather than fail
        case "$libname" in
            libcuda.so.1|libz.so.1)
                echo "setup_jvm_env: required $src not found; cannot bootstrap" >&2
                exit 1
                ;;
            *)
                echo "setup_jvm_env: optional $src not found; skipping" >&2
                continue
                ;;
        esac
    fi
    if [ ! -L "$dst" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
    fi
done

export LD_LIBRARY_PATH="${JVM_LIBS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export JVM_MEMORY="${JVM_MEMORY:-4g}"
