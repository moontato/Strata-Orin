# Native ARM64 / CUDA SM87 build

**Commands to execute on the AGX Orin, not commands validated on this machine.** Target the fork's known development stack: JetPack 6.2, Linux aarch64, CUDA 12.6, GCC/G++ 11. Other JetPack/CUDA combinations are unqualified. A board sold as 64 GB reports somewhat less usable RAM; use actual values, not the product label.

## 1. Check the installed compute stack

```sh
uname -m                         # must be aarch64
head -n 1 /etc/nv_tegra_release
/usr/local/cuda-12.6/bin/nvcc --version
/usr/bin/g++-11 --version
free -h
sudo nvpmodel -q                 # record, do not silently change the power mode
```

Install **JetPack's own development packages** (including nvcc, CUDA headers, cuBLAS and the driver API development library) if missing, using NVIDIA's JetPack instructions for this L4T release. Do not install a desktop NVIDIA driver, CUDA 13, an SBSA driver stack, or desktop CUDA Python wheels over JetPack. `/usr/local/cuda-12.6` below must identify that installed toolkit, not an arbitrary newer `/usr/local/cuda` symlink. Do not link/run against a `libcuda` stub.

Host tools and a CMake new enough for Strata (Ubuntu 22.04's default CMake may be too old):

```sh
sudo apt-get update
sudo apt-get install -y build-essential gcc-11 g++-11 git ninja-build \
  python3-venv python3-pip curl
python3 -m venv "$HOME/strata-build-tools"
"$HOME/strata-build-tools/bin/python" -m pip install 'cmake==3.28.4'
export PATH="$HOME/strata-build-tools/bin:$PATH"
cmake --version                    # Strata requires >= 3.24
```

No Python inference framework is required to build or benchmark the C++ engine. These commands do not install the Python web server or vision dependencies.

## 2. Transfer the prepared branch to `/mnt/ssd/Strata-Orin`

This branch already contains the four patches. **Do not run `git am` again on it.** It has not been pushed to GitHub; cloning the fork's default branch would get the wrong source.

To transfer the committed branch without pushing, run this on the development laptop (after reviewing the local commits):

```sh
git bundle create "$HOME/Strata-Orin-v0.1.45.bundle" \
  jetson-orin-v0.1.45 refs/tags/v0.1.45
git bundle verify "$HOME/Strata-Orin-v0.1.45.bundle"
```

Copy that bundle to the Orin. On the Orin, choose an SSD location writable by your account. Do not overwrite an existing checkout:

```sh
export WORK=/mnt/ssd/Strata-Orin
test ! -e "$WORK" || { echo "$WORK already exists; inspect it first"; exit 1; }
# Replace /path/to with the location of the copied bundle.
git clone --branch jetson-orin-v0.1.45 /path/to/Strata-Orin-v0.1.45.bundle "$WORK"
cd "$WORK"
export PATCHDIR="$WORK/ports/jetson-orin-v0.1.45"
test "$(git branch --show-current)" = jetson-orin-v0.1.45
test "$(git rev-parse 'v0.1.45^{commit}')" = 477ac13cbe8e582c20484871ce459d20d82df09d
git merge-base --is-ancestor 477ac13cbe8e582c20484871ce459d20d82df09d HEAD
test -z "$(git status --porcelain)"
python3 "$PATCHDIR/check_preservation.py" --repo "$WORK"
git log --oneline 477ac13cbe8e582c20484871ce459d20d82df09d..HEAD
```

The clone's `origin` initially points to the local bundle, not GitHub. After the owner reviews and publishes this branch, cloning **`moontato/Strata-Orin` with `--branch jetson-orin-v0.1.45`** is an alternative. Do not use a default-branch clone or `git pull` from upstream main.

For a **patch-only reproduction**, use a separate pristine checkout, not the prepared branch. Keep `PATCHDIR` pointing at this bundle outside that checkout:

```sh
git clone https://github.com/Niko1221/Strata.git "$HOME/strata-0145-patch-replay"
cd "$HOME/strata-0145-patch-replay"
git switch --detach 477ac13cbe8e582c20484871ce459d20d82df09d
test -z "$(git status --porcelain)"
while IFS= read -r patch; do git am "$PATCHDIR/$patch" || exit 1; done < "$PATCHDIR/series"
python3 "$PATCHDIR/check_preservation.py" --repo "$PWD"
cd "$WORK"
```

Do not check out current main, rebase onto 0.2, replace `src/` from the older Orin fork, or run the unported one-click setup/update scripts.

## 3. Pin the same ggml dependency as v0.1.45

The vendored block headers and codebooks are **not** a replacement for the full ggml sources. The release uses a pinned llama.cpp checkout both for CPU support and GPU MMQ template instantiation. Fetch this exact revision; do not substitute latest llama.cpp.

```sh
export LLAMA_SRC=/mnt/ssd/llama-strata-pin
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
git -C "$LLAMA_SRC" checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
```

Alternatively omit `STRATA_GGML_DIR` below and CMake will fetch that same pinned commit. No changes to the dependency are part of the patch series.

## 4. Configure and build natively

Use a fresh build directory. Two compile jobs are a conservative starting point, **not a measured optimum**. Stop other memory-heavy services with their owner's approval. Do not rely on swap to establish GPU-resident capacity.

```sh
cd "$WORK"
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build-orin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_PORTABLE=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_BUILD_TESTS=ON

cmake --build build-orin --target strata strata-q36 strata-q36ref \
  strata-device strata_kq -j2
# Compile the test/diagnostic targets too. A failure is a port failure to investigate,
# not a reason to switch off MMQ/native experts or to use a CPU inference backend.
cmake --build build-orin -j2
```

**v0.1.45 option naming:** the CMake cache option `STRATA_PREFILL_MMQ` controls the **HIP-only** path. Setting it to `ON` with HIP disabled is a configure error. `OFF` above does **not** disable CUDA MMQ: the CUDA branch builds `strata_mmq` and defines `STRATA_PREFILL_MMQ=1` for `strata_prefill` when `STRATA_NATIVE_EXPERTS=ON`. `STRATA_MMQ_KQUANTS=ON` builds the separate CUDA K-quant library. Verify the MMQ startup log and CUDA trace; do not substitute FP16 prefill for the required MMQ run.

`libstrata_kq.so` must remain next to the engine: v0.1.45 dynamically loads its K-quant MMQ prompt library. Shipping only `strata`/`strata-q36` can silently select a slower **GPU** FP16 path. Preserve and verify the shared library. `strata-q36ref` is a validation reference, not the throughput target.

Do **not** use `STRATA_NATIVE_EXPERTS=OFF`, disable MMQ/K-quants, force old kernels, disable graphs, force small chunks, or turn on CPU offload merely to make the build run. Diagnostic eager/old-kernel runs in VERIFY.md are separate correctness experiments, never benchmark results.

## 5. Compile standalone capability and memory tests

These do not need a model or the Python server. The probe uses both runtime and driver APIs. The selftest uses managed allocation only for a sequential-ownership check; production allocations remain unchanged.

```sh
mkdir -p build-orin/diagnostics
/usr/bin/g++-11 -O2 -std=c++20 -Iinclude -I"$CUDA_HOME/include" \
  tools/cuda_platform_probe.cpp src/platform/memory.cpp \
  -L"$CUDA_HOME/lib64" -lcudart -lcuda \
  -o build-orin/diagnostics/strata-cuda-probe
"$CUDA_HOME/bin/nvcc" -O2 -std=c++17 -arch=sm_87 -ccbin /usr/bin/g++-11 \
  -Iinclude tools/cuda_memory_selftest.cu -lcuda \
  -o build-orin/diagnostics/strata-cuda-memory-test
```

JetPack's system library paths should supply the real `libcuda.so`; if the development linker symlink is missing, repair the JetPack development installation rather than adding a stub directory to the runtime library path.

## 6. Obtain a release-supported Qwen3.6 GGUF

Start with the release's pinned **UD-Q4_K_XL** (22,853,663,008 bytes). Loading this file uses Qwen3.6's native GGUF path, not Flash-Next packing. Download requires about 23 GB plus build/storage headroom. Do not infer final runtime RAM demand from the file size: packed decode copies, sessions and prompt buffers add allocations.

```sh
mkdir -p /mnt/ssd/models/qwen36
export MODEL=/mnt/ssd/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
curl -fL --retry 3 -o "$MODEL" \
  'https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
printf '%s  %s\n' \
  55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1 \
  "$MODEL" | sha256sum -c -
```

The revision, size and digest are from v0.1.45 `setup.py`. Keep the same GGUF for the llama.cpp comparison. UD-Q6_K_XL and Q8_0 remain in the code, but do not assume their larger runtime footprints fit until measured.
