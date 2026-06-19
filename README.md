# llama.cpp gfx906 Slot Save/Restore

> Fork of [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) with automatic slot state persistence for seamless model hot-swapping.

## What This Adds

This fork patches two upstream PRs onto the gfx906 furnace branch:

| PR | Feature | Status |
|----|---------|--------|
| [#20819](https://github.com/ggml-org/llama.cpp/pull/20819) | Checkpoint persistence for hybrid/recurrent models | ✅ Adapted |
| [#20822](https://github.com/ggml-org/llama.cpp/pull/20822) | Auto save/restore slots on SIGTERM/startup | ✅ Adapted + fixed |

### What It Does

- **Auto-save**: When the server receives SIGTERM (model swap, TTL expiry, shutdown), slot state + checkpoints are automatically saved to disk
- **Auto-restore**: When the server starts, previously saved slot state is automatically restored
- **Crash-safe**: Atomic write (temp file → rename) prevents corrupted saves if killed mid-write

### Performance Impact

| Scenario | Cold Start | After Restore | Speedup |
|----------|-----------|---------------|---------|
| 24-token prompt | 941ms | 439ms | **53% faster** |
| Cache hit rate | 0 | 20/24 tokens | ✅ |

## Hardware Requirements

- **GPU**: AMD Radeon Pro VII (MI50) — gfx906 architecture, 2× 16GB HBM2
- **CPU**: AMD Ryzen 5 5600 or similar (12 threads)
- **RAM**: 46GB+
- **OS**: Ubuntu 24.04 LTS
- **ROCm**: 7.1.1 (recommended — 7.2.3 has garbled output regression)

## Build

```bash
# Install ROCm 7.1.1 (see gfx906 wiki for details)
# https://skyne98.github.io/wiki-gfx906/installing_ROCm_7.x.html

# Clone
git clone --recursive git@github.com:362132718/llamacpp-gfx906-slot-save.git
cd llamacpp-gfx906-slot-save

# Build
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_NO_VMM=ON
cmake --build . --target llama-server -j$(nproc)
```

## Usage with llama-swap

### llama-swap Config

```yaml
---
version: "1.13"
listen: :8080
env:
  TURBOPREFILL: "1"
  HSA_ENABLE_SDMA: "0"
models:
  Qwen3.6-27B.gguf:
    proxy: "http://127.0.0.1:${PORT}"
    cmd: |
      /path/to/build/bin/llama-server \
        -m /path/to/model.gguf \
        -ngl 999 --host 0.0.0.0 --port ${PORT} \
        -sm layer --tensor-split 48/52 -fa on \
        -c 131072 \
        --cache-type-k q4_0 --cache-type-v q4_0 \
        --threads 12 --threads-batch 12 \
        --api-key YOUR_API_KEY \
        --slot-save-path /path/to/slots/ \
        --slots
    ttl: 900
    maxLoadingTime: 60
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--slot-save-path /path/` | **Required** — enables auto save/restore |
| `--slots` | Enable slot API (required for save/restore) |
| `-sm layer` | Layer-split mode (required for TurboPrefill) |
| `--cache-type-k q4_0 --cache-type-v q4_0` | Compressed KV cache (gfx906 VRAM limited) |

### How It Works

```
Model loaded → auto_restore_slots() → restores from disk (if exists)
SIGTERM received → auto_save_slots() → saves to {slot_save_path}/{model_stem}
下次请求 → 启动 → auto_restore → cache hit → skip prefill
```

Each model saves independently:
```
/path/to/slots/
├── Qwen3.6-27B-UD-Q6_K_XL           # KV state
├── Qwen3.6-27B-UD-Q6_K_XL.checkpoints  # Recurrent state (hybrid models)
├── Qwen3.6-35B-A3B-...
└── ...
```

## gfx906 Known Limitations

| Limitation | Workaround |
|------------|------------|
| SDMA unstable | `HSA_ENABLE_SDMA=0` |
| HIP Graphs crash | Do not use `--gpu-layers` with graphs |
| mul_mat_q incompatible | Do not use `--no-mul-mat-q` |
| Optimal ubatch | `-ub 1024` (higher is slower) |
| Prefill ceiling | ~330 tok/s (PCIe 3.0 x8 bottleneck) |

## Credits

- [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) — Base fork with gfx906 kernel optimizations
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) — Custom FA/MMQ kernels
- [European-tech](https://github.com/European-tech) — PR #20819 and #20822 upstream
- [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap) — Model hot-swapping proxy

## License

Same as upstream llama.cpp (MIT).
