# llama.cpp gfx906 Slot Save/Restore

> A fork of [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) with automatic Slot state persistence for model hot-swapping.

## What This Fork Does

This fork merges two upstream PRs into the gfx906 furnace branch, with adaptation and bug fixes:

| PR | Feature | Status |
|----|---------|--------|
| [#20819](https://github.com/ggml-org/llama.cpp/pull/20819) | Checkpoint persistence for hybrid/recursive models | ✅ Adapted |
| [#20822](https://github.com/ggml-org/llama.cpp/pull/20822) | Auto save/restore Slots on SIGTERM/startup | ✅ Adapted + Fixed |

### Core Features

- **Auto Save**: On SIGTERM (model switch, TTL expiry, shutdown), automatically saves Slot state + Checkpoint to disk
- **Auto Restore**: On server startup, automatically restores previously saved Slot state from disk
- **Crash Safe**: Uses atomic writes (write to .tmp then rename) to prevent file corruption if killed mid-write
- **File Validation**: Checks file size before restoring; corrupted files are skipped without crashing

### Performance Improvement

| Scenario | Cold Start | After Restore | Improvement |
|----------|-----------|---------------|-------------|
| 24 token prompt | 941ms | 439ms | **53% faster** |
| Cache hit | 0 | 20/24 tokens | ✅ |

## Hardware Requirements

- **GPU**: AMD Radeon Pro VII (MI50) — gfx906 architecture, 2× 16GB HBM2
- **CPU**: AMD Ryzen 5 5600 or equivalent (12 threads)
- **RAM**: 46GB+
- **OS**: Ubuntu 24.04 LTS
- **ROCm**: 7.1.1 (recommended — 7.2.3 has long prompt garbled output regression)

## Build

```bash
# Install ROCm 7.1.1 (see gfx906 wiki)
# https://skyne98.github.io/wiki-gfx906/installing_ROcm_7.x.html

# Clone
git clone --recursive git@github.com:362132718/llamacpp-gfx906-slot-save.git
cd llamacpp-gfx906-slot-save

# Build
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_NO_VMM=ON
cmake --build . --target llama-server -j$(nproc)
```

## Using with llama-swap

### llama-swap Configuration Example

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

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `--slot-save-path /path/` | **Required** — Enables auto save/restore |
| `--slots` | Enables Slot API (required for save/restore) |
| `-sm layer` | Layer-split mode (required for TurboPrefill) |
| `--cache-type-k q4_0 --cache-type-v q4_0` | Compressed KV cache (saves VRAM on gfx906) |
| `-fa on` | Flash Attention (required for slot save/restore) |
| `-ub 1024` | Micro-batch size (gfx906 optimal) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slot restore fails | Check file permissions in `--slot-save-path` directory |
| Model OOM on GPU 0 | Adjust `--tensor-split` (e.g., 45/55 instead of 50/50) |
| Long prompt garbled output | Use ROCm 7.1.1, avoid 7.2.3 |
| SDMA crash | Set `HSA_ENABLE_SDMA=0` |

## Credits

- [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) — GFX906 performance optimizations
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Upstream project
- PR #20819 and #20822 — Slot persistence features
