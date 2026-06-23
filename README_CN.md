# llama.cpp gfx906 Slot Save/Restore

> 基于 [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) 的 fork，增加了模型热切换时的 Slot 状态自动持久化功能。

## 这个 Fork 做了什么

在 gfx906 furnace 分支上合入了两个上游 PR 并做了适配和修复：

| PR | 功能 | 状态 |
|----|------|------|
| [#20819](https://github.com/ggml-org/llama.cpp/pull/20819) | 混合/递归模型的 Checkpoint 持久化 | ✅ 已适配 |
| [#20822](https://github.com/ggml-org/llama.cpp/pull/20822) | SIGTERM/启动时自动保存/恢复 Slot | ✅ 已适配 + 修复 |

### 核心功能

- **自动保存**：收到 SIGTERM（模型切换、TTL 过期、关机）时，自动将 Slot 状态 + Checkpoint 保存到磁盘
- **自动恢复**：服务器启动时，自动从磁盘恢复之前保存的 Slot 状态
- **崩溃安全**：采用原子写入（先写 .tmp 再 rename），防止写入中途被杀导致文件损坏
- **文件校验**：恢复前检查文件大小，损坏文件自动跳过不崩溃

### 性能提升

| 场景 | 冷启动 | 恢复后 | 提升 |
|------|--------|--------|------|
| 24 token prompt | 941ms | 439ms | **快 53%** |
| Cache 命中 | 0 | 20/24 tokens | ✅ |

## 硬件要求

- **GPU**：AMD Radeon Pro VII (MI50) — gfx906 架构，2× 16GB HBM2
- **CPU**：AMD Ryzen 5 5600 或同级别（12 线程）
- **内存**：46GB+
- **系统**：Ubuntu 24.04 LTS
- **ROCm**：7.1.1（推荐 — 7.2.3 有长 prompt 乱码回归）

## 编译

```bash
# 安装 ROCm 7.1.1（参考 gfx906 wiki）
# https://skyne98.github.io/wiki-gfx906/installing_ROCm_7.x.html

# 克隆
git clone --recursive git@github.com:362132718/llamacpp-gfx906-slot-save.git
cd llamacpp-gfx906-slot-save

# 编译
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_NO_VMM=ON
cmake --build . --target llama-server -j$(nproc)
```

## 配合 llama-swap 使用

### llama-swap 配置示例

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

### 关键参数说明

| 参数 | 作用 |
|------|------|
| `--slot-save-path /path/` | **必填** — 启用自动保存/恢复 |
| `--slots` | 启用 Slot API（保存/恢复必需） |
| `-sm layer` | Layer-split 模式（TurboPrefill 必需） |
| `--cache-type-k q4_0 --cache-type-v q4_0` | 压缩 KV 缓存（gfx906 显存有限） |

### 工作流程

```
模型加载 → auto_restore_slots() → 从磁盘恢复（如果存在）
收到 SIGTERM → auto_save_slots() → 保存到 {slot_save_path}/{model_stem}
下次请求 → 启动 → auto_restore → cache 命中 → 跳过 prefill
```

每个模型独立保存：
```
/path/to/slots/
├── Qwen3.6-27B-UD-Q6_K_XL           # KV 状态（~158MB）
├── Qwen3.6-27B-UD-Q6_K_XL.checkpoints  # 循环层状态（混合模型必需，~156MB）
├── Qwen3.6-35B-A3B-...
└── ...
```

## gfx906 已知限制

| 限制 | 解决方案 |
|------|----------|
| SDMA 不稳定 | `HSA_ENABLE_SDMA=0` |
| HIP Graphs 崩溃 | 不要使用 HIP Graphs |
| mul_mat_q 不兼容 | 不要使用 `--no-mul-mat-q` |
| 最优 ubatch | `-ub 1024`（更大反而更慢） |
| 预填充天花板 | ~330 tok/s（PCIe 3.0 x8 带宽瓶颈） |

## 致谢

- [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace) — 基础 Fork，包含 gfx906 内核优化
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) — 自定义 FA/MMQ 内核
- [European-tech](https://github.com/European-tech) — 上游 PR #20819 和 #20822 作者
- [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap) — 模型热切换代理

## 许可证

与上游 llama.cpp 一致（MIT）。
