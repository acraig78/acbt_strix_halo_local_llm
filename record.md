# Strix Halo llama.cpp — Benchmark Record

Results from setting up and benchmarking llama.cpp on this machine, per `PLAN.md`.

**Hardware:** AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), Radeon 8060S iGPU, 124GiB RAM visible.
**Software:** Omarchy (Arch), kernel `7.1.9-arch1-2`, llama.cpp tip-of-main commit `64e9bceb2`
(build 10858), `amdgpu.gttsize=114688 ttm.pages_limit=29360128` (~112GiB GTT ceiling).

---

## Environment verification

- `vulkaninfo --summary`: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `driverID = DRIVER_ID_MESA_RADV`.
- `rocminfo | grep -i gfx`: `gfx1151`, detected natively (rocm-hip-sdk 7.2.4), no
  `HSA_OVERRIDE_GFX_VERSION` needed.
- `ggml_cuda_init` (HIP backend) reports the full 114688 MiB GTT ceiling visible to ROCm, not just
  Vulkan — confirms the Phase 1 GTT config applies to both backends.
- `sbctl verify`: UKI (`/boot/EFI/Linux/omarchy_linux.efi`) signed correctly after the
  `limine-mkinitcpio` rebuild.

## Benchmark results

All runs: `llama-bench -m <model> -ngl 999 -fa 1`, full GPU offload, flash attention on.

### Sanity test — Qwen2.5-7B-Instruct (Q4_K_M, bartowski)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 1363.58 ± 4.60 t/s | 48.06 ± 0.15 t/s |
| ROCm/HIP | 1578.81 ± 21.11 t/s | 46.68 ± 0.04 t/s |
| ROCm/HIP + `ROCBLAS_USE_HIPBLASLT=1` | 1581.07 ± 15.20 t/s | 46.67 ± 0.05 t/s |

`ROCBLAS_USE_HIPBLASLT=1` made no measurable difference at 7B scale (within noise) — likely only
matters for larger matmuls.

### Gemma 4 31B (dense, `UD-Q6_K_XL`, 25.62 GiB, 30.70B params)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 224.49 ± 0.84 t/s | 7.73 ± 0.02 t/s |
| ROCm/HIP | 297.89 ± 1.74 t/s | 7.66 ± 0.02 t/s |

ROCm wins prefill by ~33%; generation is a tie (within noise). **Backend chosen: HIP.**

Sanity check against theoretical memory-bandwidth ceiling: Strix Halo's ~256GB/s bandwidth ÷
25.62GiB model size ≈ 10 t/s theoretical max for a bandwidth-bound dense model at generation time.
7.73 t/s is ~77% of that ceiling — reasonable real-world efficiency.

### Gemma 4 26B-A4B (MoE, `UD-Q6_K_XL`, 21.68 GiB, 25.23B total / ~3.8B active, vision-capable)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 1091.02 ± 11.44 t/s | 49.37 ± 0.13 t/s |
| ROCm/HIP | 1131.86 ± 31.07 t/s | 44.42 ± 0.04 t/s |

ROCm wins prefill by ~3.7%; Vulkan wins generation by ~11%. **Backend chosen: Vulkan** (generation
throughput matters more for interactive/agentic use).

### Qwen3.6-35B-A3B (MoE + Gated-DeltaNet hybrid, `UD-Q6_K_XL`, 29.65 GiB, 34.66B params)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 1019.51 ± 10.52 t/s | 55.75 ± 0.37 t/s |
| ROCm/HIP | 952.22 ± 15.25 t/s | 48.33 ± 0.12 t/s |

Vulkan wins both metrics — the only model of the four where this happened. **Backend chosen:
Vulkan.**

**Finding — DeltaNet is GPU-accelerated on this build, on both backends.** Earlier public
information (as of the plan's initial draft) held that llama.cpp's Vulkan backend had no compute
shader for the `GATED_DELTA_NET` op (all such layers running on CPU — see
[ggml-org/llama.cpp#20354](https://github.com/ggml-org/llama.cpp/issues/20354)), and that the
ROCm/HIP kernel, while present, was untuned for gfx1151 and roughly CPU-speed. Neither holds on
this build: the Phase 3 Vulkan build log shows `Generate vulkan shaders for gated_delta_net.comp`,
and the numbers above are clean GPU-bound throughput on both backends — in fact *higher* than the
plain-MoE Gemma 4 26B-A4B comparison model, not lower. Re-verify after future
`git pull && rebuild` cycles, since gfx1151 kernel work is still active upstream and this could
regress or further improve.

### Qwen3.8-27B (dense-ish + Gated-DeltaNet hybrid, `UD-Q6_K_XL`, 23.55 GiB, 27.32B params, vision-capable)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 206.89 ± 0.39 t/s | 8.67 ± 0.01 t/s |
| ROCm/HIP | 343.43 ± 0.24 t/s | 8.67 ± 0.00 t/s |

ROCm wins prefill decisively (~+66%); generation is an exact tie. Same pattern as Gemma 4 31B
(also dense-ish) rather than the sparse-MoE pattern seen with Qwen3.6-35B-A3B and Gemma 4 26B-A4B —
consistent with this model having little/no MoE sparsity despite sharing Qwen3.6's DeltaNet hybrid
architecture. Throughput is in the same range as Gemma 4 31B dense (207-343 t/s prefill, 8.67 t/s
gen vs. Gemma's 224-298 t/s prefill, 7.7 t/s gen), confirming clean GPU-bound inference — not the
CPU-fallback speed the architecture would produce without DeltaNet GPU support (a CPU-only 27B
would be well under 1 t/s). **Backend chosen: HIP.**

### Qwen3-Coder-30B-A3B (plain MoE, no DeltaNet, `UD-Q6_K_XL`, 24.53 GiB, 30.53B params)

| Backend | Prefill (pp512) | Gen (tg128) |
|---|---|---|
| Vulkan | 1118.66 ± 5.79 t/s | 66.75 ± 0.07 t/s |
| ROCm/HIP | 997.27 ± 16.80 t/s | 57.92 ± 0.04 t/s |

Vulkan wins both metrics (prefill +12.2%, gen +15.2%) — same pattern as Qwen3.6-35B-A3B. **Backend
chosen: Vulkan.** This is the best generation throughput of any model benchmarked on this machine —
consistent with it being a clean plain-MoE with no DeltaNet overhead.

## Cross-model comparison

Same quant (`UD-Q6_K_XL`), same backend where possible, full GPU offload:

| Model | Size | Total params | Active params | Best prefill | Best gen |
|---|---|---|---|---|---|
| Gemma 4 31B (dense) | 25.62 GiB | 30.70B | 30.70B | 297.89 t/s (HIP) | 7.73 t/s (Vulkan) |
| Gemma 4 26B-A4B (MoE) | 21.68 GiB | 25.23B | ~3.8B | 1131.86 t/s (HIP) | 49.37 t/s (Vulkan) |
| Qwen3.8-27B (dense-ish+DeltaNet) | 23.55 GiB | 27.32B | ~27B | 343.43 t/s (HIP) | 8.67 t/s (either) |
| Qwen3.6-35B-A3B (MoE+DeltaNet) | 29.65 GiB | 34.66B | ~3B | 1019.51 t/s (Vulkan) | 55.75 t/s (Vulkan) |
| Qwen3-Coder-30B-A3B (plain MoE) | 24.53 GiB | 30.53B | ~3B | 1118.66 t/s (Vulkan) | 66.75 t/s (Vulkan) |

MoE sparse activation gives roughly 5-6x the generation throughput of the dense/dense-ish models
despite similar on-disk size, because only a few billion of the total parameters are active per
token — dramatically reducing the memory bandwidth this bandwidth-bound iGPU has to move per
forward pass. On this hardware, prefer MoE architectures over dense ones whenever output quality is
comparable. Qwen3.8-27B shows this clearly: despite sharing Qwen3.6-35B-A3B's DeltaNet hybrid
architecture, its lack of MoE sparsity puts its throughput in the same range as the fully-dense
Gemma 4 31B, not the sparse-MoE Qwen3.6/Gemma-26B-A4B numbers.

## Backend selection summary

| Model | Backend | Reason |
|---|---|---|
| Gemma 4 31B | HIP | Wins prefill by ~33%; gen tied |
| Gemma 4 26B-A4B | Vulkan | Wins gen by ~11% |
| Qwen3.6-35B-A3B | Vulkan | Wins both prefill and gen |
| Qwen3.8-27B | HIP | Wins prefill by ~66%; gen exact tie |
| Qwen3-Coder-30B-A3B | Vulkan | Wins prefill by ~12%, gen by ~15% |

Pattern observed: **dense/dense-ish models favor HIP** (wins prefill decisively, generation tied);
**sparse-MoE models mostly favor Vulkan** — clearly for both Qwen MoE models, for generation only
on Gemma's MoE (ROCm edges prefill there). No universal winner; benchmark each model. Both plain-MoE
Qwen models (Qwen3.6-35B-A3B and Qwen3-Coder-30B-A3B) land in a similar performance class
(~1000-1100 t/s prefill, ~55-67 t/s gen on Vulkan) despite one carrying DeltaNet hybrid attention
and the other not — reinforcing that DeltaNet itself is no longer a meaningful bottleneck on this
build, sparsity is what drives throughput.

No single backend wins uniformly across models on this hardware — always benchmark each model
individually rather than assuming one backend is categorically better.

## Context-window sweep (2026-09-09)

Question: how far can `-c` (context size) be pushed on this hardware before running out of GTT/RAM?
Method: stopped `llama-swap`, launched each model directly with its chosen backend/flags from
`config.yaml`, and tried `-c` at that model's native `n_ctx_train` (262144 for all five GGUFs, per
metadata) first, rather than binary-searching up from the old config values — every model loaded
and served successfully on the first attempt, so no search was needed.

| Model | n_ctx_train | Old `-c` | New `-c` | Limiting factor | Peak GTT / RAM at max context |
|---|---|---|---|---|---|
| Gemma 4 31B | 262144 | 65536 | 262144 | native max reached, not OOM | ~50GiB GTT / 63GiB RAM |
| Gemma 4 26B-A4B | 262144 | 65536 | 262144 | native max reached | ~29GiB GTT / 40GiB RAM |
| Qwen3.6-35B-A3B | 262144 | 32768 | 262144 | native max reached | ~34GiB GTT / 44GiB RAM |
| Qwen3.8-27B | 262144 | 32768 | 262144 | native max reached | ~33GiB GTT / 45GiB RAM |
| Qwen3-Coder-30B-A3B | 262144 | 65536 | 262144 | native max reached | ~49GiB GTT / 60GiB RAM |

**Finding: this machine's 112GiB GTT ceiling / 124GiB RAM is not the constraint for any of these
five models at full native context.** The worst case (Gemma 4 31B, the largest dense model) used
only ~50GiB of the 112GiB GTT budget — under half. Even with the heaviest model loaded at max
context, there's ~60GiB+ of headroom left, enough to load a second smaller model concurrently. The
binding constraint is each model's own trained context length (262144 tokens), not system
resources.

`config.yaml` and `opencode.json`'s `limit.context` were updated to `262144` for all five models as
a result (previous values — 32768/65536 — were an unmeasured caution, not an observed limit).
Pushing past 262144 would require RoPE/YaRN context extension (quality trade-off, untested here),
not more memory.
