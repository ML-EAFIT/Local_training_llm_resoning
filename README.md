# LLM pipeline: from raw web text to a chat model, on 2× RTX 2070 Super

This pipeline follows the lecture notes in `new_lessons_transformers/` and
`old_lessons_transformers/`. It trains a decoder-only Transformer from scratch and
post-trains it into a small assistant. Every stage is measured: data, tokenizer,
hardware scaling, scaling laws, training dynamics, each model component,
generation, long context, each post-training step, and energy.

```
1_data_extraction → 2_data_clean → 3_tokenization → 4_pre_train → 5_generation → 6_mid_training → 7_post_training
                                                      (sweep, DDP, probes)                        SFT → RM → DPO → GRPO → self-distill
```

## Run

```bash
bash run_pipeline.sh --smoke        # tiny end-to-end check, ~20-30 min, writes work_smoke/
tmux new -s llm 'bash run_pipeline.sh'   # full run, ~24 h, writes work/
FROM=6 bash run_pipeline.sh         # rerun from stage 6 onward (earlier stages are skipped if done)
python3 -m pytest -q tests          # unit tests (model, losses, MinHash, engine)
python3 5_generation/chat.py --ckpt work/7_post_training/final.pt         # chat with the final model
```

Every step writes a marker to `work/markers/`, and long trainings checkpoint every 20–30 minutes,
so after a crash or power loss you can just run `run_pipeline.sh` again.
To leave headroom for machine stability, each process caps its VRAM at 85% (`LLM_GPU_MEM_FRAC`) and CPU
pools use 24 workers (`LLM_WORKERS`). The report is `work/report/index.html`, a single file that works offline.

Useful environment variables: `LLM_WORK` (output directory), `GRID_CO2` (kg/kWh, default 0.40),
`KWH_PRICE` (default 0.20), `SWEEP_SCALE` (scales the scaling-sweep budgets),
`GPU_PEAK_TFLOPS` (MFU denominator, default 36).

## What each lesson concept maps to

| Lesson concept | Where |
|---|---|
| Web/code/math/wiki mix, high-quality mid-training pool | `1_data_extraction/extract.py` |
| Heuristic filters, PII scrub, exact + MinHash-LSH dedup, eval decontamination | `2_data_clean/` |
| Byte-level BPE, special/role/think tokens, bytes-per-token, vocab trade-off | `3_tokenization/` |
| Decoder-only: RoPE, GQA, pre-norm RMSNorm, SwiGLU, sliding window, sparse MoE + aux loss | `common/model.py` |
| Teacher forcing, cross-entropy, bits per byte, WSD schedule, fp16 mixed precision | `common/trainer.py`, `4_pre_train/pretrain.py` |
| Data parallelism (DDP), gradient accumulation with `no_sync`, ZeRO-1, MFU | `common/dist.py`, `4_pre_train/ddp_benchmark.py` |
| Kaplan / Chinchilla scaling laws, IsoFLOP, ablations (positions, MHA/GQA/MQA, norms, MoE) | `4_pre_train/scaling_sweep.py`, `fit_scaling.py` |
| Layer/head ablation, logit lens, attention maps/entropy, embedding analogies | `eval/probe.py` |
| KV cache, greedy/beam/top-k/top-p/temperature, guided decoding, speculative decoding | `5_generation/engine.py` |
| Mid-training, YaRN RoPE scaling, curriculum, best-fit packing, needle-in-a-haystack | `6_mid_training/` |
| SFT with loss only on response tokens, LoRA on attention + MLP | `7_post_training/sft.py`, `common/lora.py` |
| Bradley-Terry reward model, Best-of-N | `7_post_training/reward_model.py`, `best_of_n.py` |
| DPO (β = 0.1, frozen reference) | `7_post_training/dpo.py` |
| RLVR with GRPO, Dr.GRPO/DAPO fixes, asymmetric clip, k3 KL | `7_post_training/grpo.py`, `math_tasks.py` |
| On-policy self-distillation, reverse KL | `7_post_training/self_distill.py` |
| Benchmarks, pass@k, cons@k, LLM-as-judge with position swap | `eval/` |
| Energy: NVML (measured), CPU via RAPL or a TDP estimate, CO₂, cost | `common/energy.py` |

## Hardware notes

- Turing GPUs have no bf16, so training uses fp16 autocast with a GradScaler and fp32 master weights.
- The two GPUs talk over PCIe across NUMA nodes. With gradient accumulation there is one all-reduce per step,
  so communication stays under 1% of step time.
- RAPL (`/sys/class/powercap`) is not readable without root here, so CPU energy is an estimate
  (utilisation × TDP). The report labels it as an estimate. GPU energy comes from the NVML hardware counter.

## Honest expectations

A ~124M-parameter model trained on ~3B tokens ends up near GPT-2-small:
- fluent text, basic chat and instruction following
- small arithmetic with a chain of thought after RL
- short Python completions

HumanEval and GSM8K stay low, and it is not a general assistant. The report shows exactly where it lands.
