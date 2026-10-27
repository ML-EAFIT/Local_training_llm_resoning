#!/usr/bin/env bash
# End-to-end pipeline: 1_data_extraction -> ... -> 7_post_training -> eval -> report.
# Every stage writes a done-marker, so re-running resumes after a crash or power loss.
#   bash run_pipeline.sh            full run (~24 h on 2x RTX 2070 Super)
#   bash run_pipeline.sh --smoke    tiny end-to-end run (~20 min)
#   FROM=4 bash run_pipeline.sh     (re)start from stage 4 (earlier markers are kept)
#   TO=7 bash run_pipeline.sh       stop after stage 7 (only stages not yet done run)
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
if [[ "${1:-}" == "--smoke" ]]; then export LLM_SMOKE=1; fi
if [[ "${LLM_SMOKE:-0}" == "1" ]]; then WORK=${LLM_WORK:-$ROOT/work_smoke}; else WORK=${LLM_WORK:-$ROOT/work}; fi
export LLM_WORK=$WORK CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK/markers" "$WORK/logs"
PY=python3
TR="torchrun --standalone --nproc_per_node=2"
FROM=${FROM:-0}
TO=${TO:-99}

step() {  # step <name> <stage-number> <command...>
  local name=$1 num=$2; shift 2
  local marker="$WORK/markers/$name.done"
  if (( num > TO )); then echo "== skip $name (after stage $TO)"; return; fi
  if [[ -f "$marker" && "$num" -lt "$FROM" ]] || [[ -f "$marker" && "$FROM" == 0 ]]; then
    echo "== skip $name (done)"; return
  fi
  echo "== $(date +%H:%M:%S) start $name"
  local t0=$(date +%s)
  set +e
  "$@" 2>&1 | { grep --line-buffered -v -E "UserWarning|warnings.warn|max_autotune_gemm" || true; } \
    | tee "$WORK/logs/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then echo "!! $name failed (exit $rc), see $WORK/logs/$name.log"; exit $rc; fi
  echo "$(( $(date +%s) - t0 ))" > "$marker"
  echo "== $(date +%H:%M:%S) done $name in $(( $(date +%s) - t0 ))s"
}

disk_check() {
  local free=$(df --output=avail -BG "$WORK" | tail -1 | tr -dc 0-9)
  if (( free < 8 )); then echo "!! only ${free}G free on disk, stopping"; exit 1; fi
}

# 1-3: data
step extract        1 $PY 1_data_extraction/extract.py;           disk_check
step clean          2 $PY 2_data_clean/clean.py;                  disk_check
step tokenizer      3 $PY 3_tokenization/train_tokenizer.py
step tokenize       3 $PY 3_tokenization/tokenize_shards.py;      disk_check
step tokenizer_eval 3 $PY 3_tokenization/eval_tokenizer.py

# 4: hardware benchmark, scaling sweep, main pre-training, probes
step bench_single   4 $PY 4_pre_train/ddp_benchmark.py --phase single
step bench_ddp      4 $TR 4_pre_train/ddp_benchmark.py --phase ddp
step bench_pick     4 $PY 4_pre_train/ddp_benchmark.py --phase pick
step sweep          4 $PY 4_pre_train/scaling_sweep.py
step fit_scaling    4 $PY 4_pre_train/fit_scaling.py
step pretrain       4 $TR 4_pre_train/pretrain.py;                disk_check
step fit_scaling2   4 $PY 4_pre_train/fit_scaling.py
step probe_base     4 $PY eval/probe.py --ckpt "$WORK/4_pre_train/final.pt" --tag base
probe_snaps() {
  for s in "$WORK"/4_pre_train/snap_*.pt; do
    [[ -f "$s" ]] || continue
    t=$(basename "$s" .pt)
    $PY eval/probe.py --ckpt "$s" --tag "$t" --light
    $PY eval/run_all.py --ckpt "$s" --tag "pre_$t" --mode base --quick
  done
}
step probe_snaps    4 probe_snaps

# 5: generation
step generation     5 $PY 5_generation/eval_generation.py --ckpt "$WORK/4_pre_train/final.pt" --tag base
step eval_base      5 $PY eval/run_all.py --ckpt "$WORK/4_pre_train/final.pt" --tag base --mode base

# 6: mid-training
step mid_data       6 $PY 6_mid_training/build_mid_data.py;       disk_check
step needle_before  6 $PY 6_mid_training/needle_eval.py --ckpt "$WORK/4_pre_train/final.pt" --tag before
step needle_yarn    6 $PY 6_mid_training/needle_eval.py --ckpt "$WORK/4_pre_train/final.pt" --tag before_yarn --rope_scale 4
step midtrain       6 $TR 6_mid_training/midtrain.py
step needle_after   6 $PY 6_mid_training/needle_eval.py --ckpt "$WORK/6_mid_training/final.pt" --tag after
step eval_mid       6 $PY eval/run_all.py --ckpt "$WORK/6_mid_training/final.pt" --tag mid --mode base

# 7: post-training
step sft            7 $TR 7_post_training/sft.py
step sft_lora       7 $TR 7_post_training/sft.py --lora
step reward_model   7 $TR 7_post_training/reward_model.py
step dpo            7 $TR 7_post_training/dpo.py
step grpo           7 $TR 7_post_training/grpo.py
step distill        7 $TR 7_post_training/self_distill.py
step best_of_n      7 $PY 7_post_training/best_of_n.py

# evaluation of every post-training stage (two at a time, one per GPU)
eval_post() {
  CUDA_VISIBLE_DEVICES=0 LLM_ENERGY_SHARE=0.5 $PY eval/run_all.py --ckpt "$WORK/7_post_training/sft/final.pt" --tag sft &
  CUDA_VISIBLE_DEVICES=1 LLM_ENERGY_SHARE=0.5 $PY eval/run_all.py --ckpt "$WORK/7_post_training/sft_lora/final.pt" --tag sft_lora &
  wait
  CUDA_VISIBLE_DEVICES=0 LLM_ENERGY_SHARE=0.5 $PY eval/run_all.py --ckpt "$WORK/7_post_training/dpo/final.pt" --tag dpo &
  CUDA_VISIBLE_DEVICES=1 LLM_ENERGY_SHARE=0.5 $PY eval/run_all.py --ckpt "$WORK/7_post_training/grpo/final.pt" --tag grpo &
  wait
  $PY eval/run_all.py --ckpt "$WORK/7_post_training/distill/final.pt" --tag distill
}
step eval_post      8 eval_post
step probe_final    8 $PY eval/probe.py --ckpt "$WORK/7_post_training/final.pt" --tag final --light
step gen_final      8 $PY 5_generation/eval_generation.py --ckpt "$WORK/7_post_training/final.pt" --tag final
step judge          8 $PY eval/run_all.py --judge "dpo:sft,grpo:dpo,distill:grpo,distill:sft,sft_lora:sft"
step report         9 $PY report/build_report.py
if (( TO < 9 )); then echo "== stopped after stage $TO"; else echo "== pipeline complete: $WORK/report/index.html"; fi
