"""Central configuration: paths, model presets and per-stage hyper-parameters.

Every script imports this module, so `--smoke` (or env LLM_SMOKE=1) switches the
whole pipeline to a tiny configuration that runs end-to-end in minutes.
"""
import os
import sys
from dataclasses import dataclass, asdict, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.environ.get("LLM_SMOKE", "0") == "1" or "--smoke" in sys.argv
WORK = os.environ.get("LLM_WORK", os.path.join(ROOT, "work_smoke" if SMOKE else "work"))


# Resource limits: the machine can shut off under heavy load, so leave headroom.
MAX_WORKERS = int(os.environ.get("LLM_WORKERS", "24"))           # CPU processes (of 64 threads)
GPU_MEM_FRAC = float(os.environ.get("LLM_GPU_MEM_FRAC", "0.85"))  # cap per-process VRAM (8 GB cards)


def limit_gpu_memory():
    """Call once per process after choosing the device."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRAC, torch.cuda.current_device())


def stage_dir(name):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    return d


RAW_DIR = os.path.join(WORK, "1_data_extraction")
CLEAN_DIR = os.path.join(WORK, "2_data_clean")
TOK_DIR = os.path.join(WORK, "3_tokenization")
PRE_DIR = os.path.join(WORK, "4_pre_train")
GEN_DIR = os.path.join(WORK, "5_generation")
MID_DIR = os.path.join(WORK, "6_mid_training")
POST_DIR = os.path.join(WORK, "7_post_training")
EVAL_DIR = os.path.join(WORK, "eval")
REPORT_DIR = os.path.join(WORK, "report")

# Special tokens (Lecture 1: BOS/EOS/PAD + role tokens; Lecture 3: think tokens)
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|system|>", "<|user|>",
                  "<|assistant|>", "<|end|>", "<|think|>", "<|/think|>"]
VOCAB_SIZE = 4096 if SMOKE else 32768  # fits uint16


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4            # GQA (Lecture 2); = n_head -> MHA, 1 -> MQA
    d_model: int = 768
    d_ff: int = 2048              # SwiGLU hidden
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rope_scale: float = 1.0       # YaRN scale factor (Stage 6)
    rope_orig_len: int = 1024     # context length the RoPE frequencies were trained at
    pos_emb: str = "rope"         # rope | learned | none (ablation)
    norm: str = "rmsnorm"         # rmsnorm | layernorm
    prenorm: bool = True          # pre-norm vs post-norm (ablation)
    window: int = 0               # sliding window size for local layers (0 = off)
    local_every: int = 2          # every k-th layer is global, others local (if window>0)
    n_experts: int = 0            # 0 -> dense FFN, else sparse MoE
    moe_top_k: int = 2
    moe_shared: bool = True
    moe_aux_coef: float = 0.01
    dropout: float = 0.0
    grad_ckpt: bool = False

    def to_dict(self):
        return asdict(self)


def model_preset(name):
    """Named sizes used by the scaling sweep and the main run."""
    presets = {
        # name: (n_layer, d_model, n_head, n_kv_head)
        "nano":   (2, 128, 4, 2),
        "micro":  (4, 192, 6, 2),
        "tiny":   (4, 256, 8, 4),
        "small":  (6, 384, 6, 2),
        "medium": (8, 512, 8, 4),
        "large":  (10, 640, 10, 5),
        "main":   (12, 768, 12, 4),
    }
    L, d, h, kv = presets[name]
    return ModelConfig(n_layer=L, d_model=d, n_head=h, n_kv_head=kv,
                       d_ff=int(8 * d / 3 / 64 + 0.999) * 64)


MAIN_MODEL = "tiny" if SMOKE else "main"


@dataclass
class PretrainConfig:
    model: str = MAIN_MODEL
    seq_len: int = 256 if SMOKE else 1024
    micro_batch: int = 8 if SMOKE else 8
    global_batch_tokens: int = 16384 if SMOKE else 524288
    lr: float = 3e-3 if SMOKE else 1.2e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 20 if SMOKE else 500
    decay_frac: float = 0.15       # WSD: last 15% of steps decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    hours: float = 0.05 if SMOKE else 15.0   # wall-clock slot -> token budget
    max_tokens: int = 3_000_000 if SMOKE else 0  # 0 = derive from benchmark + hours
    eval_every: int = 25 if SMOKE else 250
    eval_tokens: int = 65536 if SMOKE else 2_097_152
    bench_every: int = 50 if SMOKE else 2000
    ckpt_minutes: float = 30.0
    compile: bool = not SMOKE
    zero1: bool = False


# Energy accounting assumptions (documented in the report)
ENERGY = dict(
    cpu_tdp_w=250.0,          # Threadripper 2990WX TDP (RAPL not readable without root)
    cpu_idle_frac=0.25,
    rest_of_system_w=60.0,    # board, RAM, disks, fans (estimate)
    grid_kg_co2_per_kwh=float(os.environ.get("GRID_CO2", "0.40")),
    price_per_kwh=float(os.environ.get("KWH_PRICE", "0.20")),
    gpu_fp16_peak_tflops=float(os.environ.get("GPU_PEAK_TFLOPS", "36.0")),  # 2070S fp16 tensor (fp32 acc)
)
