"""FLOP accounting: ~6N per token for forward+backward, plus attention (Kaplan/Chinchilla)."""
from common.config import ENERGY


def flops_per_token(cfg, seq_len, n_params_nonemb):
    # 6N matmul FLOPs (fwd 2N + bwd 4N) + LM head + attention score/value FLOPs
    attn = 6 * cfg.n_layer * seq_len * cfg.d_model  # 2 * 2 * L * T * d fwd, x3 for bwd (causal ~half, keep upper bound/2)
    head = 6 * cfg.d_model * cfg.vocab_size
    return 6 * n_params_nonemb + head + attn


def mfu(tokens_per_s, fpt, n_gpus):
    peak = ENERGY["gpu_fp16_peak_tflops"] * 1e12 * n_gpus
    return tokens_per_s * fpt / peak
