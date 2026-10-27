"""Stage 3a — train a byte-level BPE tokenizer (Lecture 1).

BPE: start from the 256 byte alphabet, repeatedly merge the most frequent
adjacent pair, stop at the target vocabulary size. Byte-level => no OOV.
Digits are split individually so arithmetic sees one token per digit.
"""
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import CLEAN_DIR, TOK_DIR, VOCAB_SIZE, SPECIAL_TOKENS, SMOKE, stage_dir  # noqa: E402
from common.logger import save_json, p  # noqa: E402

SAMPLE_BYTES = 20e6 if SMOKE else 600e6
# share of the tokenizer sample per domain (code/math slightly over-weighted vs. the pre-train mix
# so their merges are learned)
DOMAIN_SHARE = {"web": 0.6, "code": 0.17, "math": 0.13, "wiki": 0.10}
# GPT-4 style split, but numbers are handled by the Digits pre-tokenizer
SPLIT = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"""
         r"""\s+(?!\S)|\s+""")


def sample_texts():
    import pyarrow.parquet as pq
    rng = random.Random(0)
    for dom, share in DOMAIN_SHARE.items():
        files = sorted(glob.glob(os.path.join(CLEAN_DIR, "pre", dom, "*.parquet")))
        rng.shuffle(files)
        budget, got = SAMPLE_BYTES * share, 0
        for f in files:
            for t in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                yield t
                got += len(t)
                if got >= budget:
                    break
            if got >= budget:
                break
        p(f"tokenizer sample: {dom} {got/1e6:.1f} MB")


def build():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT), behavior="isolated"),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    return tk, trainer


def main():
    out = stage_dir("3_tokenization")
    t0 = time.time()
    tk, trainer = build()
    tk.train_from_iterator(sample_texts(), trainer=trainer)
    tk.save(os.path.join(TOK_DIR, "tokenizer.json"))
    save_json(os.path.join(out, "train_tokenizer.json"),
              {"vocab_size": tk.get_vocab_size(), "sample_bytes": SAMPLE_BYTES, "domain_share": DOMAIN_SHARE,
               "seconds": round(time.time() - t0, 1)})
    p(f"tokenizer trained: vocab {tk.get_vocab_size()} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
