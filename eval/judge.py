"""LLM-as-a-judge (old Lecture 8) with a local Qwen2.5-1.5B-Instruct.

Pairwise: prompt + response A + response B + criteria -> rationale, then verdict.
Every pair is judged in both orders to cancel position bias; disagreement = tie.
Deterministic (greedy) decoding for reproducibility.
"""
import re

import torch

JUDGE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPLATE = """You are an impartial judge. Compare two assistant responses to the user's request.
Criteria: helpfulness, correctness, relevance, clarity. Ignore length unless it hurts quality.

[User request]
{prompt}

[Response A]
{a}

[Response B]
{b}

Write one or two sentences of reasoning, then on the last line write exactly "Verdict: A", "Verdict: B" or "Verdict: tie"."""


class Judge:
    def __init__(self, device="cuda"):
        import sys
        # the installed torchaudio is built for a different torch/CUDA; the judge needs no audio,
        # so hide it from transformers (find_spec then reports it as unavailable)
        sys.modules.setdefault("torchaudio", None)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(JUDGE_ID)
        self.tk.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _verdicts(self, triples, bs=8):
        out = []
        for i in range(0, len(triples), bs):
            msgs = [[{"role": "user", "content": TEMPLATE.format(prompt=p[:1500], a=a[:1500], b=b[:1500])}]
                    for p, a, b in triples[i:i + bs]]
            texts = [self.tk.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            enc = self.tk(texts, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=self.tk.eos_token_id)
            for g in gen[:, enc["input_ids"].shape[1]:]:
                t = self.tk.decode(g, skip_special_tokens=True)
                m = re.findall(r"Verdict:\s*(A|B|tie)", t, re.I)
                out.append(((m[-1].upper() if m and m[-1].lower() != "tie" else "tie") if m else "tie", t))
        return out

    def pairwise(self, prompts, resp_a, resp_b):
        """Win-rate of A vs B with position swapping. Returns dict + examples."""
        fwd = self._verdicts(list(zip(prompts, resp_a, resp_b)))
        rev = self._verdicts(list(zip(prompts, resp_b, resp_a)))
        win = loss = tie = consistent = 0
        ex = []
        for (v1, t1), (v2, t2), pr in zip(fwd, rev, prompts):
            a_first = v1 == "A"
            a_second = v2 == "B"
            if a_first and a_second:
                win += 1; consistent += 1
            elif v1 == "B" and v2 == "A":
                loss += 1; consistent += 1
            else:
                tie += 1
                consistent += v1 == v2 == "tie"
            if len(ex) < 3:
                ex.append({"prompt": pr[:300], "verdict_ab": v1, "verdict_ba": v2, "rationale": t1[:400]})
        n = max(len(prompts), 1)
        return {"win": win / n, "loss": loss / n, "tie": tie / n, "win_rate": (win + 0.5 * tie) / n,
                "position_consistency": consistent / n, "n": len(prompts), "examples": ex}

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
