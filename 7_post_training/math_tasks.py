"""Procedurally generated, verifiable math problems with chain-of-thought solutions.

Used for SFT (with the worked chain inside <|think|> ... <|/think|>), as RL with
verifiable rewards prompts for GRPO, and as an evaluation set. Difficulty 1-5.
"""
import random
import re

NAMES = ["Tom", "Ana", "Li", "Sara", "Omar", "Mia", "Ken", "Zoe", "Ravi", "Eva"]
ITEMS = ["apples", "books", "coins", "marbles", "stickers", "pencils", "cards", "shells"]
ANS_RE = re.compile(r"answer is\s*(-?\d[\d,]*)", re.I)


def _add_chain(a, b):
    """Place-value decomposition: 47 + 38 -> 40 + 30 = 70, 7 + 8 = 15, 70 + 15 = 85."""
    if a < 10 and b < 10:
        return [f"{a} + {b} = {a + b}"]
    ta, ua, tb, ub = a - a % 10, a % 10, b - b % 10, b % 10
    return [f"{ta} + {tb} = {ta + tb}", f"{ua} + {ub} = {ua + ub}", f"{ta + tb} + {ua + ub} = {a + b}"]


def _mul_chain(a, b):
    """23 * 4 -> 20 * 4 = 80, 3 * 4 = 12, 80 + 12 = 92."""
    if a < 10:
        return [f"{a} * {b} = {a * b}"]
    t, u = a - a % 10, a % 10
    return [f"{t} * {b} = {t * b}", f"{u} * {b} = {u * b}", f"{t * b} + {u * b} = {a * b}"]


def make_problem(rng: random.Random, level: int):
    if level == 1:
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        q, ans, chain = f"What is {a} + {b}?", a + b, _add_chain(a, b)
    elif level == 2:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            q, ans, chain = f"What is {a} + {b}?", a + b, _add_chain(a, b)
        else:
            a, b = max(a, b), min(a, b)
            q, ans = f"What is {a} - {b}?", a - b
            chain = [f"{a} - {b - b % 10} = {a - (b - b % 10)}", f"{a - (b - b % 10)} - {b % 10} = {ans}"]
    elif level == 3:
        a, b = rng.randint(11, 99), rng.randint(2, 9)
        q, ans, chain = f"What is {a} * {b}?", a * b, _mul_chain(a, b)
    elif level == 4:
        n, it = rng.choice(NAMES), rng.choice(ITEMS)
        x, y, z = rng.randint(5, 60), rng.randint(2, 40), rng.randint(1, 30)
        z = min(z, x + y)
        q = f"{n} has {x} {it}. {n} gets {y} more and then gives away {z}. How many {it} does {n} have now?"
        ans = x + y - z
        chain = [f"Start with {x}.", f"{x} + {y} = {x + y}", f"{x + y} - {z} = {ans}"]
    else:
        n, it = rng.choice(NAMES), rng.choice(ITEMS)
        k, per, extra = rng.randint(2, 9), rng.randint(3, 25), rng.randint(1, 50)
        q = (f"{n} buys {k} boxes of {it} with {per} {it} in each box, and already had {extra} {it}. "
             f"How many {it} does {n} have in total?")
        ans = k * per + extra
        chain = [f"{k} boxes of {per}:"] + _mul_chain(per, k) + [f"{k * per} + {extra} = {ans}"]
    return {"question": q, "answer": ans, "chain": chain, "level": level}


def solution_text(pb):
    return "<|think|>" + "; ".join(pb["chain"]) + "<|/think|>" + f"The answer is {pb['answer']}."


def extract_answer(text):
    m = ANS_RE.search(text.replace("<|/think|>", " "))
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def reward(text, pb):
    """Verifiable reward = correctness (1.0) + format (0.1 for a think block and a parsable answer)."""
    fmt = 0.1 if "<|think|>" in text and "<|/think|>" in text and ANS_RE.search(text) else 0.0
    correct = 1.0 if extract_answer(text) == pb["answer"] else 0.0
    return correct + fmt, correct


def gsm8k_to_sft(row):
    body, ans = row["answer"].split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip().replace("\n", " ")
    return {"question": row["question"], "answer": int(ans.strip().replace(",", "")),
            "chain": [body], "level": 6}


def dataset(n, seed, levels=(1, 2, 3, 4, 5)):
    rng = random.Random(seed)
    return [make_problem(rng, rng.choice(levels)) for _ in range(n)]
