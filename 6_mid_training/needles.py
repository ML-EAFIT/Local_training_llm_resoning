"""Synthetic needle-in-a-haystack documents (training) and probes (evaluation).

Training and evaluation use disjoint name lists so the eval measures retrieval,
not memorisation of specific needles.
"""
FIRST = ["Alice", "Bruno", "Chen", "Dalia", "Emeka", "Farah", "Goran", "Hana", "Ivan", "Jun", "Kofi", "Lena",
         "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yusuf"]
EVAL_FIRST = ["Amara", "Bjorn", "Celia", "Dmitri", "Elif", "Felix", "Greta", "Hugo", "Ines", "Jonas"]
THINGS = ["secret code", "locker number", "room number", "passkey", "ticket number", "pin"]


def needle(rng, names):
    name, thing = rng.choice(names), rng.choice(THINGS)
    val = str(rng.randint(10000, 99999))
    return name, thing, val, f" The {thing} of {name} is {val}. "


def make_needle_doc(tok, rng, target_len, filler_docs=None, filler_ids=None, depth=None, names=FIRST):
    """Return token ids: filler with a needle at `depth` (0..1), then the question + answer."""
    name, thing, val, sent = needle(rng, names)
    q = tok.encode(f"\n\nQuestion: What is the {thing} of {name}?\nAnswer: The {thing} of {name} is")
    a = tok.encode(f" {val}.")
    nid = tok.encode(sent)
    room = max(16, target_len - len(q) - len(a) - len(nid) - 2)
    if filler_ids is None:
        filler = []
        while len(filler) < room:
            d = filler_docs[rng.randrange(len(filler_docs))]
            filler += [t for t in d if t != tok.eos][:room - len(filler)]
    else:
        s = rng.randrange(0, max(1, len(filler_ids) - room))
        filler = list(filler_ids[s:s + room])
    pos = int((rng.random() if depth is None else depth) * len(filler))
    ids = [tok.bos] + filler[:pos] + nid + filler[pos:] + q + a + [tok.eos]
    return ids if filler_ids is None else (ids, len(q) + len(a) + 1, val)
