"""Minimal JSONL metrics logger; every stage writes metrics.jsonl + summary.json."""
import json
import os
import time


class Logger:
    def __init__(self, out_dir, name="metrics", enabled=True, append=False):
        self.enabled = enabled
        self.out_dir = out_dir
        self.t0 = time.time()
        if enabled:
            os.makedirs(out_dir, exist_ok=True)
            self.f = open(os.path.join(out_dir, f"{name}.jsonl"), "a" if append else "w")

    def log(self, **kw):
        if not self.enabled:
            return
        kw.setdefault("time", round(time.time() - self.t0, 2))
        self.f.write(json.dumps(kw, default=float) + "\n")
        self.f.flush()

    def close(self):
        if self.enabled:
            self.f.close()


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def p(*a, **k):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True, **k)
