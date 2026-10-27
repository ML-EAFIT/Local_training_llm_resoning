"""Sandboxed execution of generated code (HumanEval / MBPP pass@k).

Each program runs in a fresh `python -I` subprocess inside a temp directory
with CPU-time, memory and file-size limits and a wall-clock timeout.
"""
import os
import resource
import subprocess
import sys
import tempfile
from math import comb


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.setsid()


def run_program(src, timeout=6.0):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def pass_at_k(n, c, k):
    """Unbiased estimator 1 - C(n-c, k) / C(n, k)  (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def truncate_completion(text):
    """Stop a function-body completion at the next top-level statement."""
    out = []
    for line in text.split("\n"):
        if out and line and not line.startswith((" ", "\t")):
            break
        out.append(line)
    return "\n".join(out)


def extract_code(text):
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code = parts[1]
            return code[len("python"):] if code.startswith("python") else code
    return text
