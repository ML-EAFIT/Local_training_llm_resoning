"""Stage 4c — fit scaling laws from the sweep (Kaplan 2020 / Chinchilla 2022).

1. IsoFLOP: per budget C fit loss vs log N with a parabola -> N*(C); fit N* = k C^a.
2. Parametric: L(N, D) = E + A / N^alpha + B / D^beta  (Huber loss in log space).
3. Predict the main run's final loss (N = its non-embedding params, D = planned tokens).
4. Ablation table: delta loss vs. baseline at equal compute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from common.config import PRE_DIR, VOCAB_SIZE, PretrainConfig, model_preset  # noqa: E402
from common.logger import save_json, load_json, p  # noqa: E402
from common.model import GPT  # noqa: E402


def fit_parametric(N, D, L):
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def pred_log(th):
        e, a, b, al, be = th
        # log(exp(e) + exp(a - al logN) + exp(b - be logD))  (log-sum-exp form, Chinchilla)
        return np.logaddexp(np.logaddexp(e, a - al * logN), b - be * logD)

    def obj(th):
        r = pred_log(th) - logL
        d = 1e-3
        return np.where(np.abs(r) < d, 0.5 * r ** 2, d * (np.abs(r) - 0.5 * d)).sum()

    best = None
    for e0 in (0.5, 1.0, 1.5):
        for al0 in (0.2, 0.4, 0.7):
            for be0 in (0.2, 0.4, 0.7):
                for a0 in (3.0, 6.0):
                    th0 = np.array([e0, a0, a0, al0, be0])
                    r = minimize(obj, th0, method="L-BFGS-B",
                                 bounds=[(-1, 2.5), (-5, 30), (-5, 30), (0.05, 1.5), (0.05, 1.5)])
                    if best is None or r.fun < best.fun:
                        best = r
    e, a, b, al, be = best.x
    return {"E": float(np.exp(e)), "A": float(np.exp(a)), "B": float(np.exp(b)), "alpha": float(al),
            "beta": float(be), "fit_loss": float(best.fun)}


def L_of(fit, N, D):
    return fit["E"] + fit["A"] / N ** fit["alpha"] + fit["B"] / D ** fit["beta"]


def main():
    sw = load_json(os.path.join(PRE_DIR, "sweep.json"))
    if not sw:
        raise SystemExit("run scaling_sweep.py first")
    iso = [r for r in sw["runs"] if r["kind"] == "isoflop"]
    abl = [r for r in sw["runs"] if r["kind"] == "ablation"]
    out = {"isoflop": {}, "ablations": {}}

    # 1. IsoFLOP parabolas
    budgets = sorted({r["budget"] for r in iso})
    opt = []
    for C in budgets:
        rs = sorted([r for r in iso if r["budget"] == C], key=lambda r: r["params_nonemb"])
        x = np.log([r["params_nonemb"] for r in rs]); y = np.array([r["val_loss"] for r in rs])
        entry = {"points": [[r["params_nonemb"], r["tokens_trained"], r["val_loss"]] for r in rs]}
        if len(rs) >= 3:
            c2, c1, c0 = np.polyfit(x, y, 2)
            if c2 > 0:
                xs = -c1 / (2 * c2)
                xs = float(np.clip(xs, x.min() - 1.0, x.max() + 1.0))
                entry.update(N_opt=float(np.exp(xs)), L_opt=float(c2 * xs ** 2 + c1 * xs + c0),
                             parabola=[float(c2), float(c1), float(c0)])
                opt.append((C, entry["N_opt"]))
        out["isoflop"][f"{C:.0e}"] = entry
    if len(opt) >= 2:
        a, logk = np.polyfit(np.log([o[0] for o in opt]), np.log([o[1] for o in opt]), 1)
        out["n_opt_power_law"] = {"a": float(a), "k": float(np.exp(logk))}

    # 2. parametric fit
    N = np.array([r["params_nonemb"] for r in iso], float)
    D = np.array([r["tokens_trained"] for r in iso], float)
    L = np.array([r["val_loss"] for r in iso], float)
    fit = fit_parametric(N, D, L)
    fit["r2"] = float(1 - ((L_of(fit, N, D) - L) ** 2).sum() / max(((L - L.mean()) ** 2).sum(), 1e-12))
    out["parametric"] = fit
    # compute-optimal split from the fit: minimise L s.t. 6ND = C (closed form, Chinchilla eq. 4)
    G = (fit["alpha"] * fit["A"] / (fit["beta"] * fit["B"])) ** (1 / (fit["alpha"] + fit["beta"]))
    out["chinchilla_closed_form"] = {"G": float(G), "a": fit["beta"] / (fit["alpha"] + fit["beta"]),
                                     "b": fit["alpha"] / (fit["alpha"] + fit["beta"])}

    # 3. prediction for the main run
    d = PretrainConfig()
    cfg = model_preset(d.model); cfg.vocab_size = VOCAB_SIZE; cfg.max_seq_len = d.seq_len
    n_main = GPT(cfg).num_params(True)
    bench = load_json(os.path.join(PRE_DIR, "ddp_benchmark.json"), {})
    d_main = bench.get("planned_tokens") or d.max_tokens or 2e9
    out["main_prediction"] = {"params_nonemb": n_main, "tokens": d_main,
                              "predicted_val_loss": float(L_of(fit, n_main, d_main)),
                              "curve": [[float(t), float(L_of(fit, n_main, t))]
                                        for t in np.logspace(np.log10(d_main) - 2, np.log10(d_main), 30)],
                              "tokens_per_param": d_main / n_main}

    # 4. ablations
    base = next((r for r in abl if r["ablation"] == "baseline"), None)
    for r in abl:
        out["ablations"][r["ablation"]] = {"val_loss": r["val_loss"], "bpb": r["bpb"], "args": r["args"],
                                           "delta_vs_baseline": (r["val_loss"] - base["val_loss"]) if base else None,
                                           "params_nonemb": r["params_nonemb"], "seconds": r["seconds"]}
    out["runs"] = sw["runs"]
    save_json(os.path.join(PRE_DIR, "scaling_fit.json"), out)
    p(f"fit: E={fit['E']:.3f} A={fit['A']:.3g} alpha={fit['alpha']:.3f} B={fit['B']:.3g} beta={fit['beta']:.3f} "
      f"R2={fit['r2']:.3f}; main predicted loss {out['main_prediction']['predicted_val_loss']:.3f}")
    for k, v in out["ablations"].items():
        p(f"  ablation {k:22s} loss {v['val_loss']:.4f}  delta {v['delta_vs_baseline']:+.4f}")


if __name__ == "__main__":
    main()
