"""Build work/report/index.html: one self-contained page (no network needed) that shows
every stage of the pipeline: data funnel, tokenizer, hardware scaling, scaling laws,
ablations, training curves, per-component analysis, generation, long context,
post-training, the capability matrix across stages, and energy.

Data comes only from the metrics.jsonl / summary.json files that the stages write.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import WORK, RAW_DIR, CLEAN_DIR, TOK_DIR, PRE_DIR, GEN_DIR, MID_DIR, POST_DIR, EVAL_DIR, \
    REPORT_DIR, SMOKE  # noqa: E402
from common.logger import load_json, read_jsonl, p  # noqa: E402


def thin(rows, n=400):
    if len(rows) <= n:
        return rows
    k = len(rows) / n
    return [rows[int(i * k)] for i in range(n)] + [rows[-1]]


def energy_of(obj):
    e = (obj or {}).get("energy") or {}
    return {k: e.get(k) for k in ("total_kwh", "gpu_kwh", "cpu_kwh", "rest_kwh", "seconds", "co2_kg", "cost",
                                  "avg_gpu_w", "peak_gpu_temp", "tokens", "j_per_token", "cpu_measured")} if e else None


def gather():
    D = {"smoke": SMOKE, "work": WORK}
    # ---------------- 1-3 data
    D["extract"] = load_json(os.path.join(RAW_DIR, "summary.json"))
    D["clean"] = load_json(os.path.join(CLEAN_DIR, "summary.json"))
    D["tokenizer"] = load_json(os.path.join(TOK_DIR, "summary.json"))
    # ---------------- 4 pre-training
    D["bench"] = load_json(os.path.join(PRE_DIR, "ddp_benchmark.json"))
    D["scaling"] = load_json(os.path.join(PRE_DIR, "scaling_fit.json"))
    D["pretrain"] = load_json(os.path.join(PRE_DIR, "summary.json"))
    rows = read_jsonl(os.path.join(PRE_DIR, "metrics.jsonl"))
    train_rows = [r for r in rows if "loss" in r]
    D["pre_train_curve"] = [{k: r.get(k) for k in ("step", "tokens", "loss", "lr", "grad_norm", "tok_per_s", "mfu",
                                                   "loss_scale", "mem_gb", "kwh")} for r in thin(train_rows)]
    D["pre_eval_curve"] = [{"tokens": r["tokens"], **r["eval"]} for r in rows if "eval" in r]
    D["pre_grad_comp"] = [{"tokens": r["tokens"], **r["grad_norm_components"]} for r in train_rows
                          if "grad_norm_components" in r]
    D["pre_power"] = thin(read_jsonl(os.path.join(PRE_DIR, "energy.jsonl")), 300)
    # ---------------- probes
    D["probes"] = {}
    for f in sorted(glob.glob(os.path.join(EVAL_DIR, "probe_*.json"))):
        pr = load_json(f)
        D["probes"][pr["tag"]] = pr
    # ---------------- 5 generation
    D["generation"] = {os.path.basename(f)[8:-5]: load_json(f) for f in glob.glob(os.path.join(GEN_DIR, "summary_*.json"))}
    # ---------------- 6 mid-training
    D["mid"] = load_json(os.path.join(MID_DIR, "summary.json"))
    D["mid_data"] = load_json(os.path.join(MID_DIR, "mid_data.json"))
    D["needle"] = {os.path.basename(f)[7:-5]: load_json(f) for f in glob.glob(os.path.join(MID_DIR, "needle_*.json"))}
    D["mid_curve"] = {}
    for f in sorted(glob.glob(os.path.join(MID_DIR, "metrics_L*.jsonl"))):
        L = os.path.basename(f)[9:-6]
        D["mid_curve"][L] = [{"step": r["step"], "loss": r["loss"]} for r in thin(read_jsonl(f), 200) if "loss" in r]
    # ---------------- 7 post-training
    post = {}
    for name in ("sft", "sft_lora", "reward_model", "dpo", "grpo", "distill", "best_of_n"):
        d = os.path.join(POST_DIR, name)
        post[name] = {"summary": load_json(os.path.join(d, "summary.json")),
                      "curve": thin(read_jsonl(os.path.join(d, "metrics.jsonl")), 300)}
    D["post"] = post
    # ---------------- capability matrix
    order = ["pre_snap_010", "pre_snap_025", "pre_snap_050", "pre_snap_075", "base", "mid", "sft", "sft_lora",
             "dpo", "grpo", "distill"]
    evals = {}
    for f in glob.glob(os.path.join(EVAL_DIR, "eval_*.json")):
        e = load_json(f)
        evals[e["tag"]] = e
    D["evals"] = {k: evals[k] for k in order if k in evals}
    D["eval_order"] = [k for k in order if k in evals]
    D["judge"] = load_json(os.path.join(EVAL_DIR, "judge.json"))
    # ---------------- energy per stage
    en = []

    def add(stage, obj):
        e = energy_of(obj)
        if e and e.get("total_kwh") is not None:
            en.append({"stage": stage, **e})
    add("1 extraction", D["extract"])
    add("2 cleaning", D["clean"])
    add("3 tokenization", load_json(os.path.join(TOK_DIR, "tokenize.json")))
    for ph in ("single", "ddp"):
        add(f"4 benchmark ({ph})", load_json(os.path.join(PRE_DIR, f"bench_{ph}.json")))
    add("4 scaling sweep", load_json(os.path.join(PRE_DIR, "sweep.json")))
    add("4 pre-training", D["pretrain"])
    for tag, g in D["generation"].items():
        add(f"5 generation eval ({tag})", g)
    add("6 mid-training", D["mid"])
    for name in ("sft", "sft_lora", "reward_model", "dpo", "grpo", "distill", "best_of_n"):
        add(f"7 {name}", post[name]["summary"])
    ev_tot = [energy_of(e) for e in evals.values() if energy_of(e)]
    if ev_tot:
        en.append({"stage": "evaluation (all)", **{k: sum((x.get(k) or 0) for x in ev_tot)
                                                   for k in ("total_kwh", "gpu_kwh", "cpu_kwh", "rest_kwh", "seconds",
                                                             "co2_kg", "cost")}})
    add("judge", D["judge"])
    D["energy"] = en
    return D


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Pipeline Report</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--neutral:#f0efec;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--good:#0ca30c;--critical:#d03b3b;--good-text:#006300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--neutral:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good-text:#0ca30c}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--neutral:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good-text:#0ca30c}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:28px 16px 8px;max-width:1180px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 4px}h3{font-size:15px;margin:0 0 2px}
.sub{color:var(--text-secondary);margin:0 0 12px;max-width:860px}
main{max-width:1180px;margin:0 auto;padding:0 16px 60px}
nav{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--border);padding:8px 16px;
overflow-x:auto;white-space:nowrap}
nav a{color:var(--text-secondary);text-decoration:none;margin-right:14px;font-size:13px}nav a:hover{color:var(--text-primary)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 14px 10px;min-width:0;position:relative}
.card .note{color:var(--muted);font-size:12px;margin:2px 0 6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px}
.tile .l{color:var(--text-secondary);font-size:12px}.tile .v{font-size:24px;font-weight:600}
.tile .d{color:var(--muted);font-size:12px}
.hero{grid-column:span 2}.hero .v{font-size:48px;line-height:1.1}
svg{display:block;max-width:100%;height:auto;overflow:visible}
svg text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--text-secondary);margin:4px 0}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:20;max-width:300px}
.tt b{font-weight:600}.tt .row{display:flex;gap:6px;align-items:center;white-space:nowrap}
.tt .sw{width:8px;height:8px;border-radius:50%;flex:none}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--text-secondary);font-weight:600}
.tblwrap{overflow-x:auto;max-height:420px;overflow-y:auto}
.toggle{position:absolute;top:10px;right:12px;font-size:11px;background:none;border:1px solid var(--border);
color:var(--text-secondary);border-radius:6px;padding:2px 7px;cursor:pointer}
.empty{color:var(--muted);font-style:italic;padding:20px 0}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--page);padding:8px;border-radius:6px;
max-height:260px;overflow:auto;margin:4px 0}
.themebtn{float:right;margin-top:4px}
.warn{border-left:3px solid var(--s4);padding:6px 10px;background:var(--surface-1);color:var(--text-secondary);margin:10px 0}
</style></head><body>
<nav id="nav"></nav>
<header><button class="toggle themebtn" id="theme">theme</button>
<h1>LLM Pipeline Report</h1>
<p class="sub" id="intro"></p></header>
<main id="main"></main>
<div class="tt" id="tt"></div>
<script>
const D = __DATA__;
const S = n => `var(--s${(n % 8) + 1})`;
const $ = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content.firstChild; };
const trim = x => x.replace(/\.?0+$/, '');
const fmt = (v, d) => { if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (typeof v !== 'number') return String(v);
  if (d !== undefined) return v.toFixed(d);
  const a = Math.abs(v);
  if (a >= 1e12) return trim((v/1e12).toFixed(2))+'T'; if (a >= 1e9) return trim((v/1e9).toFixed(2))+'B';
  if (a >= 1e6) return trim((v/1e6).toFixed(2))+'M'; if (a >= 1e4) return trim((v/1e3).toFixed(1))+'K';
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0); if (a >= 1) return trim(v.toFixed(2)); if (a === 0) return '0';
  return trim(v.toPrecision(3)); };
const pct = v => v === null || v === undefined ? '–' : (100*v).toFixed(1)+'%';
const tt = document.getElementById('tt');
/* charts are drawn after every card exists, so each SVG is sized to its card's final width */
const LATER = []; const later = f => { LATER.push(f); };
function showTT(e, html){ tt.innerHTML = html; tt.style.display='block';
  const x = Math.min(e.clientX + 14, window.innerWidth - tt.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, window.innerHeight - tt.offsetHeight - 8); tt.style.left = x+'px'; tt.style.top = y+'px'; }
function hideTT(){ tt.style.display='none'; }
function niceTicks(lo, hi, n=5){ if (!(hi > lo)) { hi = lo + 1; } const span = hi - lo, step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0))); const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag; const out = [];
  for (let v = Math.ceil(lo/step)*step; v <= hi + 1e-9*step; v += step) out.push(+v.toPrecision(12)); return out; }
function logTicks(lo, hi){ const out=[]; for (let e=Math.floor(Math.log10(lo)); e<=Math.ceil(Math.log10(hi)); e++){ const v=Math.pow(10,e); if (v>=lo*0.999 && v<=hi*1.001) out.push(v);} return out.length>=2?out:[lo,hi]; }

/* ---------------- card with optional table view ---------------- */
function card(parent, title, note, rows, cols){
  const c = $(`<div class="card"><h3>${title}</h3>${note?`<div class="note">${note}</div>`:''}<div class="body"></div></div>`);
  parent.appendChild(c); const body = c.querySelector('.body');
  if (rows && rows.length){ const b = $('<button class="toggle">table</button>'); c.appendChild(b);
    let tbl = null; b.onclick = () => { if (tbl){ tbl.remove(); tbl = null; body.style.display=''; b.textContent='table'; return; }
      tbl = tableEl(rows, cols); body.style.display='none'; c.appendChild(tbl); b.textContent='chart'; }; }
  return body; }
function tableEl(rows, cols){ if (!rows || !rows.length) return $('<div class="empty">no rows</div>'); cols = cols || Object.keys(rows[0]);
  const w = $('<div class="tblwrap"></div>'); const t = document.createElement('table');
  t.innerHTML = '<tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${typeof r[c]==='number'?fmt(r[c]):(r[c]??'–')}</td>`).join('') + '</tr>').join('');
  w.appendChild(t); return w; }
function empty(parent, msg){ parent.appendChild($(`<div class="empty">${msg||'not available yet: run this stage'}</div>`)); }
function legend(parent, series){ if (series.length < 2) return;
  parent.appendChild($('<div class="legend">' + series.map((s,i)=>`<span><i style="background:${s.color||S(i)}"></i>${s.name}</span>`).join('') + '</div>')); }

/* ---------------- line chart: series [{name, pts:[[x,y],...], dash?}] ---------------- */
function lineChart(parent, series, o={}){
  if (!o._now) return later(() => lineChart(parent, series, {...o, _now: 1}));
  series = series.filter(s => s.pts && s.pts.length); if (!series.length) return empty(parent);
  const W = Math.max(300, parent.clientWidth || 640), H = o.h || 250, m = {l: 54, r: 18, t: 10, b: 34};
  const xs = series.flatMap(s=>s.pts.map(p=>p[0])), ys = series.flatMap(s=>s.pts.map(p=>p[1])).filter(v=>v!==null && isFinite(v));
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = o.y0 ?? Math.min(...ys), y1 = o.y1 ?? Math.max(...ys);
  if (o.yzero) y0 = Math.min(0, y0); if (y1 === y0) { y1 += 1; y0 -= o.yzero ? 0 : 1; }
  const pad = (y1 - y0) * 0.05; if (!o.y0 && !o.yzero) y0 -= pad; y1 += pad;
  const lx = o.logx && x0 > 0, ly = o.logy && y0 > 0;
  const fx = v => m.l + (lx ? (Math.log(v)-Math.log(x0))/(Math.log(x1)-Math.log(x0)||1) : (v-x0)/((x1-x0)||1)) * (W-m.l-m.r);
  const fy = v => H-m.b - (ly ? (Math.log(v)-Math.log(y0))/(Math.log(y1)-Math.log(y0)) : (v-y0)/(y1-y0)) * (H-m.t-m.b);
  const xt = lx ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(2, Math.floor((W - m.l - m.r) / 70))), yt = ly ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  let g = '';
  yt.forEach(v => { if (v < y0 || v > y1) return; g += `<line x1="${m.l}" x2="${W-m.r}" y1="${fy(v)}" y2="${fy(v)}" stroke="var(--grid)"/>`+
    `<text x="${m.l-6}" y="${fy(v)+4}" text-anchor="end">${(o.yfmt||fmt)(v)}</text>`; });
  xt.forEach(v => { if (v < x0 || v > x1) return; g += `<text x="${fx(v)}" y="${H-m.b+16}" text-anchor="middle">${(o.xfmt||fmt)(v)}</text>`; });
  g += `<line x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}" stroke="var(--axis)"/>`;
  if (o.xlabel) g += `<text x="${(W+m.l)/2}" y="${H-2}" text-anchor="middle">${o.xlabel}</text>`;
  if (o.ylabel) g += `<text transform="translate(12,${(H-m.b+m.t)/2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  series.forEach((s,i) => { const col = s.color || S(i);
    const pts = s.pts.filter(p => p[1] !== null && isFinite(p[1]));
    if (!pts.length) return;
    const d = pts.map((p,j)=>(j?'L':'M')+fx(p[0]).toFixed(1)+','+fy(p[1]).toFixed(1)).join('');
    if (s.area) g += `<path d="${d}L${fx(pts[pts.length-1][0])},${H-m.b}L${fx(pts[0][0])},${H-m.b}Z" fill="${col}" opacity=".1"/>`;
    g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" ${s.dash?'stroke-dasharray="5 4"':''}/>`;
    if (s.markers || pts.length < 12) pts.forEach(p => g += `<circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="4" fill="${col}" stroke="var(--surface-1)" stroke-width="2"/>`);
    else { const p = pts[pts.length-1]; g += `<circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="4" fill="${col}" stroke="var(--surface-1)" stroke-width="2"/>`; } });
  const svg = $(`<svg viewBox="0 0 ${W} ${H}" role="img"><g>${g}</g><line class="xh" y1="${m.t}" y2="${H-m.b}" stroke="var(--axis)" style="display:none"/>`+
    `<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent"/></svg>`);
  const xh = svg.querySelector('.xh');
  svg.querySelector('rect').addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    let rows = '', xv = null;
    series.forEach((s,i) => { let best = null, bd = 1e18;
      s.pts.forEach(p => { const dd = Math.abs(fx(p[0]) - px); if (dd < bd && p[1] !== null) { bd = dd; best = p; } });
      if (best && bd < 40) { xv = xv ?? best[0]; rows += `<div class="row"><span class="sw" style="background:${s.color||S(i)}"></span>${s.name}: <b>${(o.yfmt||fmt)(best[1])}</b></div>`; } });
    if (!rows) { hideTT(); xh.style.display='none'; return; }
    xh.setAttribute('x1', fx(xv)); xh.setAttribute('x2', fx(xv)); xh.style.display='';
    showTT(e, `<div>${o.xlabel||'x'}: <b>${(o.xfmt||fmt)(xv)}</b></div>` + rows); });
  svg.querySelector('rect').addEventListener('mouseleave', () => { hideTT(); xh.style.display='none'; });
  legend(parent, series); parent.appendChild(svg); }

/* ---------------- horizontal bar chart: cats [{label, values:[...]}], names ---------------- */
function barChart(parent, cats, names, o={}){
  if (!o._now) return later(() => barChart(parent, cats, names, {...o, _now: 1}));
  cats = cats.filter(c => c.values.some(v => v !== null && v !== undefined)); if (!cats.length) return empty(parent);
  const nS = names.length, bh = Math.min(16, 22 / nS + 6), gap = 2, band = nS * (bh + gap) + 10;
  const lab = o.labelW || 150, W = Math.max(300, parent.clientWidth || 640), m = {l: lab, r: 70, t: 4, b: 22}, H = m.t + m.b + band * cats.length;
  const vals = cats.flatMap(c => c.values).filter(v => v !== null && isFinite(v));
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals); if (hi === lo) hi = lo + 1;
  const fx = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  let g = ''; niceTicks(lo, hi, Math.max(2, Math.floor((W - m.l - m.r) / 80))).forEach(v => { g += `<line x1="${fx(v)}" x2="${fx(v)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--grid)"/>`+
    `<text x="${fx(v)}" y="${H-6}" text-anchor="middle">${(o.fmt||fmt)(v)}</text>`; });
  const marks = [];
  cats.forEach((c, ci) => { const y0 = m.t + ci * band;
    g += `<text x="${m.l-8}" y="${y0 + band/2}" text-anchor="end" dominant-baseline="middle" style="fill:var(--text-secondary)">${c.label}</text>`;
    c.values.forEach((v, si) => { if (v === null || v === undefined) return; const y = y0 + 5 + si * (bh + gap);
      const xa = fx(Math.min(0, v)), xb = fx(Math.max(0, v)), w = Math.max(1, xb - xa), r = Math.min(4, w / 2, bh / 2);
      const col = (o.colors && o.colors[si]) || S(si);
      const d = v >= 0 ? `M${xa},${y}H${xb-r}Q${xb},${y} ${xb},${y+r}V${y+bh-r}Q${xb},${y+bh} ${xb-r},${y+bh}H${xa}Z`
                       : `M${xb},${y}H${xa+r}Q${xa},${y} ${xa},${y+r}V${y+bh-r}Q${xa},${y+bh} ${xa+r},${y+bh}H${xb}Z`;
      marks.push({ci, si, v}); g += `<path d="${d}" fill="${col}" data-k="${marks.length-1}"/>`;
      g += `<text x="${v >= 0 ? xb + 4 : xa - 4}" y="${y + bh/2}" text-anchor="${v >= 0 ? 'start' : 'end'}" dominant-baseline="middle">${(o.fmt||fmt)(v)}</text>`; }); });
  g += `<line x1="${fx(0)}" x2="${fx(0)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--axis)"/>`;
  const svg = $(`<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`);
  svg.querySelectorAll('path[data-k]').forEach(el => { const k = marks[+el.dataset.k];
    el.addEventListener('mousemove', e => showTT(e, `<b>${cats[k.ci].label}</b><div class="row"><span class="sw" style="background:${(o.colors&&o.colors[k.si])||S(k.si)}"></span>${names[k.si]}: <b>${(o.fmt||fmt)(k.v)}</b></div>`));
    el.addEventListener('mouseleave', hideTT); });
  legend(parent, names.map((n,i)=>({name:n, color:(o.colors&&o.colors[i])||S(i)}))); parent.appendChild(svg); }

/* ---------------- heatmap (sequential blue, or diverging blue-neutral-red) ---------------- */
const SEQ = ['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5','#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b'];
const RED = ['#fbd5d4','#f6b3b1','#ef8f8d','#e66767','#e34948','#c63c3b','#a33130'];
const BLUE = ['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#2a78d6','#1c5cab','#104281'];
function heatmap(parent, M, rows, cols, o={}){
  if (!o._now) return later(() => heatmap(parent, M, rows, cols, {...o, _now: 1}));
  if (!M || !M.length) return empty(parent);
  const flat = M.flat().filter(v => v !== null && isFinite(v)); const lo = o.lo ?? Math.min(...flat), hi = o.hi ?? Math.max(...flat);
  const W = Math.max(300, parent.clientWidth || 640), lab = o.labelW || 60, cw = (W - lab - 10) / cols.length, ch = o.ch || Math.min(26, Math.max(8, 300 / rows.length)), H = rows.length * ch + 28;
  const color = v => { if (o.diverging) { const a = Math.max(Math.abs(lo), Math.abs(hi)) || 1, t = v / a;
      if (Math.abs(t) < 0.06) return 'var(--neutral)'; const ramp = t > 0 ? RED : BLUE; return ramp[Math.min(ramp.length-1, Math.floor(Math.abs(t) * ramp.length))]; }
    const t = (v - lo) / ((hi - lo) || 1); return SEQ[Math.min(SEQ.length-1, Math.max(0, Math.floor(t * SEQ.length)))]; };
  let g = '';
  rows.forEach((r, i) => { g += `<text x="${lab-6}" y="${i*ch + ch/2}" text-anchor="end" dominant-baseline="middle">${r}</text>`;
    M[i].forEach((v, j) => { g += `<rect x="${lab + j*cw + 1}" y="${i*ch + 1}" width="${Math.max(1,cw-2)}" height="${Math.max(1,ch-2)}" rx="2" fill="${v===null?'var(--grid)':color(v)}" data-i="${i}" data-j="${j}"/>`; }); });
  const every = Math.ceil(cols.length / 16);
  cols.forEach((c, j) => { if (j % every === 0) g += `<text x="${lab + j*cw + cw/2}" y="${rows.length*ch + 14}" text-anchor="middle">${c}</text>`; });
  if (o.xlabel) g += `<text x="${(W+lab)/2}" y="${H}" text-anchor="middle">${o.xlabel}</text>`;
  const svg = $(`<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`);
  svg.querySelectorAll('rect').forEach(el => { const i = +el.dataset.i, j = +el.dataset.j;
    el.addEventListener('mousemove', e => showTT(e, `${o.rowName||'row'} <b>${rows[i]}</b> · ${o.colName||'col'} <b>${cols[j]}</b><br>${o.valName||'value'}: <b>${(o.fmt||fmt)(M[i][j])}</b>`));
    el.addEventListener('mouseleave', hideTT); });
  parent.appendChild(svg);
  parent.appendChild($(`<div class="legend"><span>${o.diverging?'blue = lower · gray ≈ 0 · red = higher':'light = low · dark = high'} (range ${(o.fmt||fmt)(lo)} … ${(o.fmt||fmt)(hi)})</span></div>`)); }

function tiles(parent, items, hero){ const t = $('<div class="tiles"></div>'); parent.appendChild(t);
  items.forEach((it, i) => t.appendChild($(`<div class="tile ${hero && i===0 ? 'hero':''}"><div class="l">${it[0]}</div><div class="v">${it[1]}</div>${it[2]?`<div class="d">${it[2]}</div>`:''}</div>`))); }

/* ---------------- sections ---------------- */
const main = document.getElementById('main'), nav = document.getElementById('nav');
window.addEventListener('error', e => { const m = document.getElementById('main');
  m.appendChild($(`<div class="warn">report script error: ${e.message}</div>`)); });
function section(id, title, sub){ nav.appendChild($(`<a href="#${id}">${title.replace(/^\d+ · /,'')}</a>`));
  const s = $(`<section id="${id}"><h2>${title}</h2>${sub?`<p class="sub">${sub}</p>`:''}</section>`); main.appendChild(s); return s; }
function gridIn(s){ const g = $('<div class="grid"></div>'); s.appendChild(g); return g; }

document.getElementById('intro').innerHTML = `A decoder-only Transformer trained from scratch on 2× RTX 2070 Super (8 GB, Turing, fp16 + DDP),
built as the lesson pipeline: 1 data extraction → 2 cleaning → 3 tokenization → 4 pre-training → 5 generation →
6 mid-training → 7 post-training (SFT, reward model, DPO, GRPO, on-policy self-distillation). Every number on this page is read from the
stage outputs in <code>${D.work}</code>.` + (D.smoke ? ' <b>Smoke run: tiny data and model, numbers only check the plumbing.</b>' : '');

/* overview */
(function(){ try { const s = section('overview', 'Overview');
  const pre = D.pretrain || {}, fe = pre.final_eval || {}, en = D.energy || [];
  const kwh = en.reduce((a, e) => a + (e.total_kwh || 0), 0), co2 = en.reduce((a, e) => a + (e.co2_kg || 0), 0);
  const cost = en.reduce((a, e) => a + (e.cost || 0), 0), hours = en.reduce((a, e) => a + (e.seconds || 0), 0) / 3600;
  const last = D.evals[D.eval_order[D.eval_order.length-1]] || {};
  tiles(s, [['Total energy (whole pipeline)', kwh ? kwh.toFixed(2) + ' kWh' : '–', `${fmt(co2,2)} kg CO₂e · ${fmt(cost,2)} cost units · ${hours.toFixed(1)} h measured`],
    ['Model', pre.params ? fmt(pre.params) + ' params' : '–', pre.params_nonemb ? fmt(pre.params_nonemb) + ' non-embedding' : ''],
    ['Pre-training tokens', fmt(pre.tokens), pre.tokens_per_param ? pre.tokens_per_param.toFixed(1) + ' tokens / param' : ''],
    ['Final val loss', fmt(fe.val_loss, 3), fe.bpb ? `${fe.bpb.toFixed(3)} bits/byte` : ''],
    ['HellaSwag (acc_norm)', pct(fe.hellaswag), 'chance 25%'],
    ['Final model', last.tag || '–', last.math ? `math ${pct(last.math.synthetic_acc)} · IF ${pct((last.instruction_following||{}).overall)}` : '']], true);
  if (!D.smoke) s.appendChild($(`<div class="warn">Scale check: a ~124M model on a few billion tokens lands near GPT-2-small. It writes fluent text, follows simple chat
    instructions, does small arithmetic with a chain of thought and completes short Python, but it is not a general assistant. The value of this run is
    the measured end-to-end pipeline, not frontier capability.</div>`)); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 1-2 data */
(function(){ try { const s = section('data', '1–2 · Data extraction & cleaning', 'Streamed from the Hugging Face Hub, then heuristic filters → PII scrub → 13-gram decontamination against every eval set → exact + MinHash-LSH near dedup.');
  const g = gridIn(s); const c = D.clean;
  if (!c) return empty(g);
  const per = Object.entries(c.per_source);
  const rows = per.map(([k, v]) => ({source: k, 'docs in': v.docs_in, 'docs out': v.docs_out, 'GB in': v.bytes_in/1e9, 'GB out': v.bytes_out/1e9, 'kept (bytes)': pct(v.kept_frac_bytes)}));
  barChart(card(g, 'Bytes before and after cleaning', 'per source / pool (pre = pre-training, mid = mid-training pool)', rows),
    per.map(([k, v]) => ({label: k, values: [v.bytes_in/1e9, v.bytes_out/1e9]})), ['raw GB', 'clean GB'], {fmt: v => v.toFixed(3)});
  const reasons = {}; per.forEach(([k, v]) => Object.entries(v.rej).forEach(([r, x]) => reasons[r] = (reasons[r]||0) + x[0]));
  Object.entries(c.dedup || {}).forEach(([k, v]) => { reasons.exact_dup = (reasons.exact_dup||0) + v.exact_dup; reasons.near_dup = (reasons.near_dup||0) + v.near_dup; });
  const rr = Object.entries(reasons).sort((a,b)=>b[1]-a[1]);
  barChart(card(g, 'Documents removed by each rule', 'all sources; a document is counted under the first rule that rejects it', rr.map(([r,n])=>({rule:r, docs:n}))),
    rr.map(([r,n]) => ({label: r, values: [n]})), ['docs removed']);
  const pii = {}; per.forEach(([k, v]) => Object.entries(v.pii || {}).forEach(([r, n]) => pii[r] = (pii[r]||0) + n));
  tiles(gridIn(s), [['Kept overall (bytes)', pct(c.total.kept_frac), `${fmt(c.total.bytes_out)}B of ${fmt(c.total.bytes_in)}B`],
    ['Eval 13-grams indexed', fmt(c.eval_ngrams), 'decontamination set'], ['Emails scrubbed', fmt(pii.email||0)], ['IPs scrubbed', fmt(pii.ip||0)],
    ['Boilerplate lines dropped', fmt(pii.boiler_lines||0)]]);
  const ex = card(gridIn(s), 'Examples of removed documents', 'first 300 characters, one per rule');
  const exs = []; per.forEach(([k, v]) => Object.entries(v.examples||{}).forEach(([r, e]) => { if (exs.length < 8 && !exs.find(x=>x.r===r)) exs.push({r, k, t: e[0]}); }));
  exs.forEach(x => ex.appendChild($(`<div><b>${x.r}</b> <span class="note">(${x.k})</span><pre></pre></div>`)).querySelector('pre').textContent = x.t); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 3 tokenizer */
(function(){ try { const s = section('tokenizer', '3 · Tokenization', 'Byte-level BPE (individual digits, GPT-4-style pre-split), trained on a stratified sample. Higher bytes/token = shorter sequences.');
  const g = gridIn(s), t = D.tokenizer; if (!t) return empty(g);
  const doms = Object.entries(t.domains);
  barChart(card(g, 'Compression: bytes per token', `ours (vocab ${fmt(t.vocab_size)}) vs GPT-2 (50K) vs cl100k (100K)`,
      doms.map(([d,v])=>({domain:d, ours:v.bytes_per_token, gpt2:v.gpt2_bytes_per_token, cl100k:v.cl100k_bytes_per_token, 'tokens/word':v.tokens_per_word}))),
    doms.map(([d, v]) => ({label: d, values: [v.bytes_per_token, v.gpt2_bytes_per_token, v.cl100k_bytes_per_token]})), ['ours', 'GPT-2', 'cl100k'], {fmt: v => v.toFixed(2)});
  lineChart(card(g, 'Token frequency by rank', 'Zipf-like rank-frequency curve on 20M training tokens (log-log)'),
    [{name: 'count', pts: t.rank_freq.map((c, i) => [Math.round(Math.pow(10, i * Math.log10(t.vocab_size) / 39)), Math.max(c, 1)])}], {logx: true, logy: true, xlabel: 'rank', ylabel: 'count'});
  tiles(gridIn(s), [['Round-trip exact', pct(t.roundtrip_exact)], ['OOV rate', pct(t.oov_rate), 'byte-level: always 0'], ['Vocab used', pct(t.vocab_used_frac)],
    ['Top-10 tokens share', pct(t.top10_token_share)], ['Train tokens', fmt((t.tokenize||{}).train_tokens)], ['Bytes / token (train)', fmt((t.tokenize||{}).bytes_per_token, 2)]]);
  const ex = card(gridIn(s), 'How sentences are split', 'pieces separated by │ ; counts for our tokenizer vs GPT-2 / cl100k');
  t.examples.forEach(e => ex.appendChild($(`<div><span class="note">${e.n_ours} tokens (GPT-2 ${e.n_gpt2}, cl100k ${e.n_cl100k})</span><pre></pre></div>`)).querySelector('pre').textContent = e.pieces.join('│')); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 4 hardware */
(function(){ try { const s = section('hardware', '4a · Hardware & distributed scaling', 'Throughput of the main model on 1 GPU vs 2 GPUs (DDP over PCIe, no NVLink), with torch.compile and ZeRO-1. Gradient accumulation means one all-reduce per optimizer step.');
  const g = gridIn(s), b = D.bench; if (!b) return empty(g);
  const lab = r => `${r.world} GPU · mb ${r.micro_batch}${r.compile?' · compile':''}${r.zero1?' · ZeRO-1':''}`;
  const rs = [...b.single, ...b.ddp];
  barChart(card(g, 'Training throughput (tokens/s)', 'out-of-memory configurations are listed as OOM in the table view',
      rs.map(r => ({config: lab(r), 'tok/s': r.ok ? r.tok_per_s : 'OOM', 'MFU': r.ok ? pct(r.mfu) : '–', 'peak GB': r.ok ? r.peak_mem_gb : '–', 'comm share': r.ok ? pct(r.comm_share) : '–'}))),
    rs.filter(r => r.ok).map(r => ({label: lab(r), values: [r.tok_per_s]})), ['tokens/s'], {labelW: 200});
  barChart(card(g, 'Peak GPU memory (GB)', 'per GPU; the cap is 85% of 8 GB for machine stability'),
    rs.filter(r => r.ok).map(r => ({label: lab(r), values: [r.peak_mem_gb]})), ['GB'], {labelW: 200, fmt: v => v.toFixed(2)});
  const best = b.best || {};
  tiles(gridIn(s), [['2-GPU scaling efficiency', pct(b.scaling_efficiency), 'tok/s(2 GPU) ÷ 2·tok/s(1 GPU)'],
    ['… if syncing every micro-step', pct(b.scaling_efficiency_naive_sync), 'naive DDP without no_sync'],
    ['Chosen config', `mb ${best.micro_batch}${best.compile?' + compile':''}`, `${fmt(best.tok_per_s)} tok/s · MFU ${pct(best.mfu)}`],
    ['All-reduce (fp32 grads)', best.t_allreduce ? (best.t_allreduce*1000).toFixed(0) + ' ms' : '–', `${pct(best.comm_share)} of a step`],
    ['Planned compute', fmt(b.planned_flops) + 'FLOPs', fmt(b.planned_tokens) + ' tokens']]); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 4 scaling laws */
(function(){ try { const s = section('scaling', '4b · Scaling laws & ablations', 'IsoFLOP sweep (Chinchilla approach 2) and a parametric fit L(N,D) = E + A/N^α + B/D^β, used to predict the main run before it starts.');
  const g = gridIn(s), f = D.scaling; if (!f) return empty(g);
  const iso = Object.entries(f.isoflop);
  lineChart(card(g, 'IsoFLOP curves: loss vs model size', 'one line per compute budget; the minimum of each curve is the compute-optimal size',
      f.runs.filter(r=>r.kind==='isoflop').map(r=>({run:r.name, 'budget FLOPs':r.budget, 'params (non-emb)':r.params_nonemb, tokens:r.tokens_trained, 'val loss':r.val_loss, kWh:r.kwh}))),
    iso.map(([C, e]) => ({name: C + ' FLOPs', pts: e.points.map(p => [p[0], p[2]]), markers: true})), {logx: true, xlabel: 'non-embedding parameters', ylabel: 'val loss'});
  const fit = f.parametric, mp = f.main_prediction;
  const actual = (D.pre_eval_curve || []).map(r => [r.tokens, r.val_loss]);
  lineChart(card(g, 'Main run: predicted vs actual validation loss', 'prediction from the sweep fit (dashed). WSD keeps LR high until the final decay, so loss sits above the fit until then.'),
    [{name: 'predicted L(N,D)', pts: mp.curve, dash: true}, {name: 'actual val loss', pts: actual}], {logx: true, xlabel: 'tokens', ylabel: 'val loss'});
  tiles(gridIn(s), [['Fit', `α=${fit.alpha.toFixed(2)} β=${fit.beta.toFixed(2)}`, `E=${fit.E.toFixed(2)} · R²=${fit.r2.toFixed(3)}`],
    ['Predicted final loss', mp.predicted_val_loss.toFixed(3), `${fmt(mp.params_nonemb)} params, ${fmt(mp.tokens)} tokens`],
    ['Actual final loss', D.pretrain ? D.pretrain.final_eval.val_loss.toFixed(3) : '–', D.pretrain ? `error ${(D.pretrain.final_eval.val_loss - mp.predicted_val_loss >= 0 ? '+' : '')}${(D.pretrain.final_eval.val_loss - mp.predicted_val_loss).toFixed(3)}` : ''],
    ['N* ∝ C^a', f.n_opt_power_law ? 'a = ' + f.n_opt_power_law.a.toFixed(2) : '–', 'Chinchilla ≈ 0.5'],
    ['Tokens / param (main)', mp.tokens_per_param.toFixed(1), 'Chinchilla ≈ 20']]);
  const ab = Object.entries(f.ablations).filter(([k]) => k !== 'baseline').sort((a,b)=>a[1].delta_vs_baseline-b[1].delta_vs_baseline);
  barChart(card(gridIn(s), 'Architecture ablations: Δ val loss vs the modern recipe', 'same compute; > 0 means worse than baseline (RoPE + GQA + pre-RMSNorm + dense SwiGLU)',
      ab.map(([k,v])=>({ablation:k, 'val loss':v.val_loss, 'Δ':v.delta_vs_baseline, params:v.params_nonemb}))),
    ab.map(([k, v]) => ({label: k, values: [v.delta_vs_baseline]})), ['Δ loss'], {labelW: 170, fmt: v => (v>=0?'+':'') + v.toFixed(3)}); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 4 training curves */
(function(){ try { const s = section('training', '4c · Pre-training dynamics', 'Next-token cross-entropy with teacher forcing, fp16 + loss scaling, WSD schedule, 2-GPU DDP.');
  const g = gridIn(s), tc = D.pre_train_curve; if (!tc.length) return empty(g);
  const ev = D.pre_eval_curve; const doms = Object.keys(ev[0] || {}).filter(k => k.startsWith('val_loss_')).map(k => k.slice(9));
  lineChart(card(g, 'Training loss', 'per optimizer step (both GPUs averaged)'), [{name: 'train loss', pts: tc.map(r => [r.tokens, r.loss])}], {xlabel: 'tokens', ylabel: 'loss'});
  lineChart(card(g, 'Validation loss per domain', 'held-out shards; the gap between domains reflects how predictable each one is'),
    doms.map(d => ({name: d, pts: ev.map(r => [r.tokens, r['val_loss_' + d]])})), {xlabel: 'tokens', ylabel: 'val loss'});
  lineChart(card(g, 'Bits per byte', 'tokenizer-independent: loss / ln2 / (bytes per token)'),
    doms.map(d => ({name: d, pts: ev.map(r => [r.tokens, r['bpb_' + d]])})), {xlabel: 'tokens', ylabel: 'bits/byte'});
  lineChart(card(g, 'HellaSwag during training', 'quick subset, length-normalised log-likelihood'),
    [{name: 'HellaSwag', pts: ev.filter(r => r.hellaswag !== undefined).map(r => [r.tokens, r.hellaswag]), markers: true}], {xlabel: 'tokens', yfmt: pct});
  lineChart(card(g, 'Learning rate (WSD)', 'warmup → stable → linear decay'), [{name: 'lr', pts: tc.map(r => [r.tokens, r.lr])}], {xlabel: 'tokens', yfmt: v => v.toExponential(1), yzero: true});
  lineChart(card(g, 'Gradient norm (before clipping at 1.0)'), [{name: 'grad norm', pts: tc.map(r => [r.tokens, r.grad_norm])}], {xlabel: 'tokens'});
  const gc = D.pre_grad_comp; if (gc.length) lineChart(card(g, 'Gradient norm by component'),
    ['embedding', 'attention', 'ffn', 'norm'].map(k => ({name: k, pts: gc.map(r => [r.tokens, r[k]])})), {xlabel: 'tokens', logy: true});
  lineChart(card(g, 'Model FLOPs utilisation (MFU)', 'dips are evaluation / checkpoint pauses'), [{name: 'MFU', pts: tc.map(r => [r.tokens, r.mfu])}], {xlabel: 'tokens', yfmt: pct, yzero: true});
  lineChart(card(g, 'fp16 loss scale', 'drops when an overflow skips a step'), [{name: 'loss scale', pts: tc.map(r => [r.tokens, r.loss_scale])}], {xlabel: 'tokens', logy: true});
  const pw = D.pre_power; if (pw.length) {
    lineChart(card(g, 'GPU power draw (W)', 'NVML, sampled every 10 s'), pw[0].gpus.map((_, i) => ({name: 'GPU ' + i, pts: pw.map(r => [r.t/3600, (r.gpus[i]||{}).w])})), {xlabel: 'hours', yzero: true});
    lineChart(card(g, 'GPU temperature (°C)'), pw[0].gpus.map((_, i) => ({name: 'GPU ' + i, pts: pw.map(r => [r.t/3600, (r.gpus[i]||{}).temp])})), {xlabel: 'hours'}); }
  lineChart(card(g, 'Cumulative energy (kWh)', 'GPU measured + CPU/system estimate'), [{name: 'kWh', pts: tc.filter(r => r.kwh).map(r => [r.tokens, r.kwh]), area: true}], {xlabel: 'tokens', yzero: true}); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* per-component */
(function(){ try { const s = section('components', '4d · Evaluating each part of the model', 'Ablations and probes on the pre-trained checkpoint: what each layer, head and component contributes.');
  const g = gridIn(s), pr = D.probes.base || D.probes.final; if (!pr) return empty(g);
  const bd = pr.breakdown.params;
  barChart(card(g, 'Parameters by component', 'embedding is tied with the LM head'), Object.entries(bd).map(([k, v]) => ({label: k, values: [v]})), ['parameters']);
  const la = pr.layer_ablation;
  barChart(card(g, 'Layer ablation: Δ loss when a sub-layer is skipped', 'larger = the layer matters more', la.map(r => ({layer: r.layer, 'attention Δ': r.attn_delta, 'FFN Δ': r.ffn_delta}))),
    la.map(r => ({label: 'layer ' + r.layer, values: [r.attn_delta, r.ffn_delta]})), ['attention', 'FFN'], {labelW: 70, fmt: v => v.toFixed(3)});
  if (pr.head_ablation) heatmap(card(g, 'Head ablation: Δ loss per attention head', 'rows = layers, columns = heads; most heads are individually redundant, a few are critical'),
    pr.head_ablation, pr.head_ablation.map((_, i) => 'L' + i), pr.head_ablation[0].map((_, j) => 'h' + j), {diverging: true, rowName: 'layer', colName: 'head', valName: 'Δ loss', fmt: v => v.toFixed(4)});
  const snaps = Object.keys(D.probes).filter(k => k.startsWith('snap_')).sort();
  lineChart(card(g, 'Logit lens: next-token accuracy decoded from each layer', 'shows where predictions form; one line per training checkpoint'),
    [...snaps, 'base'].filter(k => D.probes[k]).map(k => ({name: k === 'base' ? 'final' : k.replace('snap_', '') + '% trained', pts: D.probes[k].logit_lens.acc.map((v, i) => [i, v])})).slice(-8),
    {xlabel: 'layer (0 = embeddings)', yfmt: pct, markers: true});
  if (pr.attention) {
    heatmap(card(g, 'Attention entropy (nats) per head', 'low = focused on few tokens, high = diffuse'), pr.attention.entropy, pr.attention.entropy.map((_, i) => 'L' + i),
      pr.attention.entropy[0].map((_, j) => 'h' + j), {rowName: 'layer', colName: 'head', valName: 'entropy', fmt: v => v.toFixed(2)});
    heatmap(card(g, 'Attention mass by relative distance', 'RoPE locality: share of attention going to tokens d positions back'), pr.attention.mass_by_distance,
      pr.attention.mass_by_distance.map((_, i) => 'L' + i), pr.attention.distance_buckets.map(b => '≥' + b), {rowName: 'layer', colName: 'distance', valName: 'mass', fmt: pct, xlabel: 'tokens back'});
    const mk = Object.keys(pr.attention.maps).pop(), toks = pr.attention.map_tokens.map(t => JSON.stringify(t).slice(1, -1) || '·');
    heatmap(card(g, `Attention map: layer ${mk}, head 0`, 'sample sentence; rows = query token, columns = key token'), pr.attention.maps[mk][0], toks, toks,
      {rowName: 'query', colName: 'key', valName: 'weight', fmt: v => v.toFixed(3), labelW: 80}); }
  lineChart(card(g, 'Residual stream RMS and outliers per layer', 'max/RMS shows the activation outliers that make low-precision inference hard'),
    [{name: 'RMS', pts: pr.logit_lens.resid_rms.map((v, i) => [i, v])}, {name: 'max / RMS', pts: pr.logit_lens.max_over_rms.map((v, i) => [i, v])}], {xlabel: 'layer', markers: true});
  if (pr.embeddings) { const c = card(gridIn(s), 'Embedding space: nearest neighbours and analogies', 'cosine similarity of the (tied) input embeddings');
    c.appendChild(tableEl(Object.entries(pr.embeddings.neighbours).map(([w, n]) => ({word: w, neighbours: n.map(x => x[0] + ' ' + x[1].toFixed(2)).join(', ')}))));
    c.appendChild(tableEl(Object.entries(pr.embeddings.analogies).map(([q, a]) => ({analogy: q, expected: a.expected, top: a.top.map(x => x[0]).join(', '), 'hit@5': a['hit@5'] ? 'yes' : 'no'})))); } } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 5 generation */
(function(){ try { const s = section('generation', '5 · Generation & inference', 'Autoregressive decoding with a KV cache, sampling strategies, beam search, guided decoding and speculative decoding.');
  const g = gridIn(s), gen = D.generation.base || D.generation.final; if (!gen) return empty(g);
  const sp = Object.entries(gen.speed);
  barChart(card(g, 'Decoding speed: KV cache vs recomputing', 'tokens/s generated', sp.map(([k, v]) => ({batch: k, 'with cache': v.tok_s_cache, 'no cache': v.tok_s_nocache, 'speed-up': v.speedup}))),
    sp.map(([k, v]) => ({label: k, values: [v.tok_s_cache, v.tok_s_nocache]})), ['KV cache', 'no cache']);
  const kv = gen.kv_cache;
  barChart(card(g, 'KV-cache memory at full context', `fp16, ${kv.seq_len} tokens: MHA (${kv.n_head} KV heads) vs our GQA (${kv.n_kv_head}) vs MQA (1)`),
    [{label: 'MHA', values: [kv.mha/2**20]}, {label: 'GQA (ours)', values: [kv.gqa_current/2**20]}, {label: 'MQA', values: [kv.mqa/2**20]}], ['MiB'], {fmt: v => v.toFixed(1)});
  const st = Object.entries(gen.strategies);
  barChart(card(g, 'Decoding strategies: diversity vs repetition', 'distinct-2 = unique bigrams; repetition = repeated 4-grams within a sample',
      st.map(([k, v]) => ({strategy: k, 'distinct-1': v.distinct_1, 'distinct-2': v.distinct_2, 'repetition': v.repetition_4gram, 'mean length': v.mean_len}))),
    st.map(([k, v]) => ({label: k, values: [v.distinct_2, v.repetition_4gram]})), ['distinct-2', 'repetition (4-gram)'], {fmt: v => v.toFixed(2)});
  if (gen.speculative) { const sk = Object.entries(gen.speculative.by_k);
    barChart(card(g, 'Speculative decoding', `draft ${fmt(gen.speculative.draft_params)} params → target ${fmt(gen.speculative.target_params)}; greedy acceptance`,
        sk.map(([k, v]) => ({'draft length k': k, 'acceptance': pct(v.acceptance_rate), 'speed-up': v.speedup}))),
      sk.map(([k, v]) => ({label: 'k = ' + k, values: [v.acceptance_rate, v.speedup]})), ['acceptance rate', 'speed-up ×'], {fmt: v => v.toFixed(2)}); }
  tiles(gridIn(s), [['Guided decoding validity', pct(gen.guided.valid_guided), `unconstrained: ${pct(gen.guided.valid_unconstrained)} valid JSON`],
    ['KV cache output = recompute', gen.cache_matches_nocache ? 'yes' : 'NO'],
    ['Perplexity (web)', fmt(gen.perplexity.web, 1)], ['Perplexity (code)', fmt(gen.perplexity.code, 1)],
    ['Beam vs greedy log-prob', gen.beam_vs_greedy ? (gen.beam_vs_greedy.reduce((a, r) => a + r.beam_logprob - r.greedy_logprob, 0) / gen.beam_vs_greedy.length).toFixed(2) : '–', 'mean gain of beam search (nats)']]);
  const c = card(gridIn(s), 'Samples from the base model', 'prompt + continuation, per decoding strategy');
  Object.entries(gen.samples).forEach(([k, v]) => { c.appendChild($(`<div><b>${k}</b><pre></pre></div>`)).querySelector('pre').textContent = v.slice(0, 3).join('\n\n———\n\n'); }); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 6 mid */
(function(){ try { const s = section('mid', '6 · Mid-training & long context', 'Higher-quality data + synthetic needles + replay, best-fit packing, 1k → 2k → 4k curriculum with YaRN RoPE scaling.');
  const g = gridIn(s);
  const nd = D.needle; const tags = ['before', 'before_yarn', 'after'].filter(t => nd[t]);
  const names = {before: 'Before mid-training (no RoPE scaling)', before_yarn: 'Before, with YaRN ×4 at inference only', after: 'After mid-training'};
  tags.forEach(t => heatmap(card(g, 'Needle in a haystack: ' + names[t], `mean accuracy ${pct(nd[t].mean_accuracy)}; rows = context length, columns = needle depth`),
    nd[t].accuracy, nd[t].lengths.map(String), nd[t].depths.map(d => (d * 100) + '%'), {lo: 0, hi: 1, rowName: 'length', colName: 'depth', valName: 'accuracy', fmt: pct, labelW: 50}));
  if (tags.length) lineChart(card(g, 'Loss by token position', 'falling loss deep into the context = the model uses distant tokens'),
    tags.map(t => ({name: names[t], pts: nd[t].loss_by_position.bucket_start.map((b, i) => [Math.max(b, 1), nd[t].loss_by_position.loss[i]])})), {logx: true, xlabel: 'position', ylabel: 'loss'});
  const mc = Object.entries(D.mid_curve);
  if (mc.length) lineChart(card(g, 'Mid-training loss per curriculum phase'), mc.map(([L, r]) => ({name: 'L = ' + L, pts: r.map(x => [x.step, x.loss])})), {xlabel: 'step in phase', ylabel: 'loss'});
  const m = D.mid; if (m) { const b = m.short_val_before, a = m.short_val_after;
    barChart(card(g, 'Short-context val loss before vs after (regression check)', 'mid-training should help math/code and not hurt web text much'),
      Object.keys(b).map(k => ({label: k.replace('val_loss_', ''), values: [b[k], a[k]]})), ['before', 'after'], {fmt: v => v.toFixed(3)}); }
  if (!tags.length && !m) empty(g); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* 7 post-training */
(function(){ try { const s = section('post', '7 · Post-training', 'SFT (loss on assistant tokens only) and a LoRA comparison → Bradley-Terry reward model → DPO (β = 0.1) → GRPO with verifiable math rewards (Dr.GRPO/DAPO fixes) → on-policy self-distillation (reverse KL).');
  const g = gridIn(s), P = D.post; let any = false;
  const sft = P.sft.curve.filter(r => r.loss !== undefined), lora = P.sft_lora.curve.filter(r => r.loss !== undefined);
  if (sft.length) { any = true; lineChart(card(g, 'SFT training loss: full fine-tune vs LoRA (r = 16, attention + MLP)', 'LoRA runs for a fraction of the steps'),
    [{name: 'full fine-tune', pts: sft.map(r => [r.step, r.loss])}, {name: 'LoRA', pts: lora.map(r => [r.step, r.loss])}], {xlabel: 'step', ylabel: 'loss'});
    const fs = P.sft.summary || {}, ls = P.sft_lora.summary || {};
    const ev = P.sft.curve.filter(r => r.eval); if (ev.length) lineChart(card(g, 'SFT validation loss by data type'),
      ['chat', 'code', 'math', 'gsm8k'].map(k => ({name: k, pts: ev.map(r => [r.step, r.eval['val_loss_' + k]])})), {xlabel: 'step', ylabel: 'val loss'});
    tiles(gridIn(s), [['Full FT trainable params', fmt(fs.trainable_params)], ['LoRA trainable params', fmt(ls.trainable_params), ls.total_params ? pct(ls.trainable_params / ls.total_params) + ' of model' : ''],
      ['Peak memory full / LoRA', `${fmt(fs.peak_mem_gb, 2)} / ${fmt(ls.peak_mem_gb, 2)} GB`], ['sec / step full / LoRA', `${fmt(fs.sec_per_step, 2)} / ${fmt(ls.sec_per_step, 2)}`],
      ['Final val loss full / LoRA', `${fmt((fs.final_eval||{}).val_loss, 3)} / ${fmt((ls.final_eval||{}).val_loss, 3)}`]]); }
  const rm = P.reward_model.summary; if (rm) { any = true; const c = P.reward_model.curve.filter(r => r.acc !== undefined);
    lineChart(card(g, 'Reward model: training accuracy and loss', `held-out pairwise accuracy ${pct(rm.before.pairwise_acc)} → ${pct(rm.after.pairwise_acc)} (chance 50%)`),
      [{name: 'train loss', pts: c.map(r => [r.step, r.loss])}, {name: 'train accuracy', pts: c.map(r => [r.step, r.acc])}], {xlabel: 'step'}); }
  const dpo = P.dpo.curve.filter(r => r.margin !== undefined); if (dpo.length) { any = true;
    lineChart(card(g, 'DPO: implicit rewards β·log(π/π_ref)', 'chosen should rise relative to rejected; the margin is what the loss pushes up'),
      [{name: 'chosen', pts: dpo.map(r => [r.step, r.reward_chosen])}, {name: 'rejected', pts: dpo.map(r => [r.step, r.reward_rejected])}, {name: 'margin', pts: dpo.map(r => [r.step, r.margin])}], {xlabel: 'step'});
    lineChart(card(g, 'DPO loss and preference accuracy', `held-out accuracy ${pct(P.dpo.summary?.before?.pref_acc)} → ${pct(P.dpo.summary?.after?.pref_acc)}`),
      [{name: 'loss', pts: dpo.map(r => [r.step, r.loss])}, {name: 'batch accuracy', pts: dpo.map(r => [r.step, r.acc])}], {xlabel: 'step'}); }
  const gr = P.grpo.curve.filter(r => r.reward !== undefined); if (gr.length) { any = true;
    lineChart(card(g, 'GRPO: accuracy and reward on verifiable math', 'reward = correct + 0.1 × format; curriculum unlocks harder levels at 60% accuracy'),
      [{name: 'accuracy', pts: gr.map(r => [r.step, r.acc])}, {name: 'reward', pts: gr.map(r => [r.step, r.reward])}], {xlabel: 'RL step', y0: 0});
    lineChart(card(g, 'GRPO: response length', 'reasoning chains grow as RL proceeds (Lecture 4); DAPO token-level normalisation prevents length inflation from wrong answers'),
      [{name: 'mean response tokens', pts: gr.map(r => [r.step, r.resp_len])}], {xlabel: 'RL step'});
    lineChart(card(g, 'GRPO: KL to reference and clip fraction'), [{name: 'KL (k3)', pts: gr.map(r => [r.step, r.kl])}, {name: 'clip fraction', pts: gr.map(r => [r.step, r.clip_frac])},
      {name: 'zero-variance groups', pts: gr.map(r => [r.step, r.zero_var_groups])}], {xlabel: 'RL step', yzero: true});
    lineChart(card(g, 'GRPO: curriculum level'), [{name: 'max level', pts: gr.map(r => [r.step, r.max_level])}], {xlabel: 'RL step'}); }
  const ds = P.distill.curve.filter(r => r.reverse_kl !== undefined); if (ds.length) { any = true;
    lineChart(card(g, 'On-policy self-distillation: reverse KL(student ‖ teacher-with-answer)', 'dense per-token signal on the student\'s own samples'),
      ['chat', 'code', 'math'].map(k => ({name: k, pts: ds.filter(r => r.kl_by_kind && r.kl_by_kind[k] !== undefined).map(r => [r.step, r.kl_by_kind[k]])})), {xlabel: 'step', yzero: true}); }
  const bon = P.best_of_n.summary; if (bon) { any = true; const cv = Object.entries(bon.curve);
    lineChart(card(g, 'Best-of-N with the reward model', 'mean RM score of the selected response vs a random sample'),
      [{name: 'best of N', pts: cv.map(([n, v]) => [+n, v.best_of_n]), markers: true}, {name: 'random sample', pts: cv.map(([n, v]) => [+n, v.random]), markers: true}], {logx: true, xlabel: 'N'}); }
  if (!any) empty(g); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* capability matrix */
(function(){ try { const s = section('capabilities', '8 · Capability matrix across stages', 'Every checkpoint evaluated the same way. Base / mid models are prompted few-shot; post-trained models use the chat template.');
  const g = gridIn(s), E = D.evals, ord = D.eval_order; if (!ord.length) return empty(g);
  const metric = {
    'HellaSwag': e => e.benchmarks?.hellaswag?.acc_norm, 'ARC-Easy': e => e.benchmarks?.arc_easy?.acc_norm, 'PIQA': e => e.benchmarks?.piqa?.acc_norm,
    'LAMBADA': e => e.benchmarks?.lambada?.acc, 'Math (synthetic)': e => e.math?.synthetic_acc, 'Math cons@8': e => e.math?.['cons@8'], 'GSM8K': e => e.math?.gsm8k_acc,
    'HumanEval pass@1': e => e.code?.['humaneval_pass@1_greedy'], 'HumanEval pass@10': e => e.code?.['humaneval_pass@10'], 'MBPP pass@1': e => e.code?.['mbpp_pass@1_greedy'],
    'Instruction following': e => e.instruction_following?.overall, 'Refusal (harmful)': e => e.safety_refusal_rate };
  const rows = ord.map(t => { const r = {stage: t, 'val bpb': E[t].val?.mean_bpb}; Object.entries(metric).forEach(([k, f]) => { const v = f(E[t]); r[k] = v === undefined ? '–' : pct(v); });
    r['RM score'] = E[t].reward_model_score; return r; });
  const c = card(g, 'All metrics by stage', 'percentages unless noted; RM score is the reward model\'s mean score on held-out prompts');
  c.appendChild(tableEl(rows));
  const groups = [['Knowledge (zero-shot)', ['HellaSwag', 'ARC-Easy', 'PIQA', 'LAMBADA']], ['Reasoning & code', ['Math (synthetic)', 'GSM8K', 'HumanEval pass@1', 'MBPP pass@1']],
    ['Behaviour', ['Instruction following', 'Refusal (harmful)', 'Math cons@8', 'HumanEval pass@10']]];
  const g2 = gridIn(s);
  groups.forEach(([title, ms]) => lineChart(card(g2, title, 'one line per metric across pipeline stages (x = stage index, see table)'),
    ms.map(mn => ({name: mn, pts: ord.map((t, i) => [i, metric[mn](E[t]) ?? null]), markers: true})), {xfmt: i => ord[Math.round(i)] || '', yfmt: pct, yzero: true}));
  const J = D.judge; if (J) { const pairs = Object.entries(J).filter(([k]) => k !== 'energy');
    barChart(card(gridIn(s), 'LLM-as-a-judge (Qwen2.5-1.5B-Instruct) pairwise win-rate', 'both answer orders judged; inconsistent verdicts count as ties; 50% = no difference',
        pairs.map(([k, v]) => ({pair: k.replace('_vs_', ' vs '), win: pct(v.win), loss: pct(v.loss), tie: pct(v.tie), 'win-rate': pct(v.win_rate), consistency: pct(v.position_consistency)}))),
      pairs.map(([k, v]) => ({label: k.replace('_vs_', ' vs '), values: [v.win_rate]})), ['win-rate (ties = ½)'], {labelW: 140, fmt: pct}); }
  const sc = card(gridIn(s), 'Chat samples by stage', 'temperature 0.7, top-p 0.9');
  ord.filter(t => E[t].samples).forEach(t => { sc.appendChild($(`<div><b>${t}</b><pre></pre></div>`)).querySelector('pre').textContent =
    Object.entries(E[t].samples).slice(0, 3).map(([q, a]) => '» ' + q + '\n' + a).join('\n\n'); }); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

/* energy */
(function(){ try { const s = section('energy', '9 · Energy & cost', 'GPU energy from the NVML energy counter (measured). CPU energy from utilisation × TDP when RAPL is not readable (estimate), plus a constant for the rest of the system.');
  const g = gridIn(s), en = D.energy; if (!en.length) return empty(g);
  barChart(card(g, 'Energy per stage (kWh)', 'GPU measured, CPU and rest-of-system estimated', en.map(e => ({stage: e.stage, 'total kWh': e.total_kwh, 'GPU kWh': e.gpu_kwh, 'CPU kWh': e.cpu_kwh,
      'rest kWh': e.rest_kwh, hours: (e.seconds || 0) / 3600, 'kg CO₂e': e.co2_kg, cost: e.cost, 'J / token': e.j_per_token}))),
    en.map(e => ({label: e.stage, values: [e.gpu_kwh, e.cpu_kwh, e.rest_kwh]})), ['GPU', 'CPU (est.)', 'rest (est.)'], {labelW: 190, fmt: v => v.toFixed(3)});
  const tot = k => en.reduce((a, e) => a + (e[k] || 0), 0); const pre = en.find(e => e.stage === '4 pre-training') || {};
  tiles(gridIn(s), [['Total', tot('total_kwh').toFixed(2) + ' kWh', `${tot('gpu_kwh').toFixed(2)} kWh of it on GPUs`], ['CO₂e', tot('co2_kg').toFixed(2) + ' kg', 'grid factor in summary assumptions'],
    ['Cost', tot('cost').toFixed(2), 'at the configured price per kWh'], ['Pre-training share', pct((pre.total_kwh || 0) / (tot('total_kwh') || 1))],
    ['Pre-training J / token', pre.j_per_token ? pre.j_per_token.toFixed(3) : '–'], ['Avg GPU power (pre-train)', pre.avg_gpu_w ? pre.avg_gpu_w.map(w => w.toFixed(0) + ' W').join(' / ') : '–']]); } catch (e) { console.error(e); main.appendChild($(`<div class="warn">section failed: ${e.message}</div>`)); } })();

LATER.forEach(f => { try { f(); } catch (e) { console.error(e); } });

/* theme toggle */
document.getElementById('theme').onclick = () => { const r = document.documentElement, dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const cur = r.dataset.theme || (dark ? 'dark' : 'light'); r.dataset.theme = cur === 'dark' ? 'light' : 'dark'; };
</script></body></html>
"""


def main():
    D = gather()
    os.makedirs(REPORT_DIR, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(D, default=float).replace("</", "<\\/"))
    path = os.path.join(REPORT_DIR, "index.html")
    with open(path, "w") as f:
        f.write(html)
    p(f"report written: {path} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
