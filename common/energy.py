"""Energy measurement.

GPU: NVML total-energy counter (mJ, measured) + power/temp/util/clock samples.
CPU: RAPL if readable, otherwise estimated from /proc/stat utilisation x TDP
(flagged as an estimate). Rest of system: constant estimate.
"""
import json
import os
import threading
import time

from common.config import ENERGY

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:  # pragma: no cover
    _NVML = False

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"


def _rapl_uj():
    try:
        with open(RAPL) as f:
            return int(f.read())
    except Exception:
        return None


def _cpu_times():
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle = vals[3] + vals[4]
    return sum(vals), idle


class EnergyMeter:
    def __init__(self, out_dir=None, interval=1.0, log_every=10.0, name="energy"):
        self.interval, self.log_every = interval, log_every
        self.out = open(os.path.join(out_dir, f"{name}.jsonl"), "w") if out_dir else None
        self.handles = []
        if _NVML:
            idx = list(range(pynvml.nvmlDeviceGetCount()))
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis:   # a single-GPU job only accounts for its own GPU (requires CUDA_DEVICE_ORDER=PCI_BUS_ID)
                idx = [int(v) for v in vis.split(",") if v.strip().isdigit()]
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
        # share of CPU / rest-of-system power attributed to this job (concurrent jobs split it)
        self.share = float(os.environ.get("LLM_ENERGY_SHARE", "1.0"))
        self._stop = threading.Event()
        self.cpu_measured = _rapl_uj() is not None
        self.samples = 0
        self.power_sum = [0.0] * len(self.handles)
        self.peak_temp = [0] * len(self.handles)
        self.util_sum = [0.0] * len(self.handles)
        self.cpu_j = 0.0
        self.tokens = 0

    def _gpu_mj(self):
        return [pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles]

    def start(self):
        self.t0 = time.time()
        self.e0 = self._gpu_mj()
        self.r0 = _rapl_uj()
        self.c0 = _cpu_times()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def _run(self):
        last_log, last_cpu = time.time(), self.c0
        last_t = time.time()
        while not self._stop.wait(self.interval):
            now = time.time()
            row = {"t": round(now - self.t0, 1), "gpus": []}
            for i, h in enumerate(self.handles):
                try:
                    pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    tp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    ut = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30
                    clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                except Exception:
                    continue
                self.power_sum[i] += pw
                self.util_sum[i] += ut
                self.peak_temp[i] = max(self.peak_temp[i], tp)
                row["gpus"].append({"w": round(pw, 1), "temp": tp, "util": ut, "mem_gb": round(mem, 2), "mhz": clk})
            if not self.cpu_measured:
                c = _cpu_times()
                dt_tot, dt_idle = c[0] - last_cpu[0], c[1] - last_cpu[1]
                util = 1 - dt_idle / max(dt_tot, 1)
                w = ENERGY["cpu_tdp_w"] * (ENERGY["cpu_idle_frac"] + (1 - ENERGY["cpu_idle_frac"]) * util)
                self.cpu_j += w * (now - last_t)
                row["cpu_util"] = round(util, 3)
                last_cpu = c
            last_t = now
            self.samples += 1
            if self.out and now - last_log >= self.log_every:
                self.out.write(json.dumps(row) + "\n")
                self.out.flush()
                last_log = now

    def add_tokens(self, n):
        self.tokens += n

    def summary(self):
        dt = time.time() - self.t0
        e1 = self._gpu_mj()
        gpu_j = [(b - a) / 1000.0 for a, b in zip(self.e0, e1)]
        if self.cpu_measured:
            r1 = _rapl_uj()
            cpu_j = (r1 - self.r0) / 1e6 if r1 >= self.r0 else 0.0
        else:
            cpu_j = self.cpu_j
        cpu_j *= self.share
        rest_j = ENERGY["rest_of_system_w"] * dt * self.share
        total_j = sum(gpu_j) + cpu_j + rest_j
        kwh = total_j / 3.6e6
        n = max(self.samples, 1)
        return {
            "seconds": round(dt, 1),
            "gpu_joules": [round(x, 1) for x in gpu_j],
            "gpu_kwh": sum(gpu_j) / 3.6e6,
            "cpu_kwh": cpu_j / 3.6e6,
            "cpu_measured": self.cpu_measured,
            "rest_kwh": rest_j / 3.6e6,
            "total_kwh": kwh,
            "avg_gpu_w": [round(s / n, 1) for s in self.power_sum],
            "avg_gpu_util": [round(s / n, 1) for s in self.util_sum],
            "peak_gpu_temp": self.peak_temp,
            "avg_total_w": total_j / max(dt, 1e-9),
            "co2_kg": kwh * ENERGY["grid_kg_co2_per_kwh"],
            "cost": kwh * ENERGY["price_per_kwh"],
            "tokens": self.tokens,
            "j_per_token": (total_j / self.tokens) if self.tokens else None,
            "assumptions": ENERGY,
        }

    def stop(self):
        self._stop.set()
        self._th.join(timeout=5)
        s = self.summary()
        if self.out:
            self.out.close()
        return s


class NullMeter:
    def start(self):
        return self

    def add_tokens(self, n):
        pass

    def stop(self):
        return {}

    def summary(self):
        return {}
