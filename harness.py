"""
Harness de mesure — Phase 1
Carte 1 : Adaptive Compute / Dynamic Scheduler GPU

Deux modes :
  --mode baseline         : mesure le padding naïf sur toutes les distributions
                            → génère baseline_results.csv
                            C'est le zéro de mesure. Ne pas modifier ce CSV à la main.

  --mode compare_scheduling : compare padding / token_aligned / warp_aligned
                              sur les mêmes distributions
                              → génère compare_results.csv
                              Premier signal empirique : est-ce que warp_aligned > token_aligned ?
                              Si non, l'overhead de partition domine — investiguer avec nsight.

Usage :
    python harness.py --mode baseline --batch 256 --seq 64 --d_model 512 --T_max 12 --dtype bfloat16
    python harness.py --mode compare_scheduling --batch 256 --seq 64 --d_model 512 --T_max 12
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    batch_size: int = 256
    seq_len: int = 64
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    T_max: int = 12
    warp_size: int = 32
    warmup: int = 50
    runs: int = 200
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}

DistName = Literal["uniform", "gaussian_low", "gaussian_high", "bimodal", "constant_max"]

DISTRIBUTIONS: list[DistName] = [
    "constant_max",
    "gaussian_high",
    "bimodal",
    "uniform",
    "gaussian_low",
]


# ---------------------------------------------------------------------------
# Distributions de profondeurs
# ---------------------------------------------------------------------------

def generate_depths(batch_size: int, T_max: int, dist: DistName, device: torch.device) -> torch.Tensor:
    """
    Retourne (batch_size,) d'entiers dans [1, T_max].
    Chaque valeur = nombre de couches que ce sample va traverser.

    constant_max  : tous à T_max — borne de référence, 0% gaspillé
    gaussian_high : moyenne 3*T_max/4 — ~25% gaspillé
    bimodal       : 50% à T_max/4 + 50% à 3*T_max/4 — ~50% gaspillé
    uniform       : [1, T_max] équiprobable — ~50% gaspillé
    gaussian_low  : moyenne T_max/4 — ~75% gaspillé, meilleur cas théorique
    """
    if dist == "constant_max":
        depths = torch.full((batch_size,), T_max, dtype=torch.long)
    elif dist == "gaussian_high":
        raw = torch.normal(3 * T_max / 4, T_max / 8, (batch_size,))
        depths = raw.round().long().clamp(1, T_max)
    elif dist == "gaussian_low":
        raw = torch.normal(T_max / 4, T_max / 8, (batch_size,))
        depths = raw.round().long().clamp(1, T_max)
    elif dist == "bimodal":
        half = batch_size // 2
        low  = torch.normal(T_max / 4,     T_max / 12, (half,))
        high = torch.normal(3 * T_max / 4, T_max / 12, (batch_size - half,))
        depths = torch.cat([low, high]).round().long().clamp(1, T_max)
    elif dist == "uniform":
        depths = torch.randint(1, T_max + 1, (batch_size,))
    else:
        raise ValueError(f"Distribution inconnue : {dist}")
    return depths.to(device)


def depth_stats(depths: torch.Tensor, T_max: int) -> dict:
    d = depths.float()
    ecr = (d.mean() / T_max).item()
    return {
        "mean":                    round(d.mean().item(), 2),
        "std":                     round(d.std().item(), 2),
        "min":                     int(d.min().item()),
        "max":                     int(d.max().item()),
        "effective_compute_ratio": round(ecr, 3),
        "compute_wasted_pct":      round((1 - ecr) * 100, 1),
        "theoretical_speedup":     round(1.0 / ecr, 2) if ecr > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Modèle : couche unique récurrente (Universal Transformer style)
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff2(F.gelu(self.ff1(x))))


class HaltHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq, d_model) → (batch, seq)
        return torch.sigmoid(self.linear(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Stratégies de forward
# ---------------------------------------------------------------------------

def padded_forward(
    layer: TransformerLayer,
    halt_head: HaltHead,
    x: torch.Tensor,
    depths: torch.Tensor,
    T_max: int,
) -> torch.Tensor:
    """
    Padding naïf : T_max couches sur tout le batch, quelles que soient les profondeurs.
    C'est ce que font tous les papiers ACT/Universal Transformer existants.
    Toutes les distributions ont la même latence — c'est le problème.
    """
    h = x.clone()
    for _ in range(T_max):
        h = layer(h)
        _ = halt_head(h)
    return h


def token_aligned_forward(
    layer: TransformerLayer,
    halt_head: HaltHead,
    x: torch.Tensor,
    depths: torch.Tensor,
    T_max: int,
) -> torch.Tensor:
    """
    Skip par token individuel : à chaque couche, les tokens dont depth <= t
    ne passent pas dans la couche (leur état est figé).

    En pratique sur GPU, ça ne change rien à la latence — le warp exécute quand même
    32 threads même si 30 sont des noops. La divergence de warp annule le gain théorique.

    Cette stratégie est là pour démontrer empiriquement que token_aligned ≈ padding.
    Si ce n'est pas le cas, investiguer (peut-être que le compilateur optimise mieux qu'attendu).
    """
    h = x.clone()
    batch_size = x.shape[0]

    for t in range(T_max):
        # Masque des tokens encore actifs à cette couche
        active_mask = (depths > t)  # (batch,)

        if active_mask.any():
            h_new = layer(h)
            _ = halt_head(h_new)
            # Mise à jour sélective — inactive sur GPU (pas de gain réel)
            active_mask_expanded = active_mask.view(-1, 1, 1).expand_as(h)
            h = torch.where(active_mask_expanded, h_new, h)

    return h


def warp_aligned_forward(
    layer: TransformerLayer,
    halt_head: HaltHead,
    x: torch.Tensor,
    depths: torch.Tensor,
    T_max: int,
    warp_size: int = 32,
) -> torch.Tensor:
    """
    Skip par groupe de warp_size samples.
    Un groupe est actif si AU MOINS UN sample du groupe a depth > t.
    Un groupe est stoppé seulement si TOUS ses samples ont depth <= t.

    C'est la seule granularité qui évite la divergence de warp.
    Le gain est réel seulement quand des groupes entiers sont stoppés —
    ce qui correspond à des samples entiers de difficulté homogène.

    Note : cette implémentation est une simulation Python du comportement warp-aligned.
    Le vrai scheduler CUDA (scheduler/scheduler.py) fait la même chose mais sans
    exécuter la couche sur les groupes inactifs. Ici on exécute quand même
    (pour mesurer le comportement, pas l'accélération finale).
    La version accélérée viendra en étape 3.
    """
    h = x.clone()
    batch_size = x.shape[0]
    n_warps = batch_size // warp_size

    for t in range(T_max):
        # Profondeur min par groupe : groupe actif si min(depths[group]) > t
        depths_grouped = depths[:n_warps * warp_size].view(n_warps, warp_size)
        group_active = depths_grouped.min(dim=1).values > t  # (n_warps,)

        if group_active.any():
            h_new = layer(h)
            _ = halt_head(h_new)

            # Mise à jour uniquement pour les groupes actifs
            group_active_expanded = (
                group_active.unsqueeze(1)
                             .expand(-1, warp_size)
                             .reshape(n_warps * warp_size)
            )
            if batch_size > n_warps * warp_size:
                # Tokens restants si batch_size n'est pas divisible par warp_size
                remainder = torch.ones(batch_size - n_warps * warp_size, dtype=torch.bool, device=x.device)
                group_active_expanded = torch.cat([group_active_expanded, remainder])

            mask = group_active_expanded.view(-1, 1, 1).expand_as(h)
            h = torch.where(mask, h_new, h)

    return h


# ---------------------------------------------------------------------------
# Timer CUDA
# ---------------------------------------------------------------------------

class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)


# ---------------------------------------------------------------------------
# Benchmark d'une stratégie sur une distribution
# ---------------------------------------------------------------------------

def benchmark_one(
    layer: TransformerLayer,
    halt_head: HaltHead,
    cfg: Config,
    dist: DistName,
    strategy: str,
) -> dict:
    device = torch.device(cfg.device)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=device, dtype=cfg.dtype)

    forward_fn = {
        "padding":       lambda h, d: padded_forward(layer, halt_head, h, d, cfg.T_max),
        "token_aligned": lambda h, d: token_aligned_forward(layer, halt_head, h, d, cfg.T_max),
        "warp_aligned":  lambda h, d: warp_aligned_forward(layer, halt_head, h, d, cfg.T_max, cfg.warp_size),
    }[strategy]

    # Warmup
    for _ in range(cfg.warmup):
        depths = generate_depths(cfg.batch_size, cfg.T_max, dist, device)
        with torch.no_grad():
            forward_fn(x, depths)
    torch.cuda.synchronize()

    # Mesure
    latencies = []
    for _ in range(cfg.runs):
        depths = generate_depths(cfg.batch_size, cfg.T_max, dist, device)
        with CudaTimer() as timer:
            with torch.no_grad():
                forward_fn(x, depths)
        latencies.append(timer.elapsed_ms)

    lats = torch.tensor(latencies)
    mean_ms = lats.mean().item()
    total_tokens = cfg.batch_size * cfg.seq_len
    throughput = total_tokens / (mean_ms / 1000.0)

    sample_depths = generate_depths(cfg.batch_size, cfg.T_max, dist, device)
    ds = depth_stats(sample_depths, cfg.T_max)

    return {
        "strategy":                    strategy,
        "dist":                        dist,
        "depth_mean":                  ds["mean"],
        "depth_std":                   ds["std"],
        "effective_compute_ratio":     ds["effective_compute_ratio"],
        "compute_wasted_pct":          ds["compute_wasted_pct"],
        "theoretical_speedup":         ds["theoretical_speedup"],
        "latency_mean_ms":             round(mean_ms, 3),
        "latency_std_ms":              round(lats.std().item(), 3),
        "latency_p95_ms":              round(lats.quantile(0.95).item(), 3),
        "throughput_tokens_per_sec":   round(throughput),
        "gpu":                         torch.cuda.get_device_name(0),
    }


# ---------------------------------------------------------------------------
# Mode baseline
# ---------------------------------------------------------------------------

def run_baseline(cfg: Config) -> list[dict]:
    device = torch.device(cfg.device)
    layer     = TransformerLayer(cfg.d_model, cfg.n_heads, cfg.d_ff).to(device).to(cfg.dtype)
    halt_head = HaltHead(cfg.d_model).to(device).to(cfg.dtype)

    results = []
    for dist in DISTRIBUTIONS:
        print(f"  {dist}...", end=" ", flush=True)
        r = benchmark_one(layer, halt_head, cfg, dist, "padding")
        results.append(r)
        print(f"{r['latency_mean_ms']:.1f} ms")

    print_baseline_report(results, cfg)
    return results


def print_baseline_report(results: list[dict], cfg: Config):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"BASELINE — Padding naïf sur {results[0]['gpu']}")
    print(f"batch={cfg.batch_size}  seq={cfg.seq_len}  d_model={cfg.d_model}  T_max={cfg.T_max}  dtype={cfg.dtype}")
    print(sep)
    print(f"{'Distribution':<20} {'Depth mean±std':>16} {'Wasted%':>8} {'Latency ms':>12} {'Tokens/s':>12} {'Speedup pot.':>13}")
    print("-" * 83)
    for r in results:
        depth_str = f"{r['depth_mean']}±{r['depth_std']}"
        vs = "(ref)" if r["dist"] == "constant_max" else f"×{r['theoretical_speedup']}"
        print(
            f"{r['dist']:<20} {depth_str:>16} {r['compute_wasted_pct']:>7}% "
            f"{r['latency_mean_ms']:>10.1f} {r['throughput_tokens_per_sec']:>12,} {vs:>13}"
        )
    print(sep)
    print("""
NOTE : toutes les distributions ont la même latence avec le padding naïf.
C'est exactement le problème. Le scheduler doit briser cette invariance.
'Speedup pot.' = accélération théorique maximale si le scheduler était parfait.
""")


# ---------------------------------------------------------------------------
# Mode compare_scheduling
# ---------------------------------------------------------------------------

def run_compare(cfg: Config) -> list[dict]:
    device = torch.device(cfg.device)
    layer     = TransformerLayer(cfg.d_model, cfg.n_heads, cfg.d_ff).to(device).to(cfg.dtype)
    halt_head = HaltHead(cfg.d_model).to(device).to(cfg.dtype)

    strategies = ["padding", "token_aligned", "warp_aligned"]
    results = []

    for dist in DISTRIBUTIONS:
        for strategy in strategies:
            print(f"  {dist} / {strategy}...", end=" ", flush=True)
            r = benchmark_one(layer, halt_head, cfg, dist, strategy)
            results.append(r)
            print(f"{r['latency_mean_ms']:.1f} ms")

    print_compare_report(results, cfg)
    return results


def print_compare_report(results: list[dict], cfg: Config):
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"COMPARE SCHEDULING sur {results[0]['gpu']}")
    print(f"batch={cfg.batch_size}  seq={cfg.seq_len}  d_model={cfg.d_model}  T_max={cfg.T_max}  warp_size={cfg.warp_size}")
    print(sep)

    strategies = ["padding", "token_aligned", "warp_aligned"]

    for dist in DISTRIBUTIONS:
        dist_results = {r["strategy"]: r for r in results if r["dist"] == dist}
        ref_lat = dist_results["padding"]["latency_mean_ms"]

        print(f"\n{dist}  (effective_compute_ratio={dist_results['padding']['effective_compute_ratio']})")
        print(f"  {'Strategy':<16} {'Latency ms':>12} {'vs padding':>12}")
        print(f"  {'-'*42}")
        for s in strategies:
            r = dist_results[s]
            ratio = ref_lat / r["latency_mean_ms"]
            marker = ""
            if s != "padding":
                if ratio > 1.1:
                    marker = " ← GAIN"
                elif ratio < 0.95:
                    marker = " ← PLUS LENT"
                else:
                    marker = " ≈ égal"
            print(f"  {s:<16} {r['latency_mean_ms']:>10.1f}   ×{ratio:.2f}{marker}")

    print(f"\n{sep}")
    print("""
INTERPRÉTATION :
  token_aligned ≈ padding  → normal (divergence de warp annule le gain)
  warp_aligned  < padding  → signal positif, scheduler viable
  warp_aligned ≈ padding   → overhead de partition trop élevé
                              → profiler avec nsight avant de continuer
                              → métriques clés : sm__throughput, l1tex__t_sector_hit_rate

Si warp_aligned est PLUS LENT que padding : le overhead domine.
  Causes possibles : partition trop fréquente, coalescence cassée, overhead sync.
  Action : nsight sur le kernel de partition, pas sur l'attention.
""")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"→ {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    choices=["baseline", "compare_scheduling"], default="baseline")
    p.add_argument("--batch",   type=int, default=256)
    p.add_argument("--seq",     type=int, default=64)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff",    type=int, default=2048)
    p.add_argument("--T_max",   type=int, default=12)
    p.add_argument("--warp_size", type=int, default=32)
    p.add_argument("--warmup",  type=int, default=50)
    p.add_argument("--runs",    type=int, default=200)
    p.add_argument("--dtype",   type=str, default="bfloat16", choices=DTYPE_MAP.keys())
    p.add_argument("--no-csv",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible.")

    cfg = Config(
        batch_size=args.batch,
        seq_len=args.seq,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        T_max=args.T_max,
        warp_size=args.warp_size,
        warmup=args.warmup,
        runs=args.runs,
        dtype=DTYPE_MAP[args.dtype],
    )

    p = torch.cuda.get_device_properties(0)
    print(f"\nGPU : {p.name}")
    print(f"HBM : {p.total_memory / 1e9:.1f} GB")
    print(f"Shared memory / SM : {p.shared_memory_per_block / 1024:.0f} KB")

    if args.mode == "baseline":
        print("\nMode : baseline (padding naïf)")
        results = run_baseline(cfg)
        if not args.no_csv:
            save_csv(results, "baseline_results.csv")

    elif args.mode == "compare_scheduling":
        print("\nMode : compare_scheduling (padding vs token_aligned vs warp_aligned)")
        results = run_compare(cfg)
        if not args.no_csv:
            save_csv(results, "compare_results.csv")
