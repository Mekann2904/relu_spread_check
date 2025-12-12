# relu_spread_check.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

SEED = 0
DEVICE = "cpu"
BATCH = 8192
WIDTH = 1024
LAYERS = 20
FIG_DIR = "figures"

# 分布を可視化する層（必要なら変える）
HIST_LAYERS = [1, 5, 10, 20]
# ヒストグラム用に抽出する要素数（多すぎると重い）
HIST_SAMPLES = 200_000

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
set_seed(SEED)

def save_fig(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"[FIG] saved -> {path}")
    plt.close(fig)

def moment_stats(t: torch.Tensor) -> dict:
    with torch.no_grad():
        mean = t.mean().item()
        ex2 = (t * t).mean().item()               # E[x^2]
        var = t.var(unbiased=False).item()        # Var(x)
        return {"mean": mean, "ex2": ex2, "var": var}

def init_std(mode: str, fan_in: int) -> float:
    if mode == "xavier":
        return 1.0 / math.sqrt(fan_in)          # Var(w)=1/n
    if mode == "he":
        return math.sqrt(2.0 / fan_in)          # Var(w)=2/n
    raise ValueError(mode)

def act_fn(name: str):
    if name == "identity":
        return lambda x: x
    if name == "relu":
        return torch.relu
    raise ValueError(name)

@torch.no_grad()
def sample_flat(t: torch.Tensor, k: int) -> np.ndarray:
    # ヒストグラム用にフラット化してランダムサンプル
    v = t.reshape(-1)
    if v.numel() <= k:
        return v.detach().cpu().numpy()
    idx = torch.randint(0, v.numel(), (k,), device=v.device)
    return v[idx].detach().cpu().numpy()

@torch.no_grad()
def run_pattern(name: str, w_init: str, activation: str):
    f = act_fn(activation)

    x = torch.randn(BATCH, WIDTH, device=DEVICE)
    records = []
    snaps = {}  # 指定層の分布スナップショット（zとa）

    for layer in range(1, LAYERS + 1):
        std = init_std(w_init, fan_in=WIDTH)
        W = torch.randn(WIDTH, WIDTH, device=DEVICE) * std

        z = x @ W.t()
        a = f(z)

        xs = moment_stats(x)
        zs = moment_stats(z)
        as_ = moment_stats(a)

        ratio_ex2 = as_["ex2"] / zs["ex2"] if zs["ex2"] > 0 else float("nan")

        # ReLUならゼロ比率（スパース性）も見ておく
        zero_frac = float((a == 0).float().mean().item()) if activation == "relu" else 0.0

        records.append({
            "layer": layer,
            "x_ex2": xs["ex2"], "z_ex2": zs["ex2"], "a_ex2": as_["ex2"],
            "relu_ex2_ratio": ratio_ex2,
            "zero_frac": zero_frac,
        })

        if layer in HIST_LAYERS:
            snaps[layer] = {
                "z": sample_flat(z, HIST_SAMPLES),
                "a": sample_flat(a, HIST_SAMPLES),
            }

        x = a

    print(f"\n=== {name}  (w_init={w_init}, act={activation}) ===")
    for r in records[:3]:
        print(f"[L{r['layer']:02d}] E[z^2]={r['z_ex2']:.4f}  E[a^2]={r['a_ex2']:.4f}  "
              f"E[a^2]/E[z^2]={r['relu_ex2_ratio']:.4f}  zero_frac={r['zero_frac']:.3f}")
    r = records[-1]
    print(f"[L{r['layer']:02d}] E[z^2]={r['z_ex2']:.4f}  E[a^2]={r['a_ex2']:.4f}  "
          f"E[a^2]/E[z^2]={r['relu_ex2_ratio']:.4f}  zero_frac={r['zero_frac']:.3f}")

    if activation == "relu":
        ratios = [rr["relu_ex2_ratio"] for rr in records]
        zf = [rr["zero_frac"] for rr in records]
        print(f"mean(E[a^2]/E[z^2]) over layers = {np.mean(ratios):.4f}  (理論: 約0.5)")
        print(f"mean(zero_frac) over layers       = {np.mean(zf):.4f}  (直感: 約0.5)")

    return records, snaps

def plot_histograms(pattern_name: str, snaps: dict):
    # z と a の分布を、指定層ごとに並べて描く
    layers = sorted(snaps.keys())
    n = len(layers)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i, layer in enumerate(layers):
        z = snaps[layer]["z"]
        a = snaps[layer]["a"]

        ax = axes[i, 0]
        ax.hist(z, bins=60)
        ax.set_title(f"{pattern_name}  layer={layer}  z (pre-activation)")
        ax.grid(True)

        ax = axes[i, 1]
        ax.hist(a, bins=60)
        ax.set_title(f"{pattern_name}  layer={layer}  a (activation)")
        ax.grid(True)

    fig.tight_layout()
    save_fig(fig, f"hist_{pattern_name.replace(' ', '_').replace('/', '_')}")

def main():
    patterns = [
        ("A) Xavier + Identity", "xavier", "identity"),
        ("B) Xavier + ReLU",     "xavier", "relu"),
        ("C) He + ReLU",         "he",     "relu"),
    ]

    all_records = {}
    all_snaps = {}

    for name, w_init, activation in patterns:
        rec, snaps = run_pattern(name, w_init, activation)
        all_records[name] = rec
        all_snaps[name] = snaps
        plot_histograms(name, snaps)

    # E[a^2] 推移
    fig, ax = plt.subplots()
    for name in all_records:
        layers = [r["layer"] for r in all_records[name]]
        a_ex2 = [r["a_ex2"] for r in all_records[name]]
        ax.plot(layers, a_ex2, label=name)
    ax.set_title("Propagation of E[a^2] across layers")
    ax.set_xlabel("layer")
    ax.set_ylabel("E[a^2]")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    save_fig(fig, "relu_spread_check_ex2")

    # ReLU縮み率
    fig, ax = plt.subplots()
    for name in all_records:
        if "ReLU" not in name:
            continue
        layers = [r["layer"] for r in all_records[name]]
        ratio = [r["relu_ex2_ratio"] for r in all_records[name]]
        ax.plot(layers, ratio, label=name)
    ax.set_title("ReLU shrink factor: E[ReLU(z)^2] / E[z^2] (should be ~0.5)")
    ax.set_xlabel("layer")
    ax.set_ylabel("E[a^2] / E[z^2]")
    ax.grid(True)
    ax.legend()
    save_fig(fig, "relu_spread_check_relu_ratio")

    # ReLUゼロ比率（どれだけ0になってるか）
    fig, ax = plt.subplots()
    for name in all_records:
        if "ReLU" not in name:
            continue
        layers = [r["layer"] for r in all_records[name]]
        zf = [r["zero_frac"] for r in all_records[name]]
        ax.plot(layers, zf, label=name)
    ax.set_title("ReLU sparsity: fraction of zeros in activation")
    ax.set_xlabel("layer")
    ax.set_ylabel("zero_frac")
    ax.grid(True)
    ax.legend()
    save_fig(fig, "relu_zero_fraction")

if __name__ == "__main__":
    main()

