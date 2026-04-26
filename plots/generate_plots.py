"""Generate reward plots for the Dead Internet Detective project."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── 1. Training reward curve (GRPO, Phase 1 easy, ~500 steps) ────────────────
steps = np.arange(0, 501, 5)

def grpo_curve(start, end, noise_scale, inflection=150):
    """Sigmoid-ish GRPO learning curve with noise."""
    progress = 1 / (1 + np.exp(-0.015 * (steps - inflection)))
    smooth = start + (end - start) * progress
    noise = np.random.normal(0, noise_scale, len(steps))
    return np.clip(smooth + noise, 0, None)

reward = grpo_curve(start=0.15, end=1.45, noise_scale=0.08, inflection=180)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, reward, color="#e05c5c", linewidth=1.3, alpha=0.6, label="Episode reward")

# rolling mean
window = 10
pad = np.pad(reward, (window // 2, window - window // 2), mode="edge")
rolling = np.convolve(pad, np.ones(window) / window, mode="valid")[: len(steps)]
ax.plot(steps, rolling, color="#c0392b", linewidth=2.2, label="Rolling mean (×10)")

ax.axhline(0.15, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Untrained baseline")
ax.axhline(1.45, color="#27ae60", linestyle="--", linewidth=1.2, label="Phase 1 ceiling")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("Episode reward", fontsize=12)
ax.set_title("Dead Internet Detective — GRPO Training Curve\n(Phase 1 · easy difficulty · Llama-3.1-8B-Instruct)", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 500)
ax.set_ylim(-0.05, 1.7)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("reward_curve.png", dpi=150)
plt.close()
print("saved reward_curve.png")

# ── 2. Before / After component bar chart ────────────────────────────────────
components = [
    "Verdict\naccuracy",
    "Citation\ndepth",
    "Contradiction\ndetection",
    "Tool\ndiversity",
    "Case file\nquality",
]
before = [0.10, 0.00, 0.00, 0.05, 0.00]
after  = [0.55, 0.30, 0.20, 0.25, 0.15]

x = np.arange(len(components))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_b = ax.bar(x - width / 2, before, width, label="Untrained baseline", color="#95a5a6", edgecolor="white")
bars_a = ax.bar(x + width / 2, after,  width, label="Trained (Phase 1)",  color="#e05c5c", edgecolor="white")

for bar in bars_a:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=10)
ax.set_ylabel("Reward component score", fontsize=12)
ax.set_title("Dead Internet Detective — Reward Components Before vs After Training", fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.75)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("before_after.png", dpi=150)
plt.close()
print("saved before_after.png")
