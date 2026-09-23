"""
visualizer.py — CrystalRover MDP Policy & Value Visualiser (for Q2c report)
COMP3702 Assignment 2, Semester 2 2026

Produces four groups of figures (all saved to ./screenshots/ subfolder):
  A) Per-level Value-function heatmap (VI-converged V(s) coloured by tile,
     aggregated over crystal-subset at each (row,col) using max |V|)
  B) Per-level Policy-arrow overlay on the same heatmap canvas
  C) Q2a — Drift / Double-move Venn-diagram style 4-quadrant probability bar
     (for the default L1 parameters drift=0.3, double=0.2 — analytical)
  D) Q2b — BOOST single-distance distribution (5 bins) vs double-move
     convolution (9 bins, d1+d2 = 0..8)

Bilingual EN / 中文 comments are used throughout, matching solution.py.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI required)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

from game_env import GameEnv
from solution import Solver


# =============================================================================
# I/O helpers
# =============================================================================

OUT_DIR = Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(exist_ok=True)


def _save(fig, name: str) -> str:
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


# =============================================================================
# Tile-drawing helpers (map grid legend)
# =============================================================================

def _draw_grid_legend(ax, env: GameEnv):
    """Draw the raw tile underlay: rock=black, launch=orange, lava=red,
    crystal=cyan ring, crater=purple hatch, ground=light grey."""
    R, C = _shape(env)
    base = np.full((R, C, 3), 0.95)  # off-white ground
    for r in range(R):
        for c in range(C):
            t = env.grid_data[r][c]
            if t == env.ROCK_TILE:
                base[r, c] = (0.18, 0.20, 0.25)          # dark grey rock
            elif t == env.LAUNCH_TILE:
                base[r, c] = (1.00, 0.78, 0.28)          # orange pad
            elif t == env.LAVA_TILE:
                base[r, c] = (0.86, 0.18, 0.18)          # red lava
            elif t == env.CRYSTAL_TILE:
                base[r, c] = (0.45, 0.92, 0.92)          # cyan crystal
            elif t == env.CRATER_TILE:
                base[r, c] = (0.70, 0.55, 0.90)          # light purple crater
    ax.imshow(base, aspect="equal", interpolation="nearest", zorder=0)
    # Tile hatch overlays
    for r in range(R):
        for c in range(C):
            t = env.grid_data[r][c]
            if t == env.CRATER_TILE:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           fill=False, hatch="xx",
                                           linewidth=0, color=(0.45, 0.30, 0.70),
                                           zorder=1, alpha=0.55))
            if t == env.LAVA_TILE:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           fill=False, hatch="///",
                                           linewidth=0, color=(0.55, 0.0, 0.0),
                                           zorder=1, alpha=0.55))
    # Coordinates
    ax.set_xticks(np.arange(C) - 0.5, minor=True)
    ax.set_yticks(np.arange(R) - 0.5, minor=True)
    ax.grid(which="minor", color="k", linewidth=0.5, alpha=0.4)
    ax.set_xticks(range(C))
    ax.set_yticks(range(R))
    ax.set_xlim(-0.55, C - 0.45)
    ax.set_ylim(R - 0.45, -0.55)
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")


def _shape(env: GameEnv):
    return env.n_rows, env.n_cols  # alias to keep rest of file concise
# =============================================================================
# Value heatmap  (aggregator: max_abs V over crystal subsets that land at (r,c))
# =============================================================================

_ACTION_TO_DX_DY = {
    GameEnv.WALK_LEFT:  (-1, 0),
    GameEnv.WALK_RIGHT: (+1, 0),
    GameEnv.WALK_UP:    (0, -1),
    GameEnv.WALK_DOWN:  (0, +1),
    GameEnv.JUMP_LEFT:  (-1, 0),
    GameEnv.JUMP_RIGHT: (+1, 0),
    GameEnv.JUMP_UP:    (0, -1),
    GameEnv.JUMP_DOWN:  (0, +1),
    GameEnv.BOOST_LEFT: (-1, 0),
    GameEnv.BOOST_RIGHT:(+1, 0),
    GameEnv.BOOST_UP:   (0, -1),
    GameEnv.BOOST_DOWN: (0, +1),
}


def _solve_via_vi(env: GameEnv):
    solver = Solver(env)
    solver.vi_initialise()
    iters = 0
    while True:
        solver.vi_iteration()
        iters += 1
        if solver.vi_is_converged() or iters >= 500:
            break
    return solver, iters


def _per_tile_value_aggregates(solver):
    env = solver.game_env
    R, C = _shape(env)
    max_abs = np.zeros((R, C))
    min_v = np.full((R, C), np.inf)
    max_v = np.full((R, C), -np.inf)
    counts = np.zeros((R, C), dtype=int)
    for s_idx, state in enumerate(solver.states):
        r, c = state.row, state.col
        v = float(solver.vi_v_old[s_idx])
        max_abs[r, c] = max(max_abs[r, c], abs(v))
        min_v[r, c] = min(min_v[r, c], v)
        max_v[r, c] = max(max_v[r, c], v)
        counts[r, c] += 1
    return max_abs, min_v, max_v, counts


def plot_value_heatmap(level_path: str, label: str):
    env = GameEnv(level_path)
    solver, iters = _solve_via_vi(env)
    R, C = _shape(env)
    max_abs, min_v, max_v, _counts = _per_tile_value_aggregates(solver)

    fig, axes = plt.subplots(1, 2, figsize=(max(11, C * 1.3), 6))
    # Left: signed value (V * sign of worst-case tile V) → use max_v (closest to 0 is best, most-negative = farthest from solved)
    for ax, data, title, cmap, tick_labels in [
        (axes[0], max_v, "V(s) — per-tile max (least-negative → closest to solved)",
         "YlOrRd_r", ["closer\nto 0 (good)", "↓ worse ↓", "most negative"]),
        (axes[1], max_abs, "|V(s)| — per-tile max (proxy for distance to goal / hazard)",
         LinearSegmentedColormap.from_list("haz", ["#f1faee", "#ff9f1c", "#e63946"], N=128),
         ["near goal\n(abs V small)", "↕", "far / high hazard\n(abs V large)"]),
    ]:
        _draw_grid_legend(ax, env)
        masked = np.ma.masked_where(~np.isfinite(data), data)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        im = ax.imshow(masked, alpha=0.62, zorder=2, cmap=cmap,
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Value")
        ax.set_title(title, fontsize=10)

    fig.suptitle(f"Level {label} Value heatmap (VI converged in {iters} iters, "
                 f"|S|={len(solver.states)})", fontsize=13, fontweight="bold")
    return _save(fig, f"Q2c_level{label}_value_heatmap.png")


# =============================================================================
# Policy arrows  (aggregator: pick most-frequent best action at each (r,c),
#                 over the reachable crystal-subsets for that tile)
# =============================================================================

def plot_policy_arrows(level_path: str, label: str):
    env = GameEnv(level_path)
    solver, iters = _solve_via_vi(env)
    R, C = _shape(env)
    # Vote count per (row, col) → action
    vote = np.zeros((R, C, 12), dtype=int)  # 12 nominal actions (indexed via list)
    actions_list = list(GameEnv.ACTIONS)
    for s_idx, state in enumerate(solver.states):
        if env.is_solved(state) or env.is_game_over(state):
            continue
        act = solver.vi_policy.get(s_idx, None)
        if act is None:
            continue
        a_idx = actions_list.index(act)
        vote[state.row, state.col, a_idx] += 1

    fig, ax = plt.subplots(figsize=(max(8, C * 1.1), max(5, R * 1.2)))
    _draw_grid_legend(ax, env)
    _, min_v, max_v, _c = _per_tile_value_aggregates(solver)
    # Heatmap background = max V (least-negative) per tile (policy quality proxy)
    data = np.where(np.isfinite(max_v), max_v, 0.0)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, alpha=0.45, zorder=2, cmap="YlGnBu_r",
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("V(s) max over crystal subsets")

    # Draw arrows for the majority action per tile
    for r in range(R):
        for c in range(C):
            if vote[r, c].sum() == 0:
                continue
            best_ai = int(np.argmax(vote[r, c]))
            action = actions_list[best_ai]
            dx, dy = _ACTION_TO_DX_DY[action]
            # Action-kind markers:
            #   WALK = thin black; BOOST = thick blue; JUMP = dashed purple
            if action.startswith("walk"):
                col, lw, ls = (0.0, 0.0, 0.0), 1.4, "-"
            elif action.startswith("boost"):
                col, lw, ls = (0.05, 0.30, 0.90), 2.2, "-"
            else:  # jump
                col, lw, ls = (0.55, 0.10, 0.70), 1.6, "--"
            # Short arrow proportional to consensus strength
            cons = vote[r, c, best_ai] / vote[r, c].sum()
            # Length relative to tile
            scale = 0.32 + 0.12 * cons
            x0, y0 = c, r
            x1, y1 = c + dx * scale, r + dy * scale
            arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                    arrowstyle="-|>", mutation_scale=12 * cons + 6,
                                    color=col, linestyle=ls, linewidth=lw, zorder=4)
            ax.add_patch(arrow)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=(0.0, 0.0, 0.0), lw=1.4, ls="-", label="WALK majority"),
        Line2D([0], [0], color=(0.05, 0.30, 0.90), lw=2.2, ls="-", label="BOOST majority"),
        Line2D([0], [0], color=(0.55, 0.10, 0.70), lw=1.6, ls="--", label="JUMP majority"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title(f"Level {label} Policy (VI, {iters} iters) — arrow = most frequent "
                 "best action over crystal subsets", fontsize=11)
    fig.tight_layout()
    return _save(fig, f"Q2c_level{label}_policy_arrows.png")


# =============================================================================
# Q2a — Drift × Double Venn-like quadrant breakdown (analytical, L1 params)
# =============================================================================

def plot_q2a_venn_like(drift=0.3, double_prob=0.2, label="L1-default"):
    p_nodrift_nodouble = (1 - drift) * (1 - double_prob)          # 0.7 * 0.8 = 0.56
    p_drift_only      = drift       * (1 - double_prob)          # 0.3 * 0.8 = 0.24
    p_double_only     = (1 - drift) * double_prob                # 0.7 * 0.2 = 0.14
    p_both            = drift       * double_prob                # 0.3 * 0.2 = 0.06
    # Further: inside drift-only and both, split CW / CCW = ½ each
    cats = [
        "No drift + No double\n(repeat nominal action 1×)",
        f"Drift only (P = {drift:.2f})\n→ CW ½ + CCW ½",
        f"Double only (P = {double_prob:.2f})\n→ repeat drifted action 2×",
        "Drift + Double\n(direction noisy + 2 legs)",
    ]
    vals = [p_nodrift_nodouble, p_drift_only, p_double_only, p_both]
    colors = ["#a8dadc", "#f4a261", "#e76f51", "#b5179e"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(range(4), vals, color=colors, edgecolor="black", linewidth=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}  ({v*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=9.5, rotation=8)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("Probability (over one player action-step)", fontsize=11)
    ax.set_title(f"Q2a — Drift / Double-move four-quadrant breakdown "
                 f"({label}: drift={drift}, double={double_prob})", fontsize=13, fontweight="bold")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    # Venn-style annotation (textual)
    ax.text(0.98, 0.96, "Venn-style area check:\n"
                       f"  Σ = {sum(vals):.3f} ✓ = 1.000\n"
                       f"  P(Drift|any)  = {drift:.2f} = {p_drift_only + p_both:.2f}\n"
                       f"  P(Dbl|any)    = {double_prob:.2f} = {p_double_only + p_both:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", lw=1),
            fontsize=9.5)
    return _save(fig, f"Q2a_venn_quadrants_{label}.png")


# =============================================================================
# Q2b — BOOST single d ∈ {0..4}  vs  double-move convolution d ∈ {0..8}
# =============================================================================

def plot_q2b_boost_convolution(boost_probs=(0.1, 0.3, 0.3, 0.2, 0.1), label="L1-default"):
    p1 = np.asarray(boost_probs, dtype=float)
    p1 /= p1.sum()
    p2 = np.convolve(p1, p1)  # length 9 (d = 0..8)
    d_single = np.arange(len(p1))
    d_double = np.arange(len(p2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    # Single BOOST
    bars1 = axes[0].bar(d_single, p1, color="#1d3557", edgecolor="black")
    for b, v in zip(bars1, p1):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.006,
                     f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(d_single)
    axes[0].set_xlabel("Single-BOOST distance d (tiles)")
    axes[0].set_ylabel("P(d)")
    axes[0].set_title(f"Q2b (i) — single BOOST distance distribution\n"
                      f"boost_prob = {list(boost_probs)}",
                      fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    bars2 = axes[1].bar(d_double, p2, color="#2a9d8f", edgecolor="black")
    for b, v in zip(bars2, p2):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.006,
                     f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(d_double)
    axes[1].set_xlabel("Double-BOOST distance d = d1 + d2 (tiles)")
    axes[1].set_title(f"Q2b (ii) — double BOOST convolution P(d1+d2)\n"
                      f"E[d|single]={(p1*d_single).sum():.2f}, "
                      f"E[d|double]={(p2*d_double).sum():.2f} = 2·E[d|single]",
                      fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(f"Q2b — BOOST single vs double-move distance convolution "
                 f"({label})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"Q2b_boost_convolution_{label}.png"), p1, p2


# =============================================================================
# Main entry point — generate ALL figures
# =============================================================================

LEVELS = [("testcases/L1.txt", "L1"),
          ("testcases/L2.txt", "L2"),
          ("testcases/L3.txt", "L3"),
          ("testcases/L4.txt", "L4")]


def main():
    print("[visualizer] Output folder:", OUT_DIR)
    # (A)(B) Per-level heatmaps & policy arrows
    for lvl_path, lbl in LEVELS:
        print(f"  -> rendering Level {lbl} value heatmap ...")
        p = plot_value_heatmap(lvl_path, lbl)
        print("      saved:", p)
        print(f"  -> rendering Level {lbl} policy arrows ...")
        p = plot_policy_arrows(lvl_path, lbl)
        print("      saved:", p)

    # (C) Q2a quadrant breakdown
    for lbl, drift, dbl in [("L1-default", 0.3, 0.2), ("L2-default", 0.3, 0.4)]:
        p = plot_q2a_venn_like(drift, dbl, label=lbl)
        print(f"  -> Q2a {lbl} saved:", p)

    # (D) Q2b boost convolution
    for lbl, bp in [("L1..L2-default", (0.1, 0.3, 0.3, 0.2, 0.1))]:
        p, p1, p2 = plot_q2b_boost_convolution(bp, label=lbl)
        print(f"  -> Q2b {lbl} saved:", p)
        print("      single BOOST P(d):   ", np.round(p1, 4).tolist())
        print("      double BOOST P(d1+d2):", np.round(p2, 4).tolist())

    print("\n[visualizer] DONE. All figures generated under", OUT_DIR)


if __name__ == "__main__":
    main()
