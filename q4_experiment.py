"""
q4_experiment.py — Q4 parameter-sweep controlled experiment (3 x 3 grid)
COMP3702 Assignment 2, Semester 2 2026

Experiment design (rubric Q4):
  Sweep 2 parameters on the LARGEST level L4.txt (which has lava hazards):
    (1) env.random_drift_prob   ∈ {0.20, 0.30, 0.40}   -- axis 1
    (2) env.game_over_penalty   ∈ {100, 500, 1000}     -- axis 2
  That is 9 (drift, penalty) combinations.   For each combination we:
    (A) Solve once via VI (Solver.vi_plan_offline) + once via PI (pi_plan_offline)
    (B) Roll out the extracted greedy policy for 100 stochastic episodes using
        the support code's own GameEnv.step / is_solved / is_game_over dynamics.
    (C) Record two metrics per (algo, drift, penalty) run:
          * avg_reward    = mean discounted-less 100-ep total reward
          * success_rate  = (# episodes that end in is_solved==True) / 100

Outputs (all written alongside THIS script, i.e. the coursework repo root):
  * q4_results.csv       -- tidy long-format table with columns
                           [drift_prob, game_over_penalty, algorithm,
                            avg_reward, success_rate, n_episodes]
  * Q4_heatmap_VI.png    -- 2-column heatmap: left=avg_reward | right=success_rate,
                           rows=penalty, cols=drift_prob (Value Iteration policy)
  * Q4_heatmap_PI.png    -- same layout for the Policy Iteration policy

Bilingual EN / 中文 comments are used throughout, matching solution.py / visualizer.py.
双语（英文/中文）注释贯穿本脚本，与 solution.py 和 visualizer.py 一致。
"""

import os
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend (no GUI required)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from game_env import GameEnv, GameState
from solution import Solver


# =============================================================================
# Global configuration
# =============================================================================
OUT_DIR = Path(__file__).parent                 # save in repo root
LEVEL_PATH = OUT_DIR / "testcases" / "L4.txt"   # largest level with lava hazards
DRIFT_PROBS = [0.20, 0.30, 0.40]                # axis 1 (3 levels)
GO_PENALTIES = [100, 500, 1000]                 # axis 2 (3 levels)
N_EPISODES = 100                                # per cell (rubric default)

CSV_PATH = OUT_DIR / "q4_results.csv"
FIG_VI_PATH = OUT_DIR / "Q4_heatmap_VI.png"
FIG_PI_PATH = OUT_DIR / "Q4_heatmap_PI.png"

if not LEVEL_PATH.exists():
    raise FileNotFoundError(
        f"L4 testcase required for Q4 sweep not found at: {LEVEL_PATH}"
    )


# =============================================================================
# Environment factory — build an L4 GameEnv then override the two sweep params
# =============================================================================
def make_env_with_overrides(drift_prob: float, go_penalty: float) -> GameEnv:
    """Build a fresh GameEnv instance from L4.txt then patch two float fields.

    Rationale: game_env.py stores random_drift_prob, random_double_prob,
    game_over_penalty, gamma, epsilon, etc. as plain instance attributes on
    __init__ from the level header.   Writing to those attributes BEFORE any
    Solver/episode creation patches every stochastic path through the dynamics.
    中文：game_env.py 在 __init__ 时把 drift 概率、game_over 惩罚等读成普通
    实例属性；在构造 Solver / 跑 episodes 之前直接修改这些属性，就能把整
    个转移分布改到实验目标参数，不用改 source game_env.py。
    """
    env = GameEnv(str(LEVEL_PATH))
    env.random_drift_prob = float(drift_prob)
    env.game_over_penalty = float(go_penalty)
    # double-move probability is left at the L4 header default for this Q4
    # rubric sweep (sweep only the two explicit parameters).
    return env


# =============================================================================
# Episode rollout
# =============================================================================
def rollout_policy(env: GameEnv, solver: Solver, use_pi: bool) -> tuple[float, bool]:
    """Roll out a single full episode under the solver's greedy (VI or PI) policy.

    Uses the support-code native ``env.perform_action(state, action)`` call which
    returns ``(new_state, reward, error_msg_or_None)`` and internally applies
    drift/double-move/BOOST probability sampling exactly like ``tester.py`` does
    for the Gradescope 1000-episode grading rollout.
    中文：调用 game_env.perform_action(state, action) 直接使用 support code 自带的
    drift × double × BOOST 三层随机采样实现，与 tester.py 1000 轮评分 rollout 的
    环境完全一致。

    Returns
    -------
    total_reward : float
        Undiscounted stepwise total of rewards collected during the episode.
    solved : bool
        True  -> episode ended because ``env.is_solved(state)`` became True
                 (all crystals collected + rover on E, no game_over triggered).
        False -> episode ended in ``env.is_game_over(state)`` or step-budget
                 exhaustion / invalid-action loop.
    """
    state = env.get_init_state()
    total_reward = 0.0
    # Safety cap matching the tester: 1000 steps max per episode so a stuck
    # policy on walls can't loop forever.
    for _ in range(1000):
        if env.is_solved(state):
            return total_reward, True
        if env.is_game_over(state):
            return total_reward, False
        action = (
            solver.pi_select_action(state) if use_pi
            else solver.vi_select_action(state)
        )
        # GameEnv.perform_action signature = (state, action) -> (state, reward, err_msg)
        # Handles drift, double, boost, lava, craters, crystals internally.
        state, reward, _err = env.perform_action(state, action)
        total_reward += reward
        if env.is_solved(state):
            return total_reward, True
        if env.is_game_over(state):
            return total_reward, False
    # Falls through on step-budget exhaustion → classify as "not solved".
    return total_reward, False


def multi_rollout(env: GameEnv, solver: Solver, use_pi: bool, n_eps: int) -> tuple[float, float]:
    """Run ``n_eps`` independent episodes → (mean_reward, success_rate).

    Episodes are independent because ``env.step`` samples fresh RNG draws each
    call; the support-code RNG is module-level and already seeded inside
    ``GameEnv.__init__`` from the level header.
    """
    rewards = np.zeros(n_eps, dtype=np.float64)
    success = 0
    for i in range(n_eps):
        r, solved = rollout_policy(env, solver, use_pi)
        rewards[i] = r
        if solved:
            success += 1
    return float(rewards.mean()), float(success) / float(n_eps)


# =============================================================================
# Full 3 x 3 sweep driver
# =============================================================================
def run_full_sweep() -> list[dict]:
    """Execute the 9-combination sweep for BOTH algorithms; return list of dicts."""
    rows: list[dict] = []
    for drift_prob in DRIFT_PROBS:
        for go_penalty in GO_PENALTIES:
            print(
                f"[q4] cell (drift={drift_prob:.2f}, penalty={go_penalty:4d}) "
                f"start -------------------------------------------------"
            )
            # --- VI arm ------------------------------------------------------
            env_vi = make_env_with_overrides(drift_prob, go_penalty)
            sol_vi = Solver(env_vi)
            sol_vi.vi_plan_offline()
            mean_r_vi, succ_vi = multi_rollout(env_vi, sol_vi, use_pi=False,
                                               n_eps=N_EPISODES)
            rows.append({
                "drift_prob": f"{drift_prob:.2f}",
                "game_over_penalty": f"{go_penalty:d}",
                "algorithm": "VI",
                "avg_reward": f"{mean_r_vi:.3f}",
                "success_rate": f"{succ_vi:.3f}",
                "n_episodes": f"{N_EPISODES}",
            })
            print(
                f"[q4]   VI: avg_reward={mean_r_vi:7.2f}  "
                f"success={succ_vi*100:5.1f}% / {N_EPISODES} episodes"
            )
            # --- PI arm ------------------------------------------------------
            env_pi = make_env_with_overrides(drift_prob, go_penalty)
            sol_pi = Solver(env_pi)
            sol_pi.pi_plan_offline()
            mean_r_pi, succ_pi = multi_rollout(env_pi, sol_pi, use_pi=True,
                                               n_eps=N_EPISODES)
            rows.append({
                "drift_prob": f"{drift_prob:.2f}",
                "game_over_penalty": f"{go_penalty:d}",
                "algorithm": "PI",
                "avg_reward": f"{mean_r_pi:.3f}",
                "success_rate": f"{succ_pi:.3f}",
                "n_episodes": f"{N_EPISODES}",
            })
            print(
                f"[q4]   PI: avg_reward={mean_r_pi:7.2f}  "
                f"success={succ_pi*100:5.1f}% / {N_EPISODES} episodes"
            )
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["drift_prob", "game_over_penalty", "algorithm",
                  "avg_reward", "success_rate", "n_episodes"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[q4] CSV saved: {path}")


# =============================================================================
# Heatmap rendering (VI or PI matrix, 2-column reward / success rate)
# =============================================================================
def _rows_to_numpy(rows: list[dict], algorithm: str) -> tuple[np.ndarray, np.ndarray]:
    """Filter rows for one algo; return two len(DRIFT_PROBS)×len(GO_PENALTIES)
    arrays: M_reward[i,j] and M_success[i,j] where
      i = drift_prob index (0..2)
      j = go_penalty index (0..2).
    """
    n_d = len(DRIFT_PROBS)
    n_p = len(GO_PENALTIES)
    M_r = np.full((n_d, n_p), np.nan)
    M_s = np.full((n_d, n_p), np.nan)
    for row in rows:
        if row["algorithm"] != algorithm:
            continue
        i = DRIFT_PROBS.index(float(row["drift_prob"]))
        j = GO_PENALTIES.index(int(row["game_over_penalty"]))
        M_r[i, j] = float(row["avg_reward"])
        M_s[i, j] = float(row["success_rate"]) * 100.0   # percent
    assert not np.isnan(M_r).any() and not np.isnan(M_s).any(), \
        f"Missing cells for algorithm={algorithm} (sweep incomplete)"
    return M_r, M_s


def plot_pair_heatmap(rows: list[dict], algorithm: str, save_path: Path) -> None:
    """Render a 2-panel heatmap for one algorithm; save as PNG."""
    M_r, M_s = _rows_to_numpy(rows, algorithm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 1]})
    xtick_labels = [r"$p_{\mathrm{drift}}$=" + f"{p:.2f}" for p in DRIFT_PROBS]
    ytick_labels = [r"$\lambda_{\mathrm{GO}}$=" + f"{p}" for p in GO_PENALTIES]

    # ---- left panel: avg_reward (higher = better, -1 is best achievable)
    cmap_r = "YlOrRd_r"   # reversed so "least negative reward (best)" ≈ warm yellow
    vmin_r = float(np.nanmin(M_r))
    vmax_r = float(np.nanmax(M_r))
    norm_r = Normalize(vmin=vmin_r, vmax=vmax_r)
    im_r = axes[0].imshow(M_r, cmap=cmap_r, norm=norm_r, aspect="auto")
    axes[0].set_title(f"Q4 — Average reward (100 episodes) — {algorithm}")
    axes[0].set_xticks(range(len(DRIFT_PROBS))); axes[0].set_xticklabels(xtick_labels, rotation=15, ha="right")
    axes[0].set_yticks(range(len(GO_PENALTIES))); axes[0].set_yticklabels(ytick_labels)
    axes[0].set_ylabel("Game-over penalty (lava)  ↑")
    axes[0].set_xlabel("Random drift probability  →")
    for i in range(len(DRIFT_PROBS)):
        for j in range(len(GO_PENALTIES)):
            axes[0].text(j, i, f"{M_r[i,j]:.1f}",
                         ha="center", va="center",
                         color="white" if (M_r[i,j] - vmin_r) / max((vmax_r-vmin_r), 1e-9) < 0.5 else "black",
                         fontsize=10, fontweight="bold")
    cbar_r = fig.colorbar(im_r, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_r.set_label("Mean total reward / episode")

    # ---- right panel: success rate (%) — higher = better
    cmap_s = "RdYlGn"
    norm_s = Normalize(vmin=0.0, vmax=100.0)
    im_s = axes[1].imshow(M_s, cmap=cmap_s, norm=norm_s, aspect="auto")
    axes[1].set_title(f"Q4 — Episodes that end SOLVED — {algorithm}")
    axes[1].set_xticks(range(len(DRIFT_PROBS))); axes[1].set_xticklabels(xtick_labels, rotation=15, ha="right")
    axes[1].set_yticks(range(len(GO_PENALTIES))); axes[1].set_yticklabels(ytick_labels)
    axes[1].set_ylabel("Game-over penalty (lava)  ↑")
    axes[1].set_xlabel("Random drift probability  →")
    for i in range(len(DRIFT_PROBS)):
        for j in range(len(GO_PENALTIES)):
            axes[1].text(j, i, f"{M_s[i,j]:.0f}%",
                         ha="center", va="center",
                         color="white" if M_s[i,j] < 60 else "black",
                         fontsize=10, fontweight="bold")
    cbar_s = fig.colorbar(im_s, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_s.set_label("Success rate (%)")

    suptitle_str = (
        "COMP3702 A2 Q4 — L4 parameter sweep (3 drift x 3 penalty, "
        "n_episodes = " + str(N_EPISODES) + ", " + str(algorithm) + " policy)"
    )
    fig.suptitle(suptitle_str, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[q4] heatmap saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================
def main():
    print(f"[q4] Output folder: {OUT_DIR}")
    rows = run_full_sweep()
    save_csv(rows, CSV_PATH)
    plot_pair_heatmap(rows, "VI", FIG_VI_PATH)
    plot_pair_heatmap(rows, "PI", FIG_PI_PATH)
    print("[q4] DONE. CSV + two PNG artefacts written alongside script.")


if __name__ == "__main__":
    main()
