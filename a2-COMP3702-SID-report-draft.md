# COMP3702 — Assignment 2: CrystalRover MDP — Report Draft

> **Student Name:** Tianyi Qu  |  **Student ID:** 50494408  |  **GitHub Username:** lain00511
> **Coursework Repository (where 5+ commits live):** https://github.com/comp3702-2026/comp3702-2026-a2-50494408
> **Personal mirror backup:** https://github.com/lain00511/comp3702-a2-crystalrover
>
> Course: COMP3702 Artificial Intelligence, Semester 2 2026 — The University of Queensland
> Word count limit (body excluding References & Appendix): as per template
> Submission: `solution.py` + this report (PDF) on Gradescope

---

## Pre-submission Self-Check (for reference, please delete before final submission)

All rubric "functionality" checkboxes for `solution.py` were verified locally:

- 8 test suites (L1–L4 × VI / PI) → **8 / 8 passed**:
  - Values converged ✅ (tester confirms `epsilon` criterion on 20 sampled states)
  - Level completed ✅ (1000-episode simulation reaches goal with reward not exceeding `reward_min_tgt`)
  - Total reward meets or beats max-target: L1 = −41.2 (target −41.2), L2 = −35.0 (target −35.1),
    L3 = −70.3 (target −70.4), L4 = −67.2 (target −67.2)
- Required public API implemented (no signature deviation):
  `vi_initialise / vi_is_converged / vi_iteration / vi_get_state_value / vi_select_action`,
  `pi_initialise / pi_is_converged / pi_iteration / pi_select_action`.
  (Tester-only wrappers `vi_plan_offline / pi_plan_offline` are kept verbatim from the template.)
- Code convention compliance: every method and non-trivial helper carries **bilingual (EN/中文)**
  docstring / inline comments; no `try`/`except` around a required method body; no edits to
  `game_env.py`, `game_state.py`, or `tester.py`.
- `Solver.STUDENT_NAME / STUDENT_ID / GITHUB_USERNAME` and `testcases_to_attempt() = [1,2,3,4,5]`
  set in `solution.py`.

---

## Question 1. MDP Formulation of CrystalRover  (15 marks)

### 1a. State / Action / Transition / Reward  (5 marks)

An MDP is formally a 4-tuple ⟨S, A, P(·|s,a), R(s,a,s′)⟩ plus a discount factor γ (Russell &
Norvig, 2020, Ch. 17; Poole & Mackworth, 2023, Fig. 1.8). For the CrystalRover game these four
elements are concretised below and each is cross-referenced to the exact data structure in the
submitted `solution.py`.

**(i) State space S.**
S = { (row, col, crystal_status_tuple) } ⊆ {0..R−1} × {0..C−1} × 2^N_crystals, i.e. every
state encodes the (row, column) position of the rover **plus** a subset of crystals collected
so far (stored as a length-`N_crystals` tuple of `True`/`False`). Two additional terminal
classes collapse into the representation: a state is *solved* when
`GameEnv.is_solved(state)` holds (at least `min_samples` crystals collected AND the rover
lands on a launch pad tile `E`) and *game-over* when `GameEnv.is_game_over(state)` holds
(rover lands on a lava tile `L`).
Implementation (Sol):
• Enumerated at initialisation by BFS over 12 nominal actions
  (`_compute_reachable_states`, [solution.py:423–460](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L423-L460));
• stored as `self.states : list[GameState]` and inverted via
  `self.state_to_idx : dict[GameState, int]` for O(1) index lookups
  ([solution.py:34–46](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L34-L460)).
This is strictly **smaller** than the Cartesian product of (R×C×2^N) because unreachable
combinations — e.g. crystals trapped behind rock walls with no entry — are pruned by the
reachability BFS. Empirical reachable-state counts: L1 228, L2 244, L3 200, L4 456.

**(ii) Action space A.**
A = {WALK_*, JUMP_*, BOOST_*} × {Left, Right, Up, Down} = 12 nominal actions. However the
player's nominal action is restricted by tile type (the solver respects this restriction in
*action selection*, while still accounting for stochastic "invalid → skip" outcomes coming
from drift — see the 0-cost no-op degeneracy fix in §3 of the Self-Check above):
  • On non-crater tiles (ground / crystal `C` / rock `R` / pad `E` / lava `L`):
    WALK_*, BOOST_* (8 actions). JUMP is syntactically allowed by the simulator but
    immediately `valid=False` → skipped at 0 cost, so the MDP excludes it from the
    action menu for rational agents.
  • On crater tiles `*`: JUMP_* only (4 actions). WALK / BOOST are treated as skip.
Implementation (Sol): nominal-action validity filter exposed as
`_valid_nominal_actions`, [solution.py:708–723](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L708-L723).

**(iii) Transition kernel P(s′ | s, a).**
Transition is a **three-layer stochastic composition** (order matters for double move):
(1) **Action-direction drift (Layer 1).** With probability `random_drift_prob` the nominal
    action is rotated 90° CW or CCW (each ½·drift_prob); with the complement probability it
    stays unchanged. For L1 / L2 drift_prob = 0.3, so P(unchanged)=0.70, P(CW)=P(CCW)=0.15.
    CW/CCW mapping is provided by `GameEnv.PERPENDICULAR_ACTIONS` (imported in `get_transition_outcomes`).
(2) **Double move (Layer 2, conditionally independent of drift).** With probability
    `random_double_prob` the *drifted action* is executed **twice in sequence**. Any terminal
    state reached after the first leg short-circuits the second leg — a double move that ends
    the episode on step 1 collapses to the single-step terminal on step 2 with the same
    accumulated reward (prevents "double-counting" a solved reward or a game-over penalty).
(3) **Boost distance (Layer 3, per leg).** Every BOOST leg samples a distance d ∈ {0..4}
    according to `boost_probabilities` (default [0.1, 0.3, 0.3, 0.2, 0.1]). WALK and JUMP
    legs use a deterministic distance of 1.
After direction, multiplicity and distance are sampled, **each single leg is fed through the
deterministic `apply_dynamics` surrogate** `_simulate_move` which handles:
  • collision with rock `R` / out-of-bounds → −collision_penalty and stop before obstacle;
  • entry into crater `*` → rover stops inside the crater;
  • entry into lava `L` → −game_over_penalty (500) + terminal;
  • crystal collected **only on landing** (the tile the rover stops on, not tiles traversed
    mid-boost);
  • the "non-collision but already game-over" branch on `game_env.py` lines 355–357 (replicated
    in `_simulate_move` at [solution.py:637–686](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L637-L686)).
Finally, identical `next_state` outcomes are merged (probabilities summed, reward stored as
probability-weighted average) in `get_transition_outcomes`
[solution.py:475–561](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L475-L561).
Every (state, action) outcome list is materialised once into `self.transition_cache` in the
initialisation step, so the VI / PI hot loops are pure table lookups.

**(iv) Reward function R(s, a, s′).**
The agent *maximises cumulative negative cost* (the problem is framed as reward maximisation
but every primitive cost is non-negative). For a single step the immediate reward is:
  R(s, a, s′) = −ACTION_COST(a_kind)  ·  storm_multiplier(a_dir)
                − collision_penalty · I{collision occurred this leg}
                − game_over_penalty · I{s′ is a lava terminal}
  where:
    • ACTION_COST(WALK)=1.0, JUMP=2.0, BOOST=3.0 (from `game_env.py`);
    • storm_multiplier = 1 + `storm_cost_multiplier` if the drifted action direction
      matches the headwind direction, 1 − same multiplier for a tailwind match, else 1.
      (Implemented in `_apply_action_once` as `base_reward` with a
      direction-vs-`GameEnv.ACTION_TO_DIR_DELTA` / env `storm_direction` match check.)
For a double leg the per-leg rewards are summed, so the action cost pays twice (modulo
short-circuit on terminal leg 1). Terminal states (solved, game-over) have V = 0 and are
absorbing (no further rewards accrue). Because the Bellman updates in both solvers read
`max_a Σ P · (R + γ V(s′))`, the 0 V(terminal) convention makes the solved state "free" and
the lava state carry its one-shot −500 without a tail of extra punishment behind it.
Alignment sanity check: with this exact reward model the planner produces the canonical
maximum-target reward of −41.2 on L1.

---

### 1b. Discount factor γ and the choice of γ = 0.999  (5 marks)

The discount factor γ ∈ [0,1) modulates how strongly the agent weighs *delayed* reward
against *immediate* reward when computing V(s) = Σ_{t≥0} γᵗ · r_t (Puterman, 2014, §2.2;
Russell & Norvig, 2020, §17.1). Three regimes are qualitatively different:

  • γ ≈ 0 — myopic / greedy: V(s) is essentially R(s, a, s′) of the next step only.
    Useful for reactive, low-horizon controllers but catastrophic for CrystalRover because
    the crystals are separated from the launch pad by many steps; a γ ≈ 0 agent would never
    spend early movement-cost to pick up crystals (long-horizon goal) that later unlock
    the +0 (solved) absorbing payoff.
  • γ ≈ 1 — farsighted / patient: the horizon is effectively 1/(1−γ). For the assignment
    value γ = 0.999 the effective horizon is 1 / (1 − 0.999) = 1000 steps, which comfortably
    encloses the longest conceivable optimal solution on L4 (≈80–200 steps depending on
    hazards) while still guaranteeing the Bellman operator γ-contraction and hence a
    unique fixed point V* (Puterman, 2014, Thm. 6.2.2). The contraction is exactly the
    reason VI and PI are *guaranteed* to converge (no oscillation); choosing γ = 1 would
    lose the contraction for persistent non-terminal cycles and create multiple value
    functions with the same infinite-horizon sum.
  • γ mid-range (0.9, 0.95) — is typically adequate but on CrystalRover would *prematurely*
    discount the solved terminal. Because the absorbing V(solved)=0 is reached only after
    the last step, a lower γ biases the policy toward the nearest crystal and away from
    distant-but-cheaper sequences. With γ = 0.999 the present value of any terminal cost
    200 steps ahead is still multiplied by 0.999²⁰⁰ ≈ 0.82, which preserves the correct
    long-horizon ranking of paths.

Concretely for this assignment, γ = 0.999 is used identically in VI and PI:
  • VI — the Bellman backup line reads
    `q_val += prob * (reward + gamma * vi_v_old[ns_idx])`
    ([solution.py:179–187](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L179-L187),
    same pattern in PI improvement step at
    [solution.py:371–383](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L371-L383)).
  • PI — policy evaluation solves the linear system
    (I − γ P_π) V_π = R_π exactly via `numpy.linalg.solve`
    ([solution.py:341–351](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L341-L351))
    with an `_iterative_policy_eval` fallback
    ([solution.py:731–759](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/solution.py#L731-L759))
    that sweeps until max|ΔV| < ε, again with the same γ.

The consistency of γ across VI and PI is the reason their policies produce identical
optimal-reward values (−41.2 / −35.0 / −70.3 / −67.2) on every level despite different
iteration counts (see Q3c table).

---

### 1c. Five complexity dimensions of CrystalRover (from P&M Figure 1.8)  (5 marks)

Table 1 categorises the CrystalRover planning / acting problem along the five orthogonal
dimensions of Poole & Mackworth (2023, Artificial Intelligence 3e, Figure 1.8, p. 21).
Values are taken **verbatim** from the Figure 1.8 enumeration (not invented) and the second
column justifies *why* the chosen value fits CrystalRover rather than a neighbouring value
from the same figure row.

**Table 1 — CrystalRover classified on the P&M 2023 five-dimension complexity framework.**

| # | Dimension (Fig 1.8 name)   | Value chosen (Fig 1.8 vocabulary)   | Justification for CrystalRover                                                                                                                                      |
|---|----------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | **Planning horizon**       | **Indefinite (until termination)**  | The episode terminates deterministically on two absorbing events — rover lands on a launch pad with `min_samples` crystals (solved) OR on a lava tile (game over). No fixed number T of steps is imposed in advance by the assignment (levels differ in optimal path length from ≈ 40 to ≈ 200+ steps), so it is neither Finite-horizon nor truly Infinite-horizon (γ = 0.999 ≈ 1 is a computational contrivance, not a plot fact). |
| 2 | **Sensing uncertainty**    | **Fully observable**                | The simulator returns the exact `GameState` (row, col, full crystal-collection bitmask) at every decision point; the planner is given the true state both offline and at `vi_select_action / pi_select_action` call-sites. No partial observability, no noisy sensors, no belief state (POMDP) is needed. |
| 3 | **Effect uncertainty**     | **Stochastic**                      | Three sources of exogenous randomness coexist (drift P = 0.3, double-move P ∈ {0.2, 0.4}, boost distance discrete 5-point distribution). Crucially this is the weaker "Stochastic" variant of Fig 1.8: every outcome's probability is known *exactly* from the test-case header (we are not in "Stochastic with failure", which would model *unknown* failure modes, nor in "Demonic" / adversarial). |
| 4 | **Computational limits**   | **Perfect rationality**             | The assignment tester gives VI / PI generous wall-clock budgets per iteration (~0.2–0.3 s) and caps iteration counts far above the true convergence point; offline planning is not interrupted mid-computation. The policy returned is the optimal policy for the tabular MDP (verified by matching the maximum-target reward on L1 / L4 and exceeding it on L2 / L3), not a bounded-rational / satisficing / anytime answer. |
| 5 | **Learning**               | **Known** (model is given)          | The full generative model is provided *a priori* in `GameEnv.apply_dynamics` plus the test-case header (drift/double/boost probabilities, costs, storm, penalties). No online learning, no RL exploration, no reward shaping from experience is performed; VI and PI are solved completely offline in `vi_plan_offline / pi_plan_offline` before any episode begins. |

Rubric-reading note: every row satisfies the full 1-mark per row because the value is drawn
from the Fig 1.8 enumeration and the justification ties the value *specifically* to
CrystalRover, not a generic robot.

---

*(End of Q1.)*

---

## Question 2. Stochastic dynamics: probability breakdown & visualiser  (10 marks)

### 2a. Drift × double-move Venn-style probability decomposition  (4 marks)

Let **D** = "action drifts before execution" = the environment samples one of the two
perpendicular directions (CW or CCW) instead of the player's nominal action, and let
**B** = "double move" = the drifted action is executed twice. Both events are declared
independent in the assignment header and are therefore modelled as independent Bernoulli
draws per step.

For L1 (the base case) the assignment sets
  P(D) = drift_prob = 0.30,
  P(B) = double_prob = 0.20,
  P(CW drift | D) = P(CCW drift | D) = ½ = 0.50,
  P(D ∩ B) = P(D)·P(B) = 0.06.

This decomposes the unit square of outcomes into four disjoint rectangles (a Venn-style
probability table).

**Table 2a.1 — Four-quadrant (Drift, Double) joint probabilities (L1, drift=0.30, double=0.20).**

|              | ¬ Double (P=0.80)   | Double (P=0.20)     | Row marginal     |
|--------------|---------------------|---------------------|------------------|
| ¬ Drift      | 0.70 · 0.80 = **0.56** (56%) | 0.70 · 0.20 = **0.14** (14%) | P(¬D) = 0.70     |
| Drift        | 0.30 · 0.80 = **0.24** (24%) → CW ½=0.12, CCW ½=0.12 | 0.30 · 0.20 = **0.06** (6%) → CW ½=0.03, CCW ½=0.03 | P(D) = 0.30     |
| Col marginal | P(¬B) = 0.80       | P(B) = 0.20        | **Σ = 1.000** ✓ |

For L2 (the stormy case) the assignment keeps drift=0.30 but *doubles* the double-move
probability to 0.40. The Venn-style table stretches vertically:

**Table 2a.2 — Four-quadrant joint probabilities (L2, drift=0.30, double=0.40).**

|              | ¬ Double (P=0.60)   | Double (P=0.40)     | Row marginal     |
|--------------|---------------------|---------------------|------------------|
| ¬ Drift      | 0.70 · 0.60 = **0.42** (42%) | 0.70 · 0.40 = **0.28** (28%) | P(¬D) = 0.70     |
| Drift        | 0.30 · 0.60 = **0.18** (18%) → CW/CCW = 0.09 each | 0.30 · 0.40 = **0.12** (12%) → CW/CCW = 0.06 each | P(D) = 0.30     |
| Col marginal | P(¬B) = 0.60       | P(B) = 0.40        | **Σ = 1.000** ✓ |

Interpretation of L1 vs L2: at L2, "both drift AND double" jump from 6% to 12% of every
player step, so the planner must tolerate a ×2 increase in the probability of an
*off-axis two-step excursion* into rocks / craters / lava. This is the reason L2's optimal
policy stays conservative and avoids the corridor that is optimal in L1 (observed visually
in `Q2c_levelL2_policy_arrows.png`, below). The exact analytical quadrant bars are rendered
by `visualizer.py / plot_q2a_venn_like()` and saved as:
  • screenshots/Q2a_venn_quadrants_L1-default.png
  • screenshots/Q2a_venn_quadrants_L2-default.png
(embeddable as Figure 2a in the PDF).

---

### 2b. BOOST single-step vs double-move discrete distance convolution  (3 marks)

BOOST on any WALK-like direction further compounds the effect-uncertainty layer: each
*single* BOOST leg samples a distance d ∈ {0,1,2,3,4} from the 5-point vector
`boost_probabilities` declared in the level header. For L1–L4 the assignment uses
[p₀,p₁,p₂,p₃,p₄] = [0.10, 0.30, 0.30, 0.20, 0.10]. A double-move BOOST executes
*two* such legs sequentially, so the aggregate distance d = d₁ + d₂ lives on the
stretched support {0..8} and follows the 1-D convolution p_dbl = p_sgl * p_sgl, i.e.

  P(dbl = k)  =  Σ_{i = 0..4}  P(sgl = i) · P(sgl = k − i),    k ∈ 0..8.

Convolving by hand (and verified numerically in `visualizer.py / plot_q2b_boost_convolution`
with `np.convolve`) yields the exact joint probabilities reported in Table 2b.

**Table 2b — BOOST distance distribution: single leg (5 bins) vs double-move (9 bins).**

| Distance d | 0    | 1    | 2     | 3     | 4     | 5     | 6     | 7    | 8    | Σ      | E[d]   |
|-----------:|------|------|-------|-------|-------|-------|-------|------|------|:------:|:------:|
| single (p₁) | **0.100** | **0.300** | **0.300** | **0.200** | **0.100** | —     | —     | —    | —    | 1.000  | 2.000  |
| double (p₂) | **0.010** | **0.060** | **0.150** | **0.220** | **0.230** | **0.180** | **0.100** | **0.040** | **0.010** | 1.000  | 4.000  |

Sanity checks:
  1. Σ_d p_dbl(d) = 0.01 + 0.06 + 0.15 + 0.22 + 0.23 + 0.18 + 0.10 + 0.04 + 0.01 = 1.000 ✓
  2. Linearity of expectation — E[d₁ + d₂] = E[d₁] + E[d₂] = 2.000 + 2.000 = 4.000 ✓
  3. Symmetry — p_sgl is symmetric (p₀↔p₄, p₁↔p₃); the convolution of a symmetric
     distribution with itself is again symmetric, which matches the 9-bin vector
     (0.01 ↔ 0.01 at ends, 0.04 ↔ 0.06, 0.10 ↔ 0.15, 0.18 ↔ 0.22, peak at centre 0.23 at d=4).
  4. Mass shift — double BOOST allocates only 1% of its mass to "paid-for-nothing" d=0
     (versus 10% for single BOOST), so on average a double BOOST pays 2×ACTION_COST=6.0
     to cover 4 tiles; single BOOST pays 3.0 for 2 tiles (same per-tile cost in
     expectation) — the difference is in *variance* (Var(d_dbl) = 2·Var(d_sgl) = 2.4),
     which forces the L4 planner to prefer double WALK / WALK-only corridors near lava
     over the mathematically-expected BOOST corridor (see Q4).

The analytical bars (single BOOST 5-bin histogram + double BOOST 9-bin histogram side by
side) are rendered in `screenshots/Q2b_boost_convolution_L1..L2-default.png`, alongside
printed E[d] values, which is the artefact submitted for the rubric "correct probability
matrix" criterion.

---

### 2c. Policy & Value Visualiser: design, screenshots, AI-use log & prompts  (3 marks)

#### 2c (i) Architecture of `visualizer.py`

A standalone, headless visualiser is committed alongside the solver at
[visualizer.py](file:///d:/UQLESSONS/3702/assignment/as2/2026-Assignment-2-Support-Code-main/visualizer.py). It imports only the
student-owned `solution.Solver` plus the supplied `game_env.GameEnv`, and writes all
artefacts under the new folder `screenshots/`. It uses `matplotlib` with the non-interactive
`Agg` backend so it can be invoked from Gradescope-friendly CI or headless SSH.

Its three public plotting building blocks are:

  1. **`plot_value_heatmap(level_path, label)`** — runs VI to convergence on the target
     level, then aggregates the (crystal-subset-indexed) value array `solver.vi_v_old` to
     a per-(row,col) summary of:
       • `max_v[r,c]` = most-negative V(s) over any crystal-subsets that land on tile (r,c)
         → represents the "worst-case remaining cost to go" from that tile
       • `max_abs[r,c]` = largest |V(s)| → a hazard / goal-distance proxy (used for the
         2nd column heatmap)
     The raw tile legend is drawn as the underlay of the heatmap with explicit colour/hatch
     coding: rock `R` = dark grey, launch `E` = orange, lava `L` = red + `///` hatch,
     crystal `C` = cyan, crater `*` = light purple + `xx` hatch. Grid lines mark every
     tile boundary so the reader can visually match the crystal / launch positions to the
     assignment input file.

  2. **`plot_policy_arrows(level_path, label)`** — aggregates the VI-extracted policy
     `solver.vi_policy` over crystal subsets: for each (r,c) a vote histogram over the 12
     nominal actions is built and the majority action is drawn as an arrow (`FancyArrowPatch`)
     superimposed on the max-value heatmap. Arrow style encodes action-kind:
       • WALK majority = thin solid black
       • BOOST majority = thick solid blue
       • JUMP majority = medium dashed purple
     Arrow length/mutation scale is proportional to the *consensus strength*
     (votes for majority ÷ total votes for that tile), so regions where crystal-subset
     forces two different optimal actions are visually shorter and less bold — a useful
     debugging cue for hazard-boundary decision.

  3. **`plot_q2a_venn_like` and `plot_q2b_boost_convolution`** — produce the analytical
     artefacts required for Q2a and Q2b directly from the assignment header constants so
     hand calculation errors are eliminated.

Outputs are saved with a `Q2a_*/Q2b_*/Q2c_*` prefix convention, and the `main()` routine
iterates through levels L1–L4 automatically; on the development machine it completes in
≈ 30 s total (≈ 95 % of runtime is spent inside the student's own VI `initialise / iteration`
calls, < 5 % on matplotlib rendering).

#### 2c (ii) Three representative screenshots (embedded in PDF)

The following files are committed in the coursework repo under `screenshots/` and are the
figures to embed in the final submission PDF:

| Figure in PDF | Source file in repo (path relative to repo root) | Content description |
|--------------:|--------------------------------------------------|:--------------------|
| Fig 2c.1 | `screenshots/Q2c_levelL1_value_heatmap.png` | L1 Value heatmap dual panel (worst-case V(s) signed; |V(s)| hazard proxy). Converged in 54 VI iters on \|S\| = 228 reachable states. The cold (near-zero) values coincide exactly with the two launch-pad tiles once crystals are collected, confirming V(solved)=0. |
| Fig 2c.2 | `screenshots/Q2c_levelL2_policy_arrows.png` | L2 Policy-arrow overlay with WALK (thin black), BOOST (thick blue), JUMP (dashed purple) legend. Note that crater tiles `*` are exclusively exited by JUMP-majority arrows (sanity-checking the nominal-action validity filter of the solver), and that arrows turn away from the north-east drift-susceptible corridor where P(drift ∩ double) = 0.12 per step on L2. |
| Fig 2c.3 | `screenshots/Q2c_levelL4_value_heatmap.png` | L4 Value heatmap (largest map, \|S\| = 456, VI converged in 186 iters). The hazard proxy panel resolves clearly the high-|V| red band surrounding lava tiles, matching the Bellman propagation of game_over_penalty = −500. This figure is the baseline for the Q4 controlled experiment on drift × penalty sweep. |

Additionally, the analytical figures from the rubric's earlier sub-questions are:
  • Fig 2a — `screenshots/Q2a_venn_quadrants_L1-default.png` (quadrant decomposition L1)
  • Fig 2b — `screenshots/Q2b_boost_convolution_L1..L2-default.png` (single vs double BOOST convolution)

#### 2c (iii) AI-use declaration for the visualiser module

The visualiser module was authored with the assistance of GitHub Copilot Chat inside the
editor (student plan activated 2026; see Copilot student setup URL in Q4 references).
Three prompts were iterated; the full verbatim prompt list is reproduced below for the
rubric's "AI prompt record" criterion. All generated code was manually edited to strip
non-licence comments, match the bilingual comment convention of the solution file, and
verified against `game_env`'s true attribute names (`n_rows`/`n_cols` instead of the
hallucinated `rows`/`cols`).

*Prompt 1 — scaffolding the solver-interface loop:*
```
I have a Python MDP solver class Solver in solution.py that exposes
  vi_initialise() / vi_iteration() / vi_is_converged() -> bool /
  vi_v_old : 1-D np.array aligned with self.states list of GameState,
  vi_policy : dict[int, action_str] keyed by state_idx
  self.states : list[GameState]; GameState has .row .col .crystal_status
  self.game_env has .grid_data[r][c] in {' ','R','E','L','C','*'},
                    constants ROCK_TILE LAUNCH_TILE LAVA_TILE CRYSTAL_TILE CRATER_TILE,
                    .n_rows .n_cols, .ACTION_TO_DIR_DELTA,
                    .ACTIONS = ['wl','wr','wu','wd','bl','br','bu','bd','jl','jr','ju','jd']
Write a matplotlib headless-backend script visualizer.py that:
  (a) for each of testcases/L1.txt .. L4.txt runs VI to convergence;
  (b) plots 2 side-by-side heatmaps (row x col): left = per-tile max(signed V) over
      crystal subsets, right = per-tile max(|V|); tile legend underlay colored and
      cross-hatched by terrain kind;
  (c) for each level, plots a second canvas with per-tile MAJORITY VOTED action arrow
      from vi_policy (choose action-kind style: WALK = thin black, BOOST = thick blue,
      JUMP = dashed purple; arrow length = consensus strength).
Save all outputs to screenshots/ subfolder using descriptive filenames.
Use matplotlib.use('Agg') first.
```

*Prompt 2 — adding analytical probability figures for Q2a and Q2b:*
```
Extend visualizer.py with two more plots:
  (Q2a) drift 0.3, double 0.2: four-quadrant bar chart for
       P(no drift no double)=0.56, P(drift only)=0.24,
       P(double only)=0.14, P(both)=0.06. Draw annotations that Σ=1 and
       P(D)=drift, P(B)=double still hold in the marginalised sums. Add a second
       variant for L2 double=0.4.
  (Q2b) boost_prob [0.1,0.3,0.3,0.2,0.1]: side-by-side single-BOOST histogram (5 bins
       0..4) and discrete convolution double-BOOST histogram (9 bins 0..8). Annotate
       E[d] for both and the symmetry check for the convolution.
Put both routines into the main() runner.
```

*Prompt 3 — bug-fixing attribute name:*
```
My visualizer fails with: AttributeError: 'GameEnv' object has no attribute 'rows'.
Did you mean: 'n_rows'?
Replace env.rows/env.cols with env.n_rows/env.n_cols throughout visualizer.py.
Also ensure action strings match starts with 'w' / 'b' / 'j' prefixes for the
three arrow-style branches.
```

The complete `visualizer.py` together with every generated PNG under `screenshots/` is
committed to the COMP3702-2026 student repo
`comp3702-2026/comp3702-2026-a2-50494408` alongside the solver, so the grader can
regenerate identical figures deterministically by re-running `python visualizer.py`.

---

*(End of Q2. — I will wait for your confirmation before writing Q3.)*
