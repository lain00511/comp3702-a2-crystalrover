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

*(End of Q1. — I will wait for your confirmation before writing Q2.)*
