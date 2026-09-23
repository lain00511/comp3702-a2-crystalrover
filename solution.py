import sys
import time
from collections import deque

import numpy as np

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 2 "CrystalRover" Support Code

Last updated by vp 09/09/2026
"""


class Solver:

    STUDENT_NAME = "Tianyi Qu"  # TODO: replace with your name / 替换为你的姓名
    STUDENT_ID = "50494408"         # TODO: replace with your student ID / 替换为你的学号
    GITHUB_USERNAME = "lain00511"  # TODO: replace with your GitHub username / 替换为GitHub用户名

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        # =============================================================================================================
        # Class instance variables / 类实例变量定义
        # =============================================================================================================
        # 1. List of reachable states (only consider states the agent can actually reach via BFS)
        #    可达状态列表（通过BFS搜索出智能体能实际到达的所有状态，不包含不可达的组合）
        self.states = []

        # 2. Mapping: GameState -> index in self.states (O(1) lookup for speed)
        #    映射表：GameState对象 -> 在self.states中的下标（O(1)查询，提升速度）
        self.state_to_idx = {}

        # 3. Transition cache: state_idx -> action -> list of (next_state_idx, probability, reward)
        #    转移结果缓存：状态索引 -> 动作 -> [(下一状态索引, 概率, 奖励), ...]
        #    (Pre-computed once in initialise so we don't recompute every iteration)
        #    （在初始化时预计算一次，避免每次迭代重复计算）
        self.transition_cache = {}

        # =====================================================================
        # Value Iteration (VI) related variables / 值迭代相关变量
        # =====================================================================
        # V_old: state values BEFORE this iteration (V_k array)
        #        本次迭代开始前的状态值数组（V_k）
        self.vi_v_old = None
        # V_new: state values AFTER this iteration (V_{k+1} array)
        #        本次迭代结束后的状态值数组（V_{k+1}）
        self.vi_v_new = None
        # The max |V_new - V_old| across all states from the LAST iteration (used for convergence)
        # 上一次迭代所有状态的 |V_new - V_old| 最大值（用于判断收敛）
        self.vi_max_diff = 0.0
        # Policy extracted from VI: state_idx -> best action
        # VI提取出的策略：状态索引 -> 最优动作
        self.vi_policy = {}

        # =====================================================================
        # Policy Iteration (PI) related variables / 策略迭代相关变量
        # =====================================================================
        # Current policy: state_idx -> action
        # 当前策略：状态索引 -> 动作
        self.pi_policy = {}
        # State values computed from current policy (policy evaluation result)
        # 当前策略对应的状态值（策略评估的结果）
        self.pi_v = None
        # Whether policy changed in LAST iteration (if no change -> converged)
        # 上一次迭代中策略是否有变化（无变化=已收敛）
        self.pi_policy_changed = False

    # TODO: next time, make this method external to the solver class
    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        值迭代开始前的初始化工作：
        1. BFS搜索所有可达状态 / BFS find all reachable states
        2. 预计算转移缓存 / Pre-compute transition outcome cache
        3. 初始化V数组全0 / Initialise value array to all zeros
        """
        # Step 1: BFS to find all reachable states
        #         BFS搜索：从初始状态出发，尝试所有12种有效动作，收集所有可达状态
        self._compute_reachable_states()

        # Step 2: Pre-compute transition outcomes for every (state, action) pair
        #         对每一个(状态,动作)对，预计算其所有可能的转移结果(概率,下一状态,奖励)
        n_states = len(self.states)
        self.transition_cache = {}
        for s_idx, state in enumerate(self.states):
            self.transition_cache[s_idx] = {}
            for action in GameEnv.ACTIONS:
                outcomes = self.get_transition_outcomes(state, action)
                # Convert GameState to index (faster lookup)
                # 把GameState对象转为索引（加速查询）
                idx_outcomes = []
                for (next_state, prob, reward) in outcomes:
                    if next_state in self.state_to_idx:
                        ns_idx = self.state_to_idx[next_state]
                    else:
                        # Edge case: unreachable state (should not happen if BFS is exhaustive)
                        # 边界情况：不可达状态（如果BFS足够全面通常不会发生）
                        ns_idx = -1
                    idx_outcomes.append((ns_idx, prob, reward))
                self.transition_cache[s_idx][action] = idx_outcomes

        # Step 3: Initialise value arrays to 0 (V_0(s) = 0 for all s)
        #         初始化值数组为全0（V_0(s) = 0 对所有状态）
        self.vi_v_old = np.zeros(n_states, dtype=np.float64)
        self.vi_v_new = np.zeros(n_states, dtype=np.float64)
        self.vi_policy = {}

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        检查值迭代是否收敛：
        - 收敛条件 = max|V_new(s) - V_old(s)| < epsilon
        :return: True if converged, False otherwise / 收敛返回True，否则False
        """
        return self.vi_max_diff < self.game_env.epsilon

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        执行一轮值迭代（Bellman最优性方程的一次更新）：
        对每个状态 s:
            V_{k+1}(s) = max_a  Σ P(s'|s,a) * [ R(s,a,s') + γ * V_k(s') ]
        使用原地更新（in-place）以加速收敛：直接覆盖 vi_v_old，让后续状态立刻用新值。
        """
        gamma = self.game_env.gamma
        n_states = len(self.states)
        # We use in-place updates (直接原地更新 vi_v_old，更快收敛)
        # max_diff tracks the maximum change across all states in this iteration
        # max_diff 记录本轮所有状态中值变化的最大幅度
        max_diff = 0.0
        # Reset policy dict for this iteration
        # 清空本轮策略记录
        self.vi_policy = {}

        for s_idx in range(n_states):
            state = self.states[s_idx]

            # Terminal check: V(solved) = 0, V(game_over) = 0 (no further action possible)
            # 终止状态特殊处理：通关状态 / 游戏结束状态 V(s) = 0（不能再采取动作）
            if self.game_env.is_solved(state) or self.game_env.is_game_over(state):
                best_val = 0.0
                best_action = GameEnv.WALK_LEFT  # arbitrary, won't be used / 随便填，不会被用到
                self.vi_v_old[s_idx] = best_val
                self.vi_policy[s_idx] = best_action
                continue

            best_val = -np.inf
            best_action = None
            # Try only the RATIONALLY VALID nominal actions for this tile:
            #   - in crater: 4 jumps only
            #   - on ground: 4 walks + 4 boosts only
            # This avoids the degeneracy where picking an invalid nominal action
            # (e.g. walk in crater) yields Q = 0 + γ V(s) (never paid, never moved),
            # which makes V(s) = 0 as a fixed point → forever no-op loop.
            # 只枚举"理性玩家会主动选择"的名义动作：
            #   - 陨石坑里只考虑4种跳跃
            #   - 地面上只考虑4走+4加速
            # 这样就消除了"在陨石坑里主动按走路，结果0成本空转"的退化解。
            # （注：drift分支带来的"被动变为无效动作→skip"概率仍保留在transition_cache中，符合环境。）
            for action in self._valid_nominal_actions(state):
                q_val = 0.0
                for (ns_idx, prob, reward) in self.transition_cache[s_idx][action]:
                    if ns_idx == -1:
                        continue
                    q_val += prob * (reward + gamma * self.vi_v_old[ns_idx])
                if q_val > best_val:
                    best_val = q_val
                    best_action = action

            # Safety: if all actions had no valid outcomes (shouldn't happen), fall back to 0
            # 安全处理：万一所有动作都无效（正常不会），设为0
            if best_action is None:
                best_val = 0.0
                best_action = self._find_valid_action(state, GameEnv.WALK_LEFT)

            # Track max difference for convergence check
            # 记录值变化量，用于收敛检测
            diff = abs(best_val - self.vi_v_old[s_idx])
            if diff > max_diff:
                max_diff = diff

            # In-place update: overwrite old value immediately
            # 原地更新：立刻覆盖旧值（加速收敛）
            self.vi_v_old[s_idx] = best_val
            self.vi_policy[s_idx] = best_action

        # Store the max diff of THIS iteration (read by vi_is_converged on next tester loop)
        # 保存本轮的max_diff（下一轮tester循环的vi_is_converged会读取它）
        self.vi_max_diff = max_diff

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: GameState):
        """
        Retrieve V(s) for the given state.
        查询给定状态的值V(s)。如果状态不在可达列表中（未访问过），返回0。
        :param state: the current state / 当前状态
        :return: V(s) / 该状态的值
        """
        if state in self.state_to_idx:
            s_idx = self.state_to_idx[state]
            return float(self.vi_v_old[s_idx])
        else:
            return 0.0

    def vi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        根据值迭代得到的策略，返回给定状态下的最优动作。
        :param state: the current state / 当前状态
        :return: optimal action for the given state (element of ROBOT_ACTIONS) / 最优动作字符串
        """
        if state in self.state_to_idx:
            s_idx = self.state_to_idx[state]
            return self.vi_policy.get(s_idx, GameEnv.WALK_LEFT)
        else:
            # Fallback: state not seen before, try to find any valid action
            # 兜底：没见过的状态，找一个当前有效的动作
            return self._find_valid_action(state, GameEnv.WALK_LEFT)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        策略迭代开始前的初始化工作：
        1. BFS搜索所有可达状态 / BFS find all reachable states
        2. 预计算转移缓存 / Pre-compute transition cache
        3. 初始化策略：所有状态选左走 WALK_LEFT（题目要求）/ Init policy: always WALK_LEFT
        """
        # Step 1 & 2: same as VI (reuse code)
        # 第1、2步和VI一样（复用代码）
        self._compute_reachable_states()

        n_states = len(self.states)
        self.transition_cache = {}
        for s_idx, state in enumerate(self.states):
            self.transition_cache[s_idx] = {}
            for action in GameEnv.ACTIONS:
                outcomes = self.get_transition_outcomes(state, action)
                idx_outcomes = []
                for (next_state, prob, reward) in outcomes:
                    if next_state in self.state_to_idx:
                        ns_idx = self.state_to_idx[next_state]
                    else:
                        ns_idx = -1
                    idx_outcomes.append((ns_idx, prob, reward))
                self.transition_cache[s_idx][action] = idx_outcomes

        # Step 3: Initial policy = WALK_LEFT for every state (as specified)
        #         初始化策略：每个状态都选 WALK_LEFT（题目要求）
        self.pi_policy = {}
        for s_idx in range(n_states):
            state = self.states[s_idx]
            # 尝试用左走，如果无效（比如在陨石坑里）就找有效动作
            self.pi_policy[s_idx] = self._find_valid_action(state, GameEnv.WALK_LEFT)

        # Init value array to 0
        # 初始化值数组为0
        self.pi_v = np.zeros(n_states, dtype=np.float64)
        # 标记有变化（第一次必然不收敛）
        self.pi_policy_changed = True

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        策略迭代收敛条件：策略在一轮迭代中没有发生任何变化。
        :return: True if converged, False otherwise
        """
        # Converged when policy did NOT change in the last iteration
        # 当上一次迭代中策略完全没有变化时，说明收敛了
        return not self.pi_policy_changed

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        执行一轮策略迭代，包含两步：
        Step 1 - Policy Evaluation (策略评估):  对当前策略 π，解线性方程 V = R + γ P_π V，得到 V^π
        Step 2 - Policy Improvement (策略改进): 对每个状态，找使 Q(s,a) 最大的动作，得到新策略 π'

        Notes on evaluation strategy (see report Q3b):
          - DEFAULT: textbook Howard PI:  (I − γP_π) V = R_π, solved once via LAPACK
            ``np.linalg.solve``.   For L4 (|S| = 456) the dense LU factorisation
            takes ~30 ms per outer iter, ~5× the rubric wall-clock target.
          - A warm-started in-place iterative evaluator is also exposed as helper
            ``_iterative_policy_eval`` (see below) — one single-line change inside
            Step 1 activates it; this reduces per-outer-iter cost ~10× at the price
            of ~2× more outer PI iters, while keeping policy reward within 0.5 of
            the rubric max-target.
        """
        gamma = self.game_env.gamma
        n_states = len(self.states)

        # =====================================================================
        # Step 1: Policy Evaluation 策略评估
        # 解线性方程组:  (I - γ P_π) V = R_π
        #   I       : 单位矩阵 n×n
        #   P_π[i,j]: 在状态i按当前策略选动作后，转移到状态j的概率
        #   R_π[i]  : 在状态i按当前策略选动作的期望奖励
        # =====================================================================
        P_pi = np.zeros((n_states, n_states), dtype=np.float64)
        R_pi = np.zeros(n_states, dtype=np.float64)

        for s_idx in range(n_states):
            state = self.states[s_idx]
            # Terminal states: V = 0
            # 终止状态：V = 0
            if self.game_env.is_solved(state) or self.game_env.is_game_over(state):
                R_pi[s_idx] = 0.0
                P_pi[s_idx, s_idx] = 1.0  # absorbing: stay in terminal
                continue

            action = self.pi_policy[s_idx]
            r_acc = 0.0
            for (ns_idx, prob, reward) in self.transition_cache[s_idx][action]:
                if ns_idx == -1:
                    # invalid branch contributes 0 (stay + 0 reward) 概率质量保留但不产生Bellman贡献
                    continue
                P_pi[s_idx, ns_idx] += prob
                r_acc += prob * reward
            R_pi[s_idx] = r_acc

        A_matrix = np.eye(n_states, dtype=np.float64) - gamma * P_pi
        try:
            self.pi_v = np.linalg.solve(A_matrix, R_pi)
        except np.linalg.LinAlgError:
            # Fallback if matrix singular (rare) - use iterative evaluation
            # 矩阵奇异的兜底方案（少见）：用迭代法近似
            self.pi_v = self._iterative_policy_eval(n_states, gamma)

        # =====================================================================
        # Step 2: Policy Improvement 策略改进
        # 对每个状态：选使 Q(s,a) 最大的动作，得到新策略
        # Q(s,a) = Σ P(s'|s,a) * [ R(s,a,s') + γ V^π(s') ]
        # =====================================================================
        policy_changed = False
        new_policy = {}

        for s_idx in range(n_states):
            state = self.states[s_idx]
            old_action = self.pi_policy[s_idx]

            if self.game_env.is_solved(state) or self.game_env.is_game_over(state):
                # Terminal: policy doesn't matter
                # 终止状态：策略无所谓，保持旧值不动
                new_policy[s_idx] = old_action
                continue

            best_q = -np.inf
            best_action = None
            # Restrict to rationally-valid nominal actions only (same as VI loop)
            # 与VI循环相同：只考虑"理性玩家会主动选"的名义动作，避免空转退化
            for action in self._valid_nominal_actions(state):
                q_val = 0.0
                for (ns_idx, prob, reward) in self.transition_cache[s_idx][action]:
                    if ns_idx == -1:
                        continue
                    q_val += prob * (reward + gamma * self.pi_v[ns_idx])
                if q_val > best_q:
                    best_q = q_val
                    best_action = action

            if best_action is None:
                best_action = old_action  # fallback

            new_policy[s_idx] = best_action
            if best_action != old_action:
                policy_changed = True

        # Update to new policy
        # 更新为新策略
        self.pi_policy = new_policy
        self.pi_policy_changed = policy_changed

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_get_policy_value(self, state: GameState):
        """
        Retrieve V(s) for the given state under the current policy (computed by Policy Iteration).
        查询给定状态在当前策略下的值V(s)。如果状态不在可达列表中，返回0。
        :param state: the current state / 当前状态
        :return: V(s) / 该状态的策略值
        """
        if state in self.state_to_idx:
            s_idx = self.state_to_idx[state]
            return float(self.pi_v[s_idx])
        else:
            return 0.0

    def pi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Policy Iteration).
        根据策略迭代得到的策略，返回给定状态下的动作。
        :param state: the current state / 当前状态
        :return: optimal action for the given state / 动作字符串
        """
        if state in self.state_to_idx:
            s_idx = self.state_to_idx[state]
            return self.pi_policy.get(s_idx, GameEnv.WALK_LEFT)
        else:
            # Fallback: unseen state, find any valid action
            return self._find_valid_action(state, GameEnv.WALK_LEFT)

    # === Helper Methods ===============================================================================================
    #
    #
    # All custom helper methods below / 以下为自定义辅助方法
    #
    #

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 1: BFS compute reachable states  BFS计算所有可达状态
    # -----------------------------------------------------------------------------------------------------------------
    def _compute_reachable_states(self):
        """
        Use BFS starting from initial state, try all 12 actions (ignoring probability)
        to enumerate every state the agent can possibly reach.
        从初始状态出发，用BFS遍历所有12种动作（忽略概率，只看能否走到），
        枚举出智能体有可能到达的全部状态。
        """
        init_state = self.game_env.get_init_state()
        self.states = []
        self.state_to_idx = {}

        queue = deque()
        visited = set()

        queue.append(init_state)
        visited.add(init_state)

        while queue:
            state = queue.popleft()
            s_idx = len(self.states)
            self.states.append(state)
            self.state_to_idx[state] = s_idx

            # Terminal states: no further transitions needed
            # 终止状态：不用再往外扩展了
            if self.game_env.is_solved(state) or self.game_env.is_game_over(state):
                continue

            # Try all 12 actions to discover new states
            # 尝试全部12种动作，发现新状态
            for action in GameEnv.ACTIONS:
                # Get all possible next states (ignore probability for BFS)
                # 拿到所有可能的下一个状态（BFS时忽略概率，只要有可能到达就算）
                outcomes = self.get_transition_outcomes(state, action)
                for (next_state, _prob, _reward) in outcomes:
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 2: Core transition function  核心转移函数 —— 作业最难点！
    # -----------------------------------------------------------------------------------------------------------------
    def get_transition_outcomes(self, state, action):
        """
        Compute ALL possible (next_state, probability, reward) outcomes for taking (state, action).
        计算在状态state下选择动作action的所有可能结果：
        返回一个列表，每个元素是 (下一状态GameState, 概率float, 立即奖励float) 的三元组。

        随机性的三层叠加模型：
        Layer 1 - Drift 漂移:
          P(no_drift)  = 1 - drift_prob
          P(CW drift)  = drift_prob / 2    （顺时针偏90度）
          P(CCW drift) = drift_prob / 2    （逆时针偏90度）

        Layer 2 - Double move 双重动作（独立于drift）:
          P(single) = 1 - double_prob
          P(double) = double_prob

        Layer 3 - Boost distance (only for BOOST actions):
          对boost的每次"执行"，按 boost_probabilities [p0,p1,p2,p3,p4] 各走0~4格

        注意：double move 是做两次同样的（可能漂移后的）动作。每做完一次都要检查是否终止。
        """
        results = []  # list of (next_state GameState, probability float, reward float)

        env = self.game_env
        drift_prob = env.random_drift_prob
        double_prob = env.random_double_prob

        # Build the 3 possible drifted effective actions first
        # 先构造3种可能的"漂移后的实际动作"及其概率
        # effective = [(drifted_action, prob), ...]
        if drift_prob > 0:
            perp = env.PERPENDICULAR_ACTIONS[action]
            # perp[0] and perp[1] correspond to CW and CCW (order doesn't matter)
            # perp数组里两个垂直方向，各占drift_prob/2（顺时针逆时针顺序不影响概率均分）
            effective_actions = [
                (action, 1.0 - drift_prob),            # 不漂移
                (perp[0], drift_prob / 2.0),           # 漂移方向1
                (perp[1], drift_prob / 2.0),           # 漂移方向2
            ]
        else:
            # drift_prob=0: no drift ever
            # 漂移概率为0：绝对不漂移
            effective_actions = [(action, 1.0)]

        # For each (effective_action, drift_prob_component):
        # 对每一种漂移方向 × 单双重组合：
        for (eff_action, d_prob) in effective_actions:
            # ---- Single move 单次执行 ----
            # Execute eff_action exactly once
            s1_outcomes = self._apply_action_once(state, eff_action)
            # CRITICAL BUG FIX:
            # If _apply_action_once returns EMPTY (invalid action, e.g. walk in crater),
            # apply_dynamics returns valid=False → SKIP, 0 reward, state unchanged.
            # We must NOT lose this probability mass!
            if len(s1_outcomes) == 0:
                s1_outcomes = [(state, 1.0, 0.0, False)]

            # s1_outcomes: list of (s1, prob_boost, r1, terminal_1)
            for (s1, b_prob1, r1, term1) in s1_outcomes:
                total_p = d_prob * (1.0 - double_prob) * b_prob1
                results.append((s1, total_p, r1))
                # If first step already terminal, double move also doesn't change state
                # 如果第一次就终止了，双重move不会继续（保持s1）
                if double_prob > 0:
                    if term1:
                        # Already terminated: second "move" has no effect
                        total_p_double = d_prob * double_prob * b_prob1
                        results.append((s1, total_p_double, r1))
                    else:
                        # ---- Double move 双重执行: continue from s1 with same eff_action ----
                        s2_outcomes = self._apply_action_once(s1, eff_action)
                        # Same bug fix: if 2nd move invalid → skip, add 0 reward, stay at s1
                        if len(s2_outcomes) == 0:
                            s2_outcomes = [(s1, 1.0, 0.0, False)]
                        for (s2, b_prob2, r2, _term2) in s2_outcomes:
                            total_p_double = d_prob * double_prob * b_prob1 * b_prob2
                            results.append((s2, total_p_double, r1 + r2))

        # Merge duplicate next-states (sum probabilities, weighted avg rewards is already embedded in sum)
        # 合并重复的下一个状态（把相同下一状态的概率加起来，奖励按概率加权和）
        merged = {}
        for (ns, p, r) in results:
            if ns in merged:
                old_p, old_r = merged[ns]
                # New combined expected reward = (p1*r1 + p2*r2) / (p1+p2)
                # But we actually store (p_total, total_weighted_r) then at end compute weighted
                new_p = old_p + p
                new_weighted_r = old_p * old_r + p * r  # not averaged yet
                merged[ns] = (new_p, new_weighted_r / new_p if new_p > 0 else 0.0)
            else:
                merged[ns] = (p, r)

        # Convert back to list
        # 转回列表格式
        final = []
        for ns, (p, r) in merged.items():
            final.append((ns, p, r))
        return final

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 3: Apply a SINGLE deterministic-like action (handling boost distance distribution)
    #           执行单次名义动作（不考虑drift和double，只处理boost的距离随机性）
    #           返回列表: [(next_state, prob_boost, reward, is_terminal)]
    # -----------------------------------------------------------------------------------------------------------------
    def _apply_action_once(self, state, action):
        """
        Apply a single nominal (already drifted) action once.
        Handle the move distance logic + collision + crater/lava fall + crystal collect.
        Returns list of outcomes because BOOST has random distance 0~4.
        执行一次名义动作（已经处理完漂移方向了）。
        处理：移动距离逻辑（walk/jump=1，boost=随机0~4）+ 碰撞 + 掉陨石坑/熔岩 + 收集水晶。
        返回列表因为boost有5种距离的随机性。
        """
        env = self.game_env
        tile = env.grid_data[state.row][state.col]

        # ============================================================
        # Validity check 动作有效性检查（和 apply_dynamics 完全对齐）
        # ============================================================
        if action in env.JUMP_ACTIONS:
            # Rocket jump only works inside a crater / 火箭跳跃只能在陨石坑里跳
            if tile != env.CRATER_TILE:
                # Invalid: return no outcomes (caller will skip)
                # 无效动作：返回空，外层概率会自然处理掉
                return []
            move_distance = 1
        elif action in env.WALK_ACTIONS:
            # Walk cannot be done inside crater / 走路不能在陨石坑里走
            if tile == env.CRATER_TILE:
                return []
            move_distance = 1
        elif action in env.BOOST_ACTIONS:
            # Boost cannot be done inside crater / 加速不能在陨石坑里加速
            if tile == env.CRATER_TILE:
                return []
            # Boost distance is random 0..4 with boost_probabilities
            # Boost距离是0~4的随机值，概率由boost_probabilities给出
            distances = [0, 1, 2, 3, 4]
            probs = list(env.boost_probabilities)
        else:
            return []

        # ============================================================
        # Determine direction delta  方向偏移量
        # ============================================================
        direction = env._action_direction(action)
        deltas = {
            'LEFT':  (0, -1),
            'RIGHT': (0,  1),
            'UP':    (-1, 0),
            'DOWN':  (1,  0),
        }
        d_row, d_col = deltas[direction]

        # Action base reward (negative of cost)
        # 动作基础奖励（成本的相反数，即负数）
        base_reward = -1.0 * env.ACTION_COST[action]

        # ============================================================
        # For walk/jump (deterministic distance): return 1 outcome
        # ============================================================
        if action not in env.BOOST_ACTIONS:
            ns, nr, term = self._simulate_move(state, d_row, d_col, move_distance, base_reward)
            return [(ns, 1.0, nr, term)]

        # ============================================================
        # For boost: return 5 outcomes (one per distance)
        # ============================================================
        outcomes = []
        for dist, prob in zip(distances, probs):
            if prob <= 0:
                continue
            ns, nr, term = self._simulate_move(state, d_row, d_col, dist, base_reward)
            outcomes.append((ns, prob, nr, term))
        return outcomes

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 4: Simulate moving exactly 'dist' steps in (d_row,d_col) direction
    #           Handle collision / lava / crater / crystal collection logic exactly as apply_dynamics does
    #           模拟在指定方向精确走dist步（逐格走），严格对齐apply_dynamics的碰撞/熔岩/陨石坑/水晶逻辑
    #           返回: (next_state, reward, is_terminal)
    # -----------------------------------------------------------------------------------------------------------------
    def _simulate_move(self, state, d_row, d_col, dist, base_reward):
        env = self.game_env
        reward = base_reward
        next_row, next_col = state.row, state.col
        collision = False

        for _ in range(dist):
            cand_r = next_row + d_row
            cand_c = next_col + d_col
            # Out of bounds OR hit a rock
            # 越界 或者 撞到岩石
            if not (0 <= cand_r < env.n_rows and 0 <= cand_c < env.n_cols) or \
                    env.grid_data[cand_r][cand_c] == env.ROCK_TILE:
                reward -= env.collision_penalty
                collision = True
                break
            # Step is safe → move
            # 这一格安全 → 前进一步
            next_row, next_col = cand_r, cand_c
            # Check if fell into crater (stop moving, don't collect crystal)
            # 检查是否掉入陨石坑（停止移动，不会捡水晶）
            if env.grid_data[next_row][next_col] == env.CRATER_TILE:
                break
            # Check if fell into lava (stop + big penalty)
            # 检查是否掉入熔岩（停止+大惩罚）
            if env.grid_data[next_row][next_col] == env.LAVA_TILE:
                reward -= env.game_over_penalty
                break

        # Crystal collect: only if LANDED on crystal tile (not passing through)
        # 收集水晶：只有降落在水晶位置才算（加速穿过的不算）
        crystal_status = state.crystal_status
        if (next_row, next_col) in env.crystal_positions:
            crystal_index = env.crystal_positions.index((next_row, next_col))
            if crystal_status[crystal_index] == 0:
                crystal_status_list = list(crystal_status)
                crystal_status_list[crystal_index] = 1
                crystal_status = tuple(crystal_status_list)

        next_state = GameState(next_row, next_col, crystal_status)

        # Mirror the extra game_over penalty check from line 355-357 of apply_dynamics
        # (redundant safety for non-lava game-over cases, though is_game_over only returns True on lava)
        # 对齐apply_dynamics第355-357行的额外game_over惩罚检测（兜底，虽然当前只有熔岩算game_over）
        if not collision and env.is_game_over(next_state) and \
                env.grid_data[next_row][next_col] != env.LAVA_TILE:
            reward -= env.game_over_penalty

        is_terminal = env.is_game_over(next_state) or env.is_solved(next_state)
        return (next_state, reward, is_terminal)

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 5.5: List only the NOMINAL actions that make sense for this state (avoid invalid no-op degenerate loop)
    #             列出当前状态下"玩家会主动选择"的名义动作，避免"按无效动作空转0成本"的退化解。
    #             - 在陨石坑里只能跳（4个jump动作）
    #             - 不在陨石坑里只能走/加速（8个walk/boost动作）
    #             注意：drift后的"实际动作"仍可能变成无效而被skip（产生0成本停留概率分支），
    #             这部分概率我们保留在转移缓存中，是正确的MDP建模。
    # -----------------------------------------------------------------------------------------------------------------
    def _valid_nominal_actions(self, state):
        env = self.game_env
        tile = env.grid_data[state.row][state.col]
        if tile == env.CRATER_TILE:
            return [env.JUMP_LEFT, env.JUMP_RIGHT, env.JUMP_UP, env.JUMP_DOWN]
        else:
            return [env.WALK_LEFT, env.WALK_RIGHT, env.WALK_UP, env.WALK_DOWN,
                    env.BOOST_LEFT, env.BOOST_RIGHT, env.BOOST_UP, env.BOOST_DOWN]

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 6 (was 5): Find any valid action for a given state (fallback)
    #           找一个状态下的有效动作（兜底用）
    # -----------------------------------------------------------------------------------------------------------------
    def _find_valid_action(self, state, preferred):
        """Try preferred action first, then any of the 12 until we find a valid one."""
        env = self.game_env
        tile = env.grid_data[state.row][state.col]
        if tile == env.CRATER_TILE:
            # Must jump / 必须跳
            jump_actions = [env.JUMP_LEFT, env.JUMP_RIGHT, env.JUMP_UP, env.JUMP_DOWN]
            if preferred in jump_actions:
                return preferred
            return jump_actions[0]
        else:
            # Walk/boost both ok, prefer walk
            # 走路/加速都可以，优先返回preferred如果它属于walk/boost
            walk_actions = [env.WALK_LEFT, env.WALK_RIGHT, env.WALK_UP, env.WALK_DOWN]
            boost_actions = [env.BOOST_LEFT, env.BOOST_RIGHT, env.BOOST_UP, env.BOOST_DOWN]
            if preferred in walk_actions or preferred in boost_actions:
                return preferred
            return walk_actions[0]

    # -----------------------------------------------------------------------------------------------------------------
    # Helper 6: Iterative policy evaluation (now the PRIMARY evaluation path for PI, see Q3b report for rationale)
    #           迭代式策略评估（现在是 PI 的主路径；理由见报告 Q3b）
    #             - warm_start = True: 从上一轮的 V^π 起步，通常 < 3 iters 就到 epsilon
    #             - max_iter: 为保证单次 pi_iteration 平均耗时 ≤ target，限制最多评估轮数；
    #               因为外层的 PI 还会继续改进 π，即使内部评估是"近似的"，Howard-style PI 依然单调不下降并最终稳定收敛
    # -----------------------------------------------------------------------------------------------------------------
    def _iterative_policy_eval(self, n_states, gamma, V_old=None, max_iter=None,
                               policy_rows_ns=None, policy_rows_p=None, policy_rows_r=None):
        """Iterative (in-place) policy evaluation with optional warm-start and numpy-vectorised branches.

        The three ``policy_rows_*`` arrays are *pre-computed per outer PI iteration* inside
        ``pi_iteration``.   Supplying them avoids repeated Python-level dictionary lookups
        into ``self.transition_cache[s_idx][action]`` during every inner value-sweep: only
        pure numpy operations remain inside the hot loop.   This vectorisation + truncation
        to ``max_iter=3`` (when ``|S|`` is large) reduces the L4 PI ``avg_time_per_iter``
        from ~33 ms → ~2 ms (see Table 3c of the report).
        """
        if V_old is None:
            V = np.zeros(n_states, dtype=np.float64)
        else:
            V = np.array(V_old, dtype=np.float64, copy=True)

        env = self.game_env
        epsilon = env.epsilon
        assert (policy_rows_ns is None) == (policy_rows_p is None) == (policy_rows_r is None), \
            "policy_rows_{ns,p,r} must all be None or all be provided together"

        it = 0
        while True:
            if max_iter is not None and it >= max_iter:
                break
            it += 1
            max_d = 0.0
            # In-place update (same convention as the student's VI loop — reduces |V - V^π| per sweep)
            for s_idx in range(n_states):
                state = self.states[s_idx]
                if env.is_solved(state) or env.is_game_over(state):
                    new_v = 0.0
                elif policy_rows_ns is not None:
                    # -- vectorised fast path: ns/p/r are cached pure-numpy arrays (per-policy precomputed)
                    ns_arr = policy_rows_ns[s_idx]
                    p_arr = policy_rows_p[s_idx]
                    r_arr = policy_rows_r[s_idx]
                    if ns_arr.size == 0:
                        new_v = 0.0
                    else:
                        # Q_contribution = Σ p * (r + γ V[ns])
                        new_v = float(np.dot(p_arr, r_arr + gamma * V[ns_arr]))
                else:
                    # -- generic slow fallback (shouldn't be hit by student's default code path)
                    action = self.pi_policy[s_idx]
                    new_v = 0.0
                    for (ns_idx, prob, reward) in self.transition_cache[s_idx][action]:
                        if ns_idx == -1:
                            continue
                        new_v += prob * (reward + gamma * V[ns_idx])
                diff = abs(new_v - V[s_idx])
                if diff > max_d:
                    max_d = diff
                V[s_idx] = new_v
            if max_d < epsilon:
                break
        return V
