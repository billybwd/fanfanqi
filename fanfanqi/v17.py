"""StrategicV17AI: V16 with broader tactical coverage and stricter time control."""

import copy
import random
import time

from .v16 import (
    CANNON_HIDDEN,
    CAPTURE,
    DEFAULT_PARAMS as V16_DEFAULT_PARAMS,
    FLIP,
    HPC,
    KING,
    MOVE,
    Position,
    Searcher as V16Searcher,
    SOLDIER,
    T_RANK,
    TYPE,
    _Timeout,
    merge_params,
    move_to_action,
)


DEFAULT_PARAMS = copy.deepcopy(V16_DEFAULT_PARAMS)
DEFAULT_PARAMS.update({
    # Maximum hidden-cannon captures considered at each quiescence node.
    'q_hidden_caps': 1,
    # Number of consecutive hidden-cannon captures explored in quiescence.
    'q_hidden_depth': 1,
    # Check the clock more often than V16 during recursive search.
    'time_check_interval': 32,
    # Progressive widening while many hidden pieces remain.
    'early_pool_threshold': 20,
    'mid_pool_threshold': 12,
    'early_root_flips': 6,
    'mid_root_flips': 7,
    'early_node_flips': 2,
    'mid_node_flips': 3,
    # Stratified samples for non-critical outcomes; kings and soldiers stay exact.
    'early_chance_samples': 6,
    'mid_chance_samples': 8,
})


class Searcher(V16Searcher):
    """V16 search with better flip ordering, hidden-capture quiescence and timing."""

    def __init__(self, params):
        super().__init__(params)
        self.time_check_interval = max(1, int(params.get('time_check_interval', 32)))

    def _tick(self):
        self.nodes += 1
        if self.nodes % self.time_check_interval == 0 and time.perf_counter() >= self.deadline:
            raise _Timeout()

    def _phase_limits(self, pos):
        pool_size = sum(pos.pool)
        if pool_size >= int(self.p.get('early_pool_threshold', 20)):
            return (int(self.p.get('early_root_flips', 6)),
                    int(self.p.get('early_node_flips', 2)))
        if pool_size >= int(self.p.get('mid_pool_threshold', 12)):
            return (int(self.p.get('mid_root_flips', 7)),
                    int(self.p.get('mid_node_flips', 3)))
        return (int(self.p['root_flips']), int(self.p['node_flips']))

    def _chance_sample_count(self, pos):
        pool_size = sum(pos.pool)
        if pool_size >= int(self.p.get('early_pool_threshold', 20)):
            return max(0, int(self.p.get('early_chance_samples', 6)))
        if pool_size >= int(self.p.get('mid_pool_threshold', 12)):
            return max(0, int(self.p.get('mid_chance_samples', 8)))
        return 0

    def _chance_weights(self, pos):
        """Return outcome probabilities, sampling common outcomes in early play.

        King and soldier outcomes are always enumerated exactly. The remaining
        pool is estimated with deterministic systematic sampling shared across
        candidate actions from the same position.
        """
        pool_size = sum(pos.pool)
        if not pool_size:
            return {}
        sample_count = self._chance_sample_count(pos)
        if not sample_count:
            return {code: pos.pool[code] / pool_size
                    for code in range(1, 15) if pos.pool[code]}

        protected = [code for code in range(1, 15)
                     if pos.pool[code] and TYPE[code] in (KING, SOLDIER)]
        remaining = [code for code in range(1, 15)
                     if pos.pool[code] and code not in protected]
        remaining_size = sum(pos.pool[code] for code in remaining)
        weights = {code: pos.pool[code] / pool_size for code in protected}
        if not remaining_size:
            return weights
        if remaining_size <= sample_count:
            weights.update({code: pos.pool[code] / pool_size for code in remaining})
            return weights

        # Stable state hash gives the same sample sequence for every candidate
        # action from this position, reducing noise in action comparisons.
        seed = 2166136261
        values = (*pos.b, *pos.pool, pos.hp[0], pos.hp[1], pos.turn, pos.flip)
        for value in values:
            if isinstance(value, float):
                parts = value.as_integer_ratio()
            else:
                parts = (int(value),)
            for part in parts:
                seed = ((seed ^ part) * 16777619) & 0xFFFFFFFF
        offset = random.Random(seed).random()
        points = [(i + offset) * remaining_size / sample_count
                  for i in range(sample_count)]
        sampled = {code: 0 for code in remaining}
        cumulative = 0
        point_index = 0
        for code in remaining:
            cumulative += pos.pool[code]
            while point_index < sample_count and points[point_index] < cumulative:
                sampled[code] += 1
                point_index += 1
        sample_mass = remaining_size / pool_size
        weights.update({code: sample_mass * count / sample_count
                        for code, count in sampled.items() if count})
        return weights

    def ordered_moves(self, pos, ply, tt_best):
        root_flips, node_flips = self._phase_limits(pos)
        old_root_flips, old_node_flips = self.p['root_flips'], self.p['node_flips']
        self.p['root_flips'], self.p['node_flips'] = root_flips, node_flips
        try:
            moves = super().ordered_moves(pos, ply, tt_best)
        finally:
            self.p['root_flips'], self.p['node_flips'] = old_root_flips, old_node_flips
        # V16 places every flip behind quiet moves via a fixed -1000 score. Keep
        # tactical captures first, then let flips compete before quiet moves.
        tactical = [m for m in moves if m[0] in (CAPTURE, CANNON_HIDDEN)]
        flips = [m for m in moves if m[0] == FLIP]
        quiet = [m for m in moves if m[0] == MOVE]
        ordered = tactical[:]
        for i in range(max(len(flips), len(quiet))):
            if i < len(flips):
                ordered.append(flips[i])
            if i < len(quiet):
                ordered.append(quiet[i])
        if tt_best in ordered:
            ordered.remove(tt_best)
            ordered.insert(0, tt_best)
        if time.perf_counter() >= self.deadline:
            raise _Timeout()
        return ordered

    def child_value(self, pos, move, depth, alpha, beta, ply):
        kind, a, c = move
        if kind not in (FLIP, CANNON_HIDDEN):
            return super().child_value(pos, move, depth, alpha, beta, ply)
        if kind == FLIP and ply >= self.p['full_chance_plies']:
            return super().child_value(pos, move, depth, alpha, beta, ply)

        weights = self._chance_weights(pos)
        value = 0.0
        for code, probability in weights.items():
            u = (pos.do_flip(a, code) if kind == FLIP
                 else pos.do_cannon_hidden(a, c, code))
            try:
                value += probability * -self.negamax(pos, depth - 1, -float('inf'), float('inf'), ply + 1)
            finally:
                pos.undo(u)
        return value

    @staticmethod
    def _fallback(pos):
        caps, hidden_caps, moves, flips = pos.gen()
        if caps:
            return (CAPTURE, *caps[0])
        if hidden_caps:
            return (CANNON_HIDDEN, *hidden_caps[0])
        if flips:
            sq = flips[0]
            return (FLIP, sq, sq)
        if moves:
            return (MOVE, *moves[0])
        return None

    def choose(self, pos, budget_ms):
        self.deadline = time.perf_counter() + max(1, budget_ms) / 1000
        self.tt.clear()
        self.nodes = 0
        try:
            ordered = self.ordered_moves(pos, 0, None)
        except _Timeout:
            fallback = self._fallback(pos)
            return fallback, {'source': 'fallback', 'depth': 0, 'score': None,
                              'nodes': self.nodes, 'timed_out': True}
        if not ordered:
            return None, {'source': 'search', 'depth': 0, 'score': None,
                          'nodes': self.nodes, 'timed_out': False}

        best, best_value, depth_done = ordered[0], None, 0
        timed_out = False
        if len(ordered) > 1:
            for depth in range(1, self.p['max_depth'] + 1):
                if depth_done:
                    ordered = [best] + [m for m in ordered if m != best]
                alpha = float('-inf')
                it_best, it_value = None, float('-inf')
                try:
                    for m in ordered:
                        if time.perf_counter() >= self.deadline:
                            raise _Timeout()
                        v = self.child_value(pos, m, depth, alpha, float('inf'), 0)
                        if v > it_value:
                            it_best, it_value = m, v
                        if v > alpha:
                            alpha = v
                except _Timeout:
                    timed_out = True
                    break
                best, best_value, depth_done = it_best, it_value, depth
        timed_out = timed_out or time.perf_counter() >= self.deadline
        info = {'source': 'search', 'depth': depth_done, 'score': best_value,
                'nodes': self.nodes, 'timed_out': timed_out,
                'multiplier': pos.multiplier()}
        if best_value is not None:
            v = best_value / self.est_multiplier(pos) ** self.lam
            if v < 0:
                v /= self.kappa
            info['win_prob'] = max(0.0, min(1.0, (v + 1) / 2))
        return best, info

    def qsearch(self, pos, alpha, beta, ply, qdepth, hidden_left=None):
        self._tick()
        if pos.no_capture_plies >= pos.no_capture_limit:
            return 0.0
        if pos.winner >= 0:
            return self.terminal_value(pos, ply)
        stand = self.leaf_value(pos)
        if qdepth <= 0 or stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

        b = pos.b
        caps = pos.captures()
        caps.sort(key=lambda ac: HPC[b[ac[1]]] * 20 - T_RANK[TYPE[b[ac[0]]]], reverse=True)
        for a, c in caps:
            u = pos.do_move(a, c)
            try:
                v = -self.qsearch(pos, -beta, -alpha, ply + 1, qdepth - 1, hidden_left)
            finally:
                pos.undo(u)
            if v >= beta:
                return v
            if v > alpha:
                alpha = v

        if hidden_left is None:
            hidden_left = int(self.p.get('q_hidden_depth', 1))
        if hidden_left > 0:
            _, hidden_caps, _, _ = pos.gen()
            # A cannon's landing square changes follow-up tactics. Rank targets
            # using the most likely pool result, then search exact outcomes for
            # only the best few.
            total = sum(pos.pool)
            if hidden_caps and total:
                likely_code = max((code for code in range(1, 15) if pos.pool[code]),
                                  key=lambda code: pos.pool[code])
                candidates = []
                for a, c in hidden_caps:
                    u = pos.do_cannon_hidden(a, c, likely_code)
                    try:
                        score = (-self.terminal_value(pos, ply + 1) if pos.winner >= 0
                                 else -self.leaf_value(pos))
                    finally:
                        pos.undo(u)
                    candidates.append((score, (a, c)))
                candidates.sort(key=lambda item: item[0], reverse=True)
                limit = max(0, int(self.p.get('q_hidden_caps', 1)))
                for _, (a, c) in candidates[:limit]:
                    value = 0.0
                    for code, probability in self._chance_weights(pos).items():
                        u = pos.do_cannon_hidden(a, c, code)
                        try:
                            value += probability * -self.qsearch(
                                pos, -float('inf'), float('inf'), ply + 1,
                                qdepth - 1, hidden_left - 1)
                        finally:
                            pos.undo(u)
                    if value >= beta:
                        return value
                    if value > alpha:
                        alpha = value
        return alpha


class StrategicV17AI:
    def __init__(self, side, params=None):
        self.side = side
        self.time_budget_ms = 2500
        self.params = merge_params(params)
        self.params.update({k: v for k, v in DEFAULT_PARAMS.items()
                            if k not in self.params})
        self.last_decision_info = None

    def get_move(self, game):
        pos = Position.from_game(game)
        move, info = Searcher(self.params).choose(pos, self.time_budget_ms)
        self.last_decision_info = info
        return move_to_action(move) if move else None
