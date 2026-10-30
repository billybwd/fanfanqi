"""StrategicV17AI: V16 with broader tactical coverage and stricter time control."""

import copy
import math
import struct
import time

from .v16 import (
    CANNON,
    CANNON_HIDDEN,
    EMPTY,
    CAPTURE,
    DEFAULT_PARAMS as V16_DEFAULT_PARAMS,
    FLIP,
    HPC,
    KING,
    MOVE,
    NSQ,
    Searcher as V16Searcher,
    SOLDIER,
    T_RANK,
    TYPE,
    _Timeout,
    merge_params,
    run_search,
)


DEFAULT_PARAMS = copy.deepcopy(V16_DEFAULT_PARAMS)
DEFAULT_PARAMS.update({
    # Maximum hidden-cannon captures considered at each quiescence node.
    'q_hidden_caps': 1,
    # Number of consecutive hidden-cannon captures explored in quiescence.
    # 0 keeps quiescence to visible captures only (hidden captures are still
    # searched exactly by the main search); expanding them here cost too much depth.
    'q_hidden_depth': 0,
    # Delta pruning in quiescence: skip a capture when even the captured HP plus
    # piece value plus this margin (HP units) cannot raise alpha. Negative disables.
    'delta_margin': 6.0,
    # Static evaluation cache entries kept per search; 0 disables.
    'eval_cache_size': 200000,
    # Cannon value depends on how crowded the board is (more screens, more
    # targets): weights.piece_value['炮'] with all 32 squares occupied, this
    # value with an empty board, linear in between. Negative keeps it fixed.
    'cannon_endgame_value': 2.0,
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
# Evaluation weights from SPSA self-play tuning (2026-10-30, 3000 game pairs; tools/tune.py): chariot and
# horse lower, king safety (king, cannons aiming at it) higher. Against the previous V16 values it scored
# -8 Elo over 300 games at 1 s/move (95% interval -45 ~ +28), i.e. no measurable loss; kept because it is
# closer to human judgement. Previous values: params/v17-before-spsa-20261030.json.
DEFAULT_PARAMS['weights'].update({'hanging': 0.62, 'pool': 0.28, 'king_cannon': 5.79})
# Fitted on self-play quiet positions (2026-10-30): evaluation 12 HP was converted to win probabilities far too
# confidently (positions it called ~1% were won 18.5% of the time, so badly losing lines were given up early,
# e.g. by walking the king into capture); the best fit is about 20. The side to move also loses noticeably more
# often than its evaluation says (fitted offset about -10 HP); half of that is applied to tempo. Both are set
# directly and still to be confirmed with tools/tune.py sprt.
DEFAULT_PARAMS['win_temperature'] = 20.0
DEFAULT_PARAMS['weights']['tempo'] = -5.0
DEFAULT_PARAMS['weights']['piece_value'] = {
    '将': 10.56, '仕': 6.05, '象': 2.94, '车': 2.09, '马': 1.55, '炮': 3.23, '兵': 1.57,
}


_MASK64 = (1 << 64) - 1


def _sample_offset(pos):
    """Systematic-sampling offset in [0, 1) for a position.

    FNV-1a over board, pool and flip score, then splitmix64. The Go engine
    computes the same value, so both engines sample identical outcomes.
    """
    h = 0xCBF29CE484222325
    for byte in (*pos.b, *pos.pool, *struct.pack('<d', float(pos.flip))):
        h = ((h ^ byte) * 0x100000001B3) & _MASK64
    z = (h + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z ^= z >> 31
    return (z >> 11) / float(1 << 53)


class Searcher(V16Searcher):
    """V16 search with better flip ordering, hidden-capture quiescence and timing."""

    def __init__(self, params):
        super().__init__(params)
        self.time_check_interval = max(1, int(params.get('time_check_interval', 32)))
        self.delta_margin = float(params.get('delta_margin', 6.0))
        self.eval_cache_size = max(0, int(params.get('eval_cache_size', 200000)))
        self.eval_cache = {}
        self.cannon_full = self.pv[CANNON]
        self.cannon_endgame = float(params.get('cannon_endgame_value', 2.0))

    def _set_cannon_value(self, pos):
        if self.cannon_endgame >= 0:
            occupied = NSQ - pos.b.count(EMPTY)
            self.pv[CANNON] = self.cannon_endgame + (self.cannon_full - self.cannon_endgame) * occupied / NSQ

    def static_eval(self, pos):
        # The evaluation only depends on board, pool, HP and side to move.
        if not self.eval_cache_size:
            self._set_cannon_value(pos)
            return super().static_eval(pos)
        key = (tuple(pos.b), tuple(pos.pool), pos.hp[0], pos.hp[1], pos.turn)
        score = self.eval_cache.get(key)
        if score is None:
            if len(self.eval_cache) >= self.eval_cache_size:
                self.eval_cache.clear()
            self._set_cannon_value(pos)
            score = self.eval_cache[key] = super().static_eval(pos)
        return score

    def eval_terms(self, pos):
        # The dynamic cannon value is linear in its two endpoints, so split the
        # cannon count into a full-board term and an endgame term.
        self._set_cannon_value(pos)
        terms = super().eval_terms(pos)
        if self.cannon_endgame >= 0:
            fullness = (NSQ - pos.b.count(EMPTY)) / NSQ
            diff = terms['piece_value.炮']
            terms['piece_value.炮'] = diff * fullness
            terms['cannon_endgame_value'] = diff * (1 - fullness)
        return terms

    def term_weights(self):
        weights = super().term_weights()
        weights['piece_value.炮'] = self.cannon_full
        if self.cannon_endgame >= 0:
            weights['cannon_endgame_value'] = self.cannon_endgame
        return weights

    def _value_from_eval(self, pos, score):
        """leaf_value with the static evaluation already computed."""
        x = max(-30.0, min(30.0, score / self.p['win_temperature']))
        return self._objective(2.0 / (1.0 + math.exp(-x)) - 1.0, self.est_multiplier(pos))

    def _tick(self):
        self.nodes += 1
        if self.nodes % self.time_check_interval == 0 and self._out_of_time():
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
        offset = _sample_offset(pos)
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
        if ply >= self.p['full_chance_plies']:
            # Approximate flips leave the board untouched, so every flip square
            # leads to the same child position; searching one is enough.
            flips = flips[:1]
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
        if self._out_of_time():
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
                        if self._out_of_time():
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
                self._report(pos, best, best_value, depth)
        timed_out = timed_out or time.perf_counter() >= self.deadline
        info = {'source': 'search', 'depth': depth_done, 'score': best_value,
                'nodes': self.nodes, 'timed_out': timed_out,
                'multiplier': pos.multiplier()}
        if best_value is not None:
            info['win_prob'] = self.win_prob(pos, best_value)
        return best, info

    def qsearch(self, pos, alpha, beta, ply, qdepth, hidden_left=None):
        self._tick()
        if pos.no_capture_plies >= pos.no_capture_limit:
            return 0.0
        if pos.winner >= 0:
            return self.terminal_value(pos, ply)
        score = self.static_eval(pos)
        stand = self._value_from_eval(pos, score)
        if qdepth <= 0 or stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

        b = pos.b
        caps = pos.captures()
        caps.sort(key=lambda ac: HPC[b[ac[1]]] * 20 - T_RANK[TYPE[b[ac[0]]]], reverse=True)
        delta = self.delta_margin >= 0
        for a, c in caps:
            gain = HPC[b[c]] + self.pv[TYPE[b[c]]] + self.delta_margin
            u = pos.do_move(a, c)
            try:
                # The objective is monotone in the evaluation for a fixed
                # multiplier, so bound the child by an optimistic evaluation
                # using the post-capture multiplier. Game-ending captures are
                # always searched.
                if (delta and pos.winner < 0
                        and self._value_from_eval(pos, score + gain) <= alpha):
                    continue
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
        self.params = merge_params(params, DEFAULT_PARAMS)  # 补全到 V17 自己的默认值（评估权重与 V16 不同）
        self.last_decision_info = None
        self.on_progress = None
        self.should_stop = None

    def searcher(self):
        return Searcher(self.params)

    def get_move(self, game):
        return run_search(self, self.searcher(), game)
