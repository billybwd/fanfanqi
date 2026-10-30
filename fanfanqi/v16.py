"""StrategicV16AI：精简的新基线（不继承 V1~V15 的规则链）。

设计要点：
1. 翻子、炮打暗子是“概率节点”：按剩余棋子池里各类棋子的数量加权平均，
   不再假设能翻出对自己最有利的棋子（V14/V15 的乐观偏差）。
2. 评估函数只有一组有名字的权重（见 DEFAULT_PARAMS['weights']），便于手调和自动调参，
   这些评估项之后也可以直接作为机器学习特征。
3. 战术交给搜索：Alpha-Beta + 只看吃子的静态搜索，不再有绕过搜索的规则覆盖。
4. 目标函数连续可调：局面价值 = (2p − 1) × 倍率^λ，结果为负时再乘 κ。
   - λ = 1（默认）：最大化“胜负 × 倍率”的期望。领先时倾向把倍率做大，落后时倾向压小；
   - λ = 0：只追求胜率（同样赢时仍略偏好大倍率）；介于两者之间即“优先赢，兼顾倍率”；
   - κ > 1：厌恶损失，更看重少输而不是多赢。
5. 整数棋盘 + 原地走子/撤销，比基于 dict 的 Game 快得多。
"""

import math
import time

from .game import COLS, COMBO_THRESHOLD, FLIP_SCORE, PIECE_CONFIG, ROWS

DEFAULT_PARAMS = {
    # λ：局面价值 = (2p − 1) × 倍率^λ。1 = 最大化 胜负×倍率 的期望；0 = 只看胜率；中间值兼顾
    'multiplier_exponent': 1.0,
    # κ：价值为负（预计输）时乘的系数。1 = 风险中性；>1 = 更看重少输
    'loss_aversion': 1.0,
    # 评估分（HP 单位）换算胜率的温度：胜率 = sigmoid(评估分 / 温度)
    'win_temperature': 12.0,
    'max_depth': 20,
    # 根节点 / 内部节点最多考虑几个翻子位置；内部节点最多几个非吃子着法
    'root_flips': 8,
    'node_flips': 3,
    'node_quiet_moves': 10,
    'qsearch_depth': 4,
    # 前几层的翻子按全部可能结果展开求期望；更深的翻子用近似（不揭示，只计翻棋分期望），
    # 否则每次翻子都要分出最多 14 个分支，搜索无法加深
    'full_chance_plies': 1,
    # 评估权重，单位约等于 HP
    'weights': {
        'hp': 1.0,                 # 双方剩余 HP 之差
        'piece_value': {           # 明子的额外价值（战斗力，HP 已单独计）
            '将': 10, '仕': 6, '象': 3, '车': 2.5, '马': 2, '炮': 3, '兵': 1.5,
        },
        'mobility': 0.3,           # 每个可走的相邻空格
        'hanging': 0.6,            # 轮到自己走时被威胁的棋子（最大的那个除外）×(HP+子力)
        'hanging_one': 0.15,       # 被威胁的最大那个棋子（通常能躲开一个）×(HP+子力)
        'pool': 0.3,               # 己方还没翻出的棋子 × 子力
        'king_soldier_adj': 8,     # 敌兵贴着将
        'king_soldier_near': 3,    # 敌兵距将 2 步
        'king_cannon': 5,          # 每个瞄着将的敌炮
        'king_trapped': 2,         # 将可逃的空格少于 2 个时，每少一个
        'tempo': 0.5,              # 轮到走的一方
    },
}

TYPES = ['将', '仕', '象', '车', '马', '炮', '兵']
KING, CANNON, SOLDIER = 0, 5, 6
EMPTY, HIDDEN = 0, 15
NSQ = ROWS * COLS
RED, BLACK = 0, 1
SIDE_KEYS = ('red', 'black')
MOVE, CAPTURE, CANNON_HIDDEN, FLIP = 1, 2, 3, 4
CODES = range(1, 15)
INF = float('inf')

T_RANK = [PIECE_CONFIG[t]['rank'] for t in TYPES]
SIDE = [-1] * 16
TYPE = [-1] * 16
HPC = [0] * 16
FLIPC = [0] * 16
for _s in (RED, BLACK):
    for _t in range(7):
        _c = _s * 7 + _t + 1
        SIDE[_c], TYPE[_c] = _s, _t
        HPC[_c] = PIECE_CONFIG[TYPES[_t]]['hp']
        FLIPC[_c] = FLIP_SCORE[TYPES[_t]]

# EAT[a][d]：非炮棋子 a 能否吃明子 d（规则同 Game.can_eat）
EAT = [[False] * 16 for _ in range(16)]
for _a in CODES:
    for _d in CODES:
        if SIDE[_a] == SIDE[_d]:
            continue
        _ta, _td = TYPE[_a], TYPE[_d]
        if _ta == SOLDIER and _td == KING:
            EAT[_a][_d] = True
        elif _ta == KING and _td == SOLDIER:
            EAT[_a][_d] = False
        else:
            EAT[_a][_d] = T_RANK[_ta] >= T_RANK[_td]

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGH = []
RAYS = []
for _sq in range(NSQ):
    _r, _c = divmod(_sq, COLS)
    NEIGH.append([(_r + dr) * COLS + _c + dc for dr, dc in _DIRS
                  if 0 <= _r + dr < ROWS and 0 <= _c + dc < COLS])
    _rays = []
    for dr, dc in _DIRS:
        ray = []
        r, c = _r + dr, _c + dc
        while 0 <= r < ROWS and 0 <= c < COLS:
            ray.append(r * COLS + c)
            r += dr
            c += dc
        _rays.append(ray)
    RAYS.append(_rays)
DIST = [[abs(a // COLS - b // COLS) + abs(a % COLS - b % COLS) for b in range(NSQ)] for a in range(NSQ)]


def piece_code(side_key, type_name):
    return SIDE_KEYS.index(side_key) * 7 + TYPES.index(type_name) + 1


class _Timeout(Exception):
    pass


class Position:
    """搜索用的紧凑局面：32 格整数棋盘 + 剩余池计数 + 血量 / 连击 / 翻棋分。"""

    __slots__ = ('b', 'pool', 'hp', 'turn', 'cur', 'mx', 'flip', 'winner',
                 'combo_score', 'no_capture_plies', 'chase_count', 'chase_target',
                 'long_chase_limit', 'no_capture_limit')

    @classmethod
    def from_game(cls, game):
        pos = cls()
        pos.b = [EMPTY] * NSQ
        for r in range(ROWS):
            for c in range(COLS):
                cell = game.board[r][c]
                if cell is None:
                    continue
                pos.b[r * COLS + c] = piece_code(cell['side'], cell['type']) if cell['face_up'] else HIDDEN
        pos.pool = [0] * 15
        for p in game.remaining_pool:
            pos.pool[piece_code(p['side'], p['type'])] += 1
        pos.hp = [game.hp['red'], game.hp['black']]
        pos.turn = SIDE_KEYS.index(game.current_turn)
        pos.cur = [game.combo['red'], game.combo['black']]
        pos.mx = [game.max_combo['red'], game.max_combo['black']]
        pos.combo_score = [game.combo_score['red'], game.combo_score['black']]
        pos.flip = game.flip_score
        pos.winner = -1 if game.winner is None else SIDE_KEYS.index(game.winner)
        pos.no_capture_plies = game.no_capture_plies
        pos.long_chase_limit = game.long_chase_limit
        pos.no_capture_limit = game.no_capture_limit
        pos.chase_count = [game.chase_count['red'], game.chase_count['black']]
        pos.chase_target = [
            -1 if game.chase_target['red'] is None else game.chase_target['red'][0] * COLS + game.chase_target['red'][1],
            -1 if game.chase_target['black'] is None else game.chase_target['black'][0] * COLS + game.chase_target['black'][1],
        ]
        return pos

    def key(self):
        return (tuple(self.b), tuple(self.pool), self.hp[0], self.hp[1], self.turn,
                self.cur[0], self.cur[1], self.mx[0], self.mx[1], self.flip,
                tuple(self.combo_score), self.no_capture_plies, tuple(self.chase_count), tuple(self.chase_target),
                self.long_chase_limit, self.no_capture_limit)

    # ---------------------------------------------------------- 倍率
    def combo_factor(self):
        points = sum(self.combo_score)
        points += sum(self.cur[s] for s in (RED, BLACK) if self.cur[s] >= COMBO_THRESHOLD)
        return points if points > 0 else 1

    def hp_diff(self):
        return abs(self.hp[0] - self.hp[1])

    def multiplier(self):
        return self.combo_factor() * (self.flip + self.hp_diff())

    # ------------------------------------------------------ 走子 / 撤销
    def _save(self, a, c, pool_code):
        return (a, self.b[a], c, self.b[c], self.hp[0], self.hp[1], self.turn,
                self.cur[0], self.cur[1], self.mx[0], self.mx[1], self.flip, self.winner,
                tuple(self.combo_score), self.no_capture_plies, tuple(self.chase_count), tuple(self.chase_target), pool_code)

    def undo(self, u):
        a, pa, c, pc, h0, h1, t, c0, c1, m0, m1, f, w, combo_score, quiet, chase_count, chase_target, pool_code = u
        self.b[a] = pa
        self.b[c] = pc
        self.hp[0], self.hp[1] = h0, h1
        self.turn = t
        self.cur[0], self.cur[1] = c0, c1
        self.mx[0], self.mx[1] = m0, m1
        self.flip = f
        self.winner = w
        self.combo_score = list(combo_score)
        self.no_capture_plies = quiet
        self.chase_count = list(chase_count)
        self.chase_target = list(chase_target)
        if pool_code:
            self.pool[pool_code] += 1

    def _hit(self, me):
        self.cur[me] += 1
        if self.cur[me] > self.mx[me]:
            self.mx[me] = self.cur[me]

    def _break_combo(self, me):
        if self.cur[me] >= COMBO_THRESHOLD:
            self.combo_score[me] += self.cur[me]
        self.cur[me] = 0

    def _check_over(self):
        # 与 Game.move_piece 一致：先判红方
        if self.hp[0] <= 0:
            self.winner = BLACK
        elif self.hp[1] <= 0:
            self.winner = RED

    def do_move(self, a, c):
        """走子或吃明子。"""
        u = self._save(a, c, 0)
        b = self.b
        me = self.turn
        q = b[c]
        b[c] = b[a]
        b[a] = EMPTY
        if q:
            self.hp[SIDE[q]] -= HPC[q]
            if SIDE[q] != me:
                self._hit(me)
            else:
                self._break_combo(me)
            self._check_over()
        else:
            self._break_combo(me)
        self.no_capture_plies = 0 if q else self.no_capture_plies + 1
        opponent = me ^ 1
        if q:
            for side in (0, 1):
                if self.chase_target[side] == c:
                    self.chase_target[side] = -1
                    self.chase_count[side] = 0
        if self.chase_target[opponent] == a:
            self.chase_target[opponent] = c
        if q:
            self.chase_target[me] = -1
            self.chase_count[me] = 0
        else:
            targets = self.capture_targets_from(c, me)
            old_target = self.chase_target[me]
            if old_target in targets:
                self.chase_count[me] += 1
            elif targets:
                self.chase_target[me] = targets[0]
                self.chase_count[me] = 1
            else:
                self.chase_target[me] = -1
                self.chase_count[me] = 0
        self.turn = me ^ 1
        return u

    def capture_targets_from(self, sq, side):
        p = self.b[sq]
        if not p or SIDE[p] != side:
            return []
        if TYPE[p] == CANNON:
            targets = []
            for ray in RAYS[sq]:
                mounted = False
                for s2 in ray:
                    q = self.b[s2]
                    if not q:
                        continue
                    if not mounted:
                        mounted = True
                    else:
                        if q != HIDDEN and SIDE[q] != side:
                            targets.append(s2)
                        break
            return targets
        return [n for n in NEIGH[sq] if self.b[n] and self.b[n] != HIDDEN and
                SIDE[self.b[n]] != side and EAT[p][self.b[n]]]

    def do_flip(self, sq, code):
        u = self._save(sq, sq, code)
        self.b[sq] = code
        self.pool[code] -= 1
        self.chase_count[self.turn] = 0
        self.chase_target[self.turn] = -1
        self._break_combo(self.turn)
        self.turn ^= 1
        return u

    def do_flip_approx(self, sq):
        """深层翻子的近似：格子仍是暗子，只换手并清连击。"""
        u = self._save(sq, sq, 0)
        self.chase_count[self.turn] = 0
        self.chase_target[self.turn] = -1
        self._break_combo(self.turn)
        self.turn ^= 1
        return u

    def do_cannon_hidden(self, a, c, code):
        """炮打暗子：暗子实为 code，直接被吃掉（可能是己方棋子）。"""
        u = self._save(a, c, code)
        me = self.turn
        self.pool[code] -= 1
        self.hp[SIDE[code]] -= HPC[code]
        self.b[c] = self.b[a]
        self.b[a] = EMPTY
        if SIDE[code] != me:
            self._hit(me)
        else:
            self._break_combo(me)
        self._check_over()
        self.no_capture_plies = 0
        opponent = me ^ 1
        if self.chase_target[opponent] == a:
            self.chase_target[opponent] = c
        self.chase_target[me] = -1
        self.chase_count[me] = 0
        self.turn = me ^ 1
        return u

    # ------------------------------------------------------------ 着法
    def gen(self):
        """返回 (吃明子, 炮打暗子, 走子, 翻子格) 四个列表。"""
        b = self.b
        me = self.turn
        caps, hidden_caps, moves, flips = [], [], [], []
        for sq in range(NSQ):
            p = b[sq]
            if p == EMPTY:
                continue
            if p == HIDDEN:
                flips.append(sq)
                continue
            if SIDE[p] != me:
                continue
            if TYPE[p] == CANNON:
                for ray in RAYS[sq]:
                    mounted = False
                    for s2 in ray:
                        q = b[s2]
                        if q:
                            if not mounted:
                                mounted = True
                            else:
                                if q == HIDDEN:
                                    hidden_caps.append((sq, s2))
                                elif SIDE[q] != me:
                                    caps.append((sq, s2))
                                break
            else:
                eat = EAT[p]
                for s2 in NEIGH[sq]:
                    q = b[s2]
                    if q == EMPTY:
                        moves.append((sq, s2))
                    elif q != HIDDEN and eat[q]:
                        caps.append((sq, s2))
        legal_moves = []
        for a, c in moves:
            u = self.do_move(a, c)
            violates = self.chase_count[me] >= self.long_chase_limit - 1
            self.undo(u)
            if not violates:
                legal_moves.append((a, c))
        return caps, hidden_caps, legal_moves, flips

    def captures(self):
        b = self.b
        me = self.turn
        caps = []
        for sq in range(NSQ):
            p = b[sq]
            if p == EMPTY or p == HIDDEN or SIDE[p] != me:
                continue
            if TYPE[p] == CANNON:
                for ray in RAYS[sq]:
                    mounted = False
                    for s2 in ray:
                        q = b[s2]
                        if q:
                            if not mounted:
                                mounted = True
                            else:
                                if q != HIDDEN and SIDE[q] != me:
                                    caps.append((sq, s2))
                                break
            else:
                eat = EAT[p]
                for s2 in NEIGH[sq]:
                    q = b[s2]
                    if q and q != HIDDEN and eat[q]:
                        caps.append((sq, s2))
        return caps

    def cannon_hits(self, csq, target):
        b = self.b
        for ray in RAYS[csq]:
            mounted = False
            for s2 in ray:
                if b[s2]:
                    if not mounted:
                        mounted = True
                    else:
                        if s2 == target:
                            return True
                        break
        return False

    def attacked(self, sq, p):
        """位于 sq 的明子 p 能否被对方明子吃到。"""
        b = self.b
        enemy = SIDE[p] ^ 1
        for n in NEIGH[sq]:
            q = b[n]
            if q and q != HIDDEN and SIDE[q] == enemy and TYPE[q] != CANNON and EAT[q][p]:
                return True
        for ray in RAYS[sq]:
            mounted = False
            for s2 in ray:
                q = b[s2]
                if q:
                    if not mounted:
                        mounted = True
                    else:
                        if q != HIDDEN and SIDE[q] == enemy and TYPE[q] == CANNON:
                            return True
                        break
        return False


class Searcher:
    def __init__(self, params):
        self.p = params
        w = params['weights']
        self.w = w
        self.pv = [w['piece_value'][t] for t in TYPES]
        self.lam = params['multiplier_exponent']
        self.kappa = params['loss_aversion']
        self.tt = {}
        self.nodes = 0
        self.deadline = INF
        # on_progress(move, info)：每完成一层迭代调用一次；should_stop()：返回 True 时尽快结束搜索
        self.on_progress = None
        self.should_stop = None

    # -------------------------------------------------------- 评估
    def static_eval(self, pos):
        """从走子方角度的评估分（约等于 HP 单位）。"""
        b = pos.b
        me = pos.turn
        op = me ^ 1
        w = self.w
        pv = self.pv
        mat = [0.0, 0.0]
        mob = [0, 0]
        kings = [-1, -1]
        soldiers = ([], [])
        cannons = ([], [])
        threatened = []
        for sq in range(NSQ):
            p = b[sq]
            if p == EMPTY or p == HIDDEN:
                continue
            s = SIDE[p]
            t = TYPE[p]
            mat[s] += pv[t]
            if t == KING:
                kings[s] = sq
            elif t == SOLDIER:
                soldiers[s].append(sq)
            elif t == CANNON:
                cannons[s].append(sq)
            if t != CANNON:
                for n in NEIGH[sq]:
                    if b[n] == EMPTY:
                        mob[s] += 1
            if s == me and pos.attacked(sq, p):
                threatened.append(HPC[p] + pv[t])

        score = w['hp'] * (pos.hp[me] - pos.hp[op]) + (mat[me] - mat[op]) + w['mobility'] * (mob[me] - mob[op])
        if threatened:
            biggest = max(threatened)
            score -= w['hanging'] * (sum(threatened) - biggest) + w['hanging_one'] * biggest

        pool_value = [0.0, 0.0]
        for c in CODES:
            if pos.pool[c]:
                pool_value[SIDE[c]] += pos.pool[c] * pv[TYPE[c]]
        score += w['pool'] * (pool_value[me] - pool_value[op])

        for s in (me, op):
            k = kings[s]
            if k < 0:
                continue
            e = s ^ 1
            danger = 0.0
            for sq2 in soldiers[e]:
                d = DIST[k][sq2]
                if d == 1:
                    danger += w['king_soldier_adj']
                elif d == 2:
                    danger += w['king_soldier_near']
            for sq2 in cannons[e]:
                if pos.cannon_hits(sq2, k):
                    danger += w['king_cannon']
            free = sum(1 for n in NEIGH[k] if b[n] == EMPTY)
            if free < 2:
                danger += w['king_trapped'] * (2 - free)
            score += -danger if s == me else danger

        return score + w['tempo']

    def eval_terms(self, pos):
        """static_eval 拆成 {权重名: 特征值}，满足 Σ 权重 × 特征值 == static_eval（用于 Texel 拟合）。

        子力项的键为 'piece_value.<类型>'。'hanging' / 'hanging_one' / 'pool' 的特征值里含当前子力值，
        对子力而言不是线性的，拟合时按固定特征处理。必须与 static_eval 保持一致（tests/selfplay 有校验）。
        """
        b = pos.b
        me = pos.turn
        op = me ^ 1
        pv = self.pv
        terms = {f'piece_value.{t}': 0.0 for t in TYPES}
        for key in ('mobility', 'hanging', 'hanging_one', 'pool', 'king_soldier_adj', 'king_soldier_near',
                    'king_cannon', 'king_trapped'):
            terms[key] = 0.0
        terms['hp'] = pos.hp[me] - pos.hp[op]
        terms['tempo'] = 1.0
        kings = [-1, -1]
        soldiers = ([], [])
        cannons = ([], [])
        threatened = []
        for sq in range(NSQ):
            p = b[sq]
            if p == EMPTY or p == HIDDEN:
                continue
            s = SIDE[p]
            t = TYPE[p]
            sign = 1 if s == me else -1
            terms[f'piece_value.{TYPES[t]}'] += sign
            if t == KING:
                kings[s] = sq
            elif t == SOLDIER:
                soldiers[s].append(sq)
            elif t == CANNON:
                cannons[s].append(sq)
            if t != CANNON:
                for n in NEIGH[sq]:
                    if b[n] == EMPTY:
                        terms['mobility'] += sign
            if s == me and pos.attacked(sq, p):
                threatened.append(HPC[p] + pv[t])
        if threatened:
            biggest = max(threatened)
            terms['hanging'] = -(sum(threatened) - biggest)
            terms['hanging_one'] = -biggest
        for c in CODES:
            if pos.pool[c]:
                terms['pool'] += (1 if SIDE[c] == me else -1) * pos.pool[c] * pv[TYPE[c]]
        for s in (me, op):
            k = kings[s]
            if k < 0:
                continue
            sign = -1 if s == me else 1
            for sq2 in soldiers[s ^ 1]:
                d = DIST[k][sq2]
                if d == 1:
                    terms['king_soldier_adj'] += sign
                elif d == 2:
                    terms['king_soldier_near'] += sign
            for sq2 in cannons[s ^ 1]:
                if pos.cannon_hits(sq2, k):
                    terms['king_cannon'] += sign
            free = sum(1 for n in NEIGH[k] if b[n] == EMPTY)
            if free < 2:
                terms['king_trapped'] += sign * (2 - free)
        return terms

    def term_weights(self):
        """与 eval_terms 的键对应的当前权重。"""
        weights = {k: v for k, v in self.w.items() if k != 'piece_value'}
        weights.update({f'piece_value.{t}': self.pv[i] for i, t in enumerate(TYPES)})
        return weights

    def est_multiplier(self, pos):
        m = pos.combo_factor() * (pos.flip + pos.hp_diff())
        return max(1.0, m)

    def _objective(self, v, multiplier):
        """胜负期望 v ∈ [−1, 1] 换算成目标值：v × 倍率^λ，为负时再乘 κ。"""
        v *= multiplier ** self.lam
        return v * self.kappa if v < 0 else v

    def leaf_value(self, pos):
        x = self.static_eval(pos) / self.p['win_temperature']
        x = max(-30.0, min(30.0, x))
        return self._objective(2.0 / (1.0 + math.exp(-x)) - 1.0, self.est_multiplier(pos))

    def terminal_value(self, pos, ply):
        m = max(1, pos.multiplier())
        # 同等结果下越快越好；λ 很小时用 log(倍率) 区分“赢得多”和“赢得少”（不超过叶子估值的上界）
        v = (1.0 + 0.01 * math.log(m)) * (1.0 - 0.001 * ply)
        return self._objective(v if pos.winner == pos.turn else -v, m)

    # -------------------------------------------------------- 排序
    def _quiet_score(self, pos, a, c):
        b = pos.b
        p = b[a]
        score = 0.0
        if pos.attacked(a, p):
            score += (HPC[p] + self.pv[TYPE[p]]) * 10
        u = pos.do_move(a, c)
        try:
            if pos.attacked(c, p):
                score -= (HPC[p] + self.pv[TYPE[p]]) * 10
            eat = EAT[p]
            for n in NEIGH[c]:
                q = b[n]
                if q and q != HIDDEN and eat[q]:
                    score += HPC[q] * 3
        finally:
            pos.undo(u)
        return score

    def _flip_score_full(self, pos, sq, outcomes, total):
        """翻开 sq 的粗略期望收益（只用于排序 / 挑选候选翻子格）。"""
        b = pos.b
        me = pos.turn
        pv = self.pv
        adj_enemy = []
        adj_own = []
        for n in NEIGH[sq]:
            q = b[n]
            if q and q != HIDDEN:
                (adj_own if SIDE[q] == me else adj_enemy).append(q)
        score = 0.0
        for code, cnt in outcomes:
            t = TYPE[code]
            if SIDE[code] == me:
                gain = pv[t]
                if any(TYPE[e] != CANNON and EAT[e][code] for e in adj_enemy):
                    gain -= HPC[code] + pv[t]
            else:
                gain = -pv[t]
                if TYPE[code] != CANNON:
                    for o in adj_own:
                        if EAT[code][o]:
                            gain -= HPC[o] + pv[TYPE[o]]
                            break
            score += cnt / total * gain
        return score

    def _flip_score_fast(self, pos, sq):
        b = pos.b
        me = pos.turn
        score = 0.0
        for n in NEIGH[sq]:
            q = b[n]
            if q and q != HIDDEN:
                score += 1.0 if SIDE[q] == me else -(1.0 + T_RANK[TYPE[q]])
        return score

    def ordered_moves(self, pos, ply, tt_best):
        caps, hidden_caps, moves, flips = pos.gen()
        b = pos.b
        root = ply == 0
        scored = []
        for a, c in caps:
            v = b[c]
            scored.append((10000 + HPC[v] * 20 + self.pv[TYPE[v]] * 5 - T_RANK[TYPE[b[a]]], (CAPTURE, a, c)))
        if hidden_caps:
            total = sum(pos.pool)
            enemy = pos.turn ^ 1
            expected = sum(pos.pool[c] * (HPC[c] if SIDE[c] == enemy else -HPC[c]) for c in CODES) / max(1, total)
            for a, c in hidden_caps:
                scored.append((5000 + expected * 20, (CANNON_HIDDEN, a, c)))
        quiet = sorted(((self._quiet_score(pos, a, c), (MOVE, a, c)) for a, c in moves),
                       key=lambda e: e[0], reverse=True)
        if not root:
            quiet = quiet[:self.p['node_quiet_moves']]
        scored.extend(quiet)
        if flips:
            if root:
                total = sum(pos.pool)
                outcomes = [(c, pos.pool[c]) for c in CODES if pos.pool[c]]
                fs = [(self._flip_score_full(pos, sq, outcomes, total), sq) for sq in flips]
                limit = self.p['root_flips']
            else:
                fs = [(self._flip_score_fast(pos, sq), sq) for sq in flips]
                limit = self.p['node_flips']
            fs.sort(key=lambda e: e[0], reverse=True)
            for s, sq in fs[:limit]:
                scored.append((-1000 + s, (FLIP, sq, sq)))
        if tt_best is not None:
            scored = [(1e9 if m == tt_best else s, m) for s, m in scored]
        scored.sort(key=lambda e: e[0], reverse=True)
        return [m for _, m in scored]

    # -------------------------------------------------------- 搜索
    def _tick(self):
        self.nodes += 1
        if (self.nodes & 255) == 0 and self._out_of_time():
            raise _Timeout()

    def _out_of_time(self):
        return time.perf_counter() > self.deadline or (self.should_stop is not None and self.should_stop())

    def win_prob(self, pos, value):
        """把搜索值换算回预估胜率。"""
        v = value / self.est_multiplier(pos) ** self.lam
        if v < 0:
            v /= self.kappa
        return max(0.0, min(1.0, (v + 1) / 2))

    def _report(self, pos, move, value, depth):
        if self.on_progress is not None:
            self.on_progress(move, {'source': 'search', 'depth': depth, 'score': value, 'nodes': self.nodes,
                                    'multiplier': pos.multiplier(), 'win_prob': self.win_prob(pos, value),
                                    'partial': True})

    def child_value(self, pos, move, depth, alpha, beta, ply):
        kind, a, c = move
        if kind == MOVE or kind == CAPTURE:
            u = pos.do_move(a, c)
            try:
                return -self.negamax(pos, depth - 1, -beta, -alpha, ply + 1)
            finally:
                pos.undo(u)
        total = sum(pos.pool)
        if kind == FLIP and ply >= self.p['full_chance_plies']:
            u = pos.do_flip_approx(a)
            try:
                return -self.negamax(pos, depth - 1, -beta, -alpha, ply + 1)
            finally:
                pos.undo(u)
        # 概率节点：按剩余池里各类棋子的数量加权平均
        value = 0.0
        for code in CODES:
            cnt = pos.pool[code]
            if not cnt:
                continue
            u = pos.do_flip(a, code) if kind == FLIP else pos.do_cannon_hidden(a, c, code)
            try:
                value += cnt / total * -self.negamax(pos, depth - 1, -INF, INF, ply + 1)
            finally:
                pos.undo(u)
        return value

    def negamax(self, pos, depth, alpha, beta, ply):
        self._tick()
        if pos.no_capture_plies >= pos.no_capture_limit:
            return 0.0
        if pos.winner >= 0:
            return self.terminal_value(pos, ply)
        if depth <= 0:
            return self.qsearch(pos, alpha, beta, ply, self.p['qsearch_depth'])

        key = pos.key()
        entry = self.tt.get(key)
        tt_best = None
        if entry is not None:
            e_depth, e_value, e_flag, tt_best = entry
            if e_depth >= depth:
                if e_flag == 0:
                    return e_value
                if e_flag == 1 and e_value >= beta:
                    return e_value
                if e_flag == 2 and e_value <= alpha:
                    return e_value

        moves = self.ordered_moves(pos, ply, tt_best)
        if not moves:
            return self.leaf_value(pos)
        alpha0 = alpha
        best = -INF
        best_move = None
        for m in moves:
            v = self.child_value(pos, m, depth, alpha, beta, ply)
            if v > best:
                best, best_move = v, m
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        flag = 2 if best <= alpha0 else (1 if best >= beta else 0)
        self.tt[key] = (depth, best, flag, best_move)
        return best

    def qsearch(self, pos, alpha, beta, ply, qdepth):
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
                v = -self.qsearch(pos, -beta, -alpha, ply + 1, qdepth - 1)
            finally:
                pos.undo(u)
            if v >= beta:
                return v
            if v > alpha:
                alpha = v
        return alpha

    def choose(self, pos, budget_ms):
        self.deadline = time.perf_counter() + budget_ms / 1000
        self.tt.clear()
        self.nodes = 0
        ordered = self.ordered_moves(pos, 0, None)
        if not ordered:
            return None, {'source': 'search', 'depth': 0, 'score': None}
        best, best_value, depth_done = ordered[0], None, 0
        if len(ordered) > 1:
            for depth in range(1, self.p['max_depth'] + 1):
                if depth_done:
                    ordered = [best] + [m for m in ordered if m != best]
                alpha = -INF
                it_best, it_value = None, -INF
                try:
                    for m in ordered:
                        v = self.child_value(pos, m, depth, alpha, INF, 0)
                        if v > it_value:
                            it_best, it_value = m, v
                        if v > alpha:
                            alpha = v
                except _Timeout:
                    break
                best, best_value, depth_done = it_best, it_value, depth
                if self._out_of_time():
                    break
                self._report(pos, best, best_value, depth)
        info = {'source': 'search', 'depth': depth_done, 'score': best_value, 'nodes': self.nodes,
                'multiplier': pos.multiplier()}
        if best_value is not None:
            info['win_prob'] = self.win_prob(pos, best_value)
        return best, info


def move_to_action(move):
    kind, a, c = move
    if kind == FLIP:
        return {'type': 'flip', 'row': a // COLS, 'col': a % COLS}
    return {
        'type': 'move' if kind == MOVE else 'capture',
        'from_row': a // COLS, 'from_col': a % COLS,
        'to_row': c // COLS, 'to_col': c % COLS,
    }


def merge_params(overrides, defaults=None):
    """把 overrides（可以只给部分参数，weights / piece_value 按键合并）补全到 defaults（默认 V16 的）上。"""
    defaults = defaults or DEFAULT_PARAMS
    params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    params['weights'] = dict(defaults['weights'])
    params['weights']['piece_value'] = dict(defaults['weights']['piece_value'])
    for k, v in (overrides or {}).items():
        if k == 'weights' and isinstance(v, dict):
            for wk, wv in v.items():
                if wk == 'piece_value' and isinstance(wv, dict):
                    params['weights']['piece_value'].update(wv)
                else:
                    params['weights'][wk] = wv
        elif k == 'objective':  # 旧版配置：'reward' / 'win'
            if 'multiplier_exponent' not in overrides:
                params['multiplier_exponent'] = 0.0 if v == 'win' else 1.0
        else:
            params[k] = v
    return params


class StrategicV16AI:
    def __init__(self, side, params=None):
        self.side = side
        self.time_budget_ms = 900
        self.params = merge_params(params)
        self.last_decision_info = None
        # on_progress(action, info)：每完成一层迭代回调当前最佳动作；should_stop()：返回 True 时提前结束
        self.on_progress = None
        self.should_stop = None

    def get_move(self, game):
        return run_search(self, Searcher(self.params), game)


def run_search(player, searcher, game):
    """用 searcher 为 player 搜索 game，接上进度回调 / 取消，返回动作。"""
    pos = Position.from_game(game)
    if player.on_progress is not None:
        searcher.on_progress = lambda move, info: player.on_progress(move_to_action(move), info)
    searcher.should_stop = player.should_stop
    move, info = searcher.choose(pos, player.time_budget_ms)
    player.last_decision_info = info
    return move_to_action(move) if move else None
