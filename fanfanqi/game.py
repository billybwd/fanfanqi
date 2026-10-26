"""翻翻棋核心游戏逻辑（移植自原项目 game.js）。

数据约定（AI 模块依赖这些约定）：
- 棋盘 ``board[row][col]`` 为 ``None``（空格）或 dict。
- 明子：``{'type', 'side', 'rank', 'hp', 'face_up': True}``。
- 暗子：``{'type': None, 'side': None, 'rank': None, 'hp': None, 'face_up': False}``。
- 动作：
    翻棋 ``{'type': 'flip', 'row', 'col', 'pool_index'?}``
    走/吃 ``{'type': 'move'|'capture', 'from_row', 'from_col', 'to_row', 'to_col', 'pool_index'?}``
"""

import random

ROWS = 8
COLS = 4

PIECE_CONFIG = {
    '将': {'rank': 6, 'hp': 30, 'count': 1},
    '仕': {'rank': 5, 'hp': 10, 'count': 2},
    '象': {'rank': 4, 'hp': 5, 'count': 2},
    '车': {'rank': 3, 'hp': 5, 'count': 2},
    '马': {'rank': 2, 'hp': 5, 'count': 2},
    '炮': {'rank': 1, 'hp': 5, 'count': 2},
    '兵': {'rank': 0, 'hp': 2, 'count': 5},
}

PIECE_NAMES = {
    'red': {'将': '帅', '仕': '仕', '象': '相', '车': '车', '马': '马', '炮': '炮', '兵': '兵'},
    'black': {'将': '将', '仕': '士', '象': '象', '车': '车', '马': '马', '炮': '炮', '兵': '卒'},
}

SIDE_NAMES = {'red': '红方', 'black': '黑方'}

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

INITIAL_HP = 60

# 倍率规则：倍率 = 连击系数 × (翻棋分 + 血量差)
# 翻棋分：每次手动翻子按翻出棋子类型累加（双方合计；炮打暗子、开局放置不算）
FLIP_SCORE = {'兵': 2, '车': 4, '马': 4, '炮': 6, '象': 8, '仕': 10, '将': 12}
# 连击：同一方连续自己的回合都吃到子（含炮打暗子）；某回合没吃子则清零。
# 双方最高连击都不超过该值时连击系数为 1；一方超过取该方最高连击；双方都超过取两者之和
COMBO_THRESHOLD = 3


def combo_factor(max_combo_red, max_combo_black):
    over = [c for c in (max_combo_red, max_combo_black) if c > COMBO_THRESHOLD]
    return sum(over) if over else 1


def compute_multiplier(hp, max_combo, flip_score):
    """返回倍率及各组成部分。血量差按 0 封底后取绝对值（被扣到负数的一方按 0 计）。"""
    factor = combo_factor(max_combo['red'], max_combo['black'])
    hp_diff = abs(max(0, hp['red']) - max(0, hp['black']))
    return {
        'combo_factor': factor,
        'flip_score': flip_score,
        'hp_diff': hp_diff,
        'multiplier': factor * (flip_score + hp_diff),
    }


def make_piece(piece_type, side, face_up=False):
    cfg = PIECE_CONFIG[piece_type]
    return {'type': piece_type, 'side': side, 'rank': cfg['rank'], 'hp': cfg['hp'], 'face_up': face_up}


def hidden_cell():
    return {'type': None, 'side': None, 'rank': None, 'hp': None, 'face_up': False}


def create_pieces_for_side(side):
    pieces = []
    for piece_type, cfg in PIECE_CONFIG.items():
        for _ in range(cfg['count']):
            pieces.append(make_piece(piece_type, side))
    return pieces


def other_side(side):
    return 'black' if side == 'red' else 'red'


def piece_name(piece):
    return PIECE_NAMES[piece['side']][piece['type']]


def _clone_cell(cell):
    return dict(cell) if cell else None


class Game:
    def __init__(self, shuffle_pool=True):
        self.board = []
        self.current_turn = 'red'
        self.hp = {'red': INITIAL_HP, 'black': INITIAL_HP}
        self.phase = 'setup'  # 'setup' | 'playing' | 'over'
        self.winner = None
        self.last_action = None
        # 开局需要放置的 4 个位置，仅固定位置，不限制红黑数量
        self.setup_positions = [
            {'row': 1, 'col': 1},
            {'row': 3, 'col': 0},
            {'row': 4, 'col': 3},
            {'row': 6, 'col': 2},
        ]
        self.setup_index = 0
        self.combo = {'red': 0, 'black': 0}       # 当前连击
        self.max_combo = {'red': 0, 'black': 0}   # 本局最高连击
        self.flip_score = 0
        self.history = []
        self.init_board(shuffle_pool)
        self.record_history()

    def init_board(self, shuffle_pool=True):
        all_pieces = create_pieces_for_side('red') + create_pieces_for_side('black')
        if shuffle_pool:
            random.shuffle(all_pieces)
        self.remaining_pool = all_pieces
        self.board = [[hidden_cell() for _ in range(COLS)] for _ in range(ROWS)]
        self._face_up_cache = {}

    def create_snapshot(self):
        la = self.last_action
        return {
            'board': [[_clone_cell(cell) for cell in row] for row in self.board],
            'current_turn': self.current_turn,
            'hp': dict(self.hp),
            'phase': self.phase,
            'winner': self.winner,
            # last_action 只由 dict/嵌套 dict 组成，且创建后不再被修改，浅拷贝即可
            'last_action': dict(la) if la else None,
            'setup_index': self.setup_index,
            'remaining_pool': [dict(p) for p in self.remaining_pool],
            'combo': dict(self.combo),
            'max_combo': dict(self.max_combo),
            'flip_score': self.flip_score,
        }

    def restore_snapshot(self, snapshot):
        self.board = [[_clone_cell(cell) for cell in row] for row in snapshot['board']]
        self.current_turn = snapshot['current_turn']
        self.hp = dict(snapshot['hp'])
        self.phase = snapshot['phase']
        self.winner = snapshot['winner']
        la = snapshot['last_action']
        self.last_action = dict(la) if la else None
        self.setup_index = snapshot['setup_index']
        self.remaining_pool = [dict(p) for p in snapshot['remaining_pool']]
        self.combo = dict(snapshot['combo'])
        self.max_combo = dict(snapshot['max_combo'])
        self.flip_score = snapshot['flip_score']
        self._face_up_cache = {}

    def get_multiplier(self):
        return compute_multiplier(self.hp, self.max_combo, self.flip_score)

    def _record_turn_combo(self, side, captured):
        if captured:
            self.combo[side] += 1
            if self.combo[side] > self.max_combo[side]:
                self.max_combo[side] = self.combo[side]
        else:
            self.combo[side] = 0

    def clone(self):
        """深拷贝当前对局（供 AI 在后台线程中搜索，避免与界面共享状态）。"""
        copy = Game.__new__(Game)
        copy.setup_positions = self.setup_positions
        copy.restore_snapshot(self.create_snapshot())
        # 历史快照创建后不会被原地修改，复制列表即可
        copy.history = list(self.history)
        return copy

    def record_history(self):
        self.history.append(self.create_snapshot())

    def can_undo(self):
        return len(self.history) > 1

    def undo_steps(self, steps=1):
        undo_count = min(max(steps, 1), len(self.history) - 1)
        if undo_count <= 0:
            return 0
        for _ in range(undo_count):
            self.history.pop()
        self.restore_snapshot(self.history[-1])
        return undo_count

    # 开局放置棋子
    def setup_placement(self, pool_index):
        if self.phase != 'setup':
            return False
        if self.setup_index >= len(self.setup_positions):
            return False
        pos = self.setup_positions[self.setup_index]
        if pool_index < 0 or pool_index >= len(self.remaining_pool):
            return False
        piece = self.remaining_pool.pop(pool_index)
        piece['face_up'] = True
        self.board[pos['row']][pos['col']] = piece
        self._face_up_cache = {}
        self.setup_index += 1
        if self.setup_index >= len(self.setup_positions):
            self.phase = 'playing'
        self.record_history()
        return True

    def get_current_setup_pos(self):
        if self.setup_index >= len(self.setup_positions):
            return None
        return self.setup_positions[self.setup_index]

    def get_piece(self, row, col):
        if row is not None and col is not None and 0 <= row < ROWS and 0 <= col < COLS:
            return self.board[row][col]
        return None

    def face_up_pieces(self, side, piece_type=None):
        """某方的明子列表（带 row/col），按局面缓存。

        性能优化：AI 评估中这是调用最频繁的查询。缓存在任何改变棋盘的方法里清空，
        因此只能通过本类的方法修改棋盘；返回的 dict 由多次调用共享，调用方不得修改。
        """
        key = (side, piece_type)
        cached = self._face_up_cache.get(key)
        if cached is None:
            cached = []
            for r, row in enumerate(self.board):
                for c, piece in enumerate(row):
                    if not piece or not piece['face_up'] or piece['side'] != side:
                        continue
                    if piece_type and piece['type'] != piece_type:
                        continue
                    cached.append({**piece, 'row': r, 'col': c})
            self._face_up_cache[key] = cached
        return cached

    # 非炮棋子的直接吃子判断
    def can_eat(self, attacker, defender):
        if not defender or attacker['side'] == defender['side']:
            return False
        if not defender['face_up']:
            return False  # 非炮不能吃未翻开的子
        if attacker['type'] == '兵' and defender['type'] == '将':
            return True
        if attacker['type'] == '将' and defender['type'] == '兵':
            return False
        return attacker['rank'] >= defender['rank']

    # 获取某个棋子的合法走法
    def get_valid_moves(self, row, col):
        piece = self.get_piece(row, col)
        if not piece or not piece['face_up'] or piece['side'] != self.current_turn:
            return []

        moves = []
        board = self.board
        if piece['type'] == '炮':
            # 炮：隔一个子跳吃，不能直接移动
            for dr, dc in DIRS:
                jumped = False
                r, c = row + dr, col + dc
                while 0 <= r < ROWS and 0 <= c < COLS:
                    target = board[r][c]
                    if target:
                        if not jumped:
                            jumped = True  # 炮架（跳板）
                        else:
                            # 目标：可吃未翻开的子或翻开的敌方子
                            if not target['face_up'] or target['side'] != piece['side']:
                                moves.append({'row': r, 'col': c, 'type': 'capture'})
                            break
                    r += dr
                    c += dc
        else:
            # 普通棋子：上下左右移一格
            for dr, dc in DIRS:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
                    continue
                target = board[nr][nc]
                if not target:
                    moves.append({'row': nr, 'col': nc, 'type': 'move'})
                elif self.can_eat(piece, target):
                    moves.append({'row': nr, 'col': nc, 'type': 'capture'})
        return moves

    # 翻棋（从池中选择要翻出的棋子）
    def flip_piece(self, row, col, pool_index):
        cell = self.get_piece(row, col)
        if not cell or cell['face_up'] or self.phase != 'playing':
            return False
        if pool_index is None or pool_index < 0 or pool_index >= len(self.remaining_pool):
            return False

        chosen = self.remaining_pool.pop(pool_index)
        chosen['face_up'] = True
        self.board[row][col] = chosen
        self._face_up_cache = {}
        self.flip_score += FLIP_SCORE[chosen['type']]
        self._record_turn_combo(self.current_turn, False)
        self.last_action = {
            'type': 'flip', 'row': row, 'col': col,
            'piece': {'type': chosen['type'], 'side': chosen['side'], 'hp': chosen['hp']},
        }
        self.switch_turn()
        self.record_history()
        return True

    # 移动/吃子（pool_index 用于炮吃未翻开的子时选择翻出的棋子）
    def move_piece(self, from_row, from_col, to_row, to_col, pool_index=None):
        if self.phase != 'playing':
            return False
        piece = self.get_piece(from_row, from_col)
        if not piece or not piece['face_up'] or piece['side'] != self.current_turn:
            return False

        valid = self.get_valid_moves(from_row, from_col)
        if not any(m['row'] == to_row and m['col'] == to_col for m in valid):
            return False

        target = self.board[to_row][to_col]
        if target:
            # 吃子
            if not target['face_up']:
                # 炮吃未翻开的子：从池中选择翻出
                if pool_index is None or pool_index < 0 or pool_index >= len(self.remaining_pool):
                    return False
                target = self.remaining_pool.pop(pool_index)
                target['face_up'] = True
                self.board[to_row][to_col] = target
            self.hp[target['side']] -= target['hp']
            self.last_action = {
                'type': 'capture',
                'from': {'row': from_row, 'col': from_col},
                'to': {'row': to_row, 'col': to_col},
                'attacker': {'type': piece['type'], 'side': piece['side']},
                'defender': {'type': target['type'], 'side': target['side'], 'hp': target['hp']},
            }
        else:
            self.last_action = {
                'type': 'move',
                'from': {'row': from_row, 'col': from_col},
                'to': {'row': to_row, 'col': to_col},
                'piece': {'type': piece['type'], 'side': piece['side']},
            }

        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        self._face_up_cache = {}
        self._record_turn_combo(piece['side'], target is not None)

        # 判断胜负
        if self.hp['red'] <= 0:
            self.phase = 'over'
            self.winner = 'black'
        elif self.hp['black'] <= 0:
            self.phase = 'over'
            self.winner = 'red'
        else:
            self.switch_turn()

        self.record_history()
        return True

    def switch_turn(self):
        self.current_turn = other_side(self.current_turn)

    # 获取当前玩家所有合法操作
    def get_all_actions(self):
        actions = []
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board[r][c]
                if not piece:
                    continue
                if not piece['face_up']:
                    actions.append({'type': 'flip', 'row': r, 'col': c})
                elif piece['side'] == self.current_turn:
                    for m in self.get_valid_moves(r, c):
                        actions.append({
                            'type': m['type'],
                            'from_row': r, 'from_col': c,
                            'to_row': m['row'], 'to_col': m['col'],
                        })
        return actions

    def get_remaining_pool(self):
        return self.remaining_pool

    def find_pool_index(self, piece_type, side):
        for i, p in enumerate(self.remaining_pool):
            if p['type'] == piece_type and p['side'] == side:
                return i
        return -1

    # 序列化状态
    def get_state(self):
        return {
            'board': [[
                ({'type': c['type'], 'side': c['side'], 'rank': c['rank'], 'hp': c['hp'], 'face_up': True}
                 if c['face_up'] else {'face_up': False}) if c else None
                for c in row] for row in self.board],
            'current_turn': self.current_turn,
            'hp': dict(self.hp),
            'phase': self.phase,
            'winner': self.winner,
            'remaining_pool': [
                {'type': p['type'], 'side': p['side'], 'rank': p['rank'], 'hp': p['hp']}
                for p in self.remaining_pool],
        }
