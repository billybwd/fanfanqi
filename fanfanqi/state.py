"""局面状态：紧凑的状态格式、被吃棋子统计、导入导出与机器学习特征编码。

状态只描述“当前这一刻”，不依赖整盘棋谱，可用于：
- 训练样本（每个局面一条）；
- 外部来源（摄像头识别、其他棋盘程序）同步局面后让 AI 给提示。

状态格式（v1）::

    {
      "v": 1,
      "turn": "red",                       # 轮到谁走
      "hp": {"red": 48, "black": 55},      # 可省略；省略时由被吃棋子推算
      "board": [                           # 8 行 × 4 列，行 0 在上
        ["?", "r车", ".", "?"],            # "?" 暗子  "." 空格  "r"/"b" + 棋子类型
        ...
      ],
      "captured": {                        # 各方“被吃掉”的棋子（不是吃到的）
        "red":   {"仕": 1, "兵": 1},
        "black": {"马": 1}
      },
      "combo": {"red": 0, "black": 2},     # 当前连击（v2，可省略，默认 0）
      "max_combo": {"red": 1, "black": 2}, # 本局最高连击（v2，可省略，默认 0）
      "flip_score": 24,                    # 双方手动翻子累计翻棋分（v2，可省略，默认 0）
      "multiplier": {...}                  # 导出时附带的倍率明细，导入时忽略（按上面各项重新计算）
    }

棋子类型统一用 将 仕 象 车 马 炮 兵；导入时也接受 帅 相 士 卒 等显示名。
剩余棋子池（暗子的可能身份）= 全部棋子 − 明子 − 被吃棋子，不需要单独给出。
"""

from .game import COLS, INITIAL_HP, PIECE_CONFIG, ROWS, Game, compute_multiplier, make_piece

STATE_VERSION = 2
PIECE_TYPES = list(PIECE_CONFIG)  # 将 仕 象 车 马 炮 兵
SIDES = ('red', 'black')
SIDE_PREFIX = {'red': 'r', 'black': 'b'}
PREFIX_SIDE = {'r': 'red', 'b': 'black', '红': 'red', '黑': 'black'}
TYPE_ALIASES = {'帅': '将', '相': '象', '士': '仕', '卒': '兵'}


# --------------------------------------------------------------- 统计
def _empty_counts():
    return {side: {t: 0 for t in PIECE_TYPES} for side in SIDES}


def face_up_counts(game):
    counts = _empty_counts()
    for row in game.board:
        for cell in row:
            if cell and cell['face_up']:
                counts[cell['side']][cell['type']] += 1
    return counts


def pool_counts(game):
    counts = _empty_counts()
    for p in game.remaining_pool:
        counts[p['side']][p['type']] += 1
    return counts


def captured_pieces(game):
    """各方被吃掉的棋子数：全部 − 明子 − 剩余池（暗子）。"""
    up, pool = face_up_counts(game), pool_counts(game)
    return {side: {t: PIECE_CONFIG[t]['count'] - up[side][t] - pool[side][t] for t in PIECE_TYPES}
            for side in SIDES}


def lost_hp(captured_side):
    return sum(PIECE_CONFIG[t]['hp'] * n for t, n in captured_side.items())


# --------------------------------------------------------------- 导出
def game_to_state(game):
    board = []
    for row in game.board:
        tokens = []
        for cell in row:
            if not cell:
                tokens.append('.')
            elif not cell['face_up']:
                tokens.append('?')
            else:
                tokens.append(SIDE_PREFIX[cell['side']] + cell['type'])
        board.append(tokens)
    captured = captured_pieces(game)
    return {
        'v': STATE_VERSION,
        'turn': game.current_turn,
        'hp': dict(game.hp),
        'board': board,
        'captured': {side: {t: n for t, n in captured[side].items() if n} for side in SIDES},
        'combo': dict(game.combo),
        'max_combo': dict(game.max_combo),
        'flip_score': game.flip_score,
        'multiplier': game.get_multiplier(),
    }


# --------------------------------------------------------------- 导入
def _parse_token(token, r, c):
    if token is None or token in ('.', '', '空'):
        return None
    if token in ('?', '暗'):
        return 'hidden'
    if len(token) != 2 or token[0] not in PREFIX_SIDE:
        raise ValueError(f'第 {r} 行第 {c} 列无法识别的棋子：{token!r}（应为 "?"、"." 或 "r车" / "b马" 这类）')
    side = PREFIX_SIDE[token[0]]
    piece_type = TYPE_ALIASES.get(token[1], token[1])
    if piece_type not in PIECE_CONFIG:
        raise ValueError(f'第 {r} 行第 {c} 列未知棋子类型：{token!r}')
    return side, piece_type


def game_from_state(state):
    """由状态构造对局。状态不自洽（数量、血量对不上）时抛出 ValueError，说明原因。"""
    board_tokens = state.get('board')
    if not isinstance(board_tokens, list) or len(board_tokens) != ROWS or any(
            not isinstance(row, list) or len(row) != COLS for row in board_tokens):
        raise ValueError(f'board 必须是 {ROWS} 行 × {COLS} 列')
    turn = state.get('turn', 'red')
    if turn not in SIDES:
        raise ValueError(f'turn 必须是 red 或 black，收到 {turn!r}')

    captured = _empty_counts()
    for side, pieces in (state.get('captured') or {}).items():
        if side not in SIDES:
            raise ValueError(f'captured 中未知阵营：{side!r}')
        for name, n in pieces.items():
            t = TYPE_ALIASES.get(name, name)
            if t not in PIECE_CONFIG or not isinstance(n, int) or n < 0:
                raise ValueError(f'captured.{side} 中无效的条目：{name}={n!r}')
            captured[side][t] += n

    game = Game(shuffle_pool=False)
    up = _empty_counts()
    hidden = empty = 0
    for r in range(ROWS):
        for c in range(COLS):
            parsed = _parse_token(board_tokens[r][c], r, c)
            if parsed is None:
                game.board[r][c] = None
                empty += 1
            elif parsed == 'hidden':
                hidden += 1  # 暗子保持 Game 初始化时的暗子格
            else:
                side, t = parsed
                game.board[r][c] = make_piece(t, side, face_up=True)
                up[side][t] += 1

    pool = []
    for side in SIDES:
        for t in PIECE_TYPES:
            left = PIECE_CONFIG[t]['count'] - up[side][t] - captured[side][t]
            if left < 0:
                raise ValueError(f"{'红' if side == 'red' else '黑'}方「{t}」明子 {up[side][t]} + 被吃 "
                                 f"{captured[side][t]} 超过总数 {PIECE_CONFIG[t]['count']}")
            pool.extend(make_piece(t, side) for _ in range(left))
    if len(pool) != hidden:
        raise ValueError(f'棋盘上有 {hidden} 个暗子，但按被吃棋子推算剩余应为 {len(pool)} 个'
                         f'（空格 {empty} 个应等于被吃总数 {sum(sum(v.values()) for v in captured.values())}）')

    hp = {side: INITIAL_HP - lost_hp(captured[side]) for side in SIDES}
    given = state.get('hp')
    if given:
        for side in SIDES:
            if side in given and given[side] != hp[side]:
                raise ValueError(f'{side} 血量 {given[side]} 与被吃棋子推算的 {hp[side]} 不符')

    combo = _side_ints(state, 'combo')
    max_combo = _side_ints(state, 'max_combo')
    for side in SIDES:
        if combo[side] > max_combo[side]:
            max_combo[side] = combo[side]  # 只给了当前连击时，最高连击至少等于它
    flip_score = state.get('flip_score', 0)
    if not isinstance(flip_score, int) or flip_score < 0:
        raise ValueError(f'flip_score 必须是非负整数，收到 {flip_score!r}')

    game.remaining_pool = pool
    game.hp = hp
    game.combo = combo
    game.max_combo = max_combo
    game.flip_score = flip_score
    game.current_turn = turn
    game.setup_index = len(game.setup_positions)
    game.phase = 'playing'
    game.winner = None
    game.last_action = None
    if hp['red'] <= 0 or hp['black'] <= 0:
        game.phase = 'over'
        game.winner = 'black' if hp['red'] <= 0 else 'red'
    game._face_up_cache = {}
    game.history = []
    game.record_history()
    return game


def _side_ints(state, key):
    raw = state.get(key) or {}
    result = {}
    for side in SIDES:
        v = raw.get(side, 0)
        if not isinstance(v, int) or v < 0:
            raise ValueError(f'{key}.{side} 必须是非负整数，收到 {v!r}')
        result[side] = v
    return result


# ------------------------------------------------------------- 特征编码
# 前 608 维与原项目 training/encode.py 完全相同（19 通道 × 32 格），之后追加：
#   608-621：各方被吃棋子数 / 该类型总数（red 7 种 + black 7 种，顺序 将仕象车马炮兵）
#   622-635：剩余池（暗子可能身份）各类型数量 / 该类型总数（同上顺序）
#   636-642：倍率相关：当前连击 红/黑 ÷5、最高连击 红/黑 ÷5、翻棋分 ÷172、血量差 ÷60、倍率 ÷500
# 原编码看不出暗子里还剩什么，追加的 28 维让模型知道“翻开大概率是什么”；
# 倍率相关 7 维让模型能学到“领先时把倍率做大、落后时压小”。
FEATURE_VERSION = 2
NUM_CHANNELS = 19
BOARD_SIZE = ROWS * COLS
BASE_SIZE = NUM_CHANNELS * BOARD_SIZE  # 608
MAX_FLIP_SCORE = 172  # 全部 32 子都被手动翻开时的翻棋分（实际最多 28 子，留有余量）
FEATURE_SIZE = BASE_SIZE + 2 * 2 * len(PIECE_TYPES) + 7  # 643


def encode_features(game):
    vec = [0.0] * FEATURE_SIZE
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            cell = game.board[r][c]
            if cell is None:
                vec[15 * BOARD_SIZE + i] = 1.0
            elif not cell['face_up']:
                vec[14 * BOARD_SIZE + i] = 1.0
            else:
                ch = SIDES.index(cell['side']) * 7 + PIECE_TYPES.index(cell['type'])
                vec[ch * BOARD_SIZE + i] = 1.0
            vec[16 * BOARD_SIZE + i] = 1.0 if game.current_turn == 'red' else 0.0
            vec[17 * BOARD_SIZE + i] = max(0.0, game.hp['red'] / INITIAL_HP)
            vec[18 * BOARD_SIZE + i] = max(0.0, game.hp['black'] / INITIAL_HP)

    captured, pool = captured_pieces(game), pool_counts(game)
    k = BASE_SIZE
    for counts in (captured, pool):
        for side in SIDES:
            for t in PIECE_TYPES:
                vec[k] = counts[side][t] / PIECE_CONFIG[t]['count']
                k += 1

    m = game.get_multiplier()
    for v in (game.combo['red'] / 5, game.combo['black'] / 5,
              game.max_combo['red'] / 5, game.max_combo['black'] / 5,
              m['flip_score'] / MAX_FLIP_SCORE, m['hp_diff'] / INITIAL_HP, m['multiplier'] / 500):
        vec[k] = v
        k += 1
    return vec
