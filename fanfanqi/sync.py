"""画面自动同步：监视屏幕上的对局画面，识别后把局面同步到程序里的对局。

两部分：
- ScreenWatcher（后台线程）：定时截屏，棋盘区域画面稳定后识别，连续两次识别一致才交出结果，
  避免把走子 / 翻子动画中间的画面当成局面；
- BoardSync：把识别结果和当前对局对上。先找“从当前局面走一两步能到达识别出的局面”的合法动作并执行，
  这样走子历史、连击、长捉和未吃子计数都和逐步录入一样；对不上（中途开始同步、漏看了好几步）
  才整体导入局面。新开一局时按开局 4 个固定位置的明子重建，翻棋分也是对的。

轮到谁以画面为准（见 vision._whose_turn）；刚走完一步的短时间内头像绿框可能还没切换，
这段时间内与“轮流走子”推出的结果不一致时以后者为准。
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .advisor import describe_last_action
from .game import COMBO_THRESHOLD, PIECE_CONFIG, ROWS, COLS, SIDE_NAMES, Game, other_side
from .state import captured_pieces, game_from_state

TURN_LAG_SEC = 1.5  # 走子后这段时间内画面上的轮次可能还没切换


def board_tokens(game):
    tokens = []
    for row in game.board:
        line = []
        for cell in row:
            if not cell:
                line.append('.')
            elif not cell['face_up']:
                line.append('?')
            else:
                line.append(('r' if cell['side'] == 'red' else 'b') + cell['type'])
        tokens.append(line)
    return tokens


def _captured(game):
    return {side: {t: n for t, n in pieces.items() if n} for side, pieces in captured_pieces(game).items()}


def _hidden_count(tokens):
    return sum(row.count('?') for row in tokens)


def _matches(game, tokens, captured):
    return board_tokens(game) == tokens and _captured(game) == captured


@dataclass
class SyncUpdate:
    kind: str                      # 'same' 未变 | 'turn' 只改了轮次 | 'step' 按动作推进 | 'new' 新开一局 | 'import' 整体导入 | 'skip' 本次不同步
    game: Game = None
    log: list = field(default_factory=list)   # 新增的棋谱行（'new' / 'import' 时是整局重新开始的棋谱）
    message: str = ''


class BoardSync:
    def __init__(self, rules=None):
        self.rules = dict(rules or {})
        self.my_side = None        # 下方玩家（自己）执哪一方，本局内记住（连击特效时可能暂时认不出）
        self._last_step = 0.0

    def _new_game(self):
        return Game(**{k: v for k, v in self.rules.items() if k in ('long_chase_limit', 'no_capture_limit')})

    def update(self, rec, game, now=None):
        now = time.monotonic() if now is None else now
        if rec.game_over:
            return SyncUpdate('skip', message='对局已结束（画面上出现「开始」按钮）')
        if rec.problems:
            return SyncUpdate('skip', message='识别不完整：' + rec.problems[0])
        tokens = [list(row) for row in rec.board]
        captured = {side: {t: n for t, n in pieces.items() if n} for side, pieces in rec.captured.items()}
        if rec.my_side:
            self.my_side = rec.my_side

        playing = game is not None and game.phase in ('playing', 'over')
        if not playing or _hidden_count(tokens) > _hidden_count(board_tokens(game)):
            return self._rebuild(rec, tokens, captured, now)

        if _matches(game, tokens, captured):
            if rec.turn and rec.turn != game.current_turn and rec.turn_certain \
                    and now - self._last_step >= TURN_LAG_SEC and game.phase == 'playing':
                game = game.clone()
                game.current_turn = rec.turn
                game.record_history()
                return SyncUpdate('turn', game, message=f'按画面纠正：轮到{SIDE_NAMES[rec.turn]}')
            return SyncUpdate('same', game)

        found = self._search(game, tokens, captured)
        if found is None:
            update = self._rebuild(rec, tokens, captured, now)
            update.message = '无法从上一局面推出这一步（可能漏看了几步），已整体导入；连击等历史从这里重新计' \
                if update.kind == 'import' else update.message
            return update
        new_game, log = found
        self._last_step = now
        return SyncUpdate('step', new_game, log)

    # ------------------------------------------------------------ 推进
    def _search(self, game, tokens, captured):
        """找一到两步合法动作把 game 变成识别出的局面。先按轮流走子，再试另一方先走（轮次曾识别错）。"""
        for first in (game.current_turn, other_side(game.current_turn)):
            start = game.clone()
            start.current_turn = first
            for g1, line1 in self._children(start, tokens, captured):
                if _matches(g1, tokens, captured):
                    return g1, [line1]
        for first in (game.current_turn, other_side(game.current_turn)):
            start = game.clone()
            start.current_turn = first
            for g1, line1 in self._children(start, tokens, captured, exact=False):
                if g1.phase != 'playing':
                    continue
                for g2, line2 in self._children(g1, tokens, captured):
                    if _matches(g2, tokens, captured):
                        return g2, [line1, line2]
        return None

    def _children(self, game, tokens, captured, exact=True):
        """当前方所有合法动作的结果。翻子 / 炮打暗子翻出什么由目标局面决定，与目标矛盾的直接跳过。
        exact=False 用于两步中的第一步：翻开的子可能马上被走开或吃掉，目标局面里那一格已经不是它，
        这时把剩余池里每种可能都试一遍（由第二步来对上）。"""
        target_lost = captured
        current_lost = _captured(game)
        for action in game.get_all_actions():
            mover = game.current_turn
            if action['type'] == 'flip':
                r, c = action['row'], action['col']
                token = tokens[r][c]
                if token == '?':
                    continue
                if token != '.':
                    outcomes = [('red' if token[0] == 'r' else 'black', token[1])]
                else:
                    outcomes = []
                if not exact:
                    outcomes += sorted({(p['side'], p['type']) for p in game.remaining_pool} - set(outcomes))
                for side, piece_type in outcomes:
                    g = game.clone()
                    idx = g.find_pool_index(piece_type, side)
                    if idx < 0 or not g.flip_piece(r, c, idx):
                        continue
                    yield g, f'{SIDE_NAMES[mover]} {describe_last_action(g.last_action)}'
                continue
            fr, fc, tr, tc = action['from_row'], action['from_col'], action['to_row'], action['to_col']
            target = game.board[tr][tc]
            if target and not target['face_up']:
                # 炮打暗子：被打掉的是被吃子栏里新增的那种
                options = [(side, t) for side in ('red', 'black') for t in PIECE_CONFIG
                           if target_lost.get(side, {}).get(t, 0) > current_lost.get(side, {}).get(t, 0)]
            else:
                options = [None]
            for option in options:
                g = game.clone()
                idx = None
                if option:
                    idx = g.find_pool_index(option[1], option[0])
                    if idx < 0:
                        continue
                if not g.move_piece(fr, fc, tr, tc, idx):
                    continue
                line = f'{SIDE_NAMES[mover]} {describe_last_action(g.last_action)}'
                if g.combo[mover] >= COMBO_THRESHOLD:
                    line += f' · {g.combo[mover]}连击'
                yield g, line

    # ------------------------------------------------------------ 重建
    def _rebuild(self, rec, tokens, captured, now):
        game = self._from_opening(tokens, captured)
        if game is not None:
            self.my_side = rec.my_side  # 新一局，执子方重新识别
            kind, log, message = 'new', ['画面同步：新对局'], '检测到新对局'
        else:
            state = rec.to_state(turn=rec.turn or 'red')
            state['rules'] = {k: v for k, v in self.rules.items() if k in ('long_chase_limit', 'no_capture_limit')}
            state['board'], state['captured'] = tokens, captured
            try:
                game = game_from_state(state)
            except ValueError as exc:
                return SyncUpdate('skip', message=f'识别结果不自洽：{exc}')
            kind, log = 'import', ['画面同步：导入局面（之前的连击、开局翻棋分无法从画面得知，按 0 计）']
            message = '已导入当前局面'
        if rec.turn and game.phase == 'playing':
            game.current_turn = rec.turn
            game.history[-1] = game.create_snapshot()
        self._last_step = now
        return SyncUpdate(kind, game, log, message)

    def _from_opening(self, tokens, captured):
        """还没有吃子、开局 4 个固定位置都是明子时，按开局重建：先放 4 个明子（计翻棋分），其余明子按翻子处理。"""
        if any(captured.values()) or any('.' in row for row in tokens):
            return None
        game = self._new_game()
        setup = [(p['row'], p['col']) for p in game.setup_positions]
        if any(tokens[r][c] in ('.', '?') for r, c in setup):
            return None
        for r, c in setup:
            token = tokens[r][c]
            idx = game.find_pool_index(token[1], 'red' if token[0] == 'r' else 'black')
            if idx < 0:
                return None
            game.setup_placement(idx)
        for r in range(ROWS):
            for c in range(COLS):
                token = tokens[r][c]
                if (r, c) in setup or token == '?':
                    continue
                idx = game.find_pool_index(token[1], 'red' if token[0] == 'r' else 'black')
                if idx < 0 or not game.flip_piece(r, c, idx):
                    return None
        return game


# ---------------------------------------------------------------- 截屏监视
def _signature(rgb, layout):
    """棋盘和两侧被吃子栏的缩略灰度图，用来判断画面是否在变化（头像倒计时框不在其中）。"""
    parts = []
    for col, row, w, h in ((2.0, 4.0, 4.0, 8.0), (-0.95, 3.4, 0.6, 4.2), (5.0, 4.6, 0.6, 4.2)):
        patch = layout.crop(rgb, col, row, w, h)
        if patch is None:
            return None
        parts.append(patch[::4, ::4].astype(np.float32).mean(axis=2).ravel())
    return np.concatenate(parts)


def _key(rec):
    return (tuple(tuple(r) for r in rec.board), repr(sorted((s, sorted(p.items())) for s, p in rec.captured.items())),
            rec.turn, rec.turn_certain, rec.my_side, rec.game_over, bool(rec.problems))


class ScreenWatcher(threading.Thread):
    """后台截屏识别。post(kind, payload)：('rec', Recognition) 识别结果，('status', 文本) 状态。

    interval  截屏间隔（秒）；settle 画面连续不变多久才识别；refresh 画面不变时多久复查一次（轮次、倒计时框）。
    画面一直在变（特效动画）超过 3 秒时也照常识别，连续两次一致才交出。"""

    def __init__(self, post, interval=0.15, settle=0.3, refresh=0.5, grab=None, templates=None):
        super().__init__(daemon=True)
        self.post = post
        self.interval, self.settle, self.refresh = interval, settle, refresh
        self.grab = grab
        self.templates = templates
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        from . import vision
        grab = self.grab or vision.grab_screen
        templates = self.templates or vision.default_templates()
        layout, last_sig, changed_at, recognized_at = None, None, 0.0, 0.0
        pending, dirty, dirty_since, poor = None, True, time.monotonic(), 0
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                rgb = vision.to_rgb(grab())
            except Exception as exc:  # noqa: BLE001 — 截屏失败（锁屏等）时稍后重试
                self.post('status', f'截屏失败：{exc}')
                self._stop_event.wait(1.0)
                continue
            if layout is None:
                layout = vision.find_board(rgb, templates)
                if layout is None:
                    self.post('status', '屏幕上没有找到棋盘（游戏窗口需要可见，不能被遮挡或最小化）')
                    self._stop_event.wait(1.0)
                    continue
                last_sig, pending = None, None
            sig = _signature(rgb, layout)
            if sig is None:
                layout = None
                continue
            if last_sig is None or sig.shape != last_sig.shape or float(np.abs(sig - last_sig).mean()) > 1.0:
                changed_at = started
                if not dirty:
                    dirty, dirty_since = True, started
            last_sig = sig
            settled = started - changed_at >= self.settle
            busy_too_long = dirty and started - dirty_since >= 3.0
            due = started - recognized_at >= (0.25 if dirty else self.refresh)
            if due and (settled or busy_too_long or not dirty):
                recognized_at = started
                rec = vision.recognize(rgb, templates, layout)
                if not rec.game_over and rec.quality < vision.GOOD_QUALITY:
                    poor += 1
                    if poor >= 2:  # 连续两次对不上：棋盘移动了或被遮挡，重新找
                        layout, poor = None, 0
                        self.post('status', '棋盘位置变化，重新定位…')
                    continue
                poor = 0
                key = _key(rec)
                if key == pending:
                    self.post('rec', rec)
                    dirty = False
                pending = key
            self._stop_event.wait(max(0.0, self.interval - (time.monotonic() - started)))
