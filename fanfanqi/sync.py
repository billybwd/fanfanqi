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
from itertools import combinations_with_replacement

import numpy as np

from .advisor import describe_last_action
from .game import COMBO_THRESHOLD, FLIP_SCORE, PIECE_CONFIG, ROWS, COLS, SIDE_NAMES, Game, other_side
from .state import captured_pieces, game_from_state

TURN_LAG_SEC = 1.5  # 走子后这段时间内画面上的轮次可能还没切换
ROOM_FACTORS = (1, 2, 4)  # 画面倍率额外乘的系数：抢红 ×2、加倍 ×2，都有 ×4
# 开局 4 个明子的翻棋分之和可能的取值（导入局面时翻棋分未知，在其中找）
FLIP_SCORES = sorted({sum(FLIP_SCORE[t] for t in combo)
                      for combo in combinations_with_replacement(PIECE_CONFIG, 4)
                      if all(combo.count(t) <= 2 * PIECE_CONFIG[t]['count'] for t in combo)})


def _combo_points(game, extra=0):
    """连击系数：已结束的达标连击分（加上 extra）+ 正在进行的 ≥3 连击；都没有时为 1。"""
    banked = sum(game.combo_score.values()) + extra
    active = sum(n for n in game.combo.values() if n >= COMBO_THRESHOLD)
    return banked, active, (banked + active) or 1


def multiplier_candidates(shown, game, factors, flips, extras=None):
    """满足 画面倍率 = F × 连击系数 × (翻棋分 + 血量差) 的候选 {(F, 翻棋分, 额外连击分)}。

    额外连击分是对局里没记到的已结束连击分（中途导入时之前的连击、或漏看的连击），加在双方已结束连击分之和上；
    已结束连击分之和只能是 0 或 ≥3。extras 给出时只检验这些取值，否则由画面倍率算出。"""
    hp_diff = game.get_multiplier()['hp_diff']
    banked0, active, _ = _combo_points(game)
    found = set()
    for f in factors:
        for flip in flips:
            base = f * (flip + hp_diff)
            if base <= 0 or shown % base:
                continue
            c = shown // base
            if c == 1:
                if active:
                    continue
                extra = -banked0
            else:
                extra = c - active - banked0
            banked = banked0 + extra
            if banked < 0 or 0 < banked < COMBO_THRESHOLD:
                continue
            if extras is None or extra in extras:
                found.add((f, flip, extra))
    return found


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
    kind: str                      # 'same' 未变 | 'turn' 只改了轮次 | 'adjust' 按画面倍率改了翻棋分/连击 | 'step' 按动作推进 | 'new' 新开一局 | 'import' 整体导入 | 'skip' 本次不同步
    game: Game = None
    log: list = field(default_factory=list)   # 新增的棋谱行（'new' / 'import' 时是整局重新开始的棋谱）
    message: str = ''


class BoardSync:
    def __init__(self, rules=None):
        self.rules = dict(rules or {})
        self.my_side = None        # 下方玩家（自己）执哪一方，本局内记住（连击特效时可能暂时认不出）
        self.room_factor = None    # 画面倍率的额外系数 F（抢红/加倍），从画面倍率反推，本局内记住
        self.trusted = False       # 对局里的翻棋分、连击是否可信（按开局重建的可信；整体导入的要靠画面倍率反推）
        self._candidates = None    # 画面倍率的候选解 {(F, 翻棋分, 额外连击分)}，跨帧取交集
        self._flip_hint = None     # 本局已确定的翻棋分（整体导入时沿用）
        self._last_step = 0.0

    def _new_game(self):
        return Game(**{k: v for k, v in self.rules.items() if k in ('long_chase_limit', 'no_capture_limit')})

    def update(self, rec, game, now=None):
        result = self._update(rec, game, now)
        if result.kind in ('new', 'import'):
            # 新对局：F 重新认；按开局重建的翻棋分、连击可信，整体导入的要靠画面倍率反推。
            # 同一局里漏看几步而整体导入时，之前确定的 F 和翻棋分整局不变，优先沿用（对不上画面时再全面找）
            if result.kind == 'new':
                self.room_factor, self._flip_hint = None, result.game.flip_score
            self.trusted, self._candidates = result.kind == 'new', None
        if result.kind != 'skip' and rec.multiplier:
            self._reconcile(result, rec.multiplier)
        return result

    def _reconcile(self, result, shown):
        """按画面倍率反推系数 F、翻棋分和没记到的连击分。F、开局翻棋分、导入前的连击分整局不变，
        所以候选解跨帧取交集（中途导入时单看一帧常有几组解，血量差变化后会收敛），所有候选一致的部分才采用。"""
        game = result.game
        started = len(game.remaining_pool) < 32 - len(game.setup_positions)  # 开局 4 子之外翻过子
        factors = [self.room_factor] if self.room_factor and started else list(ROOM_FACTORS)
        candidates = set()
        if self._candidates:
            candidates = {c for c in self._candidates if c[0] in factors and
                          c in multiplier_candidates(shown, game, [c[0]], [c[1]], [c[2]])}
        if not candidates and self.trusted:  # 翻棋分、连击都可信时只有 F 未知
            candidates = multiplier_candidates(shown, game, factors, [game.flip_score], [0])
        if not candidates and self.trusted:  # 对局里的连击和画面不符（漏看等）：翻棋分仍可信
            candidates = multiplier_candidates(shown, game, factors, [game.flip_score])
        if not candidates and self._flip_hint is not None:
            candidates = multiplier_candidates(shown, game, factors, [self._flip_hint])
        if not candidates:
            candidates = multiplier_candidates(shown, game, factors, FLIP_SCORES)
        self._candidates = candidates or None
        if not candidates:
            return

        factors, flips, extras = ({c[i] for c in candidates} for i in range(3))
        notes = []
        if len(factors) == 1 and self.room_factor != next(iter(factors)):
            self.room_factor = next(iter(factors))
            notes.append(f'倍率系数 ×{self.room_factor}')
        changed = game.clone() if result.kind == 'same' else game
        before = changed.get_multiplier()
        if len(flips) == 1 and changed.flip_score != next(iter(flips)):
            changed.flip_score = next(iter(flips))
            notes.append(f"翻棋分 {before['flip_score']} → {changed.flip_score}")
        if len(extras) == 1 and next(iter(extras)):
            extra = next(iter(extras))
            # 连击系数只看双方已结束连击分之和，记在原来分数多的一方
            side = 'red' if changed.combo_score['red'] >= changed.combo_score['black'] else 'black'
            changed.combo_score = {side: sum(changed.combo_score.values()) + extra, other_side(side): 0}
            self._candidates = {(f, flip, x - extra) for f, flip, x in candidates}
            notes.append(f"连击系数 {before['combo_factor']} → {changed.get_multiplier()['combo_factor']}")
        if len(flips) == 1:
            self._flip_hint = next(iter(flips))
        if len(flips) == 1 and len(extras) == 1:
            self.trusted = True
        if not notes:
            return
        changed.history[-1] = changed.create_snapshot()
        if result.kind == 'same' and changed.create_snapshot() != game.create_snapshot():
            result.kind, result.game = 'adjust', changed
        result.message = '；'.join(m for m in (result.message, f"按画面倍率 {shown}：{'，'.join(notes)}") if m)

    def _update(self, rec, game, now=None):
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
            rec.turn, rec.turn_certain, rec.my_side, rec.game_over, bool(rec.problems), rec.multiplier)


def _watch_region(vision, rgb, layout):
    """找到棋盘后只截它周围一块（棋盘、两侧被吃子栏、头像和血量条、右上角倍率）：
    返回 (屏幕坐标的截屏范围, 换算到这块图像里的网格, 这块图像)。"""
    h, w = rgb.shape[:2]
    left, top, right, bottom = vision.WATCH_REGION
    x0, y0 = (max(0, int(v)) for v in layout.point(left, top))
    x1, y1 = layout.point(right, bottom)
    x1, y1 = min(w, int(x1) + 1), min(h, int(y1) + 1)
    ox, oy = vision.screen_origin()
    box = layout.box and (layout.box[0] - x0, layout.box[1] - y0, layout.box[2] - x0, layout.box[3] - y0)
    shifted = vision.Layout(layout.x - x0, layout.y - y0, layout.cell, box=box)
    return (ox + x0, oy + y0, ox + x1, oy + y1), shifted, rgb[y0:y1, x0:x1]


class ScreenWatcher(threading.Thread):
    """后台截屏识别。post(kind, payload)：('rec', Recognition) 识别结果，('status', 文本) 状态。

    interval  截屏间隔（秒）；settle 画面连续不变多久才识别；refresh 画面不变时多久复查一次（轮次、倒计时框）。
    画面一直在变（特效动画）超过 3 秒时也照常识别，连续两次一致才交出。
    save_failed  画面静止却识别出问题时，把棋盘一带存到 sample_image/failed（同一画面只存一次，最多每 5 秒一张）。"""

    def __init__(self, post, interval=0.15, settle=0.3, refresh=0.5, grab=None, templates=None, save_failed=False):
        super().__init__(daemon=True)
        self.post = post
        self.interval, self.settle, self.refresh = interval, settle, refresh
        self.grab = grab
        self.templates = templates
        self.save_failed = save_failed
        self._saved = (None, 0.0)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _save_failed(self, vision, rgb, layout, rec, now):
        key, at = self._saved
        if _key(rec) == key or now - at < 5.0:
            return
        self._saved = (_key(rec), now)
        try:
            path = vision.save_frame(rgb, layout, rec.problems)
        except OSError:
            return
        if path:
            self.post('status', f'识别没把握的画面已存到 {path}')

    def run(self):
        from . import vision
        grab = self.grab or vision.grab_screen
        # 装了 mss 时，找到棋盘后只截棋盘一带（快得多）；用自定义截屏函数（测试）或没有 mss 时每次截全屏
        use_region = self.grab is None and vision.fast_grab_available()
        region = None
        templates = self.templates or vision.default_templates()
        layout, last_sig, changed_at, recognized_at = None, None, 0.0, 0.0
        pending, dirty, dirty_since, poor = None, True, time.monotonic(), 0
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                rgb = vision.to_rgb(grab(region) if region else grab())
            except Exception as exc:  # noqa: BLE001 — 截屏失败（锁屏等）时稍后重试
                self.post('status', f'截屏失败：{exc}')
                self._stop_event.wait(1.0)
                continue
            if layout is None:
                region = None
                layout = vision.find_board(rgb, templates)
                if layout is None:
                    self.post('status', '屏幕上没有找到棋盘（游戏窗口需要可见，不能被遮挡或最小化）')
                    self._stop_event.wait(1.0)
                    continue
                if use_region:
                    region, layout, rgb = _watch_region(vision, rgb, layout)
                last_sig, pending = None, None
            sig = _signature(rgb, layout)
            if sig is None:
                layout = region = None
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
                if self.save_failed and settled and rec.problems and not rec.game_over and not rec.board_obscured:
                    self._save_failed(vision, rgb, layout, rec, started)
                if rec.board_obscured:
                    # 动画/弹窗覆盖棋盘时，保留当前同步局面，等画面恢复后再识别。
                    continue
                if not rec.game_over and rec.quality < vision.GOOD_QUALITY:
                    poor += 1
                    if poor >= 2:  # 连续两次对不上：棋盘移动了或被遮挡，重新找
                        layout, region, poor = None, None, 0
                        self.post('status', '棋盘位置变化，重新定位…')
                    continue
                poor = 0
                key = _key(rec)
                if key == pending:
                    self.post('rec', rec)
                    dirty = False
                pending = key
            self._stop_event.wait(max(0.0, self.interval - (time.monotonic() - started)))
