"""校准胜率与开局（抢红 / 加倍）建议。

搜索给出的“胜率”是 搜索值 ÷ 当前倍率 截到 0~1 的结果（λ = 1 时搜索值里还含着将来倍率的增长，比值会超过 ±1），
不是真正的概率。这里用自对弈的实际胜负拟合一个逻辑回归（tools/calibrate.py fit 生成 fanfanqi/calibration.json）：

    校准胜率 = sigmoid(w0 + w1 · 搜索值/倍率^λ + w2 · 领先血量/30 + w3 · 领先血量/30 × 剩余暗子比例)

没有 calibration.json 时各函数返回 None，界面照旧显示引擎自己的胜率。
"""

import json
import math
import os

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), 'calibration.json')
OPENING_MARGIN = 0.03   # 胜率至少比 50% 高出这么多才建议抢红 / 加倍（开局估计本身有几个百分点的误差）
DECIDED_PROB = 0.95     # 校准胜率达到这个值（或领先一个将以上）标为“基本定局”
DECIDED_LEAD = 30       # 一个将的血量：领先超过它，对方吃掉我的将也追不回来

_cache = {}


def load(path=CALIBRATION_PATH):
    """读取校准系数（缓存；文件不存在或格式不对时返回 None）。"""
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    if _cache.get('key') != (path, mtime):
        model = None
        if mtime is not None:
            try:
                with open(path, encoding='utf-8') as f:
                    model = json.load(f)
                if len(model.get('weights', [])) != 4:
                    model = None
            except (OSError, ValueError):
                model = None
        _cache.update(key=(path, mtime), model=model)
    return _cache['model']


def features(score, multiplier, hp_lead, hidden, lam=1.0):
    """搜索值 ÷ 倍率^λ（截在 ±3）、领先血量 / 30、领先 / 30 × 剩余暗子比例（开局 28 个暗子 = 1）。"""
    ratio = max(-3.0, min(3.0, score / max(1.0, multiplier) ** lam))
    lead = hp_lead / 30.0
    return [1.0, ratio, lead, lead * hidden / 28.0]


def probability(model, score, multiplier, hp_lead, hidden, lam=1.0):
    z = sum(w * x for w, x in zip(model['weights'], features(score, multiplier, hp_lead, hidden, lam)))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def hidden_count(game):
    return sum(1 for row in game.board for cell in row if cell and not cell['face_up'])


def win_prob(info, game, lam=1.0, model=None):
    """搜索结果 info（含 score、multiplier）对应的校准胜率，站在 game 轮到走的一方；算不了时返回 None。"""
    model = model or load()
    if not model or not info or info.get('score') is None:
        return None
    me = game.current_turn
    op = 'black' if me == 'red' else 'red'
    multiplier = info.get('multiplier') or game.get_multiplier()['multiplier']
    return probability(model, info['score'], multiplier, game.hp[me] - game.hp[op], hidden_count(game), lam)


def decided(prob, game):
    """当前轮到走的一方是否基本定局（胜率很高，或领先超过一个将）。"""
    me = game.current_turn
    op = 'black' if me == 'red' else 'red'
    return (prob is not None and prob >= DECIDED_PROB) or game.hp[me] - game.hp[op] >= DECIDED_LEAD


def is_opening(game):
    """开局 4 个明子已放好、还没有走第一步（抢红 / 加倍在这时决定）。"""
    return game.phase == 'playing' and len(game.remaining_pool) == 32 - len(game.setup_positions)


def opening_advice(red_prob, my_side=None, margin=OPENING_MARGIN):
    """按开局时红方（先手）的校准胜率给出抢红 / 加倍建议，返回几行文字。
    抢红 = 执红先走且倍率 ×2；加倍 = 倍率 ×2。胜率高于 50% 时期望收益为正，加倍就是把正的期望放大一倍，反之亦然。"""
    lines = [f'开局评估：红方（先手）胜率约 {red_prob:.0%}']
    if red_prob >= 0.5 + margin:
        lines.append('抢红：建议抢（执红更有利）')
    elif red_prob <= 0.5 - margin:
        lines.append('抢红：不抢（执黑更有利）')
    else:
        lines.append('抢红：差别不大，可不抢')
    sides = [my_side] if my_side else ['red', 'black']
    for side in sides:
        p = red_prob if side == 'red' else 1 - red_prob
        name = '执红' if side == 'red' else '执黑'
        verdict = '建议加倍' if p >= 0.5 + margin else '不加倍'
        lines.append(f'加倍（{name}，胜率约 {p:.0%}）：{verdict}')
    return lines
