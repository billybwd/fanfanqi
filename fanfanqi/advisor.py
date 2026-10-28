"""把 AI 的决策转换成“提示”。

原项目中 AI 计算出动作后直接执行（翻出的棋子由人在实物棋盘上看到后录入）。
这里 AI 不再落子，只给出建议，由使用者决定实际怎么走、并录入真实结果。
"""

import time

from . import ai as ai_module
from .game import PIECE_NAMES, SIDE_NAMES
from .v16 import StrategicV16AI
from .v17 import StrategicV17AI

AI_MODELS = ['SimpleAI'] + [f'StrategicV{i}AI' for i in range(2, 18)]
DEFAULT_MODEL = 'StrategicV17AI'
# 每步默认思考时间（原版 V15 在浏览器中为 900ms，Python 较慢，给多一些）
DEFAULT_TIME_BUDGET_MS = 2500


def create_ai(model_name, side, time_budget_ms=DEFAULT_TIME_BUDGET_MS, search_params=None, v16_params=None,
              v17_params=None, engine='python', threads=1):
    if engine == 'go':
        if model_name not in ('StrategicV16AI', 'StrategicV17AI'):
            raise ValueError('Go 引擎目前支持 StrategicV16AI 和 StrategicV17AI')
        from .go_engine import GoEngineAI
        params = v17_params if model_name == 'StrategicV17AI' else v16_params
        return GoEngineAI(model_name, side, time_budget_ms, params, threads)
    if model_name == 'StrategicV16AI':
        player = StrategicV16AI(side, v16_params)
        player.time_budget_ms = time_budget_ms
        return player
    if model_name == 'StrategicV17AI':
        player = StrategicV17AI(side, v17_params)
        player.time_budget_ms = time_budget_ms
        return player
    cls = getattr(ai_module, model_name)
    player = cls(side)
    if hasattr(player, 'time_budget_ms'):
        player.time_budget_ms = time_budget_ms
        # 搜索参数只对迭代加深版（V15）生效，避免改动 V14 等早期版本的固定参数
        for key, value in (search_params or {}).items():
            if hasattr(player, key):
                setattr(player, key, value)
    return player


def pos_text(row, col):
    """坐标显示格式与原项目一致：(列,行)，从 0 开始。"""
    return f'({col},{row})'


def piece_text(piece):
    if not piece or not piece.get('face_up'):
        return '暗子'
    side = '红' if piece['side'] == 'red' else '黑'
    return f"{side}{PIECE_NAMES[piece['side']][piece['type']]}"


def describe_action(game, action):
    """把动作描述成一句中文提示（基于动作执行前的局面）。"""
    if not action:
        return '无可行动作'
    if action['type'] == 'flip':
        return f"翻开 {pos_text(action['row'], action['col'])} 的暗子"

    mover = game.get_piece(action['from_row'], action['from_col'])
    target = game.get_piece(action['to_row'], action['to_col'])
    src = pos_text(action['from_row'], action['from_col'])
    dst = pos_text(action['to_row'], action['to_col'])
    if action['type'] == 'move':
        return f'{piece_text(mover)} {src} → {dst}'
    if target and not target['face_up']:
        return f'{piece_text(mover)} {src} 打 {dst} 的暗子'
    return f'{piece_text(mover)} {src} 吃 {piece_text(target)} {dst}'


def describe_last_action(last_action):
    """描述 game.last_action（动作执行后的记录）。"""
    if not last_action:
        return ''
    t = last_action['type']
    if t == 'flip':
        p = last_action['piece']
        return f"翻开 {pos_text(last_action['row'], last_action['col'])} → {piece_text({**p, 'face_up': True})}"
    a = last_action.get('attacker') or last_action.get('piece')
    mover = piece_text({**a, 'face_up': True})
    src = pos_text(last_action['from']['row'], last_action['from']['col'])
    dst = pos_text(last_action['to']['row'], last_action['to']['col'])
    if t == 'move':
        return f'{mover} {src} → {dst}'
    d = last_action['defender']
    return f"{mover} {src} 吃 {piece_text({**d, 'face_up': True})} {dst}（-{d['hp']}）"


def compute_hint(game, model_name=DEFAULT_MODEL, time_budget_ms=DEFAULT_TIME_BUDGET_MS, search_params=None,
                 v16_params=None, v17_params=None, engine='python', threads=1, on_progress=None,
                 min_depth=0, should_stop=None):
    """为当前行动方计算一条提示。

    在对局的副本上搜索，原对局不受影响（可在后台线程调用）。
    返回 dict：action / text / side / model / elapsed_ms / info。
    on_progress(hint)：迭代加深的模型（V16 / V17，含 Go 引擎）每完成一层且深度 ≥ min_depth 时，
    用同样格式（info['partial'] 为 True）回调当前最佳着法，便于提前显示、之后随深度更新；
    min_depth 为 0 时不回调。should_stop()：返回 True 时提前结束计算（如局面已变化）。
    """
    side = game.current_turn
    work = game.clone()
    player = create_ai(model_name, side, time_budget_ms, search_params, v16_params, v17_params,
                       engine, threads)
    start = time.perf_counter()

    def make_hint(action, info):
        return {
            'action': action,
            'text': describe_action(game, action),
            'side': side,
            'side_name': SIDE_NAMES[side],
            'model': model_name,
            'elapsed_ms': (time.perf_counter() - start) * 1000,
            'info': info,
        }

    if hasattr(player, 'on_progress'):
        player.should_stop = should_stop
        if on_progress is not None and min_depth > 0:
            def progress(action, info):
                if action and info.get('depth', 0) >= min_depth:
                    on_progress(make_hint(action, info))
            player.on_progress = progress
    action = player.get_move(work)
    return make_hint(action, getattr(player, 'last_decision_info', None))
