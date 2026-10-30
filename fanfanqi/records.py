"""真人对局记录：画面同步时把每一局追加到 records/年月日-时.jsonl（按结束时间每小时一个文件，一行一局的 JSON；
不在 .gitignore 里，需要时自己提交）。

用途是积累真人对局，之后统计真人局面下胜率准不准、先手优势、对手选到引擎最佳着法的比例等
（见 README「用真实对局校准」）。只记局面和着法，不存截图（截图里有别的玩家的昵称和头像）。

每局（每行）的内容：
    started_at / ended_at   开始、结束时间
    start                   'new'（从开局看到）或 'import'（中途开始同步，start_state 是导入时的局面）
    start_state             开始记录时的局面（state.py 格式）
    moves                   每一步：side 谁走的、action 动作（含翻出 / 吃掉的子）、t 距开始的秒数、
                            hp 走后双方血量、screen_multiplier 画面倍率、hint 我方回合时给出的建议和校准胜率
    events                  中途发生的事：漏看几步后整体导入（import）、按画面纠正轮次（turn）
    my_side / room_factor   自己执哪方、抢红 / 加倍系数 F
    winner / result         胜方，以及站在自己一方的 win / loss / draw / unknown
    end_reason              结束原因（对局结束、出现结算画面、新对局开始、停止同步）

"""

import json
import os
import time

from .state import game_to_state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS_DIR = os.path.join(ROOT, 'records')


def _mover(snapshot_before):
    return snapshot_before['current_turn']


class GameRecorder:
    def __init__(self, on_status=None, directory=RECORDS_DIR):
        self.on_status = on_status or (lambda message: None)
        self.directory = directory
        self.record = None
        self._t0 = 0.0
        self._hints = {}  # 局面序号（history 长度）→ 我方回合的建议

    @property
    def active(self):
        return self.record is not None

    def start(self, game, kind, now=None):
        self._t0 = time.monotonic() if now is None else now
        self.record = {
            'version': 1,
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'start': kind,
            'start_state': game_to_state(game),
            'moves': [],
            'events': [],
        }
        self._hints = {}

    def note_hint(self, game, action, prob):
        """记下当前局面（我方回合）给出的建议，等这一步走完时附在这一步上。"""
        if self.record is not None and action:
            self._hints[len(game.history)] = {'action': action, 'win_prob': None if prob is None else round(prob, 4)}

    def step(self, game, old_len, screen_multiplier=None, now=None):
        """同步按动作推进了一到两步：history[old_len:] 是新走的步。"""
        if self.record is None:
            return
        t = round((time.monotonic() if now is None else now) - self._t0, 1)
        for i in range(max(1, old_len), len(game.history)):
            snapshot = game.history[i]
            move = {'side': _mover(game.history[i - 1]), 'action': snapshot['last_action'], 't': t,
                    'hp': dict(snapshot['hp']), 'screen_multiplier': screen_multiplier}
            hint = self._hints.pop(i, None)  # 给建议时 history 长度为 i，接下来走的正是第 i 步
            if hint:
                move['hint'] = hint
            self.record['moves'].append(move)

    def event(self, kind, game, now=None, **extra):
        if self.record is None:
            return
        t = round((time.monotonic() if now is None else now) - self._t0, 1)
        item = {'kind': kind, 't': t, **extra}
        if kind == 'import':
            item['state'] = game_to_state(game)
        self.record['events'].append(item)

    def finish(self, game, reason, my_side=None, room_factor=None, screen_multiplier=None):
        """把这一局追加到所在小时的文件（一行一局）；没有任何着法的记录（例如刚开始同步就停）不保存。
        返回文件路径或 None。"""
        record, self.record = self.record, None
        if record is None or not record['moves']:
            return None
        winner = game.winner if game is not None and game.phase == 'over' else None
        if winner is None or winner == 'draw' or not my_side:
            result = 'draw' if winner == 'draw' else 'unknown'
        else:
            result = 'win' if winner == my_side else 'loss'
        record.update({
            'ended_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'end_reason': reason,
            'my_side': my_side, 'room_factor': room_factor, 'final_screen_multiplier': screen_multiplier,
            'final_hp': dict(game.hp) if game is not None else None,
            'final_multiplier': game.get_multiplier() if game is not None else None,
            'winner': winner, 'result': result,
        })
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, time.strftime('%Y%m%d-%H') + '.jsonl')
        with open(path, 'a', encoding='utf-8') as f:  # 追加：同一小时里先保存的局不会被改动
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.on_status(f'对局记录已保存：{os.path.relpath(path, ROOT)}（{len(record["moves"])} 步，{result}）')
        return path
