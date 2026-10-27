"""AI 自对弈 / 测试工具。

翻棋和炮打暗子时，真实翻出的棋子从剩余池中随机抽取（模拟实物对局），
与 AI 搜索时假设的 pool_index 无关 —— 这正是“AI 只给提示”时的真实情况。

用法：
    python -m tests.selfplay                       # 规则测试 + AI 合法性 + V17 自对弈
    python -m tests.selfplay StrategicV17AI StrategicV16AI 10 0.3  # 新旧策略基准
"""

import random
import sys
import time

from fanfanqi.advisor import AI_MODELS, create_ai
from fanfanqi.game import Game, make_piece, other_side


def random_setup(game):
    while game.phase == 'setup':
        game.setup_placement(random.randrange(len(game.remaining_pool)))


def apply_real(game, action):
    """按动作执行，暗子的真实身份随机抽取。"""
    pool_index = random.randrange(len(game.remaining_pool)) if game.remaining_pool else None
    if action['type'] == 'flip':
        ok = game.flip_piece(action['row'], action['col'], pool_index)
    else:
        target = game.get_piece(action['to_row'], action['to_col'])
        idx = pool_index if target and not target['face_up'] else None
        ok = game.move_piece(action['from_row'], action['from_col'], action['to_row'], action['to_col'], idx)
    if not ok:
        raise AssertionError(f'非法动作: {action} (turn={game.current_turn})')


class RandomAI:
    def __init__(self, side):
        self.side = side

    def get_move(self, game):
        actions = game.get_all_actions()
        return random.choice(actions) if actions else None


def make_player(name, side, budget_ms):
    """name 可写成 StrategicV16AI:λ 或 StrategicV16AI:λ,κ（如 StrategicV16AI:0.5,1.5），
    冒号后为 V16 的 multiplier_exponent / loss_aversion；也接受旧写法 :reward / :win。"""
    if name == 'random':
        return RandomAI(side)
    model, _, objective = name.partition(':')
    strategy_params = None
    if objective in ('reward', 'win'):
        strategy_params = {'objective': objective}
    elif objective:
        lam, _, kappa = objective.partition(',')
        strategy_params = {'multiplier_exponent': float(lam)}
        if kappa:
            strategy_params['loss_aversion'] = float(kappa)
    if model == 'StrategicV17AI':
        return create_ai(model, side, budget_ms, v17_params=strategy_params)
    return create_ai(model, side, budget_ms, v16_params=strategy_params)


def play_game(red_name, black_name, budget_ms=300, max_plies=300):
    game = Game()
    random_setup(game)
    players = {'red': make_player(red_name, 'red', budget_ms), 'black': make_player(black_name, 'black', budget_ms)}
    plies = 0
    while game.phase == 'playing' and plies < max_plies:
        side = game.current_turn
        work = game.clone()
        action = players[side].get_move(work)
        if action is None:
            break
        apply_real(game, action)
        plies += 1
    multiplier = game.get_multiplier()['multiplier']
    if game.phase == 'over':
        return game.winner, plies, multiplier
    # 未分胜负：按 HP 判
    if game.hp['red'] == game.hp['black']:
        return 'draw', plies, multiplier
    return ('red' if game.hp['red'] > game.hp['black'] else 'black'), plies, multiplier


def benchmark(a, b, games, budget_ms):
    wins = {a: 0, b: 0, 'draw': 0}
    reward = {a: 0, b: 0}  # 赢得 +倍率，输得 −倍率
    start = time.time()
    for i in range(games):
        if i % 2 == 0:
            red, black = a, b
        else:
            red, black = b, a
        winner, plies, multiplier = play_game(red, black, budget_ms)
        name = 'draw' if winner == 'draw' else (red if winner == 'red' else black)
        if red == black and winner != 'draw':
            name = a if (winner == 'red') == (i % 2 == 0) else b
        wins[name] += 1
        if name != 'draw':
            loser = b if name == a else a
            reward[name] += multiplier
            reward[loser] -= multiplier
        print(f'  第{i + 1}局 红={red} 黑={black} → {winner}（{plies}步，倍率 {multiplier}）', flush=True)
    print(f'{a} 胜 {wins[a]}，{b} 胜 {wins[b]}，和 {wins["draw"]}；'
          f'倍率得分 {a} {reward[a]:+}，{b} {reward[b]:+}；用时 {time.time() - start:.0f}s')
    return wins


def run_tests():
    failures = 0

    def check(name, cond):
        nonlocal failures
        print(('  ✔ ' if cond else '  ✘ ') + name)
        if not cond:
            failures += 1

    print('规则测试')
    g = Game()
    soldier, king = make_piece('兵', 'red', True), make_piece('将', 'black', True)
    check('兵可吃将', g.can_eat(soldier, king))
    check('将不能吃兵', not g.can_eat(make_piece('将', 'red', True), make_piece('兵', 'black', True)))
    check('车吃马、马不能吃车', g.can_eat(make_piece('车', 'red', True), make_piece('马', 'black', True))
          and not g.can_eat(make_piece('马', 'red', True), make_piece('车', 'black', True)))
    check('非炮不能吃暗子', not g.can_eat(make_piece('车', 'red', True), g.board[0][0]))

    g = Game()
    g.phase = 'playing'
    g.board = [[None] * 4 for _ in range(8)]
    g.board[0][0] = make_piece('炮', 'red', True)
    g.board[2][0] = make_piece('兵', 'black', True)
    g.board[5][0] = make_piece('马', 'black', True)
    moves = g.get_valid_moves(0, 0)
    check('炮隔子跳吃', any(m['row'] == 5 and m['col'] == 0 and m['type'] == 'capture' for m in moves))
    check('炮不能空走', all(m['type'] == 'capture' for m in moves))

    g.hp['black'] = 5
    g.record_history()
    g.move_piece(0, 0, 5, 0)
    check('吃子扣血并判胜负', g.hp['black'] == 0 and g.phase == 'over' and g.winner == 'red')
    g.undo_steps(1)
    check('悔棋恢复', g.hp['black'] == 5 and g.phase == 'playing')

    print('AI 合法性（每个模型在开局后的局面返回合法动作）')
    base = Game()
    random_setup(base)
    for name in AI_MODELS:
        work = base.clone()
        t = time.perf_counter()
        action = create_ai(name, work.current_turn, 300).get_move(work)
        legal = action is not None and any(
            all(action.get(k) == a.get(k) for k in a) for a in base.get_all_actions())
        check(f'{name} 返回合法动作（{(time.perf_counter() - t) * 1000:.0f}ms）', legal)

    print('V17 完整自对弈（随机翻子）')
    for _ in range(2):
        winner, plies, _ = play_game('StrategicV17AI', 'StrategicV17AI', 150)
        check(f'对局完整结束：{winner}，{plies} 步', winner in ('red', 'black', 'draw') and plies > 0)

    print(f'\n{"全部通过" if failures == 0 else f"{failures} 项失败"}')
    return failures


if __name__ == '__main__':
    if len(sys.argv) > 5:
        random.seed(int(sys.argv[5]))
    if len(sys.argv) >= 3:
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        sec = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
        benchmark(sys.argv[1], sys.argv[2], n, sec * 1000)
    else:
        sys.exit(1 if run_tests() else 0)
