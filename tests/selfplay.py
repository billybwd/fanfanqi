"""AI 自对弈 / 测试工具。

翻棋和炮打暗子时，真实翻出的棋子从剩余池中随机抽取（模拟实物对局），
与 AI 搜索时假设的 pool_index 无关 —— 这正是“AI 只给提示”时的真实情况。

用法：
    python -m tests.selfplay                       # 规则测试 + AI 合法性 + V17 自对弈
    python -m tests.selfplay StrategicV17AI StrategicV16AI 10 0.3  # 新旧策略基准
"""

import glob
import os
import random
import sys
import time

from fanfanqi.advisor import AI_MODELS, create_ai
from fanfanqi.game import Game, make_piece, other_side
from fanfanqi.v16 import Position, Searcher as V16Searcher, merge_params
from fanfanqi.v17 import StrategicV17AI


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
    冒号后为 V16 的 multiplier_exponent / loss_aversion；也接受旧写法 :reward / :win。
    前缀 go/ 使用 Go 引擎（单线程），如 go/StrategicV17AI。
    末尾可加 +参数=值 覆盖单个参数（可多个），如 go/StrategicV17AI+search_ext=0+keep_tt=0。"""
    if name == 'random':
        return RandomAI(side)
    engine = 'python'
    if name.startswith('go/'):
        engine, name = 'go', name[3:]
    name, *overrides = name.split('+')
    model, _, objective = name.partition(':')
    strategy_params = None
    if objective in ('reward', 'win'):
        strategy_params = {'objective': objective}
    elif objective:
        lam, _, kappa = objective.partition(',')
        strategy_params = {'multiplier_exponent': float(lam)}
        if kappa:
            strategy_params['loss_aversion'] = float(kappa)
    for item in overrides:
        key, _, value = item.partition('=')
        strategy_params = strategy_params or {}
        strategy_params[key] = float(value) if '.' in value else int(value)
    if model == 'StrategicV17AI':
        return create_ai(model, side, budget_ms, v17_params=strategy_params, engine=engine)
    return create_ai(model, side, budget_ms, v16_params=strategy_params, engine=engine)


def play_game(red_name, black_name, budget_ms=300, max_plies=300, record=None):
    """record 为列表时，把每个走子前的局面（Game 副本）追加进去。"""
    game = Game()
    random_setup(game)
    players = {'red': make_player(red_name, 'red', budget_ms), 'black': make_player(black_name, 'black', budget_ms)}
    plies = 0
    while game.phase == 'playing' and plies < max_plies:
        side = game.current_turn
        work = game.clone()
        if record is not None:
            record.append(game.clone())
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
    positions = []
    for _ in range(2):
        winner, plies, _ = play_game('StrategicV17AI', 'StrategicV17AI', 150, record=positions)
        check(f'对局完整结束：{winner}，{plies} 步', winner in ('red', 'black', 'draw') and plies > 0)

    print('评估特征（Texel 拟合用）与 static_eval 一致')
    for name, searcher in (('V16', V16Searcher(merge_params(None))),
                           ('V17', StrategicV17AI('red').searcher())):
        worst = 0.0
        for game in positions:
            pos = Position.from_game(game)
            score = searcher.static_eval(pos)
            weights = searcher.term_weights()
            total = sum(weights[k] * v for k, v in searcher.eval_terms(pos).items())
            worst = max(worst, abs(total - score))
        check(f'{name}：{len(positions)} 个局面最大误差 {worst:.1e}', worst < 1e-9)

    go_parity_tests(check)
    vision_tests(check)
    print(f'\n{"全部通过" if failures == 0 else f"{failures} 项失败"}')
    return failures


def go_parity_tests(check):
    """Go 引擎关闭搜索增强（search_ext=0）、不保留置换表时，固定深度下与 Python V17 选同一步。没有 Go 时跳过。"""
    from fanfanqi.go_engine import GoEngineAI
    try:
        from fanfanqi.go_engine import _engine_binary
        _engine_binary()
    except RuntimeError as exc:
        print(f'\nGo 引擎：{exc}，跳过')
        return
    print('\nGo 引擎与 Python V17 一致（固定 2 层）')
    rng = random.Random(5)
    same = total = 0
    while total < 12:
        g = Game()
        random_setup(g)
        for _ in range(rng.randrange(4, 40)):
            if g.phase != 'playing':
                break
            apply_real(g, rng.choice(g.get_all_actions()))
        if g.phase != 'playing':
            continue
        go = GoEngineAI('StrategicV17AI', g.current_turn, 60000, {'max_depth': 2, 'search_ext': 0, 'keep_tt': 0})
        py = StrategicV17AI(g.current_turn, {'max_depth': 2})
        py.time_budget_ms = 60000
        total += 1
        same += go.get_move(g.clone()) == py.get_move(g.clone())
    check(f'{same}/{total} 个局面着法相同', same == total)


def vision_tests(check):
    """画面识别：需要 numpy 和 Pillow；有本地截图（sample_image/，不入库）时逐张识别并校验。"""
    try:
        from fanfanqi import vision
    except ImportError:
        print('\n画面识别：未安装 numpy / Pillow，跳过')
        return
    print('\n画面识别')
    board = [['?'] * 4 for _ in range(8)]
    for r, c in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)):
        board[r][c] = '.'
    result = vision.Recognition(layout=None, board=board,
                                captured={'red': {'兵': 3}, 'black': {'马': 1}})
    vision._fix_soldier_count(result, 'red')
    check('兵/卒数量角标按空格数修正（3 → 5）', result.captured['red']['兵'] == 5 and result.notes)
    sync_tests(check, vision)

    folder =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_image')
    paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not paths:
        print('  没有 sample_image/ 截图，跳过逐张识别')
        return
    from PIL import Image
    bad = []
    for path in paths:
        result = vision.recognize(Image.open(path))
        if result.problems:
            bad.append(f'{os.path.basename(path)}：{result.problems[0]}')
    check(f'{len(paths)} 张截图识别结果自洽' + ('' if not bad else '：' + bad[0]), not bad)


def sync_tests(check, vision):
    """画面同步：随机对局，每步（偶尔漏看一步）把真实局面当作识别结果交给 BoardSync，
    同步出的对局应与真实对局完全一致（含连击、开局翻棋分、长捉和未吃子计数）。"""
    from fanfanqi.sync import BoardSync, _captured, board_tokens

    def rec_of(game, turn=None, certain=True):
        return vision.Recognition(layout=None, board=board_tokens(game), captured=_captured(game),
                                  players={'bottom': 'red', 'top': 'black'},
                                  turn=turn or game.current_turn, turn_certain=certain)

    def fields(g):
        return (board_tokens(g), _captured(g), g.hp, g.combo, g.combo_score, g.flip_score,
                g.no_capture_plies, g.chase_count, g.chase_target, g.current_turn, g.phase)

    rng = random.Random(7)
    kinds, same = {}, True
    for _ in range(8):
        real = Game()
        random_setup(real)
        sync, mine, now, skipped = BoardSync(), None, 0.0, False
        for _ in range(150):
            if real.phase != 'playing':
                break
            skip = mine is not None and not skipped and rng.random() < 0.2
            skipped = skip
            if not skip:
                now += 2
                update = sync.update(rec_of(real), mine, now=now)
                kinds[update.kind] = kinds.get(update.kind, 0) + 1
                mine = update.game or mine
                same = same and fields(mine) == fields(real)
            apply_real(real, rng.choice(real.get_all_actions()))
    check(f'画面同步逐步推进与真实对局一致（{kinds}）', same and set(kinds) == {'new', 'step'})

    # 轮次：刚走完一步时画面轮次与轮流走子不符先不改，稳定后以画面为准
    sync = BoardSync()
    game = Game()
    random_setup(game)
    game = sync.update(rec_of(game), None, now=0.0).game
    other = 'black' if game.current_turn == 'red' else 'red'
    lagging = sync.update(rec_of(game, turn=other), game, now=1.0)
    corrected = sync.update(rec_of(game, turn=other), game, now=3.0)
    uncertain = sync.update(rec_of(game, turn=other, certain=False), game, now=9.0)
    check('画面轮次：走子后 1.5 秒内不改、之后按画面纠正、无直接依据时不改',
          lagging.kind == 'same' and corrected.kind == 'turn' and corrected.game.current_turn == other
          and uncertain.kind == 'same')

    # 画面倍率 = F × 倍率：从开局跟踪时反推出 F，且不会把逐步推进得到的翻棋分/连击改错
    def with_mult(rec, game, factor):
        rec.multiplier = factor * game.get_multiplier()['multiplier']
        return rec

    ok, combos = True, 0
    for i in range(6):
        factor = (1, 2, 4)[i % 3]
        real = Game()
        random_setup(real)
        sync, mine = BoardSync(), None
        for step in range(150):
            if real.phase != 'playing':
                break
            update = sync.update(with_mult(rec_of(real), real, factor), mine, now=2.0 * step)
            mine = update.game or mine
            ok = ok and fields(mine) == fields(real) and sync.room_factor == factor
            combos += real.get_multiplier()['combo_factor'] > 1
            apply_real(real, rng.choice(real.get_all_actions()))
    check(f'画面倍率：开局反推系数 F，跟踪中不误改翻棋分/连击（{combos} 步带连击）', ok)

    # 中途导入：翻棋分、连击按 0 导入，候选解跨帧取交集，收敛到唯一解时补上；补上的值不能错
    solved = wrong = total = 0
    for i in range(30):
        factor = (1, 2, 4)[i % 3]
        real = Game()
        random_setup(real)
        for _ in range(rng.randrange(20, 60)):
            if real.phase != 'playing':
                break
            apply_real(real, rng.choice(real.get_all_actions()))
        if real.phase != 'playing':
            continue
        sync, mine, total = BoardSync(), None, total + 1
        for step in range(40):
            if real.phase != 'playing':
                break
            update = sync.update(with_mult(rec_of(real), real, factor), mine, now=2.0 * step)
            mine = update.game or mine
            apply_real(real, rng.choice(real.get_all_actions()))
        shown = factor * real.get_multiplier()['multiplier']
        if sync.room_factor not in (None, factor) or mine.flip_score not in (0, real.flip_score):
            wrong += 1
        elif sync.trusted and sync.room_factor == factor:
            ok_now = mine.get_multiplier()['multiplier'] * factor == shown or real.phase != 'playing'
            solved += mine.flip_score == real.flip_score and ok_now
    check(f'画面倍率：中途导入后跨帧反推，{solved}/{total} 局完全还原 F 和翻棋分，没有解错', wrong == 0)


if __name__ == '__main__':
    if len(sys.argv) > 5:
        random.seed(int(sys.argv[5]))
    if len(sys.argv) >= 3:
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        sec = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
        benchmark(sys.argv[1], sys.argv[2], n, sec * 1000)
    else:
        sys.exit(1 if run_tests() else 0)
