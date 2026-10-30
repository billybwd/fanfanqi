"""按对局胜负调参：SPSA 自动寻找参数，SPRT 判断新参数是否真的更强。

两个子命令，对局都用 Go 引擎、StrategicV17AI，每对局面同一个随机种子交换先后手各下一局：

sprt  A 对 B 序贯检验：每批对局后算对数似然比（LLR），越过上界判 A 更强（H1），越过下界判没有更强（H0），
      都没越过就继续，直到 --max-games。比固定局数省时间：差距明显时几百局就停，接近的才跑满。
      H0：A 比 B 强 elo0（默认 0）；H1：强 elo1（默认 20 Elo，约等于胜率 53%）；α = β = 0.05。

    python -m tools.tune sprt data/candidate.json current --think 1.0
    python -m tools.tune sprt current+star_margin=0.5 current

spsa  同时调多个参数：每对对局给所有参数加一个随机符号的扰动 ±c，θ+ 对 θ−；哪边赢，参数就往哪边挪
      lr × c（每个参数自己的 c）。并行时一批里的各对各自抽扰动，批末一起更新。参数最后取后一半迭代的
      平均值（比最后一刻的值噪声小）。日志每批一行（JSONL），中断后用同样的命令会从日志续跑。

    python -m tools.tune spsa tools/tune_params.json --pairs 6000 --out data/spsa
    → data/spsa.jsonl（日志）和 data/spsa.json（结果参数，可直接用于 sprt 或写进 config.json 的 v17 段）

参数写法（A / B 和 spsa 的 base）：current = config.json 里的 v17 段；也可以是 JSON 文件路径；
都可以在后面加 +键=值 覆盖单个参数，嵌套键用点号，如 current+weights.piece_value.兵=1.0。

spsa 的参数说明文件见 tools/tune_params.json：每项 name（点号路径）、c（扰动大小，一般取“改这么多
应该能看出强弱”的量）、min / max（取值范围），可选 start（默认取 base 里的当前值）。
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- 参数
def get_path(params, path):
    node = params
    for key in path.split('.'):
        node = node[key]
    return node


def set_path(params, path, value):
    keys = path.split('.')
    node = params
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_params(spec):
    """current / 文件路径，后面可跟 +键=值。"""
    from fanfanqi.config import load_config
    base, *overrides = spec.split('+')
    if base == 'current':
        params = copy.deepcopy(load_config()['v17'])
    else:
        with open(base if os.path.isabs(base) else os.path.join(ROOT, base), encoding='utf-8') as f:
            params = json.load(f)
    for item in overrides:
        key, _, value = item.partition('=')
        set_path(params, key, json.loads(value))
    return params


# ---------------------------------------------------------------- 对局
def play(red, black, budget_ms, seed, max_plies=300):
    """red / black 是 V17 参数；返回红方得分（胜 1、和 0.5、负 0）。翻出什么由 seed 决定，交换先后手时局面相同。"""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from fanfanqi.advisor import create_ai
    from fanfanqi.game import Game
    from tests.selfplay import apply_real, random_setup

    random.seed(seed)
    game = Game()
    random_setup(game)
    players = {side: create_ai('StrategicV17AI', side, budget_ms, v17_params=params, engine='go')
               for side, params in (('red', red), ('black', black))}
    for _ in range(max_plies):
        if game.phase != 'playing':
            break
        action = players[game.current_turn].get_move(game.clone())
        if action is None:
            break
        apply_real(game, action)
    winner = game.winner if game.phase == 'over' else (
        'draw' if game.hp['red'] == game.hp['black'] else ('red' if game.hp['red'] > game.hp['black'] else 'black'))
    return {'red': 1.0, 'black': 0.0}.get(winner, 0.5)


def for_games(params, tt_bits):
    """对局用的参数：置换表改小（每步不到 1 秒用不满默认的 2^21 个桶；十几个引擎进程同时跑时，
    大表会把内存吃满、开始换页，分配 / 清空一张表就要好几秒）。"""
    return {**params, 'tt_bits': tt_bits}


def play_pair(job):
    """一对对局：A 先执红、再执黑（同一种子）。返回 A 的两局得分；引擎出错时（引擎会自动重启）重下一次，
    仍出错返回 None，由调用方跳过这一对，不让几个小时的调参因为一次意外中断。"""
    a, b, budget_ms, seed = job
    for attempt in range(2):
        try:
            return play(a, b, budget_ms, seed), 1.0 - play(b, a, budget_ms, seed)
        except RuntimeError as exc:
            print(f'种子 {seed} 的对局出错（{"重下" if attempt == 0 else "跳过"}）：{str(exc)[:200]}', flush=True)
    return None


def keep_awake():
    """Windows 上调参期间阻止系统待机（现代待机会把进程降速到几乎停住，引擎请求随之超时）；进程结束后自动失效。
    合上笔记本盖子等强制睡眠仍然会生效。"""
    if os.name != 'nt':
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)


# ---------------------------------------------------------------- SPRT
def elo_to_score(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def llr(wins, draws, losses, elo0, elo1):
    """三项分布（胜/和/负）下 H1 对 H0 的对数似然比，正态近似（与常见引擎测试框架相同的 GSPRT 公式）。"""
    n = wins + draws + losses
    if n == 0 or wins + draws == 0 or losses + draws == 0:
        return 0.0
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score ** 2) / n
    if var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2 * score - s0 - s1) / (2 * var)


def elo_estimate(wins, draws, losses):
    n = wins + draws + losses
    score = min(max((wins + 0.5 * draws) / max(1, n), 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def cmd_sprt(args):
    a, b = load_params(args.a), load_params(args.b)
    lower, upper = math.log(args.beta / (1 - args.alpha)), math.log((1 - args.beta) / args.alpha)
    wins = draws = losses = 0
    pair, start = 0, time.time()
    verdict = None
    with ProcessPoolExecutor(args.workers) as pool:
        while verdict is None and wins + draws + losses < args.max_games:
            jobs = [(for_games(a, args.tt_bits), for_games(b, args.tt_bits), args.think * 1000, args.seed + pair + i)
                    for i in range(args.workers)]
            pair += len(jobs)
            for scores in pool.map(play_pair, jobs):
                for s in scores or ():
                    wins, draws, losses = wins + (s == 1.0), draws + (s == 0.5), losses + (s == 0.0)
            ratio = llr(wins, draws, losses, args.elo0, args.elo1)
            n = wins + draws + losses
            print(f'{n:5d} 局  A 胜 {wins} 和 {draws} 负 {losses}  Elo {elo_estimate(wins, draws, losses):+.1f}  '
                  f'LLR {ratio:+.2f} ∈ ({lower:.2f}, {upper:.2f})  {time.time() - start:.0f}s', flush=True)
            if ratio >= upper:
                verdict = f'H1：A 更强（至少约 {args.elo1:g} Elo 的假设成立）'
            elif ratio <= lower:
                verdict = f'H0：A 没有比 B 强 {args.elo1:g} Elo 以上'
    print('结论：' + (verdict or f'到 {args.max_games} 局仍无定论（差距小于检测精度）'))


# ---------------------------------------------------------------- SPSA
def cmd_spsa(args):
    with open(args.spec, encoding='utf-8') as f:
        spec = json.load(f)
    base = load_params(spec.get('base', 'current'))
    names = [p['name'] for p in spec['params']]
    theta = {p['name']: float(p.get('start', get_path(base, p['name']))) for p in spec['params']}
    log_path, out_path = args.out + '.jsonl', args.out + '.json'
    history, done = [], 0
    if os.path.exists(log_path):  # 续跑
        with open(log_path, encoding='utf-8') as f:
            history = [json.loads(line) for line in f if line.strip()]
        if history:
            theta, done = history[-1]['theta'], history[-1]['pairs']
            print(f'从日志续跑：已完成 {done} 对')

    def with_theta(values):
        params = copy.deepcopy(base)
        for name, value in values.items():
            set_path(params, name, value)
        return params

    def clamp(name, value):
        p = next(p for p in spec['params'] if p['name'] == name)
        return min(p.get('max', math.inf), max(p.get('min', -math.inf), value))

    rng = random.Random(args.seed + done)
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        while done < args.pairs:
            batch = min(args.workers, args.pairs - done)
            deltas = [{n: rng.choice((-1, 1)) for n in names} for _ in range(batch)]
            jobs = []
            for k, delta in enumerate(deltas):
                plus = {n: clamp(n, theta[n] + p['c'] * delta[n]) for n, p in zip(names, spec['params'])}
                minus = {n: clamp(n, theta[n] - p['c'] * delta[n]) for n, p in zip(names, spec['params'])}
                jobs.append((for_games(with_theta(plus), args.tt_bits), for_games(with_theta(minus), args.tt_bits),
                             args.think * 1000, args.seed + done + k))
            results = list(pool.map(play_pair, jobs))
            deltas = [d for d, r in zip(deltas, results) if r is not None]
            results = [r for r in results if r is not None]
            for delta, (s1, s2) in zip(deltas, results):
                r = (s1 + s2) - 1.0  # θ+ 两局得分减去 θ− 两局得分，再除以 2：-1 ~ 1
                for n, p in zip(names, spec['params']):
                    theta[n] = clamp(n, theta[n] + args.lr * p['c'] * r * delta[n])
            done += batch
            entry = {'pairs': done, 'theta': dict(theta),
                     'score': sum(s1 + s2 for s1, s2 in results) / (2 * max(1, len(results)))}
            history.append(entry)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            if len(history) % 10 == 0 or done >= args.pairs:
                rate = done / max(1e-9, time.time() - start)
                print(f'{done}/{args.pairs} 对  ' + '  '.join(f'{n.split(".")[-1]}={theta[n]:.3f}' for n in names)
                      + f'  （{rate * 3600:.0f} 对/小时）', flush=True)

    # 结果：后一半迭代的平均值
    tail = history[len(history) // 2:] or history
    final = {n: sum(h['theta'][n] for h in tail) / len(tail) for n in names}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(with_theta({n: round(v, 4) for n, v in final.items()}), f, ensure_ascii=False, indent=2)
    print('结果（后一半平均）：' + '  '.join(f'{n}={get_path(base, n)}→{final[n]:.3f}' for n in names))
    print(f'已写出 {out_path}；验收：python -m tools.tune sprt {os.path.relpath(out_path, ROOT)} current --think 1.0')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sprt', help='A 对 B 序贯检验')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--think', type=float, default=0.3, help='每步思考秒数')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--tt-bits', type=int, default=18, help='对局用的置换表大小（2 的幂，默认 18 ≈ 6MB）')
    p.add_argument('--elo0', type=float, default=0.0)
    p.add_argument('--elo1', type=float, default=20.0)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--max-games', type=int, default=6000)
    p.add_argument('--seed', type=int, default=100000)
    p.set_defaults(func=cmd_sprt)
    p = sub.add_parser('spsa', help='按对局胜负同时调多个参数')
    p.add_argument('spec', help='参数说明 JSON（见 tools/tune_params.json）')
    p.add_argument('--pairs', type=int, default=6000, help='对局对数（每对两局）')
    p.add_argument('--lr', type=float, default=0.02, help='每对对局的步长，单位是各参数的 c（越大越快、噪声越大）')
    p.add_argument('--think', type=float, default=0.3)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--tt-bits', type=int, default=18, help='对局用的置换表大小（2 的幂，默认 18 ≈ 6MB）')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', default=os.path.join('data', 'spsa'))
    p.set_defaults(func=cmd_spsa)
    args = parser.parse_args()
    keep_awake()
    args.func(args)


if __name__ == '__main__':
    main()
