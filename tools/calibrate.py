"""胜率校准与开局（抢红 / 加倍）分析，数据来自 tools/gen_data.py 的自对弈 JSONL。

1. eval：用 Go 引擎按固定深度评估一批局面（全部开局 + 随机抽样的中后盘局面），记下引擎胜率等，写 JSONL。
       python -m tools.calibrate eval data/round1.jsonl --out data/calib_eval.jsonl --depth 5 --workers 2
2. fit：拟合校准胜率 = sigmoid(a·搜索值/倍率 + b·领先血量/30 + c·…)，并统计开局先手胜率，写出
   fanfanqi/calibration.json（界面读取，用来显示校准胜率、给抢红 / 加倍建议）。
       python -m tools.calibrate fit data/calib_eval.jsonl

固定深度而不是固定时间：结果与机器负载无关，可以和别的任务同时跑。
"""

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fanfanqi.calibration import CALIBRATION_PATH, features, probability  # noqa: E402


def _evaluate(job):
    row, depth = job
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from fanfanqi.config import load_config
    from fanfanqi.go_engine import GoEngineAI
    from fanfanqi.state import game_from_state
    try:
        game = game_from_state(row['state'])
    except ValueError:
        return None
    if game.phase != 'playing':
        return None
    params = {**load_config()['v17'], 'max_depth': depth, 'keep_tt': 0, 'tt_bits': 18}
    ai = GoEngineAI('StrategicV17AI', game.current_turn, 600000, params)
    ai.get_move(game)
    info = ai.last_decision_info or {}
    if 'win_prob' not in info:
        return None
    s = row['state']
    me, op = s['turn'], ('black' if s['turn'] == 'red' else 'red')
    return {
        'game': row['game'], 'ply': row['ply'], 'turn': me, 'result': row['result'],
        'hp_lead': s['hp'][me] - s['hp'][op],
        'hidden': sum(r.count('?') for r in s['board']),
        'win_prob': info['win_prob'], 'score': info.get('score'), 'depth': info.get('depth'),
        'multiplier': max(1.0, float(info.get('multiplier') or s['multiplier']['multiplier'])),
        'board': s['board'] if row['ply'] == 0 else None,
    }


def cmd_eval(args):
    rows = []
    for path in args.data:
        with open(path, encoding='utf-8') as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rng = random.Random(args.seed)
    openings = [r for r in rows if r['ply'] == 0]
    others = [r for r in rows if r['ply'] > 0]
    picked = openings + rng.sample(others, min(args.sample, len(others)))
    done = 0
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, 'w', encoding='utf-8') as out:
        for item in pool.map(_evaluate, [(r, args.depth) for r in picked], chunksize=8):
            done += 1
            if item:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
            if done % 500 == 0:
                print(f'{done}/{len(picked)}', flush=True)
    print(f'已写出 {args.out}')


# ---------------------------------------------------------------- 拟合
def calibrated(model, score, multiplier, hp_lead, hidden):
    return probability(model, score, multiplier, hp_lead, hidden)


def fit_logistic(xs, ys, l2=1e-3, iterations=60):
    """牛顿法逻辑回归（特征少，直接解）。ys 为 1 胜 / 0 负（和棋按 0.5）。"""
    import numpy as np
    x, y = np.array(xs), np.array(ys)
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - y) + l2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def log_loss(ps, ys):
    return -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)) for p, y in zip(ps, ys)) / len(ys)


def reliability(ps, ys, bins=10):
    rows = []
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        sel = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi or (k == bins - 1 and p == 1.0)]
        if sel:
            rows.append((lo, hi, len(sel), sum(p for p, _ in sel) / len(sel), sum(y for _, y in sel) / len(sel)))
    return rows


def cmd_fit(args):
    with open(args.data, encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]
    games = sorted({it['game'] for it in items})
    rng = random.Random(1)
    val_games = set(rng.sample(games, len(games) // 5))
    train = [it for it in items if it['game'] not in val_games]
    val = [it for it in items if it['game'] in val_games]
    label = lambda it: (it['result'] + 1) / 2  # noqa: E731

    w = fit_logistic([features(it['score'], it['multiplier'], it['hp_lead'], it['hidden']) for it in train],
                     [label(it) for it in train])
    model = {'weights': [float(v) for v in w]}
    raw = [it['win_prob'] for it in val]
    cal = [calibrated(model, it['score'], it['multiplier'], it['hp_lead'], it['hidden']) for it in val]
    ys = [label(it) for it in val]
    print(f'训练 {len(train)} 个局面，验证 {len(val)} 个（按对局划分）')
    print(f'对数损失：引擎胜率 {log_loss(raw, ys):.4f} → 校准后 {log_loss(cal, ys):.4f}（越小越准；全猜 50% 为 0.6931）')
    for name, ps in (('引擎胜率', raw), ('校准胜率', cal)):
        print(f'{name}：预测区间 | 局面数 | 平均预测 | 实际胜率')
        for lo, hi, n, mp, act in reliability(ps, ys):
            print(f'  {lo:.0%}~{hi:.0%} | {n:5d} | {mp:6.1%} | {act:6.1%}')
    print('系数（常数、搜索值/倍率、领先/30、领先/30 × 暗子比例）：', [round(v, 3) for v in model['weights']])

    # 开局：先手（红方）胜率，以及按校准胜率分档后的实际红方胜率
    openings = [it for it in items if it['ply'] == 0]
    red_wins = [label(it) for it in openings]
    base = sum(red_wins) / len(red_wins)
    se = math.sqrt(base * (1 - base) / len(red_wins))
    print(f'开局 {len(openings)} 局：红方（先手）实际胜率 {base:.1%} ± {se:.1%}')
    cal_open = [calibrated(model, it['score'], it['multiplier'], 0, it['hidden']) for it in openings]
    for lo, hi in ((0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.01)):
        sel = [y for p, y in zip(cal_open, red_wins) if lo <= p < hi]
        if sel:
            print(f'  校准红方胜率 {lo:.0%}~{min(hi, 1):.0%}：{len(sel)} 局，实际红方胜率 {sum(sel) / len(sel):.1%}')
    # 开局单独一个模型：开局时引擎评估的区分力很弱（交叉验证见下），主要是先手的平均胜率，再按评估略作调整
    import numpy as np
    xo = np.array([[1.0, features(it['score'], it['multiplier'], 0, 28)[1]] for it in openings])
    yo = np.array(red_wins)
    order = list(range(len(openings)))
    random.Random(0).shuffle(order)
    folds = [order[k::5] for k in range(5)]
    base_ll, model_ll = [], []
    for fold in folds:
        test = set(fold)
        train_idx = [i for i in order if i not in test]
        p0 = yo[train_idx].mean()
        base_ll.append(log_loss([p0] * len(fold), yo[fold]))
        wo = fit_logistic(xo[train_idx], yo[train_idx])
        model_ll.append(log_loss(list(1 / (1 + np.exp(-xo[fold] @ wo))), yo[fold]))
    wo = fit_logistic(xo, yo)
    print(f'开局 5 折交叉验证对数损失：只用先手平均胜率 {sum(base_ll) / 5:.4f}，加上引擎开局评估 {sum(model_ll) / 5:.4f}')
    model.update({
        'opening_weights': [float(v) for v in wo],
        'source': os.path.basename(args.data), 'positions': len(items),
        'red_first_win_rate': round(base, 4), 'red_first_win_rate_se': round(se, 4),
    })
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print(f'已写出 {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('eval')
    p.add_argument('data', nargs='+')
    p.add_argument('--out', default=os.path.join('data', 'calib_eval.jsonl'))
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--sample', type=int, default=4000, help='开局之外再随机抽多少个局面')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=1)
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser('fit')
    p.add_argument('data')
    p.add_argument('--out', default=CALIBRATION_PATH)
    p.set_defaults(func=cmd_fit)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
