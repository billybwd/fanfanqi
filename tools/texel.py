"""Texel 调参：用自对弈局面的最终胜负拟合 V16/V17 评估函数的权重。

V16/V17 的评估分是各评估项的线性组合：score = Σ 权重 × 特征（见 Searcher.eval_terms），
搜索里把它换算成胜率 p = sigmoid(score / win_temperature)。这里固定 win_temperature，
用逻辑回归让 p 贴近实际胜负（赢 1 / 和 0.5 / 输 0），拟合出的系数就是新权重。

- 只用“安静”局面（轮到走的一方没有吃子可走、未分胜负），否则评估分会被一步吃子打乱；
- 按对局划分训练集 / 验证集，报告两者的对数损失，验证集变好才值得采用；
- L2 正则把权重拉向当前值，数据少时不至于跑偏；
- 'hanging' / 'hanging_one' / 'pool' 的特征值里含当前子力值，子力值变化较大时拟合两轮
  （--apply 写回配置后再跑一次）结果更准。

用法：
    python -m tools.texel data/selfplay.jsonl
    python -m tools.texel data/*.jsonl --model StrategicV17AI --l2 1e-3 --apply
"""

import argparse
import glob
import json
import random

import numpy as np

from fanfanqi.config import load_config, save_config
from fanfanqi.state import game_from_state
from fanfanqi.v16 import Position, Searcher as V16Searcher, merge_params
from fanfanqi.v17 import StrategicV17AI


def make_searcher(model, config):
    if model == 'StrategicV16AI':
        return V16Searcher(merge_params(config['v16']))
    return StrategicV17AI('red', config['v17']).searcher()


def load_positions(paths, searcher):
    """返回 (特征名列表, 特征矩阵, 标签, 对局 id 列表)，只保留安静局面。"""
    names = sorted(searcher.term_weights())
    rows, labels, games = [], [], []
    skipped = 0
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                try:
                    pos = Position.from_game(game_from_state(sample['state']))
                except (KeyError, ValueError):
                    skipped += 1
                    continue
                if pos.winner >= 0 or pos.captures():
                    continue
                terms = searcher.eval_terms(pos)
                rows.append([terms.get(n, 0.0) for n in names])
                labels.append((sample['result'] + 1) / 2)
                games.append(sample.get('game', path))
    if skipped:
        print(f'跳过 {skipped} 个无法解析的局面')
    return names, np.array(rows, dtype=float), np.array(labels, dtype=float), games


def log_loss(x, y, w, temperature):
    p = 1 / (1 + np.exp(-np.clip(x @ w / temperature, -30, 30)))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(x, y, w0, temperature, l2, free, iterations=50):
    """带 L2（拉向 w0）的牛顿法逻辑回归；free 为可调权重的布尔掩码。"""
    w = w0.copy()
    n = len(y)

    def objective(w):
        return log_loss(x, y, w, temperature) + l2 * float(np.sum((w - w0) ** 2))

    current = objective(w)
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ w / temperature, -30, 30)))
        grad = x.T @ (p - y) / (n * temperature) + 2 * l2 * (w - w0)
        hess = (x.T * (p * (1 - p))) @ x / (n * temperature ** 2) + 2 * l2 * np.eye(len(w))
        g, h = grad[free], hess[np.ix_(free, free)]
        step = np.zeros_like(w)
        step[free] = np.linalg.solve(h + 1e-9 * np.eye(len(g)), g)
        scale = 1.0
        while scale > 1e-4:
            candidate = w - scale * step
            value = objective(candidate)
            if value <= current:
                break
            scale /= 2
        else:
            break
        improvement = current - value
        w, current = candidate, value
        if improvement < 1e-10:
            break
    return w


def to_config(names, w, section):
    """拟合结果写回配置段（v16 / v17）的形式。"""
    result = json.loads(json.dumps(section))
    weights = result['weights']
    for name, value in zip(names, w):
        value = round(float(value), 4)
        if name.startswith('piece_value.'):
            weights['piece_value'][name.split('.', 1)[1]] = value
        elif name == 'cannon_endgame_value':
            result['cannon_endgame_value'] = value
        else:
            weights[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description='Texel 调参：用自对弈胜负拟合评估权重')
    parser.add_argument('data', nargs='+', help='gen_data 生成的 JSONL（支持通配符）')
    parser.add_argument('--model', default='StrategicV17AI', choices=('StrategicV16AI', 'StrategicV17AI'))
    parser.add_argument('--l2', type=float, default=1e-3, help='把权重拉向当前值的 L2 强度')
    parser.add_argument('--val', type=float, default=0.2, help='验证集对局比例')
    parser.add_argument('--fix', default='tempo', help='保持不变的权重，逗号分隔（如 tempo,hp）')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', help='把新配置段写到这个 JSON 文件')
    parser.add_argument('--apply', action='store_true', help='写回 config.json（验证集损失变差时拒绝）')
    args = parser.parse_args()

    paths = sorted({p for pattern in args.data for p in glob.glob(pattern)})
    if not paths:
        parser.error('没有找到数据文件')
    config = load_config()
    key = 'v16' if args.model == 'StrategicV16AI' else 'v17'
    searcher = make_searcher(args.model, config)
    temperature = searcher.p['win_temperature']
    names, x, y, games = load_positions(paths, searcher)
    if len(y) < 1000:
        print(f'警告：只有 {len(y)} 个安静局面，拟合结果可能不稳定（建议 2 万以上）')

    game_ids = sorted(set(games))
    random.Random(args.seed).shuffle(game_ids)
    val_games = set(game_ids[:int(len(game_ids) * args.val)])
    val = np.array([g in val_games for g in games])
    train = ~val

    weights = searcher.term_weights()
    w0 = np.array([weights[n] for n in names], dtype=float)
    fixed = {n.strip() for n in args.fix.split(',') if n.strip()}
    free = np.array([n not in fixed for n in names])
    w = fit(x[train], y[train], w0, temperature, args.l2, free)

    base_rate = float(np.clip(np.mean(y[train]), 1e-9, 1 - 1e-9))
    const = -float(np.mean(y[val] * np.log(base_rate) + (1 - y[val]) * np.log(1 - base_rate)))
    before_val = log_loss(x[val], y[val], w0, temperature)
    after_val = log_loss(x[val], y[val], w, temperature)
    print(f'{len(paths)} 个文件，{len(game_ids)} 局，安静局面 训练 {int(train.sum())} / 验证 {int(val.sum())}')
    print(f'训练集损失 {log_loss(x[train], y[train], w0, temperature):.4f} → {log_loss(x[train], y[train], w, temperature):.4f}')
    print(f'验证集损失 {before_val:.4f} → {after_val:.4f}（只猜平均胜率：{const:.4f}）')
    print(f'\n{"权重":<24}{"当前":>10}{"拟合":>10}')
    for name, old, new in zip(names, w0, w):
        mark = '' if name not in fixed else '  (固定)'
        print(f'{name:<24}{old:>10.3f}{new:>10.3f}{mark}')

    section = to_config(names, w, config[key])
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(section, f, ensure_ascii=False, indent=2)
        print(f'\n已写出 {args.out}')
    if args.apply:
        if after_val >= before_val:
            print('\n验证集损失没有下降，不写回 config.json')
        else:
            config[key] = section
            save_config(config)
            print(f'\n已写回 config.json 的 {key} 段；请用自对弈对比新旧权重后再决定是否保留')


if __name__ == '__main__':
    main()
