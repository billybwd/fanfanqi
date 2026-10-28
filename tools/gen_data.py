"""自对弈生成训练样本：每个局面一行 JSONL（状态 + AI 选择 + 最终胜负）。

样本只含当前局面的状态（见 fanfanqi/state.py），不需要整盘棋谱；
训练时用 fanfanqi.state.encode_features(game_from_state(...)) 转成 636 维特征，
或用 tools/texel.py 按 V16/V17 的评估项拟合权重。

AI 参数默认读 config.json 的 v16 / v17 段（与界面一致），拟合出新权重并写回配置后，
下一批数据自动用新权重生成。

用法：
    python -m tools.gen_data --games 50 --model StrategicV17AI --think 0.3 --out data/selfplay.jsonl
    python -m tools.gen_data --games 1000 --engine go --workers 10 --out data/selfplay.jsonl
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from fanfanqi.advisor import create_ai
from fanfanqi.config import load_config
from fanfanqi.game import Game
from fanfanqi.state import FEATURE_VERSION, STATE_VERSION, game_to_state
from tests.selfplay import apply_real, random_setup


def action_record(action):
    keys = ('type', 'row', 'col', 'from_row', 'from_col', 'to_row', 'to_col')
    return {k: action[k] for k in keys if k in action}


def model_params(model):
    """config.json 中该模型的参数（V16 / V17），其他模型返回 None。"""
    config = load_config()
    if model == 'StrategicV16AI':
        return {'v16_params': config['v16']}
    if model == 'StrategicV17AI':
        return {'v17_params': config['v17']}
    return {}


def play_and_record(model, budget_ms, epsilon, max_plies=300, engine='python', threads=1):
    params = model_params(model)
    game = Game()
    random_setup(game)
    samples = []
    while game.phase == 'playing' and len(samples) < max_plies:
        state = game_to_state(game)
        if random.random() < epsilon:
            # 少量随机着法，让数据覆盖 AI 自己不会走到的局面
            action = random.choice(game.get_all_actions())
        else:
            action = create_ai(model, game.current_turn, budget_ms, engine=engine, threads=threads,
                               **params).get_move(game.clone())
        if action is None:
            break
        samples.append({'state': state, 'move': action_record(action)})
        apply_real(game, action)

    winner = game.winner
    if winner is None and game.hp['red'] != game.hp['black']:
        winner = 'red' if game.hp['red'] > game.hp['black'] else 'black'
    final_multiplier = game.get_multiplier()['multiplier']
    for i, s in enumerate(samples):
        side = s['state']['turn']
        # result：从“轮到走的一方”看的最终结果，赢 1 / 输 -1 / 和 0
        s['result'] = 0 if winner is None else (1 if winner == side else -1)
        # reward：训练的最终评分 = 胜负 × 终局倍率（同样从轮到走的一方看）
        s['reward'] = s['result'] * final_multiplier
        s['final_multiplier'] = final_multiplier
        s['ply'] = i
        s['final_hp'] = dict(game.hp)
        s['model'] = model
        s['engine'] = engine
        s['state_version'] = STATE_VERSION
        s['feature_version'] = FEATURE_VERSION
    return samples


def _play_one(seed, model, budget_ms, epsilon, engine, threads):
    """进程池里跑一局：每局独立种子，结果可复现（Go 引擎的计时除外）。"""
    random.seed(seed)
    return play_and_record(model, budget_ms, epsilon, engine=engine, threads=threads)


def main():
    parser = argparse.ArgumentParser(description='自对弈生成训练样本（JSONL）')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--model', default='StrategicV17AI')
    parser.add_argument('--think', type=float, default=0.3, help='每步思考秒数（V15/V16/V17）')
    parser.add_argument('--epsilon', type=float, default=0.05, help='随机着法比例')
    parser.add_argument('--engine', choices=('python', 'go'), default='python', help='V16/V17 的搜索引擎')
    parser.add_argument('--threads', type=int, default=1, help='Go 引擎每步的搜索线程数')
    parser.add_argument('--workers', type=int, default=1, help='同时对弈的进程数（建议不超过物理核数）')
    parser.add_argument('--out', default=os.path.join('data', 'selfplay.jsonl'))
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    total = 0
    start = time.time()
    run_id = int(start)
    with open(args.out, 'a', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_play_one, base_seed + g, args.model, args.think * 1000, args.epsilon,
                               args.engine, args.threads): g for g in range(args.games)}
        for done, future in enumerate(as_completed(futures), 1):
            samples = future.result()
            game_id = f'{run_id}-{futures[future]}'
            for s in samples:
                s['game'] = game_id
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
            f.flush()
            total += len(samples)
            print(f'第 {done}/{args.games} 局：{len(samples)} 个局面，累计 {total}，用时 {time.time() - start:.0f}s',
                  flush=True)
    print(f'已追加到 {args.out}')


if __name__ == '__main__':
    main()
