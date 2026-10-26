"""自对弈生成训练样本：每个局面一行 JSONL（状态 + AI 选择 + 最终胜负）。

样本只含当前局面的状态（见 fanfanqi/state.py），不需要整盘棋谱；
训练时用 fanfanqi.state.encode_features(game_from_state(...)) 转成 636 维特征。

用法：
    python -m tools.gen_data --games 50 --model StrategicV15AI --think 0.3 --out data/selfplay.jsonl
"""

import argparse
import json
import os
import random
import time

from fanfanqi.advisor import create_ai
from fanfanqi.game import Game
from fanfanqi.state import FEATURE_VERSION, STATE_VERSION, game_to_state
from tests.selfplay import apply_real, random_setup


def action_record(action):
    keys = ('type', 'row', 'col', 'from_row', 'from_col', 'to_row', 'to_col')
    return {k: action[k] for k in keys if k in action}


def play_and_record(model, budget_ms, epsilon, max_plies=300):
    game = Game()
    random_setup(game)
    samples = []
    while game.phase == 'playing' and len(samples) < max_plies:
        state = game_to_state(game)
        if random.random() < epsilon:
            # 少量随机着法，让数据覆盖 AI 自己不会走到的局面
            action = random.choice(game.get_all_actions())
        else:
            action = create_ai(model, game.current_turn, budget_ms).get_move(game.clone())
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
        s['state_version'] = STATE_VERSION
        s['feature_version'] = FEATURE_VERSION
    return samples


def main():
    parser = argparse.ArgumentParser(description='自对弈生成训练样本（JSONL）')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--model', default='StrategicV16AI')
    parser.add_argument('--think', type=float, default=0.3, help='每步思考秒数（V15/V16）')
    parser.add_argument('--epsilon', type=float, default=0.05, help='随机着法比例')
    parser.add_argument('--out', default=os.path.join('data', 'selfplay.jsonl'))
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    total = 0
    start = time.time()
    with open(args.out, 'a', encoding='utf-8') as f:
        for g in range(args.games):
            samples = play_and_record(args.model, args.think * 1000, args.epsilon)
            game_id = f'{int(start)}-{g}'
            for s in samples:
                s['game'] = game_id
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
            total += len(samples)
            print(f'第 {g + 1}/{args.games} 局：{len(samples)} 个局面，累计 {total}，用时 {time.time() - start:.0f}s',
                  flush=True)
    print(f'已追加到 {args.out}')


if __name__ == '__main__':
    main()
