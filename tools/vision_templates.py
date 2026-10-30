"""从对局截图生成画面识别模板（fanfanqi/vision_templates.npz）。

模板分三种：
- cell   棋盘格子里的棋子字形（暗子 + 12 种字形，空格不需要模板）；
- badge  被吃子栏右上角的数量（2 ~ 5，只有兵/卒会超过 2）；
- button 结算界面中间的「开始」按钮（用来判断对局已结束）。

步骤：

1. 聚类：把截图中的格子（或数量角标）切出来，按字形自动分组：
       python -m tools.vision_templates cluster sample_image vision_clusters
       python -m tools.vision_templates cluster sample_image vision_badges --kind badge
   生成 overview.png（每组一行：组号 + 前几个样本）和 clusters.json。
2. 标注：编辑 clusters.json 里每组的 "label"，然后生成：
       python -m tools.vision_templates build vision_clusters
       python -m tools.vision_templates build vision_badges
   cell 的标签："?" 暗子，"r帅" "r仕" "r相" "r车" "r马" "r炮" "r兵"、"b将" "b士" "b象" "b卒" 等
   （车马炮红黑同字，标哪方都行，识别时看字迹颜色）；badge 的标签是数字 "2" ~ "5"；
   "skip" 表示不用（例如被「开始」按钮挡住的格子）。
3. 按钮：用一张出现「开始」按钮的截图：
       python -m tools.vision_templates button sample_image/xxx.png

build / button 只更新对应的模板，其余保持不变。换了棋盘皮肤或识别出错时，把新截图放进来重跑即可。
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from fanfanqi.vision import (
    COLS, OCCUPIED_TEXTURE, PANEL_LIT, ROWS, TEMPLATE_PATH, badge_present, badge_tile, button_tile, cell_patch,
    cell_signals, cell_tile, glyph, glyph_key, locate_board, normalize_label, panel_slots, slot_brightness,
    to_rgb,
)

MAX_PER_LABEL = 8  # 每类保存的模板数
VALID = {
    'cell': ['?'] + [s + t for s in 'rb' for t in '将仕象兵'] + ['车', '马', '炮'],  # glyph_key 的取值
    'badge': ['2', '3', '4', '5'],
}


def collect(folder, kind):
    """切出所有样本，返回 (图块数组, 来源列表)；来源用于 build 时按标注取回图块。"""
    tiles, sources = [], []
    for path in sorted(glob.glob(os.path.join(folder, '*.png'))):
        rgb = to_rgb(Image.open(path))
        layout = locate_board(rgb)
        if layout is None:
            print(f'跳过 {os.path.basename(path)}：没有找到棋盘')
            continue
        name = os.path.basename(path)
        if kind == 'cell':
            for row in range(ROWS):
                for col in range(COLS):
                    if cell_signals(cell_patch(rgb, layout, row, col), layout)['texture'] < OCCUPIED_TEXTURE:
                        continue  # 空格
                    tiles.append(cell_tile(rgb, layout, row, col))
                    sources.append([name, row, col])
        else:
            for i, (_, _, col, row) in enumerate(panel_slots()):
                if slot_brightness(rgb, layout, col, row) >= PANEL_LIT and badge_present(rgb, layout, col, row):
                    tiles.append(badge_tile(rgb, layout, col, row))
                    sources.append([name, i, 0])
    return np.array(tiles), sources


def distance(a, b):
    """两个图块字形的均方根差（0 ~ 约 2）。"""
    return float(np.sqrt(np.mean((glyph(a) - glyph(b)) ** 2)))


def cluster(tiles, threshold):
    """贪心聚类：依次放进距离最近且小于阈值的组，否则新开一组。"""
    features = [glyph(t) for t in tiles]
    centers, members = [], []
    for i, f in enumerate(features):
        best, best_d = None, threshold
        for k, center in enumerate(centers):
            d = float(np.sqrt(np.mean((f - center) ** 2)))
            if d < best_d:
                best, best_d = k, d
        if best is None:
            centers.append(f.copy())
            members.append([i])
        else:
            members[best].append(i)
            centers[best] += (f - centers[best]) / len(members[best])
    return members


def overview(tiles, groups, per_group=6, size=56):
    """每组一行：组号 + 前几个样本，三栏排列。"""
    from PIL import ImageDraw
    width = 40 + per_group * size
    img = Image.new('RGB', (3 * width, (len(groups) + 2) // 3 * size), 'white')
    draw = ImageDraw.Draw(img)
    for k, idx in enumerate(groups):
        x0, y0 = (k % 3) * width, (k // 3) * size
        draw.text((x0 + 4, y0 + size // 2 - 6), str(k), fill='black')
        for j, i in enumerate(idx[:per_group]):
            img.paste(Image.fromarray(tiles[i]).resize((size - 2, size - 2)), (x0 + 40 + j * size, y0))
    return img


def load_templates(path):
    if not os.path.exists(path):
        return {}
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def cmd_cluster(args):
    tiles, sources = collect(args.screenshots, args.kind)
    if not len(tiles):
        raise SystemExit('没有找到样本')
    members = sorted(cluster(tiles, args.threshold), key=len, reverse=True)
    os.makedirs(args.out, exist_ok=True)
    overview(tiles, members).save(os.path.join(args.out, 'overview.png'))
    groups = [{'id': k, 'label': '', 'count': len(idx), 'members': [sources[i] for i in idx]}
              for k, idx in enumerate(members)]
    with open(os.path.join(args.out, 'clusters.json'), 'w', encoding='utf-8') as f:
        json.dump({'kind': args.kind, 'screenshots': os.path.abspath(args.screenshots), 'groups': groups},
                  f, ensure_ascii=False, indent=1)
    print(f'{len(tiles)} 个样本分成 {len(groups)} 组，已写出 {args.out}/overview.png 和 clusters.json')


def cmd_build(args):
    with open(os.path.join(args.clusters, 'clusters.json'), encoding='utf-8') as f:
        data = json.load(f)
    kind = data.get('kind', 'cell')
    tiles, sources = collect(data['screenshots'], kind)
    index = {tuple(s): i for i, s in enumerate(sources)}
    by_label = {}
    for group in data['groups']:
        label = group['label'].strip()
        if label == 'skip':
            continue
        if not label:
            raise SystemExit(f"第 {group['id']} 组还没有标注 label")
        if kind == 'cell':
            label = glyph_key(normalize_label(label))
        elif label not in VALID['badge']:
            raise SystemExit(f"第 {group['id']} 组的标签应为 2 ~ 5，收到 {label!r}")
        by_label.setdefault(label, []).extend(tiles[index[tuple(m)]] for m in group['members'] if tuple(m) in index)

    labels, templates = [], []
    for label, items in sorted(by_label.items()):
        # 保留彼此差别最大的几个，覆盖光圈、阴影位置等不同外观
        chosen = [items[0]]
        while len(chosen) < min(MAX_PER_LABEL, len(items)):
            far = max(items, key=lambda t: min(distance(t, c) for c in chosen))
            if min(distance(far, c) for c in chosen) < 0.15:
                break
            chosen.append(far)
        labels += [label] * len(chosen)
        templates += chosen
        print(f'{label}: {len(items)} 个样本，保存 {len(chosen)} 个模板')
    missing = [x for x in VALID[kind] if x not in by_label]
    if missing:
        print('注意：缺少这些类别的样本：' + ' '.join(missing))

    path = args.out or TEMPLATE_PATH
    saved = load_templates(path)
    saved[f'{kind}_labels'] = np.array(labels)
    saved[f'{kind}_tiles'] = np.array(templates, dtype=np.uint8)
    np.savez_compressed(path, **saved)
    print(f'已更新 {path} 的 {kind} 模板')


def cmd_button(args):
    rgb = to_rgb(Image.open(args.screenshot))
    layout = locate_board(rgb)
    if layout is None:
        raise SystemExit('没有找到棋盘')
    path = args.out or TEMPLATE_PATH
    saved = load_templates(path)
    saved['button'] = button_tile(rgb, layout)
    np.savez_compressed(path, **saved)
    print(f'已更新 {path} 的「开始」按钮模板')


def main():
    parser = argparse.ArgumentParser(description='从对局截图生成画面识别模板')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cluster', help='切出样本并自动分组')
    p.add_argument('screenshots', help='截图所在文件夹（*.png）')
    p.add_argument('out', help='输出文件夹')
    p.add_argument('--kind', choices=('cell', 'badge'), default='cell')
    p.add_argument('--threshold', type=float, default=0.45, help='归为同一组的最大字形均方根差（0 ~ 约 2）')
    p.set_defaults(func=cmd_cluster)
    p = sub.add_parser('build', help='按标注生成模板')
    p.add_argument('clusters', help='cluster 步骤的输出文件夹（含已标注的 clusters.json）')
    p.add_argument('--out', help=f'模板文件路径（默认 {TEMPLATE_PATH}）')
    p.set_defaults(func=cmd_build)
    p = sub.add_parser('button', help='从结算截图保存「开始」按钮模板')
    p.add_argument('screenshot')
    p.add_argument('--out', help=f'模板文件路径（默认 {TEMPLATE_PATH}）')
    p.set_defaults(func=cmd_button)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
