"""画面识别：从对局截图读出局面，输出 state.py 的状态格式。

适配的画面：4 列 × 8 行木纹棋盘；两侧各一栏被吃棋子（亮 = 被吃过，右上角红色数字 = 数量，
左栏从上到下 将仕象车马炮兵、右栏从上到下 兵炮马车象仕将）；头像下方是血量条（左端 帥/將 图标、
红/蓝色条表示该玩家执红/执黑）；轮到谁走时头像外有绿色倒计时框；上一步的起点是黄色小圆点，
终点棋子外有金色光圈（只有光圈、没有圆点表示翻子）；对局结束时棋盘中间出现「开始」按钮。

所有位置都以“格子”为单位相对网格计算，窗口缩放后不需要重新标定。识别结果用 state.py 的
game_from_state() 校验（明子 + 被吃 ≤ 总数、暗子数 = 剩余池、空格数 = 被吃总数），
再和血量条对照，对不上时在 problems 里说明原因。

依赖 numpy 和 Pillow（主程序不需要，只有用到画面识别时才导入）。

命令行：python -m fanfanqi.vision 截图.png [--out state.json]
"""

import argparse
from itertools import product
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ROWS, COLS = 8, 4
TILE = 32            # 格子特征图边长
OCCUPIED_TEXTURE = 9  # 格子中间一圈的灰度标准差高于此值视为有子（空格 ≤5.5，有子 ≥13.4，含连击特效）
GREEN_DOT = 8        # 暗子中心 g − r 高于此值视为有绿色选中点（有点 ≈15，其他 ≤0）
RING_MIN = 15        # 光圈强度阈值（样本：普通棋子 ≤8，有光圈 ≥22）
DOT_MIN = 85         # 黄点强度阈值（样本：其他 ≤63，黄点 ≈104）
RED_INK = 40         # 字迹 r − g 高于此值为红字（样本：黑字 ≤20，红字 ≥50）
GLYPH_MAX = 0.75     # 字形距离超过此值认为没把握
PANEL_LIT = 110      # 被吃子栏图标亮度（样本：灰 ≤82，亮 ≥138）
BADGE_MIN = 0.3      # 角标区域红色像素占比（样本：无 0，有 ≥0.66）
TURN_GREEN = 0.03    # 头像外框绿色像素占比（样本：无 0，有 ≥0.16；倒计时快结束时绿框变短）
GOOD_QUALITY = 0.8   # 有子格中字形匹配可信的比例低于此值，认为不是棋盘（或棋盘被遮挡、移动了）
BUTTON_MIN = 0.6     # 与「开始」按钮模板的相关系数（样本：结算 ≥0.99，其他 ≤0.05）
HP_PER_CELL = 0.0388  # 血量条每 1 点血量的长度（格）
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'vision_templates.npz')

PIECE_TYPES = '将仕象车马炮兵'
SHARED_GLYPHS = '车马炮'  # 红黑同字，靠字迹颜色区分
_ALIASES = {'帅': '将', '帥': '将', '將': '将', '相': '象', '士': '仕', '卒': '兵', '車': '车', '馬': '马'}

# 被吃子栏：(栏, 类型顺序, 列中心, 第一个图标的行中心)，图标间隔 0.6 格
PANELS = (('left', '将仕象车马炮兵', -0.95, 1.57), ('right', '兵炮马车象仕将', 5.0, 2.79))
PANEL_STEP = 0.6
# 玩家信息：头像中心与边长、血量条图标中心、血量条满血时右端
PLAYERS = {
    'top': {'avatar': (-2.75, 0.2, 1.1), 'icon': (-3.1, 0.93), 'bar_row': 0.93, 'bar_span': (-3.4, -0.5), 'bar_full': -0.705},
    'bottom': {'avatar': (5.25, 7.35, 1.2), 'icon': (4.87, 8.26), 'bar_row': 8.26, 'bar_span': (4.7, 7.6), 'bar_full': 7.25},
}
BUTTON_BOX = (2.0, 4.45, 2.8, 0.6)  # 「开始」按钮：中心列、中心行、宽、高（格）
# 右上角倍率数字所在的行（格），在这几列范围内找（样本中数字在 7.1~7.7 列、-0.36~-0.11 行）
MULT_ROWS = (-0.42, -0.06)
MULT_COLS = (4.6, 10.0)
DIGIT_SIZE = (12, 20)  # 倍率数字图块（宽、高）
DIGIT_MAX = 0.6        # 数字字形距离超过此值认为没把握，不读倍率


@dataclass
class Layout:
    """网格左上角 (x, y) 和格子边长，单位像素；box 是找到的木色棋盘区域 (x0, y0, x1, y1)。"""
    x: float
    y: float
    cell: float
    box: tuple = None

    def point(self, col, row):
        """以格子为单位的坐标转换成像素坐标（col/row 可以是小数或负数）。"""
        return self.x + col * self.cell, self.y + row * self.cell

    def crop(self, rgb, col, row, w, h):
        """以 (col, row) 为中心、w × h 格大小的区域；超出画面时返回 None。"""
        cx, cy = self.point(col, row)
        half_w, half_h = w * self.cell / 2, h * self.cell / 2
        x0, y0 = int(round(cx - half_w)), int(round(cy - half_h))
        x1, y1 = int(round(cx + half_w)), int(round(cy + half_h))
        height, width = rgb.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        return rgb[y0:y1, x0:x1]


def normalize_label(label):
    """模板标签统一成状态格式的写法：'?'、'r将' ~ 'b兵'。"""
    if label in ('.', '?'):
        return label
    if len(label) != 2 or label[0] not in 'rb':
        raise ValueError(f'无法识别的标签：{label!r}')
    t = _ALIASES.get(label[1], label[1])
    if t not in PIECE_TYPES:
        raise ValueError(f'无法识别的棋子：{label!r}')
    return label[0] + t


def glyph_key(label):
    """模板按字形归类：车马炮不分红黑，其余字形本身就区分红黑（帥/將、仕/士、相/象、兵/卒）。"""
    return label[1] if label[1:] and label[1] in SHARED_GLYPHS else label


def to_rgb(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert('RGB'))
    return image[..., :3]


def _resize(patch, size):
    return np.asarray(Image.fromarray(np.ascontiguousarray(patch)).resize(size, Image.BILINEAR))


# --------------------------------------------------------------- 定位棋盘
def _longest_run(flags, max_gap):
    """最长的连续 True 区间 [start, end)，中间不超过 max_gap 的 False 视为连续（网格线、棋子会打断木色）。"""
    best, start, last = (0, 0), None, None
    for i, f in enumerate(flags):
        if not f:
            continue
        if start is None or i - last > max_gap + 1:
            start = i
        last = i
        if last + 1 - start > best[1] - best[0]:
            best = (start, last + 1)
    return best


def _equal_lines(profile, n, step=None):
    """在 profile 中找 n 条等距暗线：枚举间距（给定 step 时只用它）和起点，取暗像素占比之和最大的一组。
    间距按 0.25 像素、起点按 0.5 像素搜索（棋盘缩小时 0.5 像素的间距误差累计到第 8 行会偏 2 像素）。"""
    length = len(profile)
    best = (-1.0, 0.0, 0.0)
    steps = [step] if step else np.arange((length - 1) / (n - 1) * 0.75, (length - 1) / (n - 1), 0.25)
    for step in steps:
        offsets = step * np.arange(n)
        for start in np.arange(0, length - 1 - offsets[-1], 0.5):
            s = profile[(offsets + start).round().astype(int)].sum()
            if s > best[0]:
                best = (s, start, step)
    return best[1], best[2]


def _wood_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 200) & (g > 170) & (b > 120) & (r - b > 30)


def locate_board(image, wood=None):
    """找到网格：先按浅木色找出棋盘（木色像素最多的一段竖条），再在其中找 5 条竖线、9 条横线。
    阈值相对于最高的木色列，所以棋盘只占整个屏幕一小块时也能找到。找不到时返回 None。"""
    rgb = to_rgb(image).astype(np.int16)
    if wood is None:
        wood = _wood_mask(rgb)
    counts = wood.sum(axis=0)
    if counts.max() < 100:
        return None
    x0, x1 = _longest_run(counts > 0.25 * counts.max(), max(4, int(counts.max()) // 20))
    if x1 - x0 < 50:
        return None
    rows = wood[:, x0:x1].mean(axis=1) > 0.3
    y0, y1 = _longest_run(rows, max(4, (x1 - x0) // 12))
    if y1 - y0 < 100:
        return None
    dark = rgb[y0:y1, x0:x1].mean(axis=2) < 150
    cols, rows = dark.mean(axis=0), dark.mean(axis=1)
    _, sx = _equal_lines(cols, COLS + 1)
    _, sy = _equal_lines(rows, ROWS + 1)
    if abs(sx - sy) > 0.05 * sx or sx < 20:
        return Layout(0, 0, 0, box=(x0, y0, x1, y1))  # 木色区域不是棋盘，box 供 find_board 排除
    # 格子是正方形：取两个方向的平均间距，再按它重新定起点（否则起点和间距不配套，越往下偏得越多）
    cell = (sx * COLS + sy * ROWS) / (COLS + ROWS)
    gx, _ = _equal_lines(cols, COLS + 1, cell)
    gy, _ = _equal_lines(rows, ROWS + 1, cell)
    return Layout(x0 + gx, y0 + gy, cell, box=(x0, y0, x1, y1))


def find_board(image, templates=None, tries=3):
    """在整个屏幕上找棋盘：定位后用识别质量验证，不像棋盘（例如本程序自己画的棋盘）就涂掉再找。"""
    templates = templates or default_templates()
    rgb = to_rgb(image)
    wood = _wood_mask(rgb.astype(np.int16))
    for _ in range(tries):
        layout = locate_board(rgb, wood)
        if layout is None:
            return None
        if layout.cell > 0:
            try:
                result = recognize(rgb, templates, layout)
            except (ValueError, AttributeError):
                result = None
            if result is not None and (result.game_over or result.quality >= GOOD_QUALITY):
                return layout
        x0, y0, x1, y1 = layout.box
        wood[y0:y1, x0:x1] = False
    return None


# --------------------------------------------------------------- 格子
def cell_tile(rgb, layout, row, col, inner=0.62):
    """格子中心区域（避开棋子外的光圈），缩放到 TILE × TILE。"""
    return _resize(layout.crop(rgb, col + 0.5, row + 0.5, inner, inner), (TILE, TILE))


GLYPH_RETRY = 0.45  # 距离超过此值时在附近微调取样位置再匹配
# 第一轮：定位差一两个像素；第二轮：选中后抬起放大的棋子（约 1.08 倍、上下偏 0.1 格左右）
NEAR_OFFSETS = [(dx, dy, 0.62) for dy in (-0.04, -0.02, 0.0, 0.02, 0.04) for dx in (-0.04, -0.02, 0.0, 0.02, 0.04)]
LIFTED_OFFSETS = [(dx, round(dy, 2), inner) for inner in (0.67, 0.72) for dy in np.arange(-0.15, 0.16, 0.03)
                  for dx in (-0.06, -0.03, 0.0, 0.03, 0.06)]


def _erase_dot(patch, cell, dx=0.0, dy=0.0):
    """暗子正中的绿色小圆点（对方选中/悬停的标记）用周围的颜色盖掉，免得字形对不上。
    (dx, dy) 是取样框相对格子中心的偏移，圆点在取样框里的位置要反向挪。"""
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - ((w - 1) / 2 - dx * cell), yy - ((h - 1) / 2 - dy * cell)) / cell
    center = patch[d < 0.05].astype(np.int16)
    if not len(center) or float((center[:, 1] - center[:, 0]).mean()) < GREEN_DOT:
        return patch
    patch = patch.copy()
    patch[d < 0.1] = patch[(d >= 0.1) & (d < 0.14)].mean(axis=0).astype(patch.dtype)
    return patch


def _match_cell(rgb, layout, row, col, templates):
    """字形匹配；没把握时挪动、放大取样框再试，取最像的一次。"""
    tile = label = None
    d = float('inf')
    for offsets in ([(0.0, 0.0, 0.62)], NEAR_OFFSETS, LIFTED_OFFSETS):
        for dx, dy, inner in offsets:
            patch = layout.crop(rgb, col + 0.5 + dx, row + 0.5 + dy, inner, inner)
            if patch is None:
                continue
            t = _resize(_erase_dot(patch, layout.cell, dx, dy), (TILE, TILE))
            lab, dist = templates.cell_glyph(t)
            if dist < d:
                tile, label, d = t, lab, dist
        if d <= GLYPH_RETRY:
            break
    return tile, label, d


def cell_patch(rgb, layout, row, col):
    return layout.crop(rgb, col + 0.5, row + 0.5, 1.0, 1.0).astype(np.int16)


def _ring_mask(shape, cell, r0, r1):
    """以区域中心为圆心、半径 r0~r1 格的环形区域。"""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / cell
    return (d >= r0) & (d < r1)


def cell_signals(patch, layout):
    """格子的几个简单信号：
    texture  中间一圈（半径 0.12~0.36 格）的灰度标准差：有子（字迹或暗子花纹）≥13，空木板 ≤6，
             不受连击火焰、发光等整体变亮变暗的影响，用它判断有没有子；
    ring   金色光圈强度（上一步终点；半径 0.42~0.47 处 g−b 比外侧高约 25）；
    dot    中心黄点强度（上一步起点，只出现在空格；g−b 约 104，空格约 52）。"""
    def ring(r0, r1):
        return patch[_ring_mask(patch.shape, layout.cell, r0, r1)]

    def yellow(px):
        return float((px[:, 1] - px[:, 2]).mean())

    return {
        'texture': float(ring(0.12, 0.36).mean(axis=1).std()),
        'ring': yellow(ring(0.42, 0.47)) - yellow(ring(0.49, 0.53)),
        'dot': yellow(ring(0, 0.05)),
    }


def glyph(tile):
    """灰度、去均值、单位方差：只看字形，不受红黑和明暗影响。"""
    g = tile.astype(np.float32).mean(axis=2)
    g -= g.mean()
    return g / (g.std() + 1e-6)


def ink_redness(tile):
    """字迹像素的 r − g：红字约 50 以上，黑字 20 以下；没有字迹（暗子）时返回 None。"""
    t = tile.astype(np.int16)
    ink = t.mean(axis=2) < 120
    if ink.sum() < 8:
        return None
    return float((t[..., 0] - t[..., 1])[ink].mean())


# --------------------------------------------------------------- 被吃子栏
def panel_slots():
    """[(栏, 棋子类型, 列, 行)]，共 14 个图标位置。"""
    return [(name, t, col, row0 + PANEL_STEP * i)
            for name, order, col, row0 in PANELS for i, t in enumerate(order)]


def _disc(rgb, layout, col, row, radius):
    cx, cy = layout.point(col, row)
    r = radius * layout.cell
    h, w = rgb.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return rgb[y0:y1, x0:x1][np.hypot(xx - cx, yy - cy) <= r].astype(np.int16)


def slot_brightness(rgb, layout, col, row):
    """图标中心的平均亮度：被吃过的图标是亮的木色，没被吃的是暗淡的轮廓。"""
    px = _disc(rgb, layout, col, row, 0.15)
    return float(px.mean()) if len(px) else 0.0


def badge_present(rgb, layout, col, row):
    px = _disc(rgb, layout, col + 0.2, row - 0.2, 0.1)
    return len(px) > 0 and float(((px[:, 0] > 170) & (px[:, 1] < 110)).mean()) >= BADGE_MIN


def badge_tile(rgb, layout, col, row):
    """数量角标：按红色圆的位置和大小对齐后取 16 × 16，只留圆内的亮度（白色数字），圆外填成圆的颜色。
    窗口大小不同时角标的相对位置、大小略有出入，直接按固定框裁剪会把背景和图标边缘也算进去。"""
    patch = layout.crop(rgb, col + 0.2, row - 0.2, 0.4, 0.4)
    if patch is None:
        return None
    p = patch.astype(np.int16)
    red = (p[..., 0] > 150) & (p[..., 0] - p[..., 1] > 50) & (p[..., 0] - p[..., 2] > 50)
    ys, xs = np.where(red)
    if len(xs) < 10:
        return None
    cx, cy = float(np.median(xs)), float(np.median(ys))
    r = float(np.sqrt(red.sum() / np.pi)) * 1.08  # 红色面积不含白色数字，略放大
    x0, y0 = max(0, int(round(cx - r))), max(0, int(round(cy - r)))
    sub = p[y0:int(round(cy + r)) + 1, x0:int(round(cx + r)) + 1]
    h, w = sub.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = np.hypot(xx - (cx - x0), yy - (cy - y0)) <= r * 0.92
    g = sub.min(axis=2)  # 白色数字三个通道都高，红色圆的最小通道低
    g[~inside] = g[inside].min()
    g = _resize(g.astype(np.uint8), (16, 16))
    return np.repeat(g[..., None], 3, axis=2)


def button_tile(rgb, layout):
    patch = layout.crop(rgb, *BUTTON_BOX)
    return None if patch is None else _resize(patch, (96, 20))


def multiplier_tiles(rgb, layout):
    """右上角“倍率”框里的黄色数字，从左到右切成单个数字的图块（DIGIT_SIZE，按数字高度等比放进框里）。
    框贴着游戏窗口右上角，和棋盘的相对位置随窗口宽度变化，所以在棋盘右侧一段较宽的范围里找；
    连击火焰会让数字偏橙，按“亮、偏黄、蓝色低”取像素。找不到成组的数字时返回 None。"""
    h, w = rgb.shape[:2]
    x0, y0 = layout.point(MULT_COLS[0], MULT_ROWS[0])
    x1, y1 = layout.point(MULT_COLS[1], MULT_ROWS[1])
    x0, x1, y0, y1 = max(0, int(x0)), min(w, int(x1)), max(0, int(y0)), min(h, int(y1))
    if x1 - x0 < 10 or y1 - y0 < 5:
        return None
    px = rgb[y0:y1, x0:x1].astype(np.int16)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mask = (r > 190) & (g > 130) & (b < 110) & (g - b > 80)
    cols = np.where(mask.sum(axis=0) > 0)[0]
    if not len(cols):
        return None
    runs, start = [], cols[0]  # 按空白列切开：每段是一个数字
    for a, c in zip(cols, cols[1:]):
        if c - a > 1:
            runs.append((start, a + 1))
            start = c
    runs.append((start, cols[-1] + 1))
    min_height = 0.15 * layout.cell
    digits = []
    for a, c in runs:
        ys = np.where(mask[:, a:c].any(axis=1))[0]
        if len(ys) and ys[-1] - ys[0] + 1 >= min_height:
            digits.append((a, c))
    if not digits:
        return None
    # 只要靠在一起的一组（数字间距很小），取最长的一组
    groups, current = [], [digits[0]]
    for d in digits[1:]:
        if d[0] - current[-1][1] <= 0.12 * layout.cell:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)
    group = max(groups, key=len)
    rows = np.where(mask[:, group[0][0]:group[-1][1]].any(axis=1))[0]
    top, bottom = rows[0], rows[-1] + 1
    height = bottom - top
    width = int(round(height * DIGIT_SIZE[0] / DIGIT_SIZE[1]))
    if max(c - a for a, c in group) > 1.3 * width:
        return None  # 比一个数字宽得多：几个数字连在一起或者是别的东西
    width = max(width, max(c - a for a, c in group))
    tiles = []
    for a, c in group:
        sub = np.clip(r[top:bottom, a:c] - b[top:bottom, a:c], 0, 255).astype(np.uint8)  # 黄色越纯越亮
        box = np.zeros((height, width), np.uint8)
        offset = (width - (c - a)) // 2
        box[:, offset:offset + c - a] = sub
        tiles.append(np.repeat(_resize(box, DIGIT_SIZE)[..., None], 3, axis=2))
    return tiles


def _center_dialog(rgb, layout):
    """检测覆盖棋盘中央的浅色系统弹窗（例如重新登录确认框）。"""
    patch = layout.crop(rgb, 2.0, 4.0, 3.0, 3.0)
    if patch is None:
        return False
    px = patch.astype(np.int16)
    # 弹窗底色偏浅且接近中性白；正常木纹棋盘的蓝通道明显更低。
    light_neutral = (px[..., 0] > 200) & (px[..., 1] > 185) & (px[..., 2] > 165)
    return float(light_neutral.mean()) >= 0.18


# --------------------------------------------------------------- 模板
class Templates:
    def __init__(self, path=TEMPLATE_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f'没有识别模板 {path}，请先用 tools/vision_templates.py 生成')
        with np.load(path) as data:
            self.cell_labels = list(data['cell_labels'])
            self.cell = np.stack([glyph(t) for t in data['cell_tiles']])
            self.badge_labels = list(data['badge_labels']) if 'badge_labels' in data.files else []
            self.badge = np.stack([glyph(t) for t in data['badge_tiles']]) if self.badge_labels else None
            self.button = glyph(data['button']) if 'button' in data.files else None
            self.digit_labels = list(data['digit_labels']) if 'digit_labels' in data.files else []
            self.digit = np.stack([glyph(t) for t in data['digit_tiles']]) if self.digit_labels else None

    @staticmethod
    def _nearest(features, labels, tile):
        d = np.sqrt(((features - glyph(tile)) ** 2).mean(axis=(1, 2)))
        i = int(d.argmin())
        return labels[i], float(d[i])

    def cell_glyph(self, tile):
        return self._nearest(self.cell, self.cell_labels, tile)

    def badge_count(self, tile):
        if self.badge is None:
            return None, 0.0
        label, d = self._nearest(self.badge, self.badge_labels, tile)
        return int(label), d

    def multiplier(self, tiles):
        """倍率数字图块 → 整数；没有模板、没切出数字或有一位没把握时返回 None。"""
        if self.digit is None or not tiles:
            return None
        digits = []
        for tile in tiles:
            label, d = self._nearest(self.digit, self.digit_labels, tile)
            if d > DIGIT_MAX:
                return None
            digits.append(label)
        return int(''.join(digits))

    def is_button(self, tile):
        return self.button is not None and tile is not None and float((glyph(tile) * self.button).mean()) >= BUTTON_MIN


_templates = None


def default_templates():
    global _templates
    if _templates is None:
        _templates = Templates()
    return _templates


# --------------------------------------------------------------- 识别
@dataclass
class Recognition:
    layout: Layout
    board: list = field(default_factory=list)        # 8 × 4，'.' / '?' / 'r车' 等
    captured: dict = field(default_factory=dict)     # {'red': {'兵': 2}, 'black': {...}}
    players: dict = field(default_factory=dict)      # {'top': 'black', 'bottom': 'red'}，认不出为 None
    turn: str = None                                 # 轮到谁（头像绿框 / 上一步推断），认不出为 None
    hp_bar: dict = field(default_factory=dict)       # 血量条估计值 {'red': 41, 'black': None}，±1
    last_move: dict = None                           # {'from': (r, c) 或 None, 'to': (r, c)}；from 为 None 表示翻子
    turn_certain: bool = False                       # turn 是否有绿框等直接依据（False 表示换手过渡中，按“下方没绿框就不是自己”推的）
    game_over: bool = False
    board_obscured: bool = False                      # 棋盘中央有特效/弹窗遮挡，不能把识别结果用于同步
    quality: float = 1.0                             # 有子格中字形匹配可信的比例
    problems: list = field(default_factory=list)     # 不自洽、没把握的地方
    notes: list = field(default_factory=list)        # 按规则自动修正过的地方
    multiplier: int = None                           # 右上角显示的倍率（含抢红/加倍系数），认不出为 None

    @property
    def my_side(self):
        """下方玩家（自己）执哪一方。"""
        return self.players.get('bottom')

    def to_state(self, turn=None):
        """输出 state.py 的状态格式；turn 不给时用识别结果。连击、开局翻棋分等过程信息画面上没有，记为 0。"""
        turn = turn or self.turn
        if turn is None:
            raise ValueError('认不出轮到谁走，请指定 turn')
        return {
            'v': 6,
            'turn': turn,
            'board': [list(row) for row in self.board],
            'captured': {side: dict(pieces) for side, pieces in self.captured.items()},
        }


def _player_side(rgb, layout, info):
    """血量条左端图标是 帥（红字）还是 將（黑字）；没有血量条时返回 None。"""
    patch = layout.crop(rgb, *info['icon'], 0.3, 0.3)
    if patch is None:
        return None
    k = ink_redness(_resize(patch, (TILE, TILE)))
    if k is None:
        return None
    if k >= 50:
        return 'red'
    if k <= 23:
        return 'black'
    return None


def _hp_bar(rgb, layout, info, side):
    """血量条右端位置换算成血量；条太短（被图标挡住）或画面外时返回 None。
    条中间叠着白色的“33/60”文字，会盖住填充的末端，所以只看条的上下边缘（离中线 0.06~0.1 格），
    任一行是填充色就算填到这里。颜色按相对值判断：连击特效会把整条压暗。"""
    x0, y = layout.point(info['bar_span'][0], info['bar_row'])
    x1, _ = layout.point(info['bar_span'][1], info['bar_row'])
    band = int(round(0.1 * layout.cell))
    if y + band + 1 > rgb.shape[0] or y - band < 0 or x0 < 0 or x1 > rgb.shape[1]:
        return None
    rows = [int(round(y + s * k)) for s in (-1, 1) for k in range(int(round(0.06 * layout.cell)), band + 1)]
    px = rgb[rows, int(x0):int(x1)].astype(np.int16)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    if side == 'red':
        filled = (r > 180) & (r - g > 100) & (r - b > 90)
    else:
        filled = (b > 150) & (b - r > 80)
    xs = np.where(filled.any(axis=0))[0]
    if not len(xs):
        return None
    end = (x0 + xs[-1] - layout.x) / layout.cell
    hp = 60 - (info['bar_full'] - end) / HP_PER_CELL
    return int(round(hp)) if hp >= 6 else None  # 血量很低时条被图标挡住，读数不可靠


def _whose_turn(rgb, layout, players, shown):
    """轮到谁，以下方（自己）为准：下方头像有绿框就是自己，否则是对方。
    连击特效会挡住该玩家的头像框（同时血量条消失），这时看另一方：另一方没有绿框就是特效这一方。
    返回 (轮到的一方, 是否有直接依据)；双方颜色未知（开局翻第一张前）时返回 (None, False)。"""
    mine, theirs = players.get('bottom'), players.get('top')
    if mine is None:
        return None, False
    green = {where: _turn_frame(rgb, layout, info) >= TURN_GREEN for where, info in PLAYERS.items()}
    if green['bottom']:
        return mine, True
    if green['top']:
        return theirs, True
    if not shown['bottom'] and shown['top']:
        return mine, True     # 自己在连击特效中，对方没有绿框
    if shown['bottom'] and not shown['top']:
        return theirs, True   # 对方在连击特效中，自己没有绿框
    return theirs, False      # 双方都没有绿框：换手过渡中，按“下方没绿框就不是自己”


def _turn_frame(rgb, layout, info):
    """头像外框一圈的绿色像素占比。"""
    col, row, size = info['avatar']
    patch = layout.crop(rgb, col, row, size, size)
    if patch is None:
        return 0.0
    border = np.ones(patch.shape[:2], bool)
    b = int(0.12 * layout.cell)
    border[b:-b, b:-b] = False
    px = patch[border].astype(np.int16)
    return float(((px[:, 1] > 150) & (px[:, 0] < 130) & (px[:, 2] < 130)).mean())


def recognize(image, templates=None, layout=None):
    """识别一张截图。layout 可以传入上一次的结果，省去定位（窗口大小不变时）。"""
    templates = templates or default_templates()
    rgb = to_rgb(image)
    layout = layout or locate_board(rgb)
    if layout is None:
        raise ValueError('没有找到棋盘')
    result = Recognition(layout=layout)
    result.game_over = templates.is_button(button_tile(rgb, layout))
    result.multiplier = templates.multiplier(multiplier_tiles(rgb, layout))
    center_dialog = _center_dialog(rgb, layout)

    # 棋盘
    froms, tos = [], []
    occupied = good = 0
    central_bad_glyphs = 0
    for row in range(ROWS):
        tokens = []
        for col in range(COLS):
            signals = cell_signals(cell_patch(rgb, layout, row, col), layout)
            if signals['texture'] < OCCUPIED_TEXTURE:
                tokens.append('.')
                if signals['dot'] >= DOT_MIN:
                    froms.append((row, col))
                continue
            if signals['ring'] >= RING_MIN:
                tos.append((row, col))
            tile, label, d = _match_cell(rgb, layout, row, col, templates)
            if label == '?':
                tokens.append('?')
            else:
                redness = ink_redness(tile)
                ink = None if redness is None else ('r' if redness >= RED_INK else 'b')
                if label in SHARED_GLYPHS:
                    side = ink or 'r'
                    tokens.append(side + label)
                else:
                    tokens.append(label)
                    if ink is not None and ink != label[0]:
                        result.problems.append(f'第 {row} 行第 {col} 列：字形像「{label}」但字迹颜色不符')
                if ink is None:
                    result.problems.append(f'第 {row} 行第 {col} 列：看不清字迹颜色')
            occupied += 1
            if d <= GLYPH_MAX:
                good += 1
            elif not result.game_over:
                result.problems.append(f'第 {row} 行第 {col} 列：识别为 {tokens[-1]} 但把握不大（距离 {d:.2f}）')
            # 结算动画（如“获胜”大字）会覆盖棋盘中央。遮挡纹理通常与棋子字形
            # 差异很大；聚集出现时整帧应丢弃，不能将这些格子当成真实棋子。
            if 2 <= row <= 5 and d >= 0.95:
                central_bad_glyphs += 1
        result.board.append(tokens)
    result.quality = good / occupied if occupied else 0.0
    result.board_obscured = center_dialog or (central_bad_glyphs >= 3 and result.quality < 0.8)
    if len(tos) == 1:
        result.last_move = {'from': froms[0] if len(froms) == 1 else None, 'to': tos[0]}

    # 双方颜色、轮到谁、血量条。连击特效期间该玩家的血量条和 帥/將 图标被特效替换，按对家推断
    for where, info in PLAYERS.items():
        result.players[where] = _player_side(rgb, layout, info)
    shown = {where: side is not None for where, side in result.players.items()}
    known = [s for s in result.players.values() if s]
    if len(known) == 1:  # 一方认不出时，取另一方的对家
        other = 'black' if known[0] == 'red' else 'red'
        result.players = {w: s or other for w, s in result.players.items()}
    if len(set(result.players.values()) - {None}) == 1 and None not in result.players.values():
        result.problems.append('上下两位玩家识别成了同一方')
    for where, info in PLAYERS.items():
        side = result.players[where]
        if side and shown[where]:  # 图标不在（连击特效）时那里是特效画面，不是血量条
            result.hp_bar[side] = _hp_bar(rgb, layout, info, side)
    result.turn, result.turn_certain = _whose_turn(rgb, layout, result.players, shown)

    # 被吃子栏：先看亮图标的字迹颜色判断各栏是哪一方
    columns, soldier_badges = {}, []
    for name, order, col, row0 in PANELS:
        counts, redness = {}, []
        for i, t in enumerate(order):
            row = row0 + PANEL_STEP * i
            if slot_brightness(rgb, layout, col, row) < PANEL_LIT:
                continue
            n = 1
            if badge_present(rgb, layout, col, row):
                tile = badge_tile(rgb, layout, col, row)
                n, d = templates.badge_count(tile) if tile is not None else (None, 0.0)
                if n is None:
                    n = 2
                    result.problems.append('没有数量角标模板或看不清角标，按 2 个计')
                if t != '兵' and n != 2:
                    result.problems.append(f'{name} 栏「{t}」的数量识别为 {n}，最多 2 个，按 2 计')
                    n = 2
                if t == '兵':
                    soldier_badges.append(name)
            counts[t] = n
            k = ink_redness(_resize(layout.crop(rgb, col, row, 0.36, 0.36), (TILE, TILE)))
            if k is not None:
                redness.append(k)
        side = None if not redness else ('red' if np.median(redness) >= RED_INK else 'black')
        columns[name] = (side, counts)
    sides = {name: side for name, (side, _) in columns.items()}
    if sides['left'] is None and sides['right'] is not None:
        sides['left'] = 'black' if sides['right'] == 'red' else 'red'
    elif sides['right'] is None and sides['left'] is not None:
        sides['right'] = 'black' if sides['left'] == 'red' else 'red'
    elif sides['left'] is None:  # 两栏都还没有被吃的子
        sides = {'left': 'red', 'right': 'black'}
    if sides['left'] == sides['right']:
        result.problems.append('两侧被吃子栏识别成了同一方')
    result.captured = {'red': {}, 'black': {}}
    for name, (_, counts) in columns.items():
        if sides[name] in result.captured:
            result.captured[sides[name]].update(counts)

    if not result.game_over and not result.board_obscured:  # 结算界面/特效挡住棋盘时空格数不可靠
        badge_sides = [sides[name] for name in soldier_badges if sides.get(name) in result.captured]
        if badge_sides:
            _fix_soldier_count(result, badge_sides)
        _check(result)
    return result


def _fix_soldier_count(result, sides):
    """用空格总数和血量条联合核对兵/卒角标，避免只在单个角标时才能纠错。"""
    from .game import INITIAL_HP
    from .state import lost_hp

    if isinstance(sides, str):  # 兼容旧调用方式
        sides = [sides]
    sides = list(dict.fromkeys(sides))
    empty = sum(row.count('.') for row in result.board)
    current = {side: result.captured[side].get('兵', 0) for side in sides}
    total = sum(sum(c.values()) for c in result.captured.values())
    if total == empty:
        return

    candidates = []
    for counts in product(range(2, 6), repeat=len(sides)):
        adjusted_total = total + sum(counts[i] - current[side] for i, side in enumerate(sides))
        if adjusted_total != empty:
            continue
        hp_error = 0
        for side in ('red', 'black'):
            shown = result.hp_bar.get(side)
            if shown is None:
                continue
            captured = dict(result.captured[side])
            if side in sides:
                captured['兵'] = counts[sides.index(side)]
            derived = INITIAL_HP - lost_hp(captured)
            hp_error += max(0, abs(shown - derived) - 1)
        changed = sum(abs(counts[i] - current[side]) for i, side in enumerate(sides))
        candidates.append(((hp_error, changed), counts))

    if not candidates:
        return
    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    best = [counts for score, counts in candidates if score == best_score]
    if len(best) != 1:
        return  # 缺少足够证据时不猜数字
    for i, side in enumerate(sides):
        fixed = best[0][i]
        if fixed != current[side]:
            name = '红' if side == 'red' else '黑'
            result.notes.append(f"{name}方被吃兵/卒角标由 {current[side]} 修正为 {fixed}（空格数与血量条校验）")
            result.captured[side]['兵'] = fixed


def _check(result):
    """用规则约束校验识别结果；不自洽时把原因写进 problems。"""
    from .game import INITIAL_HP
    from .state import game_from_state, lost_hp
    try:
        game_from_state(result.to_state(turn=result.turn or 'red'))
    except ValueError as e:
        result.problems.append(f'局面不自洽：{e}')
    for side, shown in result.hp_bar.items():
        if shown is None:
            continue
        derived = INITIAL_HP - lost_hp(result.captured.get(side, {}))
        if abs(shown - derived) > 1:
            name = '红' if side == 'red' else '黑'
            result.problems.append(f'{name}方血量条约 {shown}，按被吃棋子推算是 {derived}')


def grab_screen(bbox=None):
    """截取屏幕（Windows / macOS）；bbox = (左, 上, 右, 下)。"""
    from PIL import ImageGrab
    return ImageGrab.grab(bbox=bbox, all_screens=True)


FAILED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_image', 'failed')


def save_frame(rgb, layout, problems, directory=FAILED_DIR, limit=200):
    """把识别没把握的画面存下来（只截棋盘、被吃子栏和头像一带，不存整个屏幕），附上问题说明，
    之后可以用 tools/vision_templates.py 补模板。目录里已有 limit 张时不再保存。返回文件路径或 None。"""
    os.makedirs(directory, exist_ok=True)
    if len([n for n in os.listdir(directory) if n.endswith('.png')]) >= limit:
        return None
    (x0, y0), (x1, y1) = layout.point(-3.8, -0.6), layout.point(8.2, 9.0)
    h, w = rgb.shape[:2]
    patch = rgb[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))]
    path = os.path.join(directory, time.strftime('%Y%m%d-%H%M%S') + f'-{int(time.time() * 1000) % 1000:03d}.png')
    Image.fromarray(np.ascontiguousarray(patch)).save(path)
    with open(path[:-4] + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(problems) + '\n')
    return path


def format_board(board):
    width = 3
    return '\n'.join(' '.join(f'{t:<{width}}' for t in row) for row in board)


def main():
    parser = argparse.ArgumentParser(description='从对局截图识别局面，输出状态 JSON')
    parser.add_argument('image', nargs='?', help='截图文件；不给时截取整个屏幕')
    parser.add_argument('--turn', choices=('red', 'black'), help='认不出轮到谁时手动指定')
    parser.add_argument('--out', help='把状态写到这个 JSON 文件（可在界面「导入局面」）')
    args = parser.parse_args()

    image = Image.open(args.image) if args.image else grab_screen()
    result = recognize(image)
    print(format_board(result.board))
    print(f"被吃：红 {result.captured.get('red')}  黑 {result.captured.get('black')}")
    print(f"上方 {result.players.get('top')} / 下方（自己）{result.players.get('bottom')}，"
          f"轮到 {result.turn}，血量条 {result.hp_bar}，画面倍率 {result.multiplier}")
    if result.last_move:
        move = result.last_move
        print('上一步：' + (f"翻子 {move['to']}" if move['from'] is None else f"{move['from']} → {move['to']}"))
    if result.game_over:
        print('对局已结束（出现「开始」按钮）')
    for n in result.notes:
        print('修正：' + n)
    for p in result.problems:
        print('问题：' + p)
    if args.out:
        state = result.to_state(turn=args.turn)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f'已写出 {args.out}')


if __name__ == '__main__':
    main()
