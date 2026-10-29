"""翻翻棋 tkinter 界面。

两种模式：
- 录入+提示：录入实物对局的每一步（翻出什么由人选择），AI 只给提示；
  也可以打开「画面同步」，从屏幕上的对局画面自动同步每一步（见 sync.py），只给自己一方提示。
- 人机对战：翻子随机，电脑自动走子；可选让 AI 给玩家提示。
"""

import json
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .advisor import AI_MODELS, compute_hint, describe_last_action, pos_text
from .config import load_config, save_config
from .game import (
    COLS, COMBO_THRESHOLD, DEFAULT_LONG_CHASE_LIMIT, DEFAULT_NO_CAPTURE_LIMIT,
    INITIAL_HP, PIECE_CONFIG, PIECE_NAMES, ROWS, SIDE_NAMES, Game, other_side,
)
from .state import captured_pieces, game_from_state, game_to_state, lost_hp

# 棋盘尺寸在启动时按屏幕大小计算（set_board_size），下面是上限
CELL = 72
MARGIN = 28
PIECE_R = 29
PIECE_FONT = 20
# 棋盘两侧的被吃子栏（参照游戏画面）：左栏自己、右栏对方；被吃过的亮、没被吃的淡，2 个以上有数量角标
SIDE_COL_W = 52
ICON_R = 17
ICON_STEP = 44
ICON_FONT = 12


def set_board_size(cell):
    """按格子边长设置棋盘、棋子、两侧被吃子栏的尺寸（比例同 72 像素的原始设计）。"""
    global CELL, MARGIN, PIECE_R, PIECE_FONT, SIDE_COL_W, ICON_R, ICON_STEP, ICON_FONT
    CELL = cell
    MARGIN = round(cell * 0.39)
    PIECE_R = round(cell * 0.40)
    PIECE_FONT = max(10, round(cell * 0.28))
    SIDE_COL_W = round(cell * 0.72)
    ICON_R = round(cell * 0.24)
    ICON_STEP = round(cell * 0.61)
    ICON_FONT = max(8, round(cell * 0.17))
CAPTURE_COLUMNS = (('left', '将仕象车马炮兵', 1.57), ('right', '兵炮马车象仕将', 2.79))  # 栏、从上到下的顺序、首个图标所在行

COLORS = {
    'board_bg': '#e8c98f',
    'grid': '#8a6a3a',
    'hidden': '#7a5a32',
    'hidden_text': '#f3e3c3',
    'face': '#fbf3e0',
    'red': '#c0392b',
    'black': '#222222',
    'selected': '#1e6fd9',
    'valid': '#2e9e4f',
    'hint': '#ff8c00',
    'last': '#fff3a8',
    'setup': '#ffe066',
}

MODE_LABELS = {'record': '录入+提示', 'vs': '人机对战'}
HINT_SIDE_LABELS = {'both': '双方', 'red': '红方', 'black': '黑方', 'off': '关闭'}
HINT_SIDE_VALUES = {'both': ('red', 'black'), 'red': ('red',), 'black': ('black',), 'off': ()}


def _key_of(labels, label):
    return next(k for k, v in labels.items() if v == label)


class PieceChooser(tk.Toplevel):
    """从剩余棋子池中选择一枚棋子（翻出的棋子 / 被炮打掉的暗子 / 开局放置）。"""

    def __init__(self, master, game, title):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.transient(master)
        self.result = -1

        ttk.Label(self, text=title, font=('Microsoft YaHei UI', 11, 'bold')).pack(padx=12, pady=(10, 6))
        for side in ('red', 'black'):
            frame = ttk.LabelFrame(self, text=SIDE_NAMES[side])
            frame.pack(fill='x', padx=12, pady=4)
            for col_idx, piece_type in enumerate(PIECE_CONFIG):
                count = sum(1 for p in game.remaining_pool if p['type'] == piece_type and p['side'] == side)
                name = PIECE_NAMES[side][piece_type]
                btn = tk.Button(
                    frame, text=f'{name}\n×{count}', width=5, height=2,
                    font=('Microsoft YaHei UI', 11, 'bold'),
                    fg=COLORS[side], disabledforeground='#bbbbbb',
                    state='normal' if count else 'disabled',
                    command=lambda t=piece_type, s=side: self._pick(game, t, s),
                )
                btn.grid(row=0, column=col_idx, padx=3, pady=4)
        ttk.Button(self, text='取消', command=self.destroy).pack(pady=(4, 10))
        self.bind('<Escape>', lambda _e: self.destroy())
        self.grab_set()
        self.focus_set()

    def _pick(self, game, piece_type, side):
        self.result = game.find_pool_index(piece_type, side)
        self.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title('翻翻棋 · AI 提示版')
        self.config = load_config()

        self.selected = None
        self.valid_moves = []
        self.move_log = []
        # 人机模式下开局是自动随机放置的，悔棋不能退回到放置阶段
        self.min_history = 1

        self.hint = None
        self.ai_token = 0
        self.ai_queue = queue.Queue()
        self.ai_busy = False  # 电脑正在思考/准备落子
        self.pondering = False  # 对方回合在后台提前计算（见 can_ponder）
        # 画面同步（录入模式）：后台线程截屏识别，结果经队列交给界面线程
        self.sync_watcher = None
        self.board_sync = None
        self.sync_synced = False
        self.sync_queue = queue.Queue()

        cfg = self.config
        self.mode_var = tk.StringVar(value=MODE_LABELS.get(cfg['mode'], MODE_LABELS['record']))
        self.player_side_var = tk.StringVar(value=SIDE_NAMES.get(cfg['player_side'], SIDE_NAMES['red']))
        self.model_var = tk.StringVar(value=cfg['ai_model'] if cfg['ai_model'] in AI_MODELS else 'StrategicV17AI')
        self.engine_var = tk.StringVar(value='Go' if cfg.get('engine') == 'go' else 'Python')
        self.threads_var = tk.IntVar(value=min(64, max(1, int(cfg.get('go_threads', 4)))))
        rules = cfg.get('rules', {})
        self.long_chase_var = tk.IntVar(value=max(2, int(rules.get('long_chase_limit', DEFAULT_LONG_CHASE_LIMIT))))
        self.no_capture_var = tk.IntVar(value=max(1, int(rules.get('no_capture_limit', DEFAULT_NO_CAPTURE_LIMIT))))
        self.budget_var = tk.DoubleVar(value=cfg['think_time_sec'])
        self.ai_budget_var = tk.DoubleVar(value=cfg['ai_think_time_sec'])
        self.hint_depth_var = tk.IntVar(value=max(0, int(cfg['hint_min_depth'])))
        self.hint_side_var = tk.StringVar(value=HINT_SIDE_LABELS.get(cfg['hint_side'], '双方'))
        self.show_on_board_var = tk.BooleanVar(value=cfg['show_hint_on_board'])
        self.ponder_var = tk.BooleanVar(value=cfg['ponder'])

        self._build_ui()
        self.fit_to_screen()
        self.new_game()
        self.root.after(100, self._poll_ai_queue)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    # ------------------------------------------------------------ settings
    @property
    def mode(self):
        return _key_of(MODE_LABELS, self.mode_var.get())

    @property
    def player_side(self):
        return _key_of(SIDE_NAMES, self.player_side_var.get())

    @property
    def ai_side(self):
        return other_side(self.player_side)

    def is_vs(self):
        return self.mode == 'vs'

    def think_budget_ms(self):
        """提示的思考时间。"""
        try:
            return max(100, float(self.budget_var.get()) * 1000)
        except (tk.TclError, ValueError):
            return self.config['think_time_sec'] * 1000

    def ai_think_budget_ms(self):
        """人机对战中电脑走子的思考时间。"""
        try:
            return max(100, float(self.ai_budget_var.get()) * 1000)
        except (tk.TclError, ValueError):
            return self.config['ai_think_time_sec'] * 1000

    def hint_min_depth(self):
        try:
            return min(20, max(0, int(self.hint_depth_var.get())))
        except (tk.TclError, ValueError):
            return self.config['hint_min_depth']

    @property
    def syncing(self):
        return self.sync_watcher is not None

    def sync_side(self):
        """画面同步时自己（画面下方玩家）执哪一方；还认不出时为 None。"""
        return self.board_sync.my_side if self.syncing else None

    def hint_sides(self):
        sides = HINT_SIDE_VALUES[_key_of(HINT_SIDE_LABELS, self.hint_side_var.get())]
        if self.syncing:
            # 画面同步只为自己一方提示（「关闭」时不提示）
            mine = self.sync_side()
            return (mine,) if mine and sides else ()
        if self.is_vs():
            # 人机对战中电脑一方直接走子，只为玩家提示
            return tuple(s for s in sides if s == self.player_side)
        return sides

    def can_ponder(self):
        """不需要提示的一方行动时是否后台提前计算：只用 Go 引擎（常驻进程保留置换表），人机对战中不做。"""
        return (self.config.get('ponder') and not self.is_vs() and bool(self.hint_sides())
                and self.engine_var.get() == 'Go' and self.model_var.get() in ('StrategicV16AI', 'StrategicV17AI'))

    def save_settings(self):
        cfg = self.config
        cfg['mode'] = self.mode
        cfg['player_side'] = self.player_side
        cfg['ai_model'] = self.model_var.get()
        cfg['engine'] = 'go' if self.engine_var.get() == 'Go' else 'python'
        try:
            cfg['go_threads'] = min(64, max(1, int(self.threads_var.get())))
        except (tk.TclError, ValueError):
            cfg['go_threads'] = 4
        try:
            cfg['rules'] = {
                'long_chase_limit': min(20, max(2, int(self.long_chase_var.get()))),
                'no_capture_limit': min(100, max(1, int(self.no_capture_var.get()))),
            }
        except (tk.TclError, ValueError):
            cfg['rules'] = {
                'long_chase_limit': DEFAULT_LONG_CHASE_LIMIT,
                'no_capture_limit': DEFAULT_NO_CAPTURE_LIMIT,
            }
        cfg['think_time_sec'] = self.think_budget_ms() / 1000
        cfg['ai_think_time_sec'] = self.ai_think_budget_ms() / 1000
        cfg['hint_min_depth'] = self.hint_min_depth()
        cfg['hint_side'] = _key_of(HINT_SIDE_LABELS, self.hint_side_var.get())
        cfg['show_hint_on_board'] = bool(self.show_on_board_var.get())
        cfg['ponder'] = bool(self.ponder_var.get())
        save_config(cfg)

    def on_setting_changed(self, _event=None):
        self.save_settings()
        self.request_ai()

    def on_rule_setting_changed(self, _event=None):
        self.save_settings()
        was_over = self.game.phase == 'over'
        self.game.configure_rules(**self.config['rules'])
        if not was_over and self.game.phase == 'over':
            self.after_action()
        else:
            self.request_ai()

    def update_engine_widgets(self):
        if self.engine_var.get() == 'Go':
            models = ['StrategicV16AI', 'StrategicV17AI']
            if self.model_var.get() not in models:
                self.model_var.set('StrategicV17AI')
            self.threads_spin.configure(state='normal')
        else:
            models = AI_MODELS
            self.threads_spin.configure(state='disabled')
        self.model_box.configure(values=models)

    def on_engine_changed(self, _event=None):
        self.update_engine_widgets()
        self.save_settings()
        self.request_ai()

    def on_mode_changed(self, _event=None):
        """模式 / 执子方变化需要开新局。"""
        if self.move_log and not messagebox.askyesno('开始新对局', '切换模式或执子方会开始新对局，确定吗？'):
            self.mode_var.set(MODE_LABELS.get(self.config['mode'], MODE_LABELS['record']))
            self.player_side_var.set(SIDE_NAMES.get(self.config['player_side'], SIDE_NAMES['red']))
            return
        self.save_settings()
        self.update_mode_widgets()
        self.new_game()

    def on_close(self):
        self.stop_sync()
        self.save_settings()
        self.root.destroy()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        style.configure('Hint.TLabel', font=('Microsoft YaHei UI', 12, 'bold'), foreground='#b35900')

        outer = ttk.Frame(self.root, padding=6)
        outer.pack(fill='both', expand=True)

        left = ttk.Frame(outer)
        left.pack(side='left', fill='y')
        right = ttk.Frame(outer, padding=(8, 0, 0, 0))
        right.pack(side='left', fill='both', expand=True)
        self.left_panel, self.right_panel = left, right

        self.hp_canvas = tk.Canvas(left, width=COLS * CELL + 2 * MARGIN + 2 * SIDE_COL_W, height=48,
                                   highlightthickness=0)
        self.hp_canvas.pack()

        w = COLS * CELL + 2 * MARGIN
        h = ROWS * CELL + 2 * MARGIN
        board_row = ttk.Frame(left)
        board_row.pack()
        self.capture_canvases = {}
        for name in ('left', 'board', 'right'):
            if name == 'board':
                self.canvas = tk.Canvas(board_row, width=w, height=h, bg=COLORS['board_bg'], highlightthickness=0)
                self.canvas.pack(side='left')
            else:
                canvas = tk.Canvas(board_row, width=SIDE_COL_W, height=h, highlightthickness=0)
                canvas.pack(side='left')
                self.capture_canvases[name] = canvas
        self.canvas.bind('<Button-1>', self.on_click)

        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, font=('Microsoft YaHei UI', 11, 'bold')).pack(pady=(4, 0))
        self.last_move_var = tk.StringVar()
        ttk.Label(left, textvariable=self.last_move_var, foreground='#666666').pack()

        btns = ttk.Frame(left)
        btns.pack(pady=(2, 0))
        ttk.Button(btns, text='悔棋', command=self.undo).pack(side='left', padx=3)
        ttk.Button(btns, text='新对局', command=self.restart).pack(side='left', padx=3)

        # 右侧：提示 / 设置 / 棋谱 三个标签页；宽度固定为棋盘区的一半（fit_to_screen 设置）
        right.pack_propagate(False)
        self.wrap_labels = []  # 随右侧宽度换行的标签

        def wrapped(parent, **kw):
            label = ttk.Label(parent, justify='left', **kw)
            self.wrap_labels.append(label)
            return label

        self.tabs = tabs = ttk.Notebook(right)
        tabs.pack(fill='both', expand=True)
        main_tab = ttk.Frame(tabs, padding=4)
        cfg_tab = ttk.Frame(tabs, padding=4)
        log_tab = ttk.Frame(tabs, padding=4)
        tabs.add(main_tab, text='提示')
        tabs.add(cfg_tab, text='设置')
        tabs.add(log_tab, text='棋谱')

        # 画面同步
        sync_box = ttk.LabelFrame(main_tab, text='画面同步', padding=4)
        sync_box.pack(fill='x', pady=(0, 4))
        self.sync_btn = ttk.Button(sync_box, text='开始同步', command=self.toggle_sync)
        self.sync_btn.pack(anchor='w')
        self.sync_status_var = tk.StringVar(value='从屏幕上的对局画面自动同步每一步（录入+提示模式；游戏窗口需可见）')
        wrapped(sync_box, textvariable=self.sync_status_var, foreground='#666666').pack(anchor='w', pady=(4, 0))

        # AI 提示
        hint_box = ttk.LabelFrame(main_tab, text='AI 提示', padding=4)
        hint_box.pack(fill='x')
        self.hint_title_var = tk.StringVar()
        self.hint_text_var = tk.StringVar()
        self.hint_meta_var = tk.StringVar()
        wrapped(hint_box, textvariable=self.hint_title_var).pack(anchor='w')
        wrapped(hint_box, textvariable=self.hint_text_var, style='Hint.TLabel').pack(anchor='w', pady=4)
        wrapped(hint_box, textvariable=self.hint_meta_var, foreground='#666666').pack(anchor='w')
        hb = ttk.Frame(hint_box)
        hb.pack(anchor='w', pady=(4, 0))
        self.follow_btn = ttk.Button(hb, text='按提示走', width=8, command=self.follow_hint)
        self.follow_btn.pack(side='left')
        ttk.Button(hb, text='重新计算', width=8, command=self.request_ai).pack(side='left', padx=4)

        mult_box = ttk.LabelFrame(main_tab, text='倍率（参考）', padding=4)
        mult_box.pack(fill='x', pady=(4, 0))
        self.mult_var = tk.StringVar()
        self.combo_var = tk.StringVar()
        wrapped(mult_box, textvariable=self.mult_var, font=('Microsoft YaHei UI', 10, 'bold')).pack(anchor='w')
        wrapped(mult_box, textvariable=self.combo_var, foreground='#666666').pack(anchor='w')

        pool_box = ttk.LabelFrame(main_tab, text='剩余棋子池（暗子）', padding=4)
        pool_box.pack(fill='x', pady=(4, 0))
        self.pool_var = tk.StringVar()
        wrapped(pool_box, textvariable=self.pool_var).pack(anchor='w')

        # 设置：标签尽量短；选项文字较长的（模式、模型）单独一行
        cfg = ttk.Frame(cfg_tab)
        cfg.pack(fill='x')
        cfg.columnconfigure(1, weight=1)
        row = 0

        def setting(label, widget, wide=False):
            nonlocal row
            lbl = ttk.Label(cfg, text=label)
            if wide:
                lbl.grid(row=row, column=0, columnspan=2, sticky='w')
                widget.grid(row=row + 1, column=0, columnspan=2, sticky='we', pady=(0, 2))
                row += 2
            else:
                lbl.grid(row=row, column=0, sticky='w', padx=(0, 4))
                widget.grid(row=row, column=1, sticky='w', pady=1)
                row += 1
            return lbl

        def spinbox(var, lo, hi, step, command, bind=None):
            box = ttk.Spinbox(cfg, from_=lo, to=hi, increment=step, textvariable=var, width=4, command=command)
            box.bind('<FocusOut>', bind or (lambda _e: command()))
            return box

        mode_box = ttk.Combobox(cfg, textvariable=self.mode_var, values=list(MODE_LABELS.values()),
                                state='readonly', width=10)
        mode_box.bind('<<ComboboxSelected>>', self.on_mode_changed)
        setting('模式', mode_box, wide=True)
        self.player_side_box = ttk.Combobox(cfg, textvariable=self.player_side_var,
                                            values=[SIDE_NAMES['red'], SIDE_NAMES['black']], state='readonly', width=5)
        self.player_side_box.bind('<<ComboboxSelected>>', self.on_mode_changed)
        self.player_side_label = setting('我执', self.player_side_box)
        self.model_box = ttk.Combobox(cfg, textvariable=self.model_var, values=AI_MODELS, state='readonly', width=16)
        self.model_box.bind('<<ComboboxSelected>>', self.on_setting_changed)
        setting('策略版本 / 模型', self.model_box, wide=True)
        self.engine_box = ttk.Combobox(cfg, textvariable=self.engine_var, values=['Python', 'Go'],
                                       state='readonly', width=6)
        self.engine_box.bind('<<ComboboxSelected>>', self.on_engine_changed)
        setting('计算引擎', self.engine_box)
        self.threads_spin = spinbox(self.threads_var, 1, 64, 1, self.on_setting_changed)
        setting('Go 线程', self.threads_spin)
        setting('长捉上限', spinbox(self.long_chase_var, 2, 20, 1, self.on_rule_setting_changed,
                                  self.on_rule_setting_changed))
        setting('和棋步数', spinbox(self.no_capture_var, 1, 100, 1, self.on_rule_setting_changed,
                                  self.on_rule_setting_changed))
        setting('提示时间(秒)', spinbox(self.budget_var, 0.3, 30, 0.5, self.save_settings))
        setting('起始层数', spinbox(self.hint_depth_var, 0, 20, 1, self.save_settings))
        self.ai_budget_spin = spinbox(self.ai_budget_var, 0.3, 30, 0.5, self.save_settings)
        setting('电脑时间(秒)', self.ai_budget_spin)
        side_box = ttk.Combobox(cfg, textvariable=self.hint_side_var, values=list(HINT_SIDE_LABELS.values()),
                                state='readonly', width=5)
        side_box.bind('<<ComboboxSelected>>', self.on_setting_changed)
        setting('提示方', side_box)
        ttk.Checkbutton(cfg, text='在棋盘上标出提示', variable=self.show_on_board_var,
                        command=lambda: (self.save_settings(), self.draw_board())).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(2, 0))
        row += 1
        ttk.Checkbutton(cfg, text='对方回合提前计算（Go）', variable=self.ponder_var,
                        command=self.on_setting_changed).grid(row=row, column=0, columnspan=2, sticky='w')
        wrapped(cfg_tab, text='自动保存到 config.json。提示时间：给提示的思考时间；起始层数：V16/V17 搜到这一层就先给提示，'
                              '之后随深度更新（0 = 只给最终结果）；和棋步数：连续未吃子的普通移子数；'
                              '电脑时间只用于人机对战。思考时间对 V15~V17 生效。'
                              '提前计算：不需要提示的一方行动时，Go 引擎在后台先算，轮到自己时提示更快更深。',
                foreground='#888888').pack(anchor='w', pady=(6, 0))
        self.update_engine_widgets()

        # 棋谱
        log_btns = ttk.Frame(log_tab)
        log_btns.pack(fill='x', pady=(0, 4))
        ttk.Button(log_btns, text='保存棋谱', width=8, command=self.save_log).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(log_btns, text='导出局面', width=8, command=self.export_state).grid(row=0, column=1)
        ttk.Button(log_btns, text='导入局面', width=8, command=self.import_state).grid(row=1, column=0, padx=(0, 4),
                                                                                    pady=(4, 0))
        log_box = ttk.Frame(log_tab)
        log_box.pack(fill='both', expand=True)
        # 用自动换行的文本框，窄面板里长着法也能看全
        self.log_text = tk.Text(log_box, height=10, width=10, wrap='char', font=('Microsoft YaHei UI', 9),
                                relief='flat', state='disabled')
        self.log_text.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(log_box, command=self.log_text.yview)
        sb.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=sb.set)

        rules = ('规则：4列×8行；将>仕>象>车>马>炮，兵可吃将、将不能吃兵；'
                 '炮隔一子跳吃（可打暗子），不能空走；每步翻一子或走一格；'
                 'HP 将30 仕10 象/车/马/炮5 兵2，双方各60，先扣完者负。')
        wrapped(log_tab, text=rules, foreground='#666666').pack(anchor='w', pady=(4, 0))
        self.update_mode_widgets()

    def fit_to_screen(self):
        """按屏幕大小定尺寸：棋盘区（棋盘 + 两侧被吃子栏）与右侧面板宽度约 2:1，窗口总宽不超过屏幕的 42%、
        高度不超过屏幕（扣除任务栏和标题栏），然后摆到屏幕右上角。"""
        root = self.root
        root.update_idletasks()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        board_h = ROWS * CELL + 2 * MARGIN
        other_h = self.left_panel.winfo_reqheight() - board_h + 12  # 血量条、状态行、按钮
        width_per_cell = COLS + 2 * 0.39 + 2 * 0.72  # 棋盘区宽度 / 格子边长
        by_w = (screen_w * 0.42 - 30) / 1.5 / width_per_cell
        by_h = (screen_h - 90 - other_h) / (ROWS + 2 * 0.39)
        set_board_size(max(40, min(72, int(min(by_w, by_h)))))
        w, h = COLS * CELL + 2 * MARGIN, ROWS * CELL + 2 * MARGIN
        self.canvas.configure(width=w, height=h)
        for canvas in self.capture_canvases.values():
            canvas.configure(width=SIDE_COL_W, height=h)
        board_w = w + 2 * SIDE_COL_W
        self.hp_canvas.configure(width=board_w)
        panel_w = board_w // 2
        self.right_panel.configure(width=panel_w)
        for label in self.wrap_labels:
            label.configure(wraplength=panel_w - 34)
        root.update_idletasks()
        x = max(0, screen_w - root.winfo_reqwidth() - 16)
        root.geometry(f'+{x}+0')

    def update_mode_widgets(self):
        state = 'readonly' if self.is_vs() else 'disabled'
        self.player_side_box.configure(state=state)
        self.ai_budget_spin.configure(state='normal' if self.is_vs() else 'disabled')
        if self.is_vs():
            self.stop_sync()
        self.sync_btn.state(['disabled'] if self.is_vs() else ['!disabled'])

    # -------------------------------------------------------------- drawing
    def cell_rect(self, row, col):
        x0 = MARGIN + col * CELL
        y0 = MARGIN + row * CELL
        return x0, y0, x0 + CELL, y0 + CELL

    def cell_center(self, row, col):
        x0, y0, x1, y1 = self.cell_rect(row, col)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def draw_board(self):
        c = self.canvas
        c.delete('all')
        g = self.game

        for col in range(COLS):
            x, _ = self.cell_center(0, col)
            c.create_text(x, MARGIN / 2, text=str(col), fill=COLORS['grid'])
        for row in range(ROWS):
            _, y = self.cell_center(row, 0)
            c.create_text(MARGIN / 2, y, text=str(row), fill=COLORS['grid'])

        highlight = {}
        la = g.last_action
        if la:
            if la['type'] == 'flip':
                highlight[(la['row'], la['col'])] = COLORS['last']
            else:
                highlight[(la['from']['row'], la['from']['col'])] = COLORS['last']
                highlight[(la['to']['row'], la['to']['col'])] = COLORS['last']
        if g.phase == 'setup':
            pos = g.get_current_setup_pos()
            if pos:
                highlight[(pos['row'], pos['col'])] = COLORS['setup']

        for row in range(ROWS):
            for col in range(COLS):
                fill = highlight.get((row, col), COLORS['board_bg'])
                c.create_rectangle(*self.cell_rect(row, col), fill=fill, outline=COLORS['grid'])

        for row in range(ROWS):
            for col in range(COLS):
                self.draw_piece(row, col, g.board[row][col])

        if self.selected:
            c.create_rectangle(*self.cell_rect(*self.selected), outline=COLORS['selected'], width=4)
        for m in self.valid_moves:
            x, y = self.cell_center(m['row'], m['col'])
            if m['type'] == 'move':
                c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=COLORS['valid'], outline='')
            else:
                c.create_rectangle(*self.cell_rect(m['row'], m['col']), outline=COLORS['valid'], width=4)

        if self.hint and self.hint.get('action') and self.show_on_board_var.get():
            self.draw_hint(self.hint['action'])

    def draw_piece(self, row, col, piece):
        if not piece:
            return
        c = self.canvas
        x, y = self.cell_center(row, col)
        if not piece['face_up']:
            c.create_oval(x - PIECE_R, y - PIECE_R, x + PIECE_R, y + PIECE_R,
                          fill=COLORS['hidden'], outline='#4a3418', width=2)
            c.create_text(x, y, text='?', fill=COLORS['hidden_text'],
                          font=('Microsoft YaHei UI', max(9, PIECE_FONT * 4 // 5), 'bold'))
            return
        color = COLORS[piece['side']]
        c.create_oval(x - PIECE_R, y - PIECE_R, x + PIECE_R, y + PIECE_R, fill=COLORS['face'], outline=color, width=3)
        c.create_oval(x - PIECE_R + 5, y - PIECE_R + 5, x + PIECE_R - 5, y + PIECE_R - 5, outline=color, width=1)
        c.create_text(x, y, text=PIECE_NAMES[piece['side']][piece['type']], fill=color,
                      font=('Microsoft YaHei UI', PIECE_FONT, 'bold'))

    def draw_hint(self, action):
        c = self.canvas
        if action['type'] == 'flip':
            x0, y0, x1, y1 = self.cell_rect(action['row'], action['col'])
            c.create_rectangle(x0 + 3, y0 + 3, x1 - 3, y1 - 3, outline=COLORS['hint'], width=4, dash=(6, 3))
            return
        fx, fy = self.cell_center(action['from_row'], action['from_col'])
        tx, ty = self.cell_center(action['to_row'], action['to_col'])
        x0, y0, x1, y1 = self.cell_rect(action['to_row'], action['to_col'])
        c.create_rectangle(x0 + 3, y0 + 3, x1 - 3, y1 - 3, outline=COLORS['hint'], width=4, dash=(6, 3))
        c.create_line(fx, fy, tx, ty, fill=COLORS['hint'], width=5, arrow='last', arrowshape=(16, 20, 7))

    def bottom_side(self):
        """画面下方（自己）一方：画面同步时是识别出的执子方，人机对战是玩家一方，录入模式默认红方。"""
        return self.sync_side() or (self.player_side if self.is_vs() else 'red')

    def draw_captured(self):
        """棋盘两侧的被吃子栏：左栏自己被吃的子，右栏对方的。"""
        g = self.game
        captured = captured_pieces(g)
        mine = self.bottom_side()
        la = g.last_action
        latest = (la['defender']['side'], la['defender']['type']) if la and la['type'] == 'capture' else None
        for name, order, first_row in CAPTURE_COLUMNS:
            side = mine if name == 'left' else other_side(mine)
            c = self.capture_canvases[name]
            c.delete('all')
            x = SIDE_COL_W / 2
            y0 = MARGIN + first_row * CELL
            c.create_text(x, y0 - ICON_R - 14, text=SIDE_NAMES[side][0], fill=COLORS[side],
                          font=('Microsoft YaHei UI', 10, 'bold'))
            for i, t in enumerate(order):
                y = y0 + i * ICON_STEP
                n = captured[side][t]
                if n:
                    c.create_oval(x - ICON_R, y - ICON_R, x + ICON_R, y + ICON_R,
                                  fill=COLORS['face'], outline=COLORS[side], width=2)
                    color = COLORS[side]
                else:
                    c.create_oval(x - ICON_R, y - ICON_R, x + ICON_R, y + ICON_R, outline='#d6d6d6')
                    color = '#cfcfcf'
                c.create_text(x, y, text=PIECE_NAMES[side][t], fill=color, font=('Microsoft YaHei UI', ICON_FONT, 'bold'))
                if n >= 2:
                    bx, by = x + ICON_R - 3, y - ICON_R + 3
                    c.create_oval(bx - 8, by - 8, bx + 8, by + 8, fill='#c0392b', outline='white')
                    c.create_text(bx, by, text=str(n), fill='white', font=('Microsoft YaHei UI', 8, 'bold'))
                if latest == (side, t):  # 最近被吃的一种
                    c.create_rectangle(x - ICON_R - 1, y + ICON_R - 9, x - ICON_R + 7, y + ICON_R - 1,
                                       fill='#c0392b', outline='')
            lost = lost_hp(captured[side])
            if lost:
                c.create_text(x, y0 + len(order) * ICON_STEP - 8, text=f'-{lost}', fill=COLORS[side],
                              font=('Microsoft YaHei UI', 9, 'bold'))

    def draw_hp(self):
        c = self.hp_canvas
        c.delete('all')
        width = COLS * CELL + 2 * MARGIN + 2 * SIDE_COL_W
        left = 92  # 标签“黑方(对方)”的宽度
        bar_w = width - left - 64
        for i, side in enumerate(('red', 'black')):
            y = 4 + i * 22
            hp = max(0, self.game.hp[side])
            label = SIDE_NAMES[side]
            if self.is_vs():
                label += '(我)' if side == self.player_side else '(电脑)'
            elif self.sync_side():
                label += '(我)' if side == self.sync_side() else '(对方)'
            c.create_text(4, y + 8, text=label, anchor='w', fill=COLORS[side], font=('Microsoft YaHei UI', 9, 'bold'))
            c.create_rectangle(left, y, left + bar_w, y + 16, fill='#eeeeee', outline='#cccccc')
            c.create_rectangle(left, y, left + bar_w * hp / INITIAL_HP, y + 16, fill=COLORS[side], outline='')
            c.create_text(left + bar_w + 8, y + 8, text=f'{hp}/{INITIAL_HP}', anchor='w')

    def refresh(self):
        g = self.game
        self.draw_hp()
        self.draw_board()
        if g.phase == 'setup':
            pos = g.get_current_setup_pos()
            self.status_var.set(f'开局放置 {g.setup_index + 1}/4：点击黄色格子 {pos_text(pos["row"], pos["col"])} 录入明子')
        elif g.phase == 'over':
            self.status_var.set(f'对局结束：{SIDE_NAMES[g.winner]}获胜')
        elif self.is_vs():
            who = '轮到你' if g.current_turn == self.player_side else '电脑思考中…'
            self.status_var.set(f'{SIDE_NAMES[g.current_turn]}行动 · {who}')
        elif self.sync_side():
            who = '轮到你' if g.current_turn == self.sync_side() else '对方行动'
            self.status_var.set(f'{SIDE_NAMES[g.current_turn]}行动 · {who}')
        else:
            self.status_var.set(f'{SIDE_NAMES[g.current_turn]}行动')
        self.last_move_var.set(f'上一步：{describe_last_action(g.last_action)}' if g.last_action else '')

        counts = []
        for side in ('red', 'black'):
            parts = []
            for t in PIECE_CONFIG:
                n = sum(1 for p in g.remaining_pool if p['type'] == t and p['side'] == side)
                if n:
                    parts.append(f'{PIECE_NAMES[side][t]}{n}')
            counts.append(f"{SIDE_NAMES[side]}：{' '.join(parts) or '无'}")
        self.pool_var.set('\n'.join(counts))

        m = g.get_multiplier()
        self.mult_var.set(f"{m['multiplier']} = 连击 {m['combo_factor']} ×（翻棋 {m['flip_score']} + 血差 {m['hp_diff']}）")
        # 每段达到 3 次的连击都累计；当前连击段达到 3 次后实时计入。
        counted = [f"{'红' if s == 'red' else '黑'}方 {m['combo_points'][s]} 分"
                   for s in ('red', 'black') if m['combo_points'][s] > 0]
        self.combo_var.set('累计连击：' + (' + '.join(counted) if counted else '无（尚无 3 连击）'))

        self.draw_captured()

        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.insert('end', '\n'.join(f'{i}. {line}' for i, line in enumerate(self.move_log, 1)))
        self.log_text.configure(state='disabled')
        self.log_text.see('end')
        self.update_hint_panel()

    # ------------------------------------------------------------------- AI
    def request_ai(self):
        """局面或设置变化后调用：作废旧结果；需要时在后台线程计算（电脑走子或给提示）。"""
        self.ai_token += 1
        self.hint = None
        self.ai_busy = False
        self.pondering = False
        g = self.game
        purpose = None
        if g.phase == 'playing':
            if self.is_vs() and g.current_turn == self.ai_side:
                purpose = 'move'
            elif g.current_turn in self.hint_sides():
                purpose = 'hint'
            elif self.can_ponder():
                purpose = 'ponder'
        if purpose:
            token = self.ai_token
            model = self.model_var.get()
            budget_ms = {'move': self.ai_think_budget_ms(), 'hint': self.think_budget_ms(),
                         'ponder': max(1.0, float(self.config['ponder_time_sec'])) * 1000}[purpose]
            min_depth = self.hint_min_depth() if purpose == 'hint' else 0
            # 提前计算只为填充引擎的置换表，不显示中间结果
            on_progress = None if purpose == 'ponder' else (lambda h: self.ai_queue.put((token, 'hint_progress', h)))
            search = dict(self.config['search'])
            v16_params = self.config['v16']
            v17_params = self.config['v17']
            engine = 'go' if self.engine_var.get() == 'Go' else 'python'
            try:
                threads = min(64, max(1, int(self.threads_var.get())))
            except (tk.TclError, ValueError):
                threads = 4
            snapshot = g.clone()
            self.ai_busy = purpose == 'move'
            self.pondering = purpose == 'ponder'

            def worker():
                try:
                    # 局面一变（token 过期）就停止计算，不和新的计算抢 CPU
                    result = compute_hint(snapshot, model, budget_ms, search, v16_params, v17_params,
                                          engine, threads, min_depth=min_depth, on_progress=on_progress,
                                          should_stop=lambda: token != self.ai_token)
                except Exception as exc:  # noqa: BLE001 — AI 出错不应影响对局
                    result = {'error': f'{type(exc).__name__}: {exc}'}
                self.ai_queue.put((token, purpose, result))

            threading.Thread(target=worker, daemon=True).start()
        self.update_hint_panel()
        self.draw_board()

    def _poll_ai_queue(self):
        self._poll_sync_queue()
        try:
            while True:
                token, purpose, result = self.ai_queue.get_nowait()
                if token != self.ai_token:
                    continue
                if purpose == 'hint_progress':
                    # 中间结果只在最终结果到达前显示，随深度更新
                    if self.hint is None or (self.hint.get('info') or {}).get('partial'):
                        self.hint = result
                        self.update_hint_panel()
                        self.draw_board()
                elif purpose == 'hint':
                    self.hint = result
                    self.update_hint_panel()
                    self.draw_board()
                elif purpose == 'ponder':
                    pass  # 结果留在引擎置换表里，界面上不用
                elif 'error' in result or not result.get('action'):
                    self.ai_busy = False
                    self.hint_meta_var.set(result.get('error', '电脑无棋可走'))
                else:
                    delay = max(0, int(self.config['ai_move_delay_ms']))
                    self.root.after(delay, lambda t=token, a=result['action']: self.apply_ai_move(t, a))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ai_queue)

    def apply_ai_move(self, token, action):
        if token != self.ai_token:
            return
        self.ai_busy = False
        # AI 搜索时假设的 pool_index 不代表真实翻出的棋子，人机模式下真实结果随机
        if action['type'] == 'flip':
            self.do_flip(action['row'], action['col'])
        else:
            self.do_move(action['from_row'], action['from_col'], action['to_row'], action['to_col'])

    def update_hint_panel(self):
        g = self.game
        self.follow_btn.state(['disabled'])
        self.hint_title_var.set('')
        self.hint_meta_var.set('')
        if g.phase != 'playing':
            self.hint_text_var.set('开局放置阶段不提供提示' if g.phase == 'setup' else '对局已结束')
            return
        side_name = SIDE_NAMES[g.current_turn]
        if self.is_vs() and g.current_turn == self.ai_side:
            self.hint_title_var.set(f'电脑（{side_name}）· {self.model_var.get()}')
            self.hint_text_var.set('电脑思考中…')
            return
        if g.current_turn not in self.hint_sides():
            self.hint_title_var.set(f'{side_name}行动')
            if self.pondering:
                self.hint_text_var.set('对方思考中')
                self.hint_meta_var.set('后台提前计算中，轮到你时提示更快更深')
            else:
                self.hint_text_var.set('（未开启该方提示）')
            return
        if self.hint is None:
            self.hint_title_var.set(f'{side_name} · {self.model_var.get()}')
            self.hint_text_var.set('AI 思考中…')
            return
        if 'error' in self.hint:
            self.hint_title_var.set(f'{side_name} · 提示计算失败')
            self.hint_text_var.set('')
            self.hint_meta_var.set(self.hint['error'])
            return
        info = self.hint.get('info')
        partial = bool(info and info.get('partial'))
        suffix = '当前建议（继续计算中）：' if partial else '建议：'
        self.hint_title_var.set(f"{self.hint['side_name']} · {self.hint['model']} {suffix}")
        self.hint_text_var.set(self.hint['text'])
        meta = [f"{'已用' if partial else '用时'} {self.hint['elapsed_ms'] / 1000:.1f}s"]
        if info:
            if info.get('source') == 'tactical':
                meta.append('依据：紧急战术（护将 / 防兵杀 / 强制吃子）')
            elif info.get('source') == 'search':
                meta.append(f"依据：Alpha-Beta 搜索 {info.get('depth')} 层")
                if info.get('win_prob') is not None:
                    meta.append(f"预估胜率 {info['win_prob']:.0%}")
                elif info.get('score') is not None:
                    meta.append(f"评估 {info['score']:+.0f}")
        action = self.hint.get('action')
        if action and action['type'] == 'flip' and not self.is_vs() and not self.syncing:
            meta.append('翻开后请在弹窗中录入实际翻出的棋子')
        self.hint_meta_var.set(' · '.join(meta))
        if action and not self.syncing:
            self.follow_btn.state(['!disabled'])

    def follow_hint(self):
        if not self.hint or not self.hint.get('action') or self.game.phase != 'playing' or self.ai_busy \
                or self.syncing:
            return
        a = self.hint['action']
        if a['type'] == 'flip':
            self.do_flip(a['row'], a['col'])
        else:
            self.do_move(a['from_row'], a['from_col'], a['to_row'], a['to_col'])

    # ---------------------------------------------------------- interaction
    def on_click(self, event):
        if self.syncing:
            return  # 局面由画面同步，手动录入会和画面对不上
        col = int((event.x - MARGIN) // CELL)
        row = int((event.y - MARGIN) // CELL)
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return
        g = self.game
        if g.phase == 'setup':
            pos = g.get_current_setup_pos()
            if pos and pos['row'] == row and pos['col'] == col:
                idx = self.choose_piece(f'开局放置 {pos_text(row, col)} 的明子')
                if idx >= 0:
                    self.place_setup(idx)
                    self.after_action()
            return
        if g.phase != 'playing':
            return
        if self.is_vs() and g.current_turn != self.player_side:
            return  # 电脑回合

        if self.selected:
            target = next((m for m in self.valid_moves if m['row'] == row and m['col'] == col), None)
            if target:
                self.do_move(self.selected[0], self.selected[1], row, col)
                return

        piece = g.get_piece(row, col)
        if piece and piece['face_up'] and piece['side'] == g.current_turn:
            if self.selected == (row, col):
                self.clear_selection()
            else:
                self.selected = (row, col)
                self.valid_moves = g.get_valid_moves(row, col)
            self.draw_board()
            return
        if piece and not piece['face_up']:
            self.do_flip(row, col)
            return
        self.clear_selection()
        self.draw_board()

    def clear_selection(self):
        self.selected = None
        self.valid_moves = []

    def choose_piece(self, title):
        chooser = PieceChooser(self.root, self.game, title)
        self.root.wait_window(chooser)
        return chooser.result

    def reveal_index(self, title):
        """暗子的真实身份：人机模式随机抽取，录入模式由用户选择。"""
        if self.is_vs():
            return random.randrange(len(self.game.remaining_pool))
        return self.choose_piece(title)

    def place_setup(self, idx):
        g = self.game
        pos = g.get_current_setup_pos()
        piece = g.remaining_pool[idx]
        g.setup_placement(idx)
        self.move_log.append(f"开局 {pos_text(pos['row'], pos['col'])} 放置 "
                             f"{PIECE_NAMES[piece['side']][piece['type']]}（{SIDE_NAMES[piece['side']]}）")

    def do_flip(self, row, col):
        g = self.game
        idx = self.reveal_index(f'{SIDE_NAMES[g.current_turn]}翻开 {pos_text(row, col)}：实际翻出的是？')
        if idx < 0:
            return
        side = SIDE_NAMES[g.current_turn]
        if g.flip_piece(row, col, idx):
            self.move_log.append(f'{side} {describe_last_action(g.last_action)}')
            self.after_action()

    def do_move(self, fr, fc, tr, tc):
        g = self.game
        target = g.get_piece(tr, tc)
        idx = None
        if target and not target['face_up']:
            idx = self.reveal_index(f'炮打 {pos_text(tr, tc)} 的暗子：被打掉的是？')
            if idx < 0:
                return
        mover = g.current_turn
        if g.move_piece(fr, fc, tr, tc, idx):
            line = f'{SIDE_NAMES[mover]} {describe_last_action(g.last_action)}'
            if g.combo[mover] >= COMBO_THRESHOLD:
                line += f' · {g.combo[mover]}连击'
            self.move_log.append(line)
            self.after_action()

    def after_action(self):
        self.clear_selection()
        if self.game.phase == 'over':
            if self.game.winner == 'draw':
                self.move_log.append(
                    f'—— 和棋（连续 {self.game.no_capture_limit} 个普通移子未吃子）——')
            else:
                self.move_log.append(f'—— {SIDE_NAMES[self.game.winner]}获胜 ——')
        self.request_ai()
        self.refresh()
        if self.game.phase == 'over':
            if self.game.winner == 'draw':
                messagebox.showinfo(
                    '对局结束', f'连续 {self.game.no_capture_limit} 个普通移子未吃子，和棋。')
                return
            if self.is_vs():
                result = '你赢了！' if self.game.winner == self.player_side else '电脑获胜。'
                messagebox.showinfo('对局结束', f'{SIDE_NAMES[self.game.winner]}获胜，{result}')
            else:
                messagebox.showinfo('对局结束', f'{SIDE_NAMES[self.game.winner]}获胜！')

    def undo(self):
        g = self.game
        if len(g.history) <= self.min_history:
            return
        if g.phase == 'over' and self.move_log and self.move_log[-1].startswith('——'):
            self.move_log.pop()
        steps = 1
        if self.is_vs():
            # 人机对战悔棋退回到玩家上一次行动之前（通常是两步：电脑一步 + 自己一步）
            steps = 0
            while len(g.history) - steps > self.min_history:
                steps += 1
                if g.history[-1 - steps]['current_turn'] == self.player_side:
                    break
            if g.history[-1 - steps]['current_turn'] != self.player_side:
                return
        g.undo_steps(steps)
        del self.move_log[len(self.move_log) - steps:]
        self.after_action()

    def new_game(self):
        self.ai_token += 1
        rules = self.config.get('rules', {})
        self.game = Game(
            long_chase_limit=rules.get('long_chase_limit', DEFAULT_LONG_CHASE_LIMIT),
            no_capture_limit=rules.get('no_capture_limit', DEFAULT_NO_CAPTURE_LIMIT),
        )
        self.move_log = []
        self.min_history = 1
        if self.is_vs():
            # 人机对战：开局 4 个固定位置随机放置明子
            while self.game.phase == 'setup':
                self.place_setup(random.randrange(len(self.game.remaining_pool)))
            self.min_history = len(self.game.history)
        self.after_action()

    def restart(self):
        if self.move_log and self.game.phase != 'over' and not messagebox.askyesno('新对局', '确定放弃当前对局并重新开始吗？'):
            return
        self.new_game()

    def export_state(self):
        if self.game.phase == 'setup':
            messagebox.showinfo('导出局面', '开局放置完成后才能导出局面')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.json', filetypes=[('JSON', '*.json')],
            initialfile=time.strftime('state_%Y%m%d_%H%M%S.json'))
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(game_to_state(self.game), f, ensure_ascii=False, indent=2)

    def import_state(self):
        """从状态文件载入局面（例如外部识别程序生成的），之后继续对局 / 给提示。"""
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                game = game_from_state(json.load(f))
        except (OSError, ValueError) as exc:
            messagebox.showerror('导入局面失败', str(exc))
            return
        self.ai_token += 1
        self.game = game
        self.long_chase_var.set(game.long_chase_limit)
        self.no_capture_var.set(game.no_capture_limit)
        self.save_settings()
        self.move_log = [f'导入局面：{path.replace(chr(92), "/").rsplit("/", 1)[-1]}']
        # 导入的局面没有更早的历史，悔棋不能越过它
        self.min_history = len(game.history)
        self.after_action()

    # ------------------------------------------------------------ 画面同步
    def toggle_sync(self):
        if self.syncing:
            self.stop_sync()
        else:
            self.start_sync()

    def start_sync(self):
        if self.is_vs():
            return
        try:
            from .sync import BoardSync, ScreenWatcher
            from .vision import default_templates
            default_templates()
        except ImportError:
            messagebox.showerror('画面同步', '画面识别需要 numpy 和 Pillow：\npip install numpy pillow')
            return
        except FileNotFoundError as exc:
            messagebox.showerror('画面同步', str(exc))
            return
        self.board_sync = BoardSync(self.config.get('rules'))
        self.sync_synced = False  # 第一次识别结果出来前，界面上的对局与画面无关
        self.sync_watcher = ScreenWatcher(lambda kind, payload: self.sync_queue.put((kind, payload)))
        self.sync_watcher.start()
        self.sync_btn.configure(text='停止同步')
        self.sync_status_var.set('正在查找屏幕上的棋盘…')
        self.clear_selection()
        self.request_ai()
        self.refresh()

    def stop_sync(self):
        if not self.syncing:
            return
        self.sync_watcher.stop()
        self.sync_watcher = None
        while not self.sync_queue.empty():
            self.sync_queue.get_nowait()
        self.sync_btn.configure(text='开始同步')
        self.sync_status_var.set('已停止同步；可以继续手动录入')
        self.request_ai()
        self.refresh()

    def _poll_sync_queue(self):
        while self.syncing:
            try:
                kind, payload = self.sync_queue.get_nowait()
            except queue.Empty:
                return
            if kind == 'status':
                self.sync_status_var.set(payload)
            else:
                self.apply_sync(payload)

    def apply_sync(self, rec):
        update = self.board_sync.update(rec, self.game if self.sync_synced else None)
        if update.kind in ('step', 'turn', 'new', 'import'):
            self.sync_synced = True
            self.game = update.game
            if update.kind in ('new', 'import'):
                self.move_log = list(update.log)
                # 同步开始前的历史不存在，悔棋不能越过它
                self.min_history = len(self.game.history)
            else:
                self.move_log.extend(update.log)
            if self.game.phase == 'over':
                self.move_log.append('—— 和棋 ——' if self.game.winner == 'draw'
                                     else f'—— {SIDE_NAMES[self.game.winner]}获胜 ——')
            self.clear_selection()
            self.request_ai()
            self.refresh()
        mine = self.sync_side()
        parts = ['同步中', f'我执{SIDE_NAMES[mine]}' if mine else '还认不出我执哪一方（翻出第一个子后确定）']
        if self.sync_synced and self.game.phase == 'playing' and mine:
            parts.append('轮到我' if self.game.current_turn == mine else '对方行动')
        if update.message:
            parts.append(update.message)
        self.sync_status_var.set(' · '.join(parts))

    def save_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json', filetypes=[('JSON', '*.json')],
            initialfile=time.strftime('fanfanqi_%Y%m%d_%H%M%S.json'))
        if not path:
            return
        data = {
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.mode,
            'player_side': self.player_side if self.is_vs() else None,
            'ai_model': self.model_var.get(),
            'winner': self.game.winner,
            'hp': self.game.hp,
            'moves': self.move_log,
            'actions': [s['last_action'] for s in self.game.history[1:]],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
