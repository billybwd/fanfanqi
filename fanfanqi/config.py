"""配置文件读写：项目根目录下的 config.json。

界面里修改的设置会自动写回该文件；也可以直接编辑它（例如调整 V15 的搜索参数），
重启程序后生效。缺失或写错的项自动使用默认值。
"""

import copy
import json
import os
from typing import Any, Dict

from .v16 import DEFAULT_PARAMS as V16_DEFAULT_PARAMS
from .v17 import DEFAULT_PARAMS as V17_DEFAULT_PARAMS

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')

DEFAULT_CONFIG: Dict[str, Any] = {
    # 'record'：录入实物对局，AI 只给提示；'vs'：人机对战，翻子随机
    'mode': 'record',
    # 人机对战中玩家执哪一方（红方先手）
    'player_side': 'red',
    'ai_model': 'StrategicV17AI',
    'engine': 'python',
    'go_threads': 4,
    'rules': {
        'long_chase_limit': 5,
        'no_capture_limit': 16,
    },
    # 每步思考时间（秒），对 V15、V16、V17 生效
    'think_time_sec': 2.5,
    # 提示方：'both' | 'red' | 'black' | 'off'。人机对战中只会提示玩家一方
    'hint_side': 'both',
    'show_hint_on_board': True,
    # 人机对战中电脑算完后再停顿多久落子（毫秒），便于看清
    'ai_move_delay_ms': 400,
    # StrategicV15AI 搜索参数（高级）。数值越大越强也越慢；含义见 README
    'search': {
        'max_iterative_depth': 7,
        'max_root_actions': 16,
        'max_node_actions': 8,
        'max_noisy_actions': 6,
        'max_flip_squares': 4,
        'max_pool_choices': 2,
        'quiescence_depth': 2,
    },
    # StrategicV16AI 参数（目标函数、搜索宽度、评估权重）；含义见 fanfanqi/v16.py
    'v16': copy.deepcopy(V16_DEFAULT_PARAMS),
    'v17': copy.deepcopy(V17_DEFAULT_PARAMS),
}


def _merge(defaults: Dict[str, Any], loaded: Any) -> Dict[str, Any]:
    result = copy.deepcopy(defaults)
    if not isinstance(loaded, dict):
        return result
    for key, default in defaults.items():
        if key not in loaded:
            continue
        value = loaded[key]
        if isinstance(default, dict):
            result[key] = _merge(default, value)
        elif isinstance(default, bool):
            if isinstance(value, bool):
                result[key] = value
        elif isinstance(default, (int, float)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = value
        elif isinstance(value, type(default)):
            result[key] = value
    return result


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(path, encoding='utf-8') as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        loaded = {}
    v16 = loaded.get('v16') if isinstance(loaded, dict) else None
    if isinstance(v16, dict) and 'objective' in v16 and 'multiplier_exponent' not in v16:
        # 旧版 'reward' / 'win' 换算成 λ
        v16['multiplier_exponent'] = 0.0 if v16['objective'] == 'win' else 1.0
    return _merge(DEFAULT_CONFIG, loaded)


def save_config(config: Dict[str, Any], path: str = CONFIG_PATH) -> None:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # 只读目录等情况下静默放弃保存，不影响使用
