"""Adapter for the optional Go search engine."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from .state import game_to_state

_BUILD_LOCK = threading.Lock()
_ENGINE_DIR = Path(__file__).resolve().parent.parent / 'goengine'


def _engine_binary():
    go = shutil.which('go')
    if not go:
        raise RuntimeError('未找到 Go 编译器。请安装 Go 1.21 或更高版本后重试。')
    source_files = (_ENGINE_DIR / 'go.mod', _ENGINE_DIR / 'main.go')
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.read_bytes())
    suffix = '.exe' if os.name == 'nt' else ''
    binary = Path(tempfile.gettempdir()) / f'fanfanqi-go-{digest.hexdigest()[:16]}{suffix}'
    if binary.is_file():
        return binary
    with _BUILD_LOCK:
        if binary.is_file():
            return binary
        result = subprocess.run(
            [go, 'build', '-trimpath', '-o', str(binary), '.'],
            cwd=str(_ENGINE_DIR), capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=180,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f'Go 引擎编译失败：{detail or result.returncode}')
    return binary


class GoEngineAI:
    def __init__(self, model_name, side, time_budget_ms, params=None, threads=1):
        self.model_name = model_name
        self.side = side
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.params = params or {}
        self.threads = min(64, max(1, int(threads)))
        self.last_decision_info = None

    def get_move(self, game):
        request = {
            'strategy': self.model_name,
            'threads': self.threads,
            'time_budget_ms': self.time_budget_ms,
            'state': game_to_state(game),
            'params': self.params,
        }
        try:
            result = subprocess.run(
                [str(_engine_binary())], input=json.dumps(request, ensure_ascii=False),
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=self.time_budget_ms / 1000 + 15,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Go 引擎进程超时') from exc
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip() or
                               f'Go 引擎退出码 {result.returncode}')
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Go 引擎返回了无效响应：{result.stdout[:300]}') from exc
        if response.get('error'):
            raise RuntimeError(response['error'])
        self.last_decision_info = response.get('info')
        return response.get('action')
