"""Adapter for the optional Go search engine."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .state import game_to_state

_BUILD_LOCK = threading.Lock()
_ENGINE_DIR = Path(__file__).resolve().parent.parent / 'goengine'


def _engine_binary():
    # 指定已编译好的引擎（例如长时间跑基准时固定版本，不受源码修改影响）
    override = os.environ.get('FANFANQI_GO_BINARY')
    if override:
        return Path(override)
    source_files = (_ENGINE_DIR / 'go.mod', _ENGINE_DIR / 'main.go')
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.read_bytes())
    suffix = '.exe' if os.name == 'nt' else ''
    binary = Path(tempfile.gettempdir()) / f'fanfanqi-go-{digest.hexdigest()[:16]}{suffix}'
    if binary.is_file():
        return binary
    # 只有需要编译时才找 Go 编译器；已编译好的引擎不依赖它
    go = shutil.which('go')
    if not go:
        raise RuntimeError('未找到 Go 编译器。请安装 Go 1.21 或更高版本后重试。')
    with _BUILD_LOCK:
        if binary.is_file():
            return binary
        # 多个进程（如 gen_data --workers）可能同时编译：各自写临时文件再改名，避免互相覆盖
        temp = binary.with_name(f'{binary.stem}-{os.getpid()}.tmp{suffix}')
        result = subprocess.run(
            [go, 'build', '-trimpath', '-o', str(temp), '.'],
            cwd=str(_ENGINE_DIR), capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=180,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f'Go 引擎编译失败：{detail or result.returncode}')
        try:
            os.replace(temp, binary)
        except OSError:
            # 另一个进程已经放好了同样的引擎（Windows 上正在运行的 exe 不能被替换）
            temp.unlink(missing_ok=True)
            if not binary.is_file():
                raise
    return binary


class GoEngineAI:
    def __init__(self, model_name, side, time_budget_ms, params=None, threads=1):
        self.model_name = model_name
        self.side = side
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.params = params or {}
        self.threads = min(64, max(1, int(threads)))
        self.last_decision_info = None
        # 与 StrategicV16AI / V17AI 相同：on_progress(action, info) 每完成一层回调；should_stop() 为 True 时结束引擎进程
        self.on_progress = None
        self.should_stop = None

    def get_move(self, game):
        request = {
            'strategy': self.model_name,
            'threads': self.threads,
            'time_budget_ms': self.time_budget_ms,
            'state': game_to_state(game),
            'params': self.params,
            'progress': self.on_progress is not None,
        }
        proc = subprocess.Popen(
            [str(_engine_binary())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
        )
        ended = threading.Event()
        why = {}

        def watch():
            # 超时或被取消时结束引擎进程；stdout 随之关闭，下面的读循环自然结束
            deadline = time.monotonic() + self.time_budget_ms / 1000 + 15
            while not ended.wait(0.05):
                if self.should_stop is not None and self.should_stop():
                    why['stopped'] = True
                elif time.monotonic() > deadline:
                    why['timeout'] = True
                else:
                    continue
                proc.kill()
                return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        response = None
        lines = []
        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            proc.stdin.close()
            for line in proc.stdout:
                lines.append(line)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'progress' in message:
                    if self.on_progress is not None:
                        progress = message['progress']
                        self.on_progress(progress.get('action'), progress.get('info') or {})
                else:
                    response = message
            stderr = proc.stderr.read()
            proc.wait()
        finally:
            ended.set()
        if why.get('stopped'):
            self.last_decision_info = {'source': 'search', 'stopped': True}
            return None
        if why.get('timeout'):
            raise RuntimeError('Go 引擎进程超时')
        if proc.returncode:
            raise RuntimeError(stderr.strip() or ''.join(lines).strip() or f'Go 引擎退出码 {proc.returncode}')
        if response is None:
            raise RuntimeError(f'Go 引擎返回了无效响应：{"".join(lines)[:300]}')
        if response.get('error'):
            raise RuntimeError(response['error'])
        self.last_decision_info = response.get('info')
        return response.get('action')
