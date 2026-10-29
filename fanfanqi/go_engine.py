"""Adapter for the optional Go search engine (a long-lived process, see _EngineProcess)."""

import collections
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


class _EngineProcess:
    """常驻的 Go 引擎进程：一行一个请求，进程不退出，置换表跨请求保留（上一步、对手思考时算过的局面直接复用）。
    同一进程内的请求按顺序执行；should_stop 为真时发 {"cmd":"stop"}，引擎立即返回当前最好的结果。"""

    def __init__(self, binary):
        self.binary = binary
        self.proc = None
        self.lock = threading.Lock()        # 一次只跑一个请求
        self.write_lock = threading.Lock()  # 请求和 stop 命令可能来自不同线程
        self.stderr_tail = collections.deque(maxlen=200)

    def _start(self):
        self.proc = subprocess.Popen(
            [str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
        )
        proc, tail = self.proc, self.stderr_tail
        tail.clear()
        # 持续读走 stderr，避免管道写满卡住引擎
        threading.Thread(target=lambda: tail.extend(proc.stderr), daemon=True).start()

    def _send(self, proc, message):
        with self.write_lock:
            proc.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
            proc.stdin.flush()

    def _kill(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def run(self, request, on_progress, should_stop, timeout_s):
        """返回 (response, stopped)。"""
        with self.lock:
            for attempt in range(2):
                if self.proc is None or self.proc.poll() is not None:
                    self._start()
                proc = self.proc
                try:
                    self._send(proc, request)
                except OSError:
                    self._kill()  # 进程已退出（例如上次被看门狗结束），重启后再试一次
                    if attempt:
                        raise
                    continue
                return self._read(proc, on_progress, should_stop, timeout_s)

    def _read(self, proc, on_progress, should_stop, timeout_s):
        ended = threading.Event()
        why = {}

        def watch():
            deadline = time.monotonic() + timeout_s
            while not ended.wait(0.05):
                if should_stop is not None and not why.get('stopped') and should_stop():
                    why['stopped'] = True
                    try:
                        self._send(proc, {'cmd': 'stop'})
                    except OSError:
                        pass
                elif time.monotonic() > deadline:
                    why['timeout'] = True
                    proc.kill()  # stdout 随之关闭，下面的读循环结束
                    return

        threading.Thread(target=watch, daemon=True).start()
        response, lines = None, []
        try:
            for line in proc.stdout:
                lines.append(line)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'progress' in message:
                    if on_progress is not None:
                        progress = message['progress']
                        on_progress(progress.get('action'), progress.get('info') or {})
                else:
                    response = message
                    break
        finally:
            ended.set()
        if response is None:
            self._kill()
            if why.get('timeout'):
                raise RuntimeError('Go 引擎进程超时')
            detail = ''.join(self.stderr_tail).strip() or ''.join(lines).strip()
            raise RuntimeError(f'Go 引擎意外退出：{detail[-500:] or proc.returncode}')
        return response, bool(why.get('stopped'))


_ENGINES = {}
_ENGINES_LOCK = threading.Lock()


def _engine_process():
    binary = _engine_binary()
    with _ENGINES_LOCK:
        engine = _ENGINES.get(str(binary))
        if engine is None:
            engine = _ENGINES[str(binary)] = _EngineProcess(binary)
        return engine


class GoEngineAI:
    def __init__(self, model_name, side, time_budget_ms, params=None, threads=1):
        self.model_name = model_name
        self.side = side
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.params = params or {}
        self.threads = min(64, max(1, int(threads)))
        self.last_decision_info = None
        # 与 StrategicV16AI / V17AI 相同：on_progress(action, info) 每完成一层回调；should_stop() 为 True 时停止搜索
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
        response, stopped = _engine_process().run(request, self.on_progress, self.should_stop,
                                                  self.time_budget_ms / 1000 + 15)
        if stopped:
            self.last_decision_info = {'source': 'search', 'stopped': True}
            return None
        if response.get('error'):
            raise RuntimeError(response['error'])
        self.last_decision_info = response.get('info')
        return response.get('action')
