#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# core/utils.py
"""
工具类模块 - Logger、Timer、DeviceInfo 等
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Optional


class Logger:
    """日志输出类"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
        self._lock = threading.Lock()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message: str, indent: int = 0):
        if self.verbose:
            with self._lock:
                prefix = " " * indent if indent > 0 else f"[{self._timestamp()}] [INFO] "
                print(f"{prefix}{message}")

    def progress(self, current: int, total: int, start_time: float, extra: str = ""):
        if self.verbose:
            with self._lock:
                elapsed = time.time() - start_time

                if total <= 0:
                    total = 1
                if current > total:
                    current = total

                percent = current / total * 100
                fps = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / fps if fps > 0 else 0

                progress_bar = self._make_progress_bar(percent)

                sys.stdout.write(f"\r[{self._timestamp()}] [PROGRESS] {progress_bar} {current}/{total} ({percent:.1f}%) | "
                      f"已用: {self._format_time(elapsed)} | 剩余: ~{self._format_time(remaining)} | "
                      f"速度: {fps:.2f} fps {extra}    ")
                sys.stdout.flush()

    def _make_progress_bar(self, percent: float, width: int = 20) -> str:
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def success(self, message: str):
        if self.verbose:
            with self._lock:
                print(f"\n[{self._timestamp()}] [SUCCESS] ✓ {message}")

    def error(self, message: str):
        with self._lock:
            print(f"\n[{self._timestamp()}] [ERROR] ✗ {message}", file=sys.stderr)

    def warning(self, message: str):
        if self.verbose:
            with self._lock:
                print(f"[{self._timestamp()}] [WARNING] ⚠ {message}")

    def newline(self):
        if self.verbose:
            print()

    def header(self, title: str):
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"{title:^80}")
            print("=" * 80 + "\n")

    def section(self, title: str):
        if self.verbose:
            print(f"\n{'─' * 40}")
            print(f"  {title}")
            print(f"{'─' * 40}")


class Timer:
    """计时器类"""

    def __init__(self):
        self.stages: Dict[str, Dict] = {}
        self.current_stage: Optional[str] = None
        self._lock = threading.Lock()

    def start(self, stage_name: str):
        with self._lock:
            self.current_stage = stage_name
            self.stages[stage_name] = {
                "start": time.time(),
                "end": None,
                "duration": 0
            }

    def stop(self, stage_name: str = None):
        with self._lock:
            stage = stage_name or self.current_stage
            if stage and stage in self.stages:
                self.stages[stage]["end"] = time.time()
                self.stages[stage]["duration"] = (
                    self.stages[stage]["end"] - self.stages[stage]["start"]
                )

    def get_duration(self, stage_name: str) -> float:
        if stage_name in self.stages:
            return self.stages[stage_name]["duration"]
        return 0

    def get_total(self) -> float:
        return sum(s["duration"] for s in self.stages.values())


class DeviceInfo:
    """设备信息获取器"""

    @staticmethod
    def get_paddle_device_info() -> Dict:
        """获取 Paddle 设备信息"""
        import paddle

        device_info = {}

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device_name = "GPU"
            try:
                gpu_info = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                if gpu_info.returncode == 0:
                    device_name = f"GPU ({gpu_info.stdout.strip().split(chr(10))[0]})"
            except:
                pass
            device_info["paddle_device"] = device_name
            device_info["paddle_use_gpu"] = True
        else:
            device_info["paddle_device"] = "CPU"
            device_info["paddle_use_gpu"] = False

        device_info["ffmpeg_decode"] = "CPU (FFmpeg)"
        device_info["ffmpeg_encode"] = "CPU (FFmpeg)"

        return device_info