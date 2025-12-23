#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理模块 - 负责视频编解码、帧提取等
"""

import json
import os
import subprocess
import sys
from typing import Dict

import numpy as np


def get_ffmpeg_path(name: str = "ffmpeg") -> str:
    """获取 ffmpeg/ffprobe 的路径"""
    if sys.platform == "win32":
        name = f"{name}.exe"

    # 当前工作目录
    local_path = os.path.join(os.getcwd(), name)
    if os.path.isfile(local_path):
        return local_path

    # 项目根目录（core 的上级目录）
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(script_dir, name)
    if os.path.isfile(script_path):
        return script_path

    # 项目根目录下的 bin 子目录
    bin_path = os.path.join(script_dir, "bin", name)
    if os.path.isfile(bin_path):
        return bin_path

    # 项目根目录下的 ffmpeg 子目录
    ffmpeg_path = os.path.join(script_dir, "ffmpeg", name)
    if os.path.isfile(ffmpeg_path):
        return ffmpeg_path

    # 返回命令名称（假设在 PATH 中）
    return name.replace(".exe", "") if sys.platform != "win32" else name


class VideoInfo:
    """视频信息数据类"""

    def __init__(self, probe_data: Dict):
        self.filename = ""
        self.filepath = ""
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.total_frames = 0
        self.duration = 0.0
        self.has_audio = False

        self._parse_probe_data(probe_data)

    def _parse_probe_data(self, probe: Dict):
        """解析 ffprobe 数据"""
        video_stream = None
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise RuntimeError("未找到视频流")

        # 解析帧率
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        self.fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

        # 解析总帧数
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            self.total_frames = int(nb_frames)
        else:
            duration = float(probe.get("format", {}).get("duration", 0))
            self.total_frames = int(duration * self.fps)

        self.width = int(video_stream.get("width", 0))
        self.height = int(video_stream.get("height", 0))
        self.duration = float(probe.get("format", {}).get("duration", 0))
        self.has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration
        }


class VideoProcessor:
    """视频处理器 - 负责视频编解码"""

    def __init__(self):
        self.ffmpeg_path = get_ffmpeg_path("ffmpeg")
        self.ffprobe_path = get_ffmpeg_path("ffprobe")

    def get_video_info(self, video_path: str) -> VideoInfo:
        """获取视频信息"""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"无法读取视频信息: {video_path}")

        probe = json.loads(result.stdout)
        video_info = VideoInfo(probe)
        video_info.filename = os.path.basename(video_path)
        video_info.filepath = video_path

        return video_info

    def create_reader(self, video_path: str):
        """创建视频读取进程"""
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-"
        ]

        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def create_writer(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        keep_audio: bool = False,
        input_path: str = None
    ):
        """创建视频写入进程"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-v", "quiet"
        ]

        if keep_audio and input_path:
            cmd.extend(["-i", input_path, "-c:a", "aac", "-map", "0:v", "-map", "1:a"])

        cmd.append(output_path)

        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_frame(self, process, width: int, height: int) -> np.ndarray:
        """从进程读取一帧"""
        frame_size = width * height * 3
        raw_frame = process.stdout.read(frame_size)

        if len(raw_frame) != frame_size:
            return None

        return np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

    def write_frame(self, process, frame: np.ndarray):
        """向进程写入一帧"""
        process.stdin.write(frame.tobytes())
