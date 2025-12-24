#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# core/result_exporter.py
"""
结果导出模块 - 负责导出JSON和统计信息
"""

import json
from typing import Dict, List


class ResultExporter:
    """结果导出器"""

    @staticmethod
    def export_json(
        output_path: str,
        video_info: Dict,
        settings: Dict,
        device_info: Dict,
        frames_data: List[Dict],
        frames_dir: str = None,
        masks_dir: str = None
    ):
        """
        导出JSON结果文件

        Args:
            output_path: 输出文件路径
            video_info: 视频信息
            settings: 处理设置
            device_info: 设备信息
            frames_data: 每帧的检测数据
            frames_dir: 帧图片目录（可选）
            masks_dir: mask图片目录（可选）
        """
        result = {
            "video_info": video_info,
            "settings": settings,
            "device_info": device_info,
            "frames": frames_data
        }

        if frames_dir:
            result["frames_dir"] = frames_dir
        
        if masks_dir:
            result["masks_dir"] = masks_dir

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_txt_summary(
        output_path: str,
        lines: List[str]
    ):
        """
        导出文本统计摘要

        Args:
            output_path: 输出文件路径
            lines: 文本行列表
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))