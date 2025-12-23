#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帧渲染模块 - 负责绘制标注框和保存帧图片
"""

import os
from typing import List, Literal

import cv2
import numpy as np


class FrameRenderer:
    """帧渲染器 - 负责绘制标注框"""

    def __init__(self, box_style: Literal["red_hollow", "green_fill", "mask"] = "red_hollow"):
        """
        初始化帧渲染器

        Args:
            box_style: 标注框样式
                - "red_hollow": 红色空心框
                - "green_fill": 绿色半透明填充
                - "mask": 黑白遮罩（字幕区域白色，其他区域黑色）
        """
        self.box_style = box_style

    def draw_boxes(self, frame: np.ndarray, boxes: List) -> np.ndarray:
        """
        在帧上绘制标注框

        Args:
            frame: 输入帧
            boxes: 文字框坐标列表

        Returns:
            绘制后的帧
        """
        if self.box_style == "mask":
            # 创建黑白遮罩：字幕区域白色，其他区域黑色
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for box in boxes:
                points = np.array(box, dtype=np.int32)
                cv2.fillPoly(mask, [points], color=255)

            # 转换为3通道BGR图像
            result_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return result_frame

        # 原有的标注框样式
        result_frame = frame.copy()

        for box in boxes:
            points = np.array(box, dtype=np.int32)

            if self.box_style == "red_hollow":
                cv2.polylines(result_frame, [points], isClosed=True,
                            color=(0, 0, 255), thickness=2)
            elif self.box_style == "green_fill":
                overlay = result_frame.copy()
                cv2.fillPoly(overlay, [points], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
                cv2.polylines(result_frame, [points], isClosed=True,
                            color=(0, 255, 0), thickness=2)

        return result_frame


class FrameSaver:
    """帧保存器 - 负责保存帧图片到文件"""

    def __init__(self, output_dir: str = None, quality: int = 95):
        """
        初始化帧保存器

        Args:
            output_dir: 输出目录，None表示不保存
            quality: JPEG 质量 (0-100)
        """
        self.output_dir = output_dir
        self.quality = quality
        self.frames_saved = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def save_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """
        保存单帧图片

        Args:
            frame: 帧数据
            frame_id: 帧ID

        Returns:
            是否成功保存
        """
        if not self.output_dir:
            return False

        filename = f"frame_{frame_id:08d}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if success:
            self.frames_saved += 1

        return success
