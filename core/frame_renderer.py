#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帧渲染模块 - 负责绘制标注框和保存帧图片
"""

import os
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np


class FrameRenderer:
    """帧渲染器 - 负责绘制标注框"""

    def __init__(self, box_style: Literal["red_hollow", "green_fill", "mask"] = "red_hollow"):
        """
        初始化帧渲染器

        Args:
            box_style: 标注框样式
                - "red_hollow": 红色空心框（同时生成 mask）
                - "green_fill": 绿色半透明填充（同时生成 mask）
                - "mask": 仅输出黑白遮罩视频（字幕区域白色，其他区域黑色）
        """
        self.box_style = box_style

    def draw_boxes(self, frame: np.ndarray, boxes: List) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        在帧上绘制标注框

        Args:
            frame: 输入帧
            boxes: 文字框坐标列表

        Returns:
            (result_frame, mask_frame) 元组：
            - result_frame: 绘制标注框后的帧
            - mask_frame: 黑白遮罩帧（mask模式下为None，因为result_frame就是mask）
        """
        # 始终生成 mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in boxes:
            points = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)
        
        # 转换为3通道BGR图像
        mask_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if self.box_style == "mask":
            # 纯 mask 模式：只返回 mask，不需要额外的 mask_frame
            return mask_frame, None

        # 标注框样式：同时返回标注帧和 mask
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

        return result_frame, mask_frame


class FrameSaver:
    """帧保存器 - 负责保存帧图片到文件"""

    def __init__(self, output_dir: str = None, quality: int = 95, save_mask: bool = False):
        """
        初始化帧保存器

        Args:
            output_dir: 输出目录，None表示不保存
            quality: JPEG 质量 (0-100)
            save_mask: 是否同时保存 mask 图片
        """
        self.output_dir = output_dir
        self.quality = quality
        self.save_mask = save_mask
        self.frames_saved = 0
        self.masks_saved = 0
        self.masks_dir = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if save_mask:
                self.masks_dir = os.path.join(output_dir, "masks")
                os.makedirs(self.masks_dir, exist_ok=True)

    def save_frame(self, frame: np.ndarray, frame_id: int, mask: np.ndarray = None) -> bool:
        """
        保存单帧图片

        Args:
            frame: 帧数据
            frame_id: 帧ID
            mask: mask 帧数据（可选）

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

        # 保存 mask
        if self.save_mask and mask is not None and self.masks_dir:
            mask_filepath = os.path.join(self.masks_dir, filename)
            mask_success = cv2.imwrite(mask_filepath, mask, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if mask_success:
                self.masks_saved += 1

        return success

    def get_stats(self) -> dict:
        """获取保存统计"""
        return {
            "frames_saved": self.frames_saved,
            "masks_saved": self.masks_saved
        }