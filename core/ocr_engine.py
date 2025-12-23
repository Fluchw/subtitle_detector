#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 引擎模块 - 负责文字检测和识别
"""

import logging
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np

# 禁用 PaddleOCR 和 PaddlePaddle 的日志输出
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)


class ImageEnhancer:
    """图像增强器"""

    @staticmethod
    def enhance(frame: np.ndarray, mode: Optional[Literal["clahe", "binary", "both"]]) -> np.ndarray:
        """
        图像预处理增强

        Args:
            frame: 原始帧 (BGR)
            mode: 增强模式

        Returns:
            增强后的帧 (BGR)
        """
        if not mode:
            return frame

        if mode == "clahe":
            return ImageEnhancer._apply_clahe(frame)
        elif mode == "binary":
            return ImageEnhancer._apply_binary(frame)
        elif mode == "both":
            enhanced = ImageEnhancer._apply_clahe(frame)
            return ImageEnhancer._apply_binary(enhanced)
        else:
            return frame

    @staticmethod
    def _apply_clahe(frame: np.ndarray) -> np.ndarray:
        """应用 CLAHE 对比度增强"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_binary(frame: np.ndarray) -> np.ndarray:
        """应用自适应二值化（保持3通道输出）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class OCREngine:
    """OCR 引擎 - 负责文字检测和识别"""

    def __init__(
        self,
        use_paddle_ocr: bool = True,
        lang: str = "ch",
        use_lightweight: bool = True,
        detect_only: bool = False,
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.0,
        enhance_mode: Optional[Literal["clahe", "binary", "both"]] = None,
        use_orientation_classify: bool = False,
        use_textline_orientation: bool = False,
        use_doc_unwarping: bool = False
    ):
        """
        初始化 OCR 引擎

        Args:
            use_paddle_ocr: True使用PaddleOCR, False使用TextDetection
            lang: PaddleOCR语言
            use_lightweight: 使用轻量级Mobile模型(True)还是Server模型(False)
            detect_only: 仅检测文字位置，不识别文字内容
            confidence_threshold: 置信度阈值(0-1)
            scale_factor: 缩放因子(0-1)
            enhance_mode: 图像预处理模式
            use_orientation_classify: 启用文档方向分类（检测整体旋转0°/90°/180°/270°）
            use_textline_orientation: 启用文本行方向检测（检测倾斜文字）
            use_doc_unwarping: 启用文档矫正（处理弯曲/透视变形）
        """
        self.use_paddle_ocr = use_paddle_ocr
        self.lang = lang
        self.use_lightweight = use_lightweight
        self.detect_only = detect_only
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.enhance_mode = enhance_mode
        self.use_orientation_classify = use_orientation_classify
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_unwarping = use_doc_unwarping

        self.model = None
        self.enhancer = ImageEnhancer()

        self._load_model()

    def _load_model(self):
        """加载OCR模型"""
        if self.detect_only:
            from paddleocr import TextDetection
            if self.use_lightweight:
                self.model = TextDetection(model_name="PP-OCRv5_mobile_det")
            else:
                self.model = TextDetection(model_name="PP-OCRv5_server_det")
            self.use_paddle_ocr = False
        elif self.use_paddle_ocr:
            from paddleocr import PaddleOCR

            if self.use_lightweight:
                self.model = PaddleOCR(
                    lang=self.lang,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=self.use_orientation_classify,
                    use_doc_unwarping=self.use_doc_unwarping,
                    use_textline_orientation=self.use_textline_orientation
                )
            else:
                self.model = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=self.use_orientation_classify,
                    use_doc_unwarping=self.use_doc_unwarping,
                    use_textline_orientation=self.use_textline_orientation
                )
        else:
            from paddleocr import TextDetection
            self.model = TextDetection(model_name="PP-OCRv5_server_det")

    def process_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        对单帧进行OCR处理

        Args:
            frame: 输入帧 (BGR)

        Returns:
            (boxes, detections)
            boxes: 文字框坐标列表
            detections: 检测结果列表 [{"bbox": [...], "text": "...", "confidence": 0.9}, ...]
        """
        # 缩放处理
        if self.scale_factor < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            process_frame = frame

        # 图像预处理增强
        process_frame = self.enhancer.enhance(process_frame, self.enhance_mode)

        boxes = []
        detections = []

        if self.use_paddle_ocr:
            boxes, detections = self._process_with_paddleocr(process_frame)
        else:
            boxes, detections = self._process_with_text_detection(process_frame)

        return boxes, detections

    def _process_with_paddleocr(self, frame: np.ndarray) -> Tuple[List, List]:
        """使用 PaddleOCR 处理"""
        boxes = []
        detections = []

        result = self.model.predict(frame)

        for res in result:
            res_dict = None
            if hasattr(res, 'json') and res.json:
                res_dict = res.json.get('res', res.json)
            elif hasattr(res, '__getitem__'):
                try:
                    res_dict = res.get('res', res) if isinstance(res, dict) else None
                except:
                    pass

            if res_dict and isinstance(res_dict, dict):
                polys = res_dict.get('rec_polys', res_dict.get('dt_polys', None))
                texts = res_dict.get('rec_texts', [])
                scores = res_dict.get('rec_scores', res_dict.get('dt_scores', []))

                if polys is not None and len(polys) > 0:
                    for i, poly in enumerate(polys):
                        score = float(scores[i]) if i < len(scores) else 1.0

                        if score < self.confidence_threshold:
                            continue

                        box = poly.tolist() if hasattr(poly, 'tolist') else list(poly)

                        if self.scale_factor < 1.0:
                            box = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box]

                        text = texts[i] if i < len(texts) else ""

                        boxes.append(box)
                        detections.append({
                            "bbox": box,
                            "text": str(text),
                            "confidence": score
                        })

        return boxes, detections

    def _process_with_text_detection(self, frame: np.ndarray) -> Tuple[List, List]:
        """使用 TextDetection 处理"""
        boxes = []
        detections = []

        output = self.model.predict(frame, batch_size=1)

        for res in output:
            polys = None
            scores = []

            if hasattr(res, 'json') and res.json:
                res_dict = res.json.get('res', res.json) if isinstance(res.json, dict) else None
                if res_dict and isinstance(res_dict, dict):
                    polys = res_dict.get('dt_polys', res_dict.get('polys', None))
                    scores = res_dict.get('dt_scores', res_dict.get('scores', []))

            if polys is None and hasattr(res, 'dt_polys') and res.dt_polys is not None:
                polys = res.dt_polys
                scores = res.dt_scores if hasattr(res, 'dt_scores') else []

            if polys is not None and len(polys) > 0:
                for i, poly in enumerate(polys):
                    score = float(scores[i]) if i < len(scores) else 1.0

                    if score < self.confidence_threshold:
                        continue

                    box_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)

                    if self.scale_factor < 1.0:
                        box_list = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box_list]

                    boxes.append(box_list)
                    detections.append({
                        "bbox": box_list,
                        "confidence": score
                    })

        return boxes, detections

    def get_model_info(self) -> str:
        """获取模型信息描述"""
        if self.detect_only:
            model_type = "Mobile(轻量)" if self.use_lightweight else "Server(精准)"
            return f"仅检测 TextDetection {model_type}"
        elif self.use_paddle_ocr:
            model_type = "Mobile(轻量)" if self.use_lightweight else "Server(精准)"
            orientation_info = []
            if self.use_orientation_classify:
                orientation_info.append("方向分类")
            if self.use_textline_orientation:
                orientation_info.append("文本行方向")
            if self.use_doc_unwarping:
                orientation_info.append("文档矫正")
            orientation_str = f" [{', '.join(orientation_info)}]" if orientation_info else ""
            return f"PaddleOCR 检测+识别 {model_type} (lang={self.lang}){orientation_str}"
        else:
            return "TextDetection"