#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoOCR - 视频文字检测工具
支持逐帧OCR检测，输出带标注框的视频和JSON坐标文件
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

# 禁用 PaddleOCR 和 PaddlePaddle 的日志输出
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)


def get_ffmpeg_path(name: str = "ffmpeg") -> str:
    """
    获取 ffmpeg/ffprobe 的路径
    优先查找当前目录，然后查找系统 PATH
    
    Args:
        name: "ffmpeg" 或 "ffprobe"
    
    Returns:
        可执行文件的路径
    """
    # Windows 下添加 .exe 后缀
    if sys.platform == "win32":
        name = f"{name}.exe"
    
    # 1. 检查当前工作目录
    local_path = os.path.join(os.getcwd(), name)
    if os.path.isfile(local_path):
        return local_path
    
    # 2. 检查脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, name)
    if os.path.isfile(script_path):
        return script_path
    
    # 3. 检查脚本所在目录的 bin 子目录
    bin_path = os.path.join(script_dir, "bin", name)
    if os.path.isfile(bin_path):
        return bin_path
    
    # 4. 检查脚本所在目录的 ffmpeg 子目录
    ffmpeg_path = os.path.join(script_dir, "ffmpeg", name)
    if os.path.isfile(ffmpeg_path):
        return ffmpeg_path
    
    # 5. 使用系统 PATH 中的版本
    return name.replace(".exe", "") if sys.platform != "win32" else name


class Logger:
    """日志输出类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def info(self, message: str, indent: int = 0):
        if self.verbose:
            prefix = " " * indent if indent > 0 else f"[{self._timestamp()}] [INFO] "
            print(f"{prefix}{message}")
    
    def progress(self, current: int, total: int, start_time: float, extra: str = ""):
        if self.verbose:
            elapsed = time.time() - start_time
            
            # 安全检查
            if total <= 0:
                total = 1
            if current > total:
                current = total
                
            percent = current / total * 100
            fps = current / elapsed if elapsed > 0 else 0
            remaining = (total - current) / fps if fps > 0 else 0
            
            progress_bar = self._make_progress_bar(percent)
            
            # 使用 sys.stdout 确保立即输出
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
            print(f"\n[{self._timestamp()}] [SUCCESS] ✓ {message}")
    
    def error(self, message: str):
        print(f"\n[{self._timestamp()}] [ERROR] ✗ {message}", file=sys.stderr)
    
    def warning(self, message: str):
        if self.verbose:
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
    """计时器类，用于统计各阶段耗时"""
    
    def __init__(self):
        self.stages: Dict[str, Dict] = {}
        self.current_stage: Optional[str] = None
    
    def start(self, stage_name: str):
        self.current_stage = stage_name
        self.stages[stage_name] = {
            "start": time.time(),
            "end": None,
            "duration": 0
        }
    
    def stop(self, stage_name: str = None):
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
    
    def get_summary(self) -> Dict[str, float]:
        total = self.get_total()
        summary = {}
        for name, data in self.stages.items():
            duration = data["duration"]
            percent = (duration / total * 100) if total > 0 else 0
            summary[name] = {"duration": duration, "percent": percent}
        return summary


class VideoOCR:
    """视频OCR检测主类"""
    
    def __init__(
        self,
        box_style: Literal["red_hollow", "green_fill"] = "red_hollow",
        keep_audio: bool = False,
        verbose: bool = True,
        use_paddle_ocr: bool = True,
        lang: str = "ch",
        use_angle_cls: bool = False,
        model_name: str = "PP-OCRv5_server_det",
        use_lightweight: bool = True,
        skip_frames: int = 0,
        detect_only: bool = False,
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.0,
        batch_size: int = 1,
        use_cache: bool = False,
        cache_iou_threshold: float = 0.85,
        save_frames: bool = False
    ):
        """
        初始化 VideoOCR
        
        Args:
            box_style: 标注框样式 - "red_hollow"(红色空心框) 或 "green_fill"(绿色半透明填充)
            keep_audio: 是否保留原视频音频
            verbose: 是否输出详细日志
            use_paddle_ocr: True使用PaddleOCR(推荐), False使用TextDetection
            lang: PaddleOCR语言，默认"ch"(中英文)
            use_angle_cls: 是否使用文本行方向分类器（会降低速度）
            model_name: TextDetection模型名称(仅use_paddle_ocr=False时有效)
            use_lightweight: 使用轻量级Mobile模型(True)还是Server模型(False)，默认True以提升速度
            skip_frames: 跳帧处理，0表示处理每一帧，N表示每N+1帧处理1帧
            detect_only: 仅检测文字位置，不识别文字内容（速度更快）
            confidence_threshold: 置信度阈值(0-1)，低于此值的检测结果会被过滤，默认0.5
            scale_factor: 缩放因子(0-1)，对高分辨率视频缩小后处理可提升速度，默认1.0不缩放
            batch_size: 批量处理帧数，>1时启用批量OCR，GPU下效果更明显，默认1
            use_cache: 是否启用结果缓存，相邻帧文字位置相似时复用结果跳过OCR
            cache_iou_threshold: 缓存IoU阈值(0-1)，框重叠度超过此值认为相同，默认0.85
            save_frames: 是否保存标注后的每一帧图片，默认False不保存
        """
        self.box_style = box_style
        self.keep_audio = keep_audio
        self.verbose = verbose
        self.use_paddle_ocr = use_paddle_ocr
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.model_name = model_name
        self.use_lightweight = use_lightweight
        self.skip_frames = skip_frames
        self.detect_only = detect_only
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_iou_threshold = cache_iou_threshold
        self.save_frames = save_frames
        
        self.logger = Logger(verbose)
        self.timer = Timer()
        
        self.model = None
        self.device_info = {}
        self.video_info = {}
        self.stats = {
            "total_detections": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "cache_hits": 0,
            "frames_saved": 0
        }
        
        # 缓存相关
        self._last_boxes = None
        self._last_detections = None
        
        # 帧保存目录
        self._frames_dir = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化模型和设备信息"""
        self.logger.header("VideoOCR 视频文字检测工具")
        self.logger.info("初始化开始...")
        
        self.timer.start("模型初始化")
        
        # 获取设备信息
        self._get_device_info()
        
        # 加载OCR模型
        self._load_model()
        
        self.timer.stop("模型初始化")
        
        self.logger.success("模型加载完成")
        if self.detect_only:
            model_type = "Mobile(轻量)" if self.use_lightweight else "Server(精准)"
            self.logger.info(f"- 模式: 仅检测 TextDetection {model_type}", indent=28)
        elif self.use_paddle_ocr:
            model_type = "Mobile(轻量)" if self.use_lightweight else "Server(精准)"
            self.logger.info(f"- 模式: PaddleOCR 检测+识别 {model_type} (lang={self.lang})", indent=28)
        else:
            self.logger.info(f"- 模式: TextDetection ({self.model_name})", indent=28)
        self.logger.info(f"- 设备: {self.device_info.get('paddle_device', 'Unknown')}", indent=28)
        if self.skip_frames > 0:
            self.logger.info(f"- 跳帧: 每 {self.skip_frames + 1} 帧处理 1 帧", indent=28)
        if self.save_frames:
            self.logger.info(f"- 保存帧: 已启用", indent=28)
        self.logger.info(f"- 耗时: {self.timer.get_duration('模型初始化'):.2f}s", indent=28)
    
    def _get_device_info(self):
        """获取设备信息"""
        import paddle
        
        # PaddlePaddle 设备信息
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device_name = "GPU"
            try:
                # 尝试获取GPU名称
                gpu_info = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                if gpu_info.returncode == 0:
                    device_name = f"GPU ({gpu_info.stdout.strip().split(chr(10))[0]})"
            except:
                pass
            self.device_info["paddle_device"] = device_name
            self.device_info["paddle_use_gpu"] = True
        else:
            self.device_info["paddle_device"] = "CPU"
            self.device_info["paddle_use_gpu"] = False
        
        # FFmpeg 设备信息（默认CPU，可以检测是否支持硬件加速）
        self.device_info["ffmpeg_decode"] = "CPU (FFmpeg)"
        self.device_info["ffmpeg_encode"] = "CPU (FFmpeg)"
        
        # 检测FFmpeg是否支持NVENC
        try:
            ffmpeg_path = get_ffmpeg_path("ffmpeg")
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True, text=True
            )
            if "h264_nvenc" in result.stdout:
                self.device_info["ffmpeg_nvenc_available"] = True
            else:
                self.device_info["ffmpeg_nvenc_available"] = False
        except:
            self.device_info["ffmpeg_nvenc_available"] = False
    
    def _load_model(self):
        """加载OCR模型"""
        if self.detect_only:
            # 仅检测模式 - 使用 TextDetection，速度最快
            from paddleocr import TextDetection
            if self.use_lightweight:
                self.model = TextDetection(model_name="PP-OCRv5_mobile_det")
            else:
                self.model = TextDetection(model_name="PP-OCRv5_server_det")
            self.use_paddle_ocr = False  # 标记为非OCR模式
        elif self.use_paddle_ocr:
            from paddleocr import PaddleOCR
            
            # 根据 use_lightweight 选择模型
            if self.use_lightweight:
                # PP-OCRv5 Mobile 轻量模型 - 速度快
                self.model = PaddleOCR(
                    lang=self.lang,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False  # 禁用以提升速度
                )
            else:
                # PP-OCRv5 Server 模型 - 精度高但较慢
                self.model = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self.use_angle_cls
                )
        else:
            from paddleocr import TextDetection
            self.model = TextDetection(model_name=self.model_name)
    
    def _get_video_info(self, video_path: str) -> Dict:
        """获取视频信息"""
        ffprobe_path = get_ffmpeg_path("ffprobe")
        cmd = [
            ffprobe_path,
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
        
        # 找到视频流
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
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        
        # 获取总帧数
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            # 如果没有帧数信息，通过时长计算
            duration = float(probe.get("format", {}).get("duration", 0))
            total_frames = int(duration * fps)
        
        info = {
            "filename": os.path.basename(video_path),
            "filepath": video_path,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "total_frames": total_frames,
            "duration": float(probe.get("format", {}).get("duration", 0)),
            "has_audio": any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
        }
        
        return info
    
    def _draw_box(self, frame: np.ndarray, boxes: List) -> np.ndarray:
        """
        在帧上绘制标注框
        
        Args:
            frame: 原始帧
            boxes: 检测到的文字框列表
        
        Returns:
            绘制后的帧
        """
        result_frame = frame.copy()
        
        for box in boxes:
            # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(box, dtype=np.int32)
            
            if self.box_style == "red_hollow":
                # 红色空心框
                cv2.polylines(result_frame, [points], isClosed=True, 
                            color=(0, 0, 255), thickness=2)
            
            elif self.box_style == "green_fill":
                # 绿色半透明填充
                overlay = result_frame.copy()
                cv2.fillPoly(overlay, [points], color=(0, 255, 0))
                # 混合原图和覆盖层，alpha=0.3 表示30%透明度
                cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
                # 再画一个边框
                cv2.polylines(result_frame, [points], isClosed=True,
                            color=(0, 255, 0), thickness=2)
        
        return result_frame
    
    def _save_frame(self, frame: np.ndarray, frame_id: int):
        """
        保存标注后的帧图片
        
        Args:
            frame: 标注后的帧
            frame_id: 帧ID
        """
        if not self.save_frames or self._frames_dir is None:
            return
        
        # 生成文件名，使用零填充确保排序正确
        filename = f"frame_{frame_id:08d}.jpg"
        filepath = os.path.join(self._frames_dir, filename)
        
        # 保存图片
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.stats["frames_saved"] += 1
    
    def _box_to_rect(self, box):
        """将4点框转换为矩形 (x_min, y_min, x_max, y_max)"""
        points = np.array(box)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return (x_min, y_min, x_max, y_max)
    
    def _compute_iou(self, box1, box2):
        """计算两个框的IoU（交并比）"""
        r1 = self._box_to_rect(box1)
        r2 = self._box_to_rect(box2)
        
        # 计算交集
        x_left = max(r1[0], r2[0])
        y_top = max(r1[1], r2[1])
        x_right = min(r1[2], r2[2])
        y_bottom = min(r1[3], r2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _check_cache_hit(self, current_boxes):
        """
        检查当前帧的检测框是否与缓存相似
        
        Args:
            current_boxes: 当前帧检测到的框列表
        
        Returns:
            bool: 是否命中缓存
        """
        if self._last_boxes is None:
            return False
        
        # 框数量不同，认为不同
        if len(current_boxes) != len(self._last_boxes):
            return False
        
        # 没有框，认为相同
        if len(current_boxes) == 0:
            return True
        
        # 计算每个框与上一帧对应框的IoU
        # 使用贪心匹配：对每个当前框找最相似的上一帧框
        used = set()
        total_iou = 0.0
        
        for curr_box in current_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for i, last_box in enumerate(self._last_boxes):
                if i in used:
                    continue
                iou = self._compute_iou(curr_box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx >= 0:
                used.add(best_idx)
                total_iou += best_iou
        
        # 平均IoU超过阈值认为相似
        avg_iou = total_iou / len(current_boxes)
        return avg_iou >= self.cache_iou_threshold
    
    def _process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Tuple[np.ndarray, Dict]:
        """
        处理单帧
        
        Args:
            frame: 原始帧
            frame_id: 帧ID
            timestamp: 时间戳
        
        Returns:
            (处理后的帧, 帧数据)
        """
        boxes = []
        detections = []
        cache_hit = False
        
        # 缩放处理
        if self.scale_factor < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            process_frame = frame
        
        if self.use_paddle_ocr:
            # PaddleOCR 3.x / PP-OCRv5 模式
            # 使用 predict 方法，返回结果对象
            result = self.model.predict(process_frame)
            
            # result 是一个生成器或列表，每个元素是一个 OCRResult 对象
            for res in result:
                # PP-OCRv5 的结果存储在 res 字典中，或者通过 json 属性访问
                res_dict = None
                
                # 尝试获取结果字典
                if hasattr(res, 'json') and res.json:
                    res_dict = res.json.get('res', res.json)
                elif hasattr(res, '__getitem__'):
                    try:
                        res_dict = res.get('res', res) if isinstance(res, dict) else None
                    except:
                        pass
                
                # 调试：第一帧打印结果
                if frame_id == 0:
                    self.logger.info(f"res_dict 类型: {type(res_dict)}")
                    if res_dict:
                        self.logger.info(f"res_dict 键: {list(res_dict.keys()) if isinstance(res_dict, dict) else 'N/A'}")
                
                if res_dict and isinstance(res_dict, dict):
                    # 获取文本框坐标 - 优先使用 rec_polys（识别后的），否则使用 dt_polys（检测的）
                    polys = res_dict.get('rec_polys', res_dict.get('dt_polys', None))
                    texts = res_dict.get('rec_texts', [])
                    scores = res_dict.get('rec_scores', res_dict.get('dt_scores', []))
                    
                    if polys is not None and len(polys) > 0:
                        for i, poly in enumerate(polys):
                            score = float(scores[i]) if i < len(scores) else 1.0
                            
                            # 置信度过滤
                            if score < self.confidence_threshold:
                                continue
                            
                            # poly 是 numpy 数组，形状为 (4, 2)
                            box = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
                            
                            # 如果使用了缩放，将坐标映射回原始尺寸
                            if self.scale_factor < 1.0:
                                box = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box]
                            
                            text = texts[i] if i < len(texts) else ""
                            
                            boxes.append(box)
                            detections.append({
                                "bbox": box,
                                "text": str(text),
                                "confidence": score
                            })
        else:
            # TextDetection 模式 (detect_only=True 或 use_paddle_ocr=False)
            output = self.model.predict(process_frame, batch_size=1)
            
            for res in output:
                # 调试：第一帧打印结果对象信息
                if frame_id == 0:
                    self.logger.info(f"TextDetection 结果类型: {type(res)}")
                    if hasattr(res, '__dict__'):
                        self.logger.info(f"TextDetection 结果属性: {list(res.__dict__.keys())}")
                    # 检查 json 属性内容
                    if hasattr(res, 'json') and res.json:
                        self.logger.info(f"json 内容键: {list(res.json.keys())}")
                        if 'res' in res.json:
                            self.logger.info(f"json['res'] 键: {list(res.json['res'].keys())}")
                
                # 尝试多种方式获取检测框
                polys = None
                scores = []
                
                # 方式1: 通过 json 属性获取
                if hasattr(res, 'json') and res.json:
                    res_dict = res.json.get('res', res.json) if isinstance(res.json, dict) else None
                    if res_dict and isinstance(res_dict, dict):
                        polys = res_dict.get('dt_polys', res_dict.get('polys', None))
                        scores = res_dict.get('dt_scores', res_dict.get('scores', []))
                        
                        # 调试：打印获取到的 polys
                        if frame_id == 0:
                            self.logger.info(f"polys 类型: {type(polys)}, 值: {polys is not None}")
                            if polys is not None:
                                self.logger.info(f"polys 长度: {len(polys)}")
                                if len(polys) > 0:
                                    self.logger.info(f"第一个 poly: {polys[0]}")
                
                # 方式2: 直接访问属性
                if polys is None and hasattr(res, 'dt_polys') and res.dt_polys is not None:
                    polys = res.dt_polys
                    scores = res.dt_scores if hasattr(res, 'dt_scores') else []
                
                if polys is None and hasattr(res, 'boxes') and res.boxes is not None:
                    polys = res.boxes
                    scores = res.scores if hasattr(res, 'scores') else []
                
                if polys is None and hasattr(res, 'polys') and res.polys is not None:
                    polys = res.polys
                    scores = res.scores if hasattr(res, 'scores') else []
                
                # 处理检测到的框
                if polys is not None and len(polys) > 0:
                    if frame_id == 0:
                        self.logger.info(f"检测到 {len(polys)} 个文字区域 (阈值: {self.confidence_threshold})")
                    
                    for i, poly in enumerate(polys):
                        score = float(scores[i]) if i < len(scores) else 1.0
                        
                        # 置信度过滤
                        if score < self.confidence_threshold:
                            continue
                        
                        box_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                        
                        # 如果使用了缩放，将坐标映射回原始尺寸
                        if self.scale_factor < 1.0:
                            box_list = [[p[0] / self.scale_factor, p[1] / self.scale_factor] for p in box_list]
                        
                        boxes.append(box_list)
                        
                        detection = {
                            "bbox": box_list,
                            "confidence": score
                        }
                        detections.append(detection)
        
        # 缓存逻辑：检查是否与上一帧相似
        if self.use_cache and self._last_boxes is not None:
            if self._check_cache_hit(boxes):
                # 命中缓存，复用上一帧结果
                boxes = self._last_boxes
                detections = self._last_detections
                cache_hit = True
                self.stats["cache_hits"] += 1
        
        # 更新缓存
        if self.use_cache and not cache_hit:
            self._last_boxes = boxes.copy() if boxes else []
            self._last_detections = detections.copy() if detections else []
        
        # 绘制标注框
        result_frame = self._draw_box(frame, boxes)
        
        # 保存帧图片
        self._save_frame(result_frame, frame_id)
        
        # 帧数据
        frame_data = {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "detections": detections
        }
        
        self.stats["total_detections"] += len(detections)
        
        return result_frame, frame_data
    
    def _process_batch(self, original_frames: List[np.ndarray], process_frames: List[np.ndarray], frame_ids: List[int], timestamps: List[float]) -> List[Tuple[np.ndarray, Dict, List, List]]:
        """
        批量处理多帧
        
        Args:
            original_frames: 原始帧列表（用于画框）
            process_frames: 处理帧列表（可能已缩放）
            frame_ids: 帧ID列表
            timestamps: 时间戳列表
        
        Returns:
            [(处理后的帧, 帧数据, boxes, detections), ...] 列表，顺序与输入一致
        """
        results = []
        
        if self.detect_only:
            # TextDetection 支持真正的批量处理，顺序与输入一致
            batch_results = list(self.model.predict(process_frames, batch_size=len(process_frames)))
            
            for idx, (orig_frame, frame_id, timestamp) in enumerate(zip(original_frames, frame_ids, timestamps)):
                boxes = []
                detections = []
                
                if idx < len(batch_results):
                    res = batch_results[idx]
                    polys = None
                    scores = []
                    
                    if hasattr(res, 'json') and res.json:
                        res_dict = res.json.get('res', res.json) if isinstance(res.json, dict) else None
                        if res_dict and isinstance(res_dict, dict):
                            polys = res_dict.get('dt_polys', res_dict.get('polys', None))
                            scores = res_dict.get('dt_scores', res_dict.get('scores', []))
                    
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
                
                self.stats["total_detections"] += len(detections)
                self.stats["frames_ocr"] = self.stats.get("frames_ocr", 0) + 1
                
                result_frame = self._draw_box(orig_frame, boxes)
                
                # 保存帧图片
                self._save_frame(result_frame, frame_id)
                
                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "detections": detections
                }
                results.append((result_frame, frame_data, boxes, detections))
        else:
            # PaddleOCR 不支持 batch_size 参数，逐帧处理
            for orig_frame, proc_frame, frame_id, timestamp in zip(original_frames, process_frames, frame_ids, timestamps):
                boxes = []
                detections = []
                
                for res in self.model.predict(proc_frame):
                    res_dict = None
                    if hasattr(res, 'json') and res.json:
                        res_dict = res.json.get('res', res.json)
                    
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
                
                self.stats["total_detections"] += len(detections)
                self.stats["frames_ocr"] = self.stats.get("frames_ocr", 0) + 1
                
                result_frame = self._draw_box(orig_frame, boxes)
                
                # 保存帧图片
                self._save_frame(result_frame, frame_id)
                
                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "detections": detections
                }
                results.append((result_frame, frame_data, boxes, detections))
        
        return results
    
    def process(
        self,
        input_path: str,
        output_dir: str = "output",
        output_video_path: str = None,
        output_json_path: str = None
    ):
        """
        处理视频
        
        Args:
            input_path: 输入视频路径
            output_dir: 输出目录，默认"output"
            output_video_path: 输出视频路径（可选，默认自动生成）
            output_json_path: 输出JSON路径（可选，默认自动生成）
        
        输出文件命名规则（当不指定路径时）：
            - 视频: {output_dir}/{输入文件名}_ocr_{时间戳}.mp4
            - JSON: {output_dir}/{输入文件名}_ocr_{时间戳}.json
            - 统计: {output_dir}/{输入文件名}_ocr_{时间戳}.txt
            - 帧图片: {output_dir}/{输入文件名}_ocr_{时间戳}_frames/frame_XXXXXXXX.jpg
        """
        # 检查输入文件
        if not os.path.exists(input_path):
            self.logger.error(f"输入文件不存在: {input_path}")
            return
        
        # 生成时间戳
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 自动生成输出路径
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果未指定输出路径，自动生成（带时间戳）
        if output_video_path is None:
            output_video_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.mp4")
        if output_json_path is None:
            output_json_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.json")
        
        # 统计文件路径
        output_txt_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.txt")
        self._output_txt_path = output_txt_path
        
        # 帧图片保存目录
        if self.save_frames:
            self._frames_dir = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}_frames")
            os.makedirs(self._frames_dir, exist_ok=True)
            self.logger.info(f"帧图片将保存到: {self._frames_dir}")
        
        # 获取视频信息
        self.logger.section("视频信息分析")
        self.timer.start("视频分析")
        
        self.video_info = self._get_video_info(input_path)
        
        self.timer.stop("视频分析")
        
        self.logger.info(f"✓ 视频信息:")
        self.logger.info(f"- 文件: {self.video_info['filename']}", indent=28)
        self.logger.info(f"- 分辨率: {self.video_info['width']}x{self.video_info['height']}", indent=28)
        self.logger.info(f"- 帧率: {self.video_info['fps']:.1f} fps", indent=28)
        self.logger.info(f"- 总帧数: {self.video_info['total_frames']}", indent=28)
        self.logger.info(f"- 时长: {self.video_info['duration']:.1f}s", indent=28)
        self.logger.info(f"- 音频: {'有' if self.video_info['has_audio'] else '无'}", indent=28)
        
        # 显示处理设置
        self.logger.section("处理设置")
        self.logger.info(f"- 标注框样式: {'红色空心框' if self.box_style == 'red_hollow' else '绿色半透明填充'}")
        self.logger.info(f"- 保留音频: {'是' if self.keep_audio and self.video_info['has_audio'] else '否'}")
        self.logger.info(f"- 保存帧图片: {'是' if self.save_frames else '否'}")
        self.logger.info(f"- 解码: {self.device_info['ffmpeg_decode']}")
        self.logger.info(f"- OCR: {self.device_info['paddle_device']}")
        self.logger.info(f"- 编码: {self.device_info['ffmpeg_encode']}")
        self.logger.info(f"- 输出目录: {output_dir}")
        
        # 开始处理
        self.logger.section("开始处理视频")
        
        width = self.video_info["width"]
        height = self.video_info["height"]
        fps = self.video_info["fps"]
        total_frames = self.video_info["total_frames"]
        
        # 获取 ffmpeg 路径
        ffmpeg_path = get_ffmpeg_path("ffmpeg")
        
        # FFmpeg 读取命令
        read_cmd = [
            ffmpeg_path,
            "-i", input_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-"
        ]
        
        # FFmpeg 写入命令
        write_cmd = [
            ffmpeg_path,
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
        
        # 如果保留音频
        if self.keep_audio and self.video_info["has_audio"]:
            write_cmd.extend(["-i", input_path, "-c:a", "aac", "-map", "0:v", "-map", "1:a"])
        
        write_cmd.append(output_video_path)
        
        # 启动进程
        self.timer.start("视频解码")
        read_process = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.timer.start("视频编码")
        write_process = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 处理每一帧
        frame_size = width * height * 3
        frames_data = []
        frame_id = 0
        
        # 如果无法获取总帧数，设置一个估计值
        if total_frames <= 0:
            self.logger.info("警告: 无法获取视频总帧数，进度显示可能不准确")
            total_frames = int(fps * 60)  # 假设1分钟
        
        self.timer.start("OCR处理")
        process_start_time = time.time()
        
        # 跳帧处理相关变量
        last_boxes = []  # 缓存上一次OCR检测的框
        last_detections = []  # 缓存上一次OCR检测的结果
        
        # 批量处理相关变量
        batch_original_frames = []  # 原始帧（用于画框）
        batch_process_frames = []   # 处理帧（可能已缩放）
        batch_frame_ids = []        # 对应的帧ID
        batch_timestamps = []       # 对应的时间戳
        
        # 显示处理模式
        mode_info = []
        if self.skip_frames == 0:
            mode_info.append("逐帧处理")
        else:
            mode_info.append(f"每{self.skip_frames + 1}帧处理1帧")
        if self.scale_factor < 1.0:
            mode_info.append(f"缩放{self.scale_factor:.1%}")
        if self.batch_size > 1:
            if self.detect_only:
                mode_info.append(f"批量{self.batch_size}")
            else:
                mode_info.append(f"批量{self.batch_size}(PaddleOCR不支持,实际逐帧)")
        if self.save_frames:
            mode_info.append("保存帧")
        
        self.logger.info(f"开始处理，预计 {total_frames} 帧 ({', '.join(mode_info)})...")
        
        try:
            while True:
                # 读取一帧
                raw_frame = read_process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    # 处理剩余的批量帧
                    if len(batch_original_frames) > 0:
                        batch_results = self._process_batch(batch_original_frames, batch_process_frames, batch_frame_ids, batch_timestamps)
                        # 直接按顺序写入（batch_results顺序已经和输入一致）
                        for result_frame, frame_data, boxes, detections in batch_results:
                            frames_data.append(frame_data)
                            write_process.stdin.write(result_frame.tobytes())
                            last_boxes = boxes
                            last_detections = detections
                    break
                
                # 转换为numpy数组
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                
                # 计算时间戳
                timestamp = frame_id / fps
                
                # 判断是否需要进行OCR处理
                if self.skip_frames == 0:
                    should_process = True
                else:
                    should_process = (frame_id % (self.skip_frames + 1) == 0)
                
                if should_process:
                    if self.batch_size > 1:
                        # 批量模式：收集帧
                        batch_original_frames.append(frame.copy())
                        
                        # 缩放处理
                        if self.scale_factor < 1.0:
                            h, w = frame.shape[:2]
                            new_w = int(w * self.scale_factor)
                            new_h = int(h * self.scale_factor)
                            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            process_frame = frame.copy()
                        
                        batch_process_frames.append(process_frame)
                        batch_frame_ids.append(frame_id)
                        batch_timestamps.append(timestamp)
                        
                        # 批量满了，处理
                        if len(batch_original_frames) >= self.batch_size:
                            batch_results = self._process_batch(batch_original_frames, batch_process_frames, batch_frame_ids, batch_timestamps)
                            
                            # 直接按顺序写入
                            for result_frame, frame_data, boxes, detections in batch_results:
                                frames_data.append(frame_data)
                                write_process.stdin.write(result_frame.tobytes())
                                last_boxes = boxes
                                last_detections = detections
                            
                            # 清空缓存
                            batch_original_frames = []
                            batch_process_frames = []
                            batch_frame_ids = []
                            batch_timestamps = []
                            
                            self.logger.progress(frame_id + 1, total_frames, process_start_time)
                    else:
                        # 单帧模式
                        result_frame, frame_data = self._process_frame(frame, frame_id, timestamp)
                        last_boxes = [d["bbox"] for d in frame_data.get("detections", [])]
                        last_detections = frame_data.get("detections", [])
                        frames_data.append(frame_data)
                        write_process.stdin.write(result_frame.tobytes())
                        self.logger.progress(frame_id + 1, total_frames, process_start_time)
                else:
                    # 跳过OCR，使用上一帧的检测结果
                    result_frame = self._draw_box(frame, last_boxes)
                    
                    # 保存跳过的帧图片（如果启用）
                    self._save_frame(result_frame, frame_id)
                    
                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "detections": last_detections,
                        "skipped": True
                    }
                    self.stats["frames_skipped"] += 1
                    frames_data.append(frame_data)
                    write_process.stdin.write(result_frame.tobytes())
                    self.logger.progress(frame_id + 1, total_frames, process_start_time)
                
                frame_id += 1
                self.stats["frames_processed"] = frame_id
        
        except Exception as e:
            self.logger.error(f"处理过程中出错: {str(e)}")
            raise
        
        finally:
            # 关闭进程
            read_process.stdout.close()
            read_process.wait()
            
            write_process.stdin.close()
            write_process.wait()
        
        self.timer.stop("OCR处理")
        self.timer.stop("视频解码")
        self.timer.stop("视频编码")
        
        self.logger.success("视频处理完成")
        
        # 保存JSON
        self.logger.section("保存结果")
        self.timer.start("保存JSON")
        
        self._save_json(frames_data, output_json_path)
        
        self.timer.stop("保存JSON")
        self.logger.info(f"✓ JSON已保存: {output_json_path}")
        
        # 输出汇总
        self._print_summary(output_video_path, output_json_path)
    
    def _save_json(self, frames_data: List[Dict], output_path: str):
        """保存JSON文件"""
        settings = {
            "box_style": self.box_style,
            "keep_audio": self.keep_audio,
            "use_paddle_ocr": self.use_paddle_ocr,
            "save_frames": self.save_frames
        }
        
        if self.use_paddle_ocr:
            settings["lang"] = self.lang
            settings["use_angle_cls"] = self.use_angle_cls
        else:
            settings["model_name"] = self.model_name
        
        result = {
            "video_info": {
                "filename": self.video_info["filename"],
                "width": self.video_info["width"],
                "height": self.video_info["height"],
                "fps": self.video_info["fps"],
                "total_frames": self.video_info["total_frames"],
                "duration": self.video_info["duration"]
            },
            "settings": settings,
            "device_info": {
                "ocr_device": self.device_info.get("paddle_device", "Unknown"),
                "ffmpeg_decode": self.device_info.get("ffmpeg_decode", "Unknown"),
                "ffmpeg_encode": self.device_info.get("ffmpeg_encode", "Unknown")
            },
            "frames": frames_data
        }
        
        # 如果保存了帧图片，记录目录路径
        if self.save_frames and self._frames_dir:
            result["frames_dir"] = self._frames_dir
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def _print_summary(self, video_path: str, json_path: str):
        """打印处理汇总并保存到txt文件"""
        lines = []  # 收集所有输出行
        
        def log_and_save(msg):
            self.logger.info(msg)
            lines.append(msg)
        
        self.logger.header("处理汇总")
        lines.append("=" * 80)
        lines.append("处理汇总")
        lines.append("=" * 80)
        
        # 使用的参数
        log_and_save("\n使用参数:")
        log_and_save(f"  ├─ 标注框样式: {self.box_style}")
        log_and_save(f"  ├─ 模型: {'Mobile(轻量)' if self.use_lightweight else 'Server(精准)'}")
        log_and_save(f"  ├─ 模式: {'仅检测' if self.detect_only else '检测+识别'}")
        log_and_save(f"  ├─ 跳帧: {self.skip_frames} (每{self.skip_frames + 1}帧处理1帧)")
        log_and_save(f"  ├─ 置信度阈值: {self.confidence_threshold}")
        log_and_save(f"  ├─ 缩放因子: {self.scale_factor}")
        log_and_save(f"  ├─ 批量大小: {self.batch_size}")
        log_and_save(f"  ├─ 结果缓存: {'启用' if self.use_cache else '禁用'}")
        log_and_save(f"  ├─ 保存帧图片: {'是' if self.save_frames else '否'}")
        if self.use_cache:
            log_and_save(f"  └─ 缓存IoU阈值: {self.cache_iou_threshold}")
        else:
            log_and_save(f"  └─ 保留音频: {'是' if self.keep_audio else '否'}")
        
        # 视频信息
        log_and_save("\n视频信息:")
        log_and_save(f"  ├─ 文件: {self.video_info.get('filename', 'N/A')}")
        log_and_save(f"  ├─ 分辨率: {self.video_info.get('width', 0)}x{self.video_info.get('height', 0)}")
        log_and_save(f"  ├─ 帧率: {self.video_info.get('fps', 0):.1f} fps")
        log_and_save(f"  ├─ 总帧数: {self.video_info.get('total_frames', 0)}")
        log_and_save(f"  └─ 时长: {self.video_info.get('duration', 0):.1f}s")
        
        # 总耗时
        total_time = self.timer.get_total()
        log_and_save(f"\n总耗时: {total_time:.1f}s")
        
        # 各阶段耗时
        log_and_save("\n各阶段耗时:")
        summary = self.timer.get_summary()
        
        stages_order = ["模型初始化", "视频分析", "OCR处理", "保存JSON"]
        for stage in stages_order:
            if stage in summary:
                data = summary[stage]
                device = ""
                if stage == "模型初始化":
                    device = f"[{self.device_info.get('paddle_device', '')}]"
                elif stage == "OCR处理":
                    device = f"[{self.device_info.get('paddle_device', '')}]"
                
                log_and_save(f"  ├─ {stage}: {data['duration']:.2f}s ({data['percent']:.1f}%) {device}")
        
        # 输出文件信息
        log_and_save("\n输出文件:")
        if os.path.exists(video_path):
            video_size = os.path.getsize(video_path) / (1024 * 1024)
            log_and_save(f"  ├─ 视频: {video_path} ({video_size:.1f} MB)")
        
        if os.path.exists(json_path):
            json_size = os.path.getsize(json_path) / 1024
            log_and_save(f"  ├─ JSON: {json_path} ({json_size:.1f} KB)")
        
        txt_path = getattr(self, '_output_txt_path', None)
        if txt_path:
            log_and_save(f"  ├─ 统计: {txt_path}")
        
        # 帧图片目录信息
        if self.save_frames and self._frames_dir:
            frames_saved = self.stats.get("frames_saved", 0)
            if os.path.exists(self._frames_dir):
                # 计算目录大小
                total_size = 0
                for f in os.listdir(self._frames_dir):
                    fp = os.path.join(self._frames_dir, f)
                    if os.path.isfile(fp):
                        total_size += os.path.getsize(fp)
                size_mb = total_size / (1024 * 1024)
                log_and_save(f"  ├─ 帧图片: {self._frames_dir} ({frames_saved} 张, {size_mb:.1f} MB)")
            else:
                log_and_save(f"  ├─ 帧图片: {self._frames_dir} ({frames_saved} 张)")
        
        log_and_save(f"  └─ (完成)")
        
        # 检测统计
        log_and_save("\n检测统计:")
        log_and_save(f"  ├─ 总帧数: {self.stats['frames_processed']}")
        
        ocr_frames = self.stats.get('frames_ocr', self.stats['frames_processed'])
        skipped_frames = self.stats.get('frames_skipped', 0)
        cache_hits = self.stats.get('cache_hits', 0)
        
        if skipped_frames > 0:
            log_and_save(f"  ├─ OCR处理帧数: {ocr_frames}")
            log_and_save(f"  ├─ 跳过帧数: {skipped_frames}")
            skip_ratio = skipped_frames / self.stats['frames_processed'] * 100 if self.stats['frames_processed'] > 0 else 0
            log_and_save(f"  ├─ 跳帧率: {skip_ratio:.1f}%")
        
        if cache_hits > 0:
            log_and_save(f"  ├─ 缓存命中: {cache_hits} 次")
            cache_ratio = cache_hits / ocr_frames * 100 if ocr_frames > 0 else 0
            log_and_save(f"  ├─ 缓存命中率: {cache_ratio:.1f}%")
        
        if self.save_frames:
            log_and_save(f"  ├─ 保存帧数: {self.stats.get('frames_saved', 0)} 张")
        
        log_and_save(f"  ├─ 检测文字区域: {self.stats['total_detections']} 个")
        avg = self.stats['total_detections'] / self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0
        log_and_save(f"  └─ 平均每帧: {avg:.1f} 个")
        
        lines.append("\n" + "=" * 80)
        self.logger.info("\n" + "=" * 80)
        
        # 保存到txt文件
        if txt_path:
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                self.logger.warning(f"保存统计文件失败: {e}")


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VideoOCR - 视频文字检测工具")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出视频路径", default=None)
    parser.add_argument("-j", "--json", help="输出JSON路径", default=None)
    parser.add_argument(
        "-s", "--style",
        choices=["red_hollow", "green_fill"],
        default="red_hollow",
        help="标注框样式: red_hollow(红色空心框) 或 green_fill(绿色半透明填充)"
    )
    parser.add_argument(
        "-a", "--audio",
        action="store_true",
        help="保留原视频音频"
    )
    parser.add_argument(
        "--text-detection",
        action="store_true",
        help="使用TextDetection模式(默认使用PaddleOCR)"
    )
    parser.add_argument(
        "-l", "--lang",
        default="ch",
        help="PaddleOCR语言，默认'ch'(中英文)"
    )
    parser.add_argument(
        "-m", "--model",
        default="PP-OCRv5_server_det",
        help="TextDetection模型名称(仅--text-detection模式有效)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式，减少输出"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="使用Server模型(精度高但较慢)，默认使用Mobile轻量模型"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="跳帧数，0=逐帧处理，N=每N+1帧处理1帧。例如--skip 4表示每5帧处理1帧"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="仅检测文字位置，不识别文字内容（速度更快）"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="置信度阈值(0-1)，低于此值的检测结果会被过滤，默认0.5"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="缩放因子(0-1)，对高分辨率视频缩小后处理可提升速度，如0.5表示缩小到50%%，默认1.0不缩放"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="批量处理帧数，>1时启用批量OCR，GPU下效果更明显，默认1"
    )
    parser.add_argument(
        "-d", "--output-dir",
        default="output",
        help="输出目录，默认'output'"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="启用结果缓存，相邻帧文字位置相似时复用结果跳过OCR"
    )
    parser.add_argument(
        "--cache-iou",
        type=float,
        default=0.85,
        help="缓存IoU阈值(0-1)，框重叠度超过此值认为相同，默认0.85"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="保存标注后的每一帧图片，方便检查（默认不保存）"
    )
    
    args = parser.parse_args()
    
    # 创建处理器并运行
    ocr = VideoOCR(
        box_style=args.style,
        keep_audio=args.audio,
        verbose=not args.quiet,
        use_paddle_ocr=not args.text_detection,
        lang=args.lang,
        model_name=args.model,
        use_lightweight=not args.server,
        skip_frames=args.skip,
        detect_only=args.detect_only,
        confidence_threshold=args.threshold,
        scale_factor=args.scale,
        batch_size=args.batch,
        use_cache=args.cache,
        cache_iou_threshold=args.cache_iou,
        save_frames=args.save_frames
    )
    
    # 处理视频（自动生成输出路径，或使用指定路径）
    ocr.process(
        input_path=args.input,
        output_dir=args.output_dir,
        output_video_path=args.output,
        output_json_path=args.json
    )


if __name__ == "__main__":
    main()