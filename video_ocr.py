#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoOCR v2 - 视频文字检测工具（优化版）
新增：图像预处理、多线程流水线
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
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
    """获取 ffmpeg/ffprobe 的路径"""
    if sys.platform == "win32":
        name = f"{name}.exe"
    
    local_path = os.path.join(os.getcwd(), name)
    if os.path.isfile(local_path):
        return local_path
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, name)
    if os.path.isfile(script_path):
        return script_path
    
    bin_path = os.path.join(script_dir, "bin", name)
    if os.path.isfile(bin_path):
        return bin_path
    
    ffmpeg_path = os.path.join(script_dir, "ffmpeg", name)
    if os.path.isfile(ffmpeg_path):
        return ffmpeg_path
    
    return name.replace(".exe", "") if sys.platform != "win32" else name


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


class VideoOCR:
    """视频OCR检测主类（优化版）"""
    
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
        save_frames: bool = False,
        enhance_mode: Optional[Literal["clahe", "binary", "both"]] = None,
        pipeline_queue_size: int = 0
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
            use_lightweight: 使用轻量级Mobile模型(True)还是Server模型(False)
            skip_frames: 跳帧处理，0表示处理每一帧，N表示每N+1帧处理1帧
            detect_only: 仅检测文字位置，不识别文字内容（速度更快）
            confidence_threshold: 置信度阈值(0-1)，低于此值的检测结果会被过滤
            scale_factor: 缩放因子(0-1)，对高分辨率视频缩小后处理可提升速度
            batch_size: 批量处理帧数，>1时启用批量OCR
            use_cache: 是否启用结果缓存，相邻帧文字位置相似时复用结果跳过OCR
            cache_iou_threshold: 缓存IoU阈值(0-1)，框重叠度超过此值认为相同
            save_frames: 是否保存标注后的每一帧图片
            enhance_mode: 图像预处理模式，None禁用，可选 "clahe"(对比度增强) / "binary"(二值化) / "both"(两者结合)
            pipeline_queue_size: 流水线队列大小，>0时启用多线程流水线处理，推荐32
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
        self.enhance_mode = enhance_mode
        self.pipeline_queue_size = pipeline_queue_size
        
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
            "frames_saved": 0,
            "frame_save_time": 0.0,
            "preprocess_time": 0.0
        }
        
        # 缓存相关
        self._last_boxes = None
        self._last_detections = None
        
        # 帧保存目录
        self._frames_dir = None
        
        # 流水线相关
        self._pipeline_stop = False
        self._decode_queue = None
        self._ocr_queue = None
        self._encode_queue = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化模型和设备信息"""
        self.logger.header("VideoOCR v2 视频文字检测工具（优化版）")
        self.logger.info("初始化开始...")
        
        self.timer.start("模型初始化")
        
        self._get_device_info()
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
        if self.enhance_mode:
            self.logger.info(f"- 图像增强: {self.enhance_mode}", indent=28)
        if self.pipeline_queue_size > 0:
            self.logger.info(f"- 多线程流水线: 已启用 (队列大小={self.pipeline_queue_size})", indent=28)
        self.logger.info(f"- 耗时: {self.timer.get_duration('模型初始化'):.2f}s", indent=28)
    
    def _get_device_info(self):
        """获取设备信息"""
        import paddle
        
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
            self.device_info["paddle_device"] = device_name
            self.device_info["paddle_use_gpu"] = True
        else:
            self.device_info["paddle_device"] = "CPU"
            self.device_info["paddle_use_gpu"] = False
        
        self.device_info["ffmpeg_decode"] = "CPU (FFmpeg)"
        self.device_info["ffmpeg_encode"] = "CPU (FFmpeg)"
        
        try:
            ffmpeg_path = get_ffmpeg_path("ffmpeg")
            result = subprocess.run([ffmpeg_path, "-encoders"], capture_output=True, text=True)
            self.device_info["ffmpeg_nvenc_available"] = "h264_nvenc" in result.stdout
        except:
            self.device_info["ffmpeg_nvenc_available"] = False
    
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
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False
                )
            else:
                self.model = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self.use_angle_cls
                )
        else:
            from paddleocr import TextDetection
            self.model = TextDetection(model_name=self.model_name)
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        图像预处理增强，提升OCR检测准确度
        
        Args:
            frame: 原始帧 (BGR)
        
        Returns:
            增强后的帧 (BGR)
        """
        if not self.enhance_mode:
            return frame
        
        start_time = time.time()
        
        if self.enhance_mode == "clahe":
            # CLAHE 对比度增强
            enhanced = self._apply_clahe(frame)
        elif self.enhance_mode == "binary":
            # 自适应二值化
            enhanced = self._apply_binary(frame)
        elif self.enhance_mode == "both":
            # 先 CLAHE 再二值化
            enhanced = self._apply_clahe(frame)
            enhanced = self._apply_binary(enhanced)
        else:
            enhanced = frame
        
        self.stats["preprocess_time"] += time.time() - start_time
        return enhanced
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """应用 CLAHE 对比度增强"""
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对 L 通道应用 CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 合并通道并转回 BGR
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _apply_binary(self, frame: np.ndarray) -> np.ndarray:
        """应用自适应二值化（保持3通道输出）"""
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 转回 BGR（3通道）
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
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
        
        video_stream = None
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            raise RuntimeError("未找到视频流")
        
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            duration = float(probe.get("format", {}).get("duration", 0))
            total_frames = int(duration * fps)
        
        return {
            "filename": os.path.basename(video_path),
            "filepath": video_path,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "total_frames": total_frames,
            "duration": float(probe.get("format", {}).get("duration", 0)),
            "has_audio": any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
        }
    
    def _draw_box(self, frame: np.ndarray, boxes: List) -> np.ndarray:
        """在帧上绘制标注框"""
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
    
    def _save_frame(self, frame: np.ndarray, frame_id: int):
        """保存标注后的帧图片"""
        if not self.save_frames or self._frames_dir is None:
            return
        
        save_start = time.time()
        filename = f"frame_{frame_id:08d}.jpg"
        filepath = os.path.join(self._frames_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.stats["frames_saved"] += 1
        self.stats["frame_save_time"] += time.time() - save_start
    
    def _process_frame_ocr(self, frame: np.ndarray, frame_id: int) -> Tuple[List, List]:
        """
        对单帧进行OCR处理（仅OCR，不画框）
        
        Returns:
            (boxes, detections)
        """
        boxes = []
        detections = []
        
        # 缩放处理
        if self.scale_factor < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            process_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            process_frame = frame
        
        # 图像预处理增强
        process_frame = self._enhance_frame(process_frame)
        
        if self.use_paddle_ocr:
            result = self.model.predict(process_frame)
            
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
        else:
            output = self.model.predict(process_frame, batch_size=1)
            
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
    
    # ==================== 流水线相关方法 ====================
    
    def _decode_worker(self, read_process, width, height, fps):
        """解码线程：从FFmpeg读取帧并放入队列"""
        frame_size = width * height * 3
        frame_id = 0
        
        try:
            while not self._pipeline_stop:
                raw_frame = read_process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                timestamp = frame_id / fps
                
                # 放入解码队列
                self._decode_queue.put((frame_id, frame.copy(), timestamp))
                frame_id += 1
        except Exception as e:
            self.logger.error(f"解码线程错误: {e}")
        finally:
            # 发送结束信号
            self._decode_queue.put(None)
    
    def _ocr_worker(self):
        """OCR线程：从解码队列取帧，处理后放入编码队列"""
        last_boxes = []
        last_detections = []
        
        try:
            while not self._pipeline_stop:
                item = self._decode_queue.get()
                if item is None:
                    break
                
                frame_id, frame, timestamp = item
                
                # 判断是否需要OCR处理
                should_process = (self.skip_frames == 0) or (frame_id % (self.skip_frames + 1) == 0)
                
                if should_process:
                    boxes, detections = self._process_frame_ocr(frame, frame_id)
                    last_boxes = boxes
                    last_detections = detections
                    self.stats["total_detections"] += len(detections)
                else:
                    boxes = last_boxes
                    detections = last_detections
                    self.stats["frames_skipped"] += 1
                
                # 构建帧数据
                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "detections": detections
                }
                if not should_process:
                    frame_data["skipped"] = True
                
                # 放入编码队列
                self._ocr_queue.put((frame_id, frame, boxes, frame_data))
                
        except Exception as e:
            self.logger.error(f"OCR线程错误: {e}")
        finally:
            self._ocr_queue.put(None)
    
    def _encode_worker(self, write_process, total_frames, process_start_time):
        """编码线程：从OCR队列取结果，画框后写入FFmpeg"""
        frames_data = []
        pending_frames = {}  # 用于保持帧顺序
        next_frame_id = 0
        
        try:
            while not self._pipeline_stop:
                item = self._ocr_queue.get()
                if item is None:
                    break
                
                frame_id, frame, boxes, frame_data = item
                
                # 缓存乱序的帧
                pending_frames[frame_id] = (frame, boxes, frame_data)
                
                # 按顺序处理
                while next_frame_id in pending_frames:
                    frame, boxes, frame_data = pending_frames.pop(next_frame_id)
                    
                    # 画框
                    result_frame = self._draw_box(frame, boxes)
                    
                    # 保存帧图片
                    self._save_frame(result_frame, next_frame_id)
                    
                    # 写入编码器
                    write_process.stdin.write(result_frame.tobytes())
                    
                    frames_data.append(frame_data)
                    self.stats["frames_processed"] = next_frame_id + 1
                    
                    # 更新进度
                    self.logger.progress(next_frame_id + 1, total_frames, process_start_time)
                    
                    next_frame_id += 1
                    
        except Exception as e:
            self.logger.error(f"编码线程错误: {e}")
        finally:
            self._encode_queue.put(frames_data)
    
    def _process_pipeline(self, input_path: str, output_video_path: str) -> List[Dict]:
        """使用多线程流水线处理视频"""
        width = self.video_info["width"]
        height = self.video_info["height"]
        fps = self.video_info["fps"]
        total_frames = self.video_info["total_frames"]
        
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
        
        if self.keep_audio and self.video_info["has_audio"]:
            write_cmd.extend(["-i", input_path, "-c:a", "aac", "-map", "0:v", "-map", "1:a"])
        
        write_cmd.append(output_video_path)
        
        # 初始化队列
        self._decode_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self._ocr_queue = queue.Queue(maxsize=self.pipeline_queue_size)
        self._encode_queue = queue.Queue(maxsize=1)
        self._pipeline_stop = False
        
        # 启动FFmpeg进程
        read_process = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        write_process = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        process_start_time = time.time()
        
        # 启动工作线程
        decode_thread = threading.Thread(
            target=self._decode_worker,
            args=(read_process, width, height, fps)
        )
        ocr_thread = threading.Thread(target=self._ocr_worker)
        encode_thread = threading.Thread(
            target=self._encode_worker,
            args=(write_process, total_frames, process_start_time)
        )
        
        decode_thread.start()
        ocr_thread.start()
        encode_thread.start()
        
        # 等待完成
        decode_thread.join()
        ocr_thread.join()
        encode_thread.join()
        
        # 获取结果
        frames_data = self._encode_queue.get()
        
        # 关闭进程
        read_process.stdout.close()
        read_process.wait()
        write_process.stdin.close()
        write_process.wait()
        
        return frames_data
    
    def _process_sequential(self, input_path: str, output_video_path: str) -> List[Dict]:
        """使用串行方式处理视频（原有逻辑）"""
        width = self.video_info["width"]
        height = self.video_info["height"]
        fps = self.video_info["fps"]
        total_frames = self.video_info["total_frames"]
        
        ffmpeg_path = get_ffmpeg_path("ffmpeg")
        
        read_cmd = [
            ffmpeg_path,
            "-i", input_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-v", "quiet",
            "-"
        ]
        
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
        
        if self.keep_audio and self.video_info["has_audio"]:
            write_cmd.extend(["-i", input_path, "-c:a", "aac", "-map", "0:v", "-map", "1:a"])
        
        write_cmd.append(output_video_path)
        
        read_process = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        write_process = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
        frame_size = width * height * 3
        frames_data = []
        frame_id = 0
        
        if total_frames <= 0:
            total_frames = int(fps * 60)
        
        process_start_time = time.time()
        
        last_boxes = []
        last_detections = []
        
        try:
            while True:
                raw_frame = read_process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                timestamp = frame_id / fps
                
                should_process = (self.skip_frames == 0) or (frame_id % (self.skip_frames + 1) == 0)
                
                if should_process:
                    boxes, detections = self._process_frame_ocr(frame, frame_id)
                    last_boxes = boxes
                    last_detections = detections
                    self.stats["total_detections"] += len(detections)
                    
                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "detections": detections
                    }
                else:
                    boxes = last_boxes
                    detections = last_detections
                    self.stats["frames_skipped"] += 1
                    
                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "detections": detections,
                        "skipped": True
                    }
                
                result_frame = self._draw_box(frame, boxes)
                self._save_frame(result_frame, frame_id)
                
                frames_data.append(frame_data)
                write_process.stdin.write(result_frame.tobytes())
                
                frame_id += 1
                self.stats["frames_processed"] = frame_id
                
                self.logger.progress(frame_id, total_frames, process_start_time)
                
        finally:
            read_process.stdout.close()
            read_process.wait()
            write_process.stdin.close()
            write_process.wait()
        
        return frames_data
    
    def process(
        self,
        input_path: str,
        output_dir: str = "output",
        output_video_path: str = None,
        output_json_path: str = None
    ):
        """处理视频"""
        if not os.path.exists(input_path):
            self.logger.error(f"输入文件不存在: {input_path}")
            return
        
        self.timer.start("总耗时")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        if output_video_path is None:
            output_video_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.mp4")
        if output_json_path is None:
            output_json_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.json")
        
        output_txt_path = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}.txt")
        self._output_txt_path = output_txt_path
        
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
        
        # 显示处理设置
        self.logger.section("处理设置")
        self.logger.info(f"- 标注框样式: {'红色空心框' if self.box_style == 'red_hollow' else '绿色半透明填充'}")
        self.logger.info(f"- 图像增强: {'启用 (' + self.enhance_mode + ')' if self.enhance_mode else '禁用'}")
        self.logger.info(f"- 多线程流水线: {'启用 (队列=' + str(self.pipeline_queue_size) + ')' if self.pipeline_queue_size > 0 else '禁用'}")
        self.logger.info(f"- 保存帧图片: {'是' if self.save_frames else '否'}")
        self.logger.info(f"- 输出目录: {output_dir}")
        
        # 开始处理
        self.logger.section("开始处理视频")
        
        mode_info = []
        if self.skip_frames == 0:
            mode_info.append("逐帧处理")
        else:
            mode_info.append(f"每{self.skip_frames + 1}帧处理1帧")
        if self.scale_factor < 1.0:
            mode_info.append(f"缩放{self.scale_factor:.0%}")
        if self.enhance_mode:
            mode_info.append(f"图像增强({self.enhance_mode})")
        if self.pipeline_queue_size > 0:
            mode_info.append("多线程流水线")
        
        self.logger.info(f"开始处理，预计 {self.video_info['total_frames']} 帧 ({', '.join(mode_info)})...")
        
        self.timer.start("OCR处理")
        
        # 选择处理方式
        if self.pipeline_queue_size > 0:
            frames_data = self._process_pipeline(input_path, output_video_path)
        else:
            frames_data = self._process_sequential(input_path, output_video_path)
        
        self.timer.stop("OCR处理")
        
        # 记录预处理耗时
        if self.enhance_mode and self.stats["preprocess_time"] > 0:
            self.timer.stages["图像预处理"] = {
                "start": 0, "end": 0,
                "duration": self.stats["preprocess_time"]
            }
            if "OCR处理" in self.timer.stages:
                self.timer.stages["OCR处理"]["duration"] -= self.stats["preprocess_time"]
        
        # 记录保存帧图片耗时
        if self.save_frames and self.stats["frame_save_time"] > 0:
            self.timer.stages["保存帧图片"] = {
                "start": 0, "end": 0,
                "duration": self.stats["frame_save_time"]
            }
            if "OCR处理" in self.timer.stages:
                self.timer.stages["OCR处理"]["duration"] -= self.stats["frame_save_time"]
        
        self.logger.success("视频处理完成")
        
        # 保存JSON
        self.logger.section("保存结果")
        self.timer.start("保存JSON")
        self._save_json(frames_data, output_json_path)
        self.timer.stop("保存JSON")
        self.logger.info(f"✓ JSON已保存: {output_json_path}")
        
        self.timer.stop("总耗时")
        
        self._print_summary(output_video_path, output_json_path)
    
    def _save_json(self, frames_data: List[Dict], output_path: str):
        """保存JSON文件"""
        settings = {
            "box_style": self.box_style,
            "keep_audio": self.keep_audio,
            "use_paddle_ocr": self.use_paddle_ocr,
            "save_frames": self.save_frames,
            "enhance_mode": self.enhance_mode,
            "pipeline_queue_size": self.pipeline_queue_size
        }
        
        if self.use_paddle_ocr:
            settings["lang"] = self.lang
        
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
        
        if self.save_frames and self._frames_dir:
            result["frames_dir"] = self._frames_dir
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def _print_summary(self, video_path: str, json_path: str):
        """打印处理汇总"""
        lines = []
        
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
        log_and_save(f"  ├─ 图像增强: {'启用 (' + self.enhance_mode + ')' if self.enhance_mode else '禁用'}")
        log_and_save(f"  ├─ 多线程流水线: {'启用 (队列=' + str(self.pipeline_queue_size) + ')' if self.pipeline_queue_size > 0 else '禁用'}")
        log_and_save(f"  ├─ 保存帧图片: {'是' if self.save_frames else '否'}")
        log_and_save(f"  └─ 保留音频: {'是' if self.keep_audio else '否'}")
        
        # 视频信息
        log_and_save("\n视频信息:")
        log_and_save(f"  ├─ 文件: {self.video_info.get('filename', 'N/A')}")
        log_and_save(f"  ├─ 分辨率: {self.video_info.get('width', 0)}x{self.video_info.get('height', 0)}")
        log_and_save(f"  ├─ 帧率: {self.video_info.get('fps', 0):.1f} fps")
        log_and_save(f"  ├─ 总帧数: {self.video_info.get('total_frames', 0)}")
        log_and_save(f"  └─ 时长: {self.video_info.get('duration', 0):.1f}s")
        
        # 总耗时
        total_time = self.timer.get_duration("总耗时")
        if total_time <= 0:
            total_time = self.timer.get_total()
        log_and_save(f"\n总耗时: {total_time:.1f}s")
        
        # 各阶段耗时
        log_and_save("\n各阶段耗时:")
        stages_order = ["模型初始化", "视频分析", "OCR处理", "图像预处理", "保存帧图片", "保存JSON"]
        stages_total = 0
        
        for stage in stages_order:
            if stage in self.timer.stages:
                duration = self.timer.stages[stage]["duration"]
                if duration < 0:
                    duration = 0
                percent = (duration / total_time * 100) if total_time > 0 else 0
                stages_total += duration
                device = ""
                if stage == "模型初始化" or stage == "OCR处理":
                    device = f"[{self.device_info.get('paddle_device', '')}]"
                log_and_save(f"  ├─ {stage}: {duration:.2f}s ({percent:.1f}%) {device}")
        
        other_time = total_time - stages_total
        if other_time > 0.1:
            other_percent = (other_time / total_time * 100) if total_time > 0 else 0
            log_and_save(f"  └─ 其他(编解码等): {other_time:.2f}s ({other_percent:.1f}%)")
        
        # 输出文件信息
        log_and_save("\n输出文件:")
        if os.path.exists(video_path):
            video_size = os.path.getsize(video_path) / (1024 * 1024)
            log_and_save(f"  ├─ 视频: {video_path} ({video_size:.1f} MB)")
        
        if os.path.exists(json_path):
            json_size = os.path.getsize(json_path) / 1024
            log_and_save(f"  ├─ JSON: {json_path} ({json_size:.1f} KB)")
        
        if hasattr(self, '_output_txt_path') and self._output_txt_path:
            log_and_save(f"  ├─ 统计: {self._output_txt_path}")
        
        if self.save_frames and self._frames_dir and os.path.exists(self._frames_dir):
            frames_saved = self.stats.get("frames_saved", 0)
            total_size = sum(os.path.getsize(os.path.join(self._frames_dir, f)) 
                           for f in os.listdir(self._frames_dir) if os.path.isfile(os.path.join(self._frames_dir, f)))
            size_mb = total_size / (1024 * 1024)
            log_and_save(f"  ├─ 帧图片: {self._frames_dir} ({frames_saved} 张, {size_mb:.1f} MB)")
        
        log_and_save(f"  └─ (完成)")
        
        # 检测统计
        log_and_save("\n检测统计:")
        log_and_save(f"  ├─ 总帧数: {self.stats['frames_processed']}")
        
        if self.stats.get('frames_skipped', 0) > 0:
            log_and_save(f"  ├─ OCR处理帧数: {self.stats['frames_processed'] - self.stats['frames_skipped']}")
            log_and_save(f"  ├─ 跳过帧数: {self.stats['frames_skipped']}")
        
        log_and_save(f"  ├─ 检测文字区域: {self.stats['total_detections']} 个")
        avg = self.stats['total_detections'] / self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0
        log_and_save(f"  └─ 平均每帧: {avg:.1f} 个")
        
        lines.append("\n" + "=" * 80)
        self.logger.info("\n" + "=" * 80)
        
        # 保存到txt文件
        if hasattr(self, '_output_txt_path') and self._output_txt_path:
            try:
                with open(self._output_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                self.logger.warning(f"保存统计文件失败: {e}")


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VideoOCR v2 - 视频文字检测工具（优化版）")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出视频路径", default=None)
    parser.add_argument("-j", "--json", help="输出JSON路径", default=None)
    parser.add_argument(
        "-s", "--style",
        choices=["red_hollow", "green_fill"],
        default="red_hollow",
        help="标注框样式"
    )
    parser.add_argument("-a", "--audio", action="store_true", help="保留原视频音频")
    parser.add_argument("--text-detection", action="store_true", help="使用TextDetection模式")
    parser.add_argument("-l", "--lang", default="ch", help="PaddleOCR语言")
    parser.add_argument("-m", "--model", default="PP-OCRv5_server_det", help="TextDetection模型名称")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")
    parser.add_argument("--server", action="store_true", help="使用Server模型")
    parser.add_argument("--skip", type=int, default=0, help="跳帧数")
    parser.add_argument("--detect-only", action="store_true", help="仅检测文字位置")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--scale", type=float, default=1.0, help="缩放因子")
    parser.add_argument("--batch", type=int, default=1, help="批量处理帧数")
    parser.add_argument("-d", "--output-dir", default="output", help="输出目录")
    parser.add_argument("--cache", action="store_true", help="启用结果缓存")
    parser.add_argument("--cache-iou", type=float, default=0.85, help="缓存IoU阈值")
    parser.add_argument("--save-frames", action="store_true", help="保存标注后的每一帧图片")
    
    # 新增参数
    parser.add_argument(
        "--enhance-mode",
        choices=["clahe", "binary", "both"],
        default=None,
        help="图像预处理模式: clahe(对比度增强) / binary(二值化) / both(两者结合)，不指定则禁用"
    )
    parser.add_argument(
        "--pipeline-queue-size",
        type=int,
        default=0,
        help="流水线队列大小，>0时启用多线程流水线，推荐32"
    )
    
    args = parser.parse_args()
    
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
        save_frames=args.save_frames,
        enhance_mode=args.enhance_mode,
        pipeline_queue_size=args.pipeline_queue_size
    )
    
    ocr.process(
        input_path=args.input,
        output_dir=args.output_dir,
        output_video_path=args.output,
        output_json_path=args.json
    )


if __name__ == "__main__":
    main()