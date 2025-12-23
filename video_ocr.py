#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoOCR v2 - 视频文字检测工具（重构版）
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional

import numpy as np

from core import *


class VideoOCR:
    """视频OCR检测主类（重构版）"""

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
        enhance_mode: Optional[Literal["clahe", "binary", "both"]] = None
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
            batch_size: 批量处理帧数，>1时启用批量OCR（保留参数，暂未使用）
            use_cache: 是否启用结果缓存（保留参数，暂未使用）
            cache_iou_threshold: 缓存IoU阈值（保留参数，暂未使用）
            save_frames: 是否保存标注后的每一帧图片
            enhance_mode: 图像预处理模式，None禁用，可选 "clahe"(对比度增强) / "binary"(二值化) / "both"(两者结合)
        """
        self.box_style = box_style
        self.keep_audio = keep_audio
        self.verbose = verbose
        self.skip_frames = skip_frames
        self.save_frames = save_frames
        self.enhance_mode = enhance_mode

        # 未使用的参数（保留用于向后兼容）
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_iou_threshold = cache_iou_threshold
        self.use_angle_cls = use_angle_cls
        self.model_name = model_name

        # 初始化核心组件
        self.logger = Logger(verbose)
        self.timer = Timer()

        self.video_processor = VideoProcessor()
        self.ocr_engine = OCREngine(
            use_paddle_ocr=use_paddle_ocr,
            lang=lang,
            use_lightweight=use_lightweight,
            detect_only=detect_only,
            confidence_threshold=confidence_threshold,
            scale_factor=scale_factor,
            enhance_mode=enhance_mode
        )
        self.renderer = FrameRenderer(box_style=box_style)
        self.frame_saver = None  # 稍后初始化

        # 设备信息和视频信息
        self.device_info = {}
        self.video_info = None

        # 统计信息
        self.stats = {
            "total_detections": 0,
            "frames_processed": 0,
            "frames_skipped": 0,
            "cache_hits": 0,
            "frames_saved": 0,
            "frame_save_time": 0.0,
            "preprocess_time": 0.0
        }

        # 输出路径
        self._output_txt_path = None
        self._frames_dir = None

        self._initialize()

    def _initialize(self):
        """初始化模型和设备信息"""
        self.logger.header("VideoOCR v2 视频文字检测工具（重构版）")
        self.logger.info("初始化开始...")

        self.timer.start("模型初始化")

        # 获取设备信息
        self.device_info = DeviceInfo.get_paddle_device_info()

        self.timer.stop("模型初始化")

        self.logger.success("模型加载完成")
        self.logger.info(f"- 模式: {self.ocr_engine.get_model_info()}", indent=28)
        self.logger.info(f"- 设备: {self.device_info.get('paddle_device', 'Unknown')}", indent=28)
        if self.enhance_mode:
            self.logger.info(f"- 图像增强: {self.enhance_mode}", indent=28)
        self.logger.info(f"- 耗时: {self.timer.get_duration('模型初始化'):.2f}s", indent=28)

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

        # 初始化帧保存器
        if self.save_frames:
            self._frames_dir = os.path.join(output_dir, f"{input_basename}_ocr_{timestamp_str}_frames")
            self.frame_saver = FrameSaver(output_dir=self._frames_dir)
            self.logger.info(f"帧图片将保存到: {self._frames_dir}")

        # 获取视频信息
        self.logger.section("视频信息分析")
        self.timer.start("视频分析")
        self.video_info = self.video_processor.get_video_info(input_path)
        self.timer.stop("视频分析")

        self._print_video_info()

        # 显示处理设置
        self._print_process_settings(output_dir)

        # 开始处理
        self.logger.section("开始处理视频")
        self._print_processing_info()

        self.timer.start("OCR处理")
        frames_data = self._process_video(input_path, output_video_path)
        self.timer.stop("OCR处理")

        # 调整耗时统计
        self._adjust_timing_stats()

        self.logger.success("视频处理完成")

        # 保存JSON
        self._save_results(frames_data, output_json_path)

        self.timer.stop("总耗时")

        self._print_summary(output_video_path, output_json_path)

    def _print_video_info(self):
        """打印视频信息"""
        self.logger.info(f"✓ 视频信息:")
        self.logger.info(f"- 文件: {self.video_info.filename}", indent=28)
        self.logger.info(f"- 分辨率: {self.video_info.width}x{self.video_info.height}", indent=28)
        self.logger.info(f"- 帧率: {self.video_info.fps:.1f} fps", indent=28)
        self.logger.info(f"- 总帧数: {self.video_info.total_frames}", indent=28)
        self.logger.info(f"- 时长: {self.video_info.duration:.1f}s", indent=28)

    def _print_process_settings(self, output_dir: str):
        """打印处理设置"""
        self.logger.section("处理设置")
        self.logger.info(f"- 标注框样式: {'红色空心框' if self.box_style == 'red_hollow' else '绿色半透明填充'}")
        self.logger.info(f"- 图像增强: {'启用 (' + self.enhance_mode + ')' if self.enhance_mode else '禁用'}")
        self.logger.info(f"- 保存帧图片: {'是' if self.save_frames else '否'}")
        self.logger.info(f"- 输出目录: {output_dir}")

    def _print_processing_info(self):
        """打印处理信息"""
        mode_info = []
        if self.skip_frames == 0:
            mode_info.append("逐帧处理")
        else:
            mode_info.append(f"每{self.skip_frames + 1}帧处理1帧")
        if self.ocr_engine.scale_factor < 1.0:
            mode_info.append(f"缩放{self.ocr_engine.scale_factor:.0%}")
        if self.enhance_mode:
            mode_info.append(f"图像增强({self.enhance_mode})")

        self.logger.info(f"开始处理，预计 {self.video_info.total_frames} 帧 ({', '.join(mode_info)})...")

    def _process_video(self, input_path: str, output_video_path: str) -> List[Dict]:
        """处理视频（串行方式）"""
        width = self.video_info.width
        height = self.video_info.height
        fps = self.video_info.fps
        total_frames = self.video_info.total_frames

        # 创建读写进程
        read_process = self.video_processor.create_reader(input_path)
        write_process = self.video_processor.create_writer(
            output_video_path,
            width,
            height,
            fps,
            self.keep_audio and self.video_info.has_audio,
            input_path
        )

        frames_data = []
        frame_id = 0

        if total_frames <= 0:
            total_frames = int(fps * 60)

        process_start_time = time.time()
        preprocess_start_time = 0

        last_boxes = []
        last_detections = []

        try:
            while True:
                frame = self.video_processor.read_frame(read_process, width, height)
                if frame is None:
                    break

                timestamp = frame_id / fps

                # 判断是否需要OCR处理
                should_process = (self.skip_frames == 0) or (frame_id % (self.skip_frames + 1) == 0)

                if should_process:
                    # OCR处理
                    preprocess_start = time.time()
                    boxes, detections = self.ocr_engine.process_frame(frame)
                    preprocess_start_time += time.time() - preprocess_start

                    last_boxes = boxes
                    last_detections = detections
                    self.stats["total_detections"] += len(detections)

                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "detections": detections
                    }
                else:
                    # 使用上一帧的结果
                    boxes = last_boxes
                    detections = last_detections
                    self.stats["frames_skipped"] += 1

                    frame_data = {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 3),
                        "detections": detections,
                        "skipped": True
                    }

                # 绘制标注框
                result_frame = self.renderer.draw_boxes(frame, boxes)

                # 保存帧图片
                if self.frame_saver:
                    save_start = time.time()
                    self.frame_saver.save_frame(result_frame, frame_id)
                    self.stats["frame_save_time"] += time.time() - save_start

                # 写入视频
                self.video_processor.write_frame(write_process, result_frame)

                frames_data.append(frame_data)
                frame_id += 1
                self.stats["frames_processed"] = frame_id

                # 更新进度
                self.logger.progress(frame_id, total_frames, process_start_time)

        finally:
            read_process.stdout.close()
            read_process.wait()
            write_process.stdin.close()
            write_process.wait()

        self.stats["preprocess_time"] = preprocess_start_time

        return frames_data

    def _adjust_timing_stats(self):
        """调整耗时统计"""
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

    def _save_results(self, frames_data: List[Dict], output_json_path: str):
        """保存结果"""
        self.logger.section("保存结果")
        self.timer.start("保存JSON")

        settings = {
            "box_style": self.box_style,
            "keep_audio": self.keep_audio,
            "use_paddle_ocr": self.ocr_engine.use_paddle_ocr,
            "save_frames": self.save_frames,
            "enhance_mode": self.enhance_mode
        }

        if self.ocr_engine.use_paddle_ocr:
            settings["lang"] = self.ocr_engine.lang

        device_info = {
            "ocr_device": self.device_info.get("paddle_device", "Unknown"),
            "ffmpeg_decode": self.device_info.get("ffmpeg_decode", "Unknown"),
            "ffmpeg_encode": self.device_info.get("ffmpeg_encode", "Unknown")
        }

        ResultExporter.export_json(
            output_json_path,
            self.video_info.to_dict(),
            settings,
            device_info,
            frames_data,
            self._frames_dir
        )

        self.timer.stop("保存JSON")
        self.logger.info(f"✓ JSON已保存: {output_json_path}")

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
        log_and_save(f"  ├─ 模型: {'Mobile(轻量)' if self.ocr_engine.use_lightweight else 'Server(精准)'}")
        log_and_save(f"  ├─ 模式: {'仅检测' if self.ocr_engine.detect_only else '检测+识别'}")
        log_and_save(f"  ├─ 跳帧: {self.skip_frames} (每{self.skip_frames + 1}帧处理1帧)")
        log_and_save(f"  ├─ 置信度阈值: {self.ocr_engine.confidence_threshold}")
        log_and_save(f"  ├─ 缩放因子: {self.ocr_engine.scale_factor}")
        log_and_save(f"  ├─ 图像增强: {'启用 (' + self.enhance_mode + ')' if self.enhance_mode else '禁用'}")
        log_and_save(f"  ├─ 保存帧图片: {'是' if self.save_frames else '否'}")
        log_and_save(f"  └─ 保留音频: {'是' if self.keep_audio else '否'}")

        # 视频信息
        log_and_save("\n视频信息:")
        log_and_save(f"  ├─ 文件: {self.video_info.filename}")
        log_and_save(f"  ├─ 分辨率: {self.video_info.width}x{self.video_info.height}")
        log_and_save(f"  ├─ 帧率: {self.video_info.fps:.1f} fps")
        log_and_save(f"  ├─ 总帧数: {self.video_info.total_frames}")
        log_and_save(f"  └─ 时长: {self.video_info.duration:.1f}s")

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

        if self._output_txt_path:
            log_and_save(f"  ├─ 统计: {self._output_txt_path}")

        if self.save_frames and self._frames_dir and os.path.exists(self._frames_dir):
            frames_saved = self.frame_saver.frames_saved if self.frame_saver else 0
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
        if self._output_txt_path:
            try:
                ResultExporter.export_txt_summary(self._output_txt_path, lines)
            except Exception as e:
                self.logger.warning(f"保存统计文件失败: {e}")


def main():
    """主函数 - 命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="VideoOCR v2 - 视频文字检测工具（重构版）")
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
    parser.add_argument("--batch", type=int, default=1, help="批量处理帧数（保留参数，暂未使用）")
    parser.add_argument("-d", "--output-dir", default="output", help="输出目录")
    parser.add_argument("--cache", action="store_true", help="启用结果缓存（保留参数，暂未使用）")
    parser.add_argument("--cache-iou", type=float, default=0.85, help="缓存IoU阈值（保留参数，暂未使用）")
    parser.add_argument("--save-frames", action="store_true", help="保存标注后的每一帧图片")
    parser.add_argument(
        "--enhance-mode",
        choices=["clahe", "binary", "both"],
        default=None,
        help="图像预处理模式: clahe(对比度增强) / binary(二值化) / both(两者结合)，不指定则禁用"
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
        enhance_mode=args.enhance_mode
    )

    ocr.process(
        input_path=args.input,
        output_dir=args.output_dir,
        output_video_path=args.output,
        output_json_path=args.json
    )


if __name__ == "__main__":
    main()
