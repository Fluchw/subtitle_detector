#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构后的模块化代码
"""


def test_imports():
    """测试所有模块是否可以正常导入"""
    print("测试模块导入...")

    try:
        from core import (
            VideoProcessor,
            VideoInfo,
            OCREngine,
            ImageEnhancer,
            FrameRenderer,
            FrameSaver,
            ResultExporter,
            Logger,
            Timer,
            DeviceInfo,
            get_ffmpeg_path
        )
        print("✓ core 模块导入成功")
    except ImportError as e:
        print(f"✗ core 模块导入失败: {e}")
        return False

    try:
        from video_ocr import VideoOCR, main
        print("✓ video_ocr 模块导入成功")
    except ImportError as e:
        print(f"✗ video_ocr 模块导入失败: {e}")
        return False

    return True


def test_logger():
    """测试日志类"""
    print("\n测试 Logger 类...")

    from core import Logger

    logger = Logger(verbose=True)
    logger.info("这是一条普通日志")
    logger.success("这是一条成功日志")
    logger.warning("这是一条警告日志")
    logger.section("这是一个章节标题")

    print("✓ Logger 测试通过")


def test_timer():
    """测试计时器类"""
    print("\n测试 Timer 类...")

    import time
    from core import Timer

    timer = Timer()
    timer.start("阶段1")
    time.sleep(0.1)
    timer.stop("阶段1")

    timer.start("阶段2")
    time.sleep(0.05)
    timer.stop("阶段2")

    total = timer.get_total()
    print(f"总耗时: {total:.3f}s")
    print(f"阶段1: {timer.get_duration('阶段1'):.3f}s")
    print(f"阶段2: {timer.get_duration('阶段2'):.3f}s")

    print("✓ Timer 测试通过")


def test_frame_renderer():
    """测试帧渲染器"""
    print("\n测试 FrameRenderer 类...")

    try:
        import numpy as np
        import cv2
        from core import FrameRenderer

        # 创建测试帧
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 创建测试框
        boxes = [
            [[10, 10], [50, 10], [50, 30], [10, 30]]
        ]

        renderer = FrameRenderer(box_style="red_hollow")
        result = renderer.draw_boxes(frame, boxes)

        assert result.shape == frame.shape
        print("✓ FrameRenderer 测试通过")

    except ImportError:
        print("⊘ 跳过 FrameRenderer 测试（缺少依赖）")


def test_module_independence():
    """测试模块独立性"""
    print("\n测试模块独立性...")

    # 测试是否可以独立导入和使用各个模块
    from core.utils import Logger, Timer
    from core.frame_renderer import FrameRenderer
    from core.result_exporter import ResultExporter

    logger = Logger(verbose=False)
    timer = Timer()
    renderer = FrameRenderer()

    print("✓ 各模块可以独立导入和使用")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("VideoOCR 重构代码测试")
    print("=" * 60)

    if not test_imports():
        print("\n✗ 基础导入测试失败，停止测试")
        return

    test_logger()
    test_timer()
    test_frame_renderer()
    test_module_independence()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
