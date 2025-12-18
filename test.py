# 1. 上传 video_ocr.py 到 Colab

# 2. 运行
from video_ocr import VideoOCR

# 创建实例
ocr = VideoOCR(
    # ===== 图像增强 =====
    enhance_mode="clahe",          # 预处理模式: None(禁用) / "clahe"(对比度增强) / "binary"(二值化) / "both"(两者结合)
    # ===== 多线程 =====
    pipeline_queue_size=64,        # 流水线队列大小: 0(禁用) / >0(启用，推荐32)
    # ===== 标注样式 =====
    box_style="green_fill",        # 标注框样式: "red_hollow"(红色空心) / "green_fill"(绿色半透明填充)
    # ===== 基础设置 =====
    keep_audio=False,              # 是否保留原视频音频
    verbose=True,                  # 是否输出详细日志
    # ===== 模型设置 =====
    use_lightweight=True,          # True: Mobile轻量模型(快) / False: Server模型(精准)
    detect_only=True,              # True: 仅检测位置(快) / False: 检测+识别文字
    # ===== 处理优化 =====
    skip_frames=0,                 # 跳帧: 0(逐帧) / N(每N+1帧处理1帧)
    confidence_threshold=0.75,     # 置信度阈值(0-1): 过滤低置信度结果，推荐0.5-0.8
    scale_factor=1,                # 缩放因子(0-1): 1.0(原尺寸) / 0.5(缩小50%，4K推荐)
    batch_size=256,                 # 批量处理帧数: 仅detect_only=True有效，GPU推荐4-8
    # ===== 缓存设置 =====
    use_cache=False,               # 结果缓存: 相邻帧相似时复用结果
    
    # cache_iou_threshold=0.85,    # 缓存IoU阈值: 框重叠度超过此值认为相同
    # ===== 输出设置 =====
    # save_frames=True,            # 保存每帧图片: 会增加处理时间和磁盘占用
)

# 处理视频
ocr.process(input_path="./t.mp4",)