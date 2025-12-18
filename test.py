# 1. 上传 video_ocr.py 到 Colab

# 2. 运行
from video_ocr import VideoOCR

# 创建实例
ocr = VideoOCR(
    box_style="red_hollow",      # 或 "green_fill"
    keep_audio=False,
    verbose=True,
    use_lightweight=True,
    skip_frames=0,
    detect_only=True,
    confidence_threshold=0.8,
    # 新增参数
    scale_factor=0.4,            # 缩放因子(0-1)，如0.5缩小到50%，4K视频推荐
    batch_size=16,                # 批量OCR帧数，GPU下可设为4-8
    use_cache=False,             # 启用结果缓存，相似帧复用结果
    # cache_iou_threshold=0.85     # 缓存IoU阈值，框重叠度超过此值认为相同
)

# 处理视频
ocr.process(input_path="./t.mp4",)