# VideoOCR 重构文档

## 重构概述

将原有的 1200+ 行单文件代码重构为模块化架构，提升可维护性和可扩展性。

## 🆕 新增功能

### Mask 模式 - 黑白遮罩输出

新增 `box_style="mask"` 参数，可生成黑白遮罩视频：
- **字幕区域**：白色实心 (255, 255, 255)
- **其他区域**：黑色实心 (0, 0, 0)

**用途**：
- 视频去字幕（作为遮罩层）
- 字幕区域定位
- 后期处理的 mask 输入

**示例**：
```python
ocr = VideoOCR(box_style="mask")
ocr.process("input.mp4")
```

**命令行**：
```bash
python video_ocr.py input.mp4 -s mask
```

## 模块划分

### 📁 core/ - 核心模块目录

#### 1. `video_processor.py` - 视频处理模块
**职责**：视频编解码、帧提取
- `VideoInfo`: 视频信息数据类
- `VideoProcessor`: 视频处理器
  - `get_video_info()`: 获取视频信息
  - `create_reader()`: 创建视频读取进程
  - `create_writer()`: 创建视频写入进程
  - `read_frame()`: 读取单帧
  - `write_frame()`: 写入单帧

#### 2. `ocr_engine.py` - OCR 引擎模块
**职责**：文字检测和识别
- `ImageEnhancer`: 图像增强器
  - `enhance()`: 图像预处理（CLAHE / 二值化）
- `OCREngine`: OCR 引擎
  - `process_frame()`: 处理单帧，返回 boxes 和 detections
  - `get_model_info()`: 获取模型信息描述

#### 3. `frame_renderer.py` - 帧渲染模块
**职责**：绘制标注框和保存帧图片
- `FrameRenderer`: 帧渲染器
  - `draw_boxes()`: 绘制标注框
- `FrameSaver`: 帧保存器
  - `save_frame()`: 保存单帧图片

#### 4. `result_exporter.py` - 结果导出模块
**职责**：导出 JSON 和统计信息
- `ResultExporter`: 结果导出器
  - `export_json()`: 导出 JSON 结果文件
  - `export_txt_summary()`: 导出文本统计摘要

#### 5. `utils.py` - 工具类模块
**职责**：日志、计时器、设备信息
- `Logger`: 日志输出类
- `Timer`: 计时器类
- `DeviceInfo`: 设备信息获取器

### 📄 video_ocr.py - 主协调类（重构后）
**职责**：组装各模块，控制处理流程
- 从 1200+ 行精简到 ~550 行
- 移除了流水线相关代码（~100 行）
- 保持所有原有功能和 API 兼容性

## 重构前后对比

| 项目 | 重构前 | 重构后 |
|------|--------|--------|
| 总代码行数 | ~1234 行 | ~800 行（拆分为 6 个文件） |
| 单文件最大行数 | 1234 行 | 550 行 |
| 类的职责数量 | 5+ | 1-2（单一职责） |
| 可测试性 | 困难 | 容易（每个模块独立测试） |
| 可扩展性 | 困难 | 容易（新增模型/格式等） |
| 流水线代码 | 保留（~100 行） | **已删除** |

## 删除的功能

### ❌ 多线程流水线处理
**原因**：
1. 增加了 ~100 行复杂度
2. Python GIL 限制，CPU 上多线程效果不佳
3. GPU 处理时主要瓶颈在 OCR，多线程意义不大
4. 维护成本高

**影响**：
- 参数 `pipeline_queue_size` 已移除
- 相关方法 `_decode_worker()`, `_ocr_worker()`, `_encode_worker()`, `_process_pipeline()` 已删除
- 使用串行方式 `_process_video()` 处理视频

## 保留的功能

✅ 所有原有功能都已保留：
- PaddleOCR / TextDetection 双引擎支持
- 图像预处理增强（CLAHE / 二值化）
- 跳帧处理
- 缩放处理
- 帧图片保存
- JSON / TXT 导出
- 详细的日志和进度显示
- 完整的统计信息

## 向后兼容性

✅ **完全兼容原有 API**

所有参数保持不变，包括：
```python
VideoOCR(
    box_style="red_hollow",
    keep_audio=False,
    verbose=True,
    use_paddle_ocr=True,
    lang="ch",
    use_angle_cls=False,  # 保留但未使用
    model_name="PP-OCRv5_server_det",
    use_lightweight=True,
    skip_frames=0,
    detect_only=False,
    confidence_threshold=0.5,
    scale_factor=1.0,
    batch_size=1,  # 保留但未使用
    use_cache=False,  # 保留但未使用
    cache_iou_threshold=0.85,  # 保留但未使用
    save_frames=False,
    enhance_mode=None
)
```

**移除的参数**：
- `pipeline_queue_size`: 多线程流水线队列大小（功能已删除）

## 使用方式

### 基本用法（与之前完全相同）
```python
from video_ocr import VideoOCR

ocr = VideoOCR(
    use_lightweight=True,
    skip_frames=2,
    enhance_mode="clahe"
)

ocr.process(
    input_path="test.mp4",
    output_dir="output"
)
```

### 命令行用法（完全相同）
```bash
python video_ocr.py input.mp4 --skip 2 --enhance-mode clahe
```

## 目录结构

```
subtitle_detector/
├── core/                          # 核心模块目录（新）
│   ├── __init__.py
│   ├── video_processor.py         # 视频编解码
│   ├── ocr_engine.py              # OCR 引擎
│   ├── frame_renderer.py          # 帧渲染
│   ├── result_exporter.py         # 结果导出
│   └── utils.py                   # 工具类
├── video_ocr.py                   # 主类（重构后）
├── video_ocr_old.py               # 原文件备份
└── REFACTORING.md                 # 本文档
```

## 优势

### 1. **可维护性提升**
- 每个模块职责明确，修改时只需关注单个文件
- 代码量减少，更易理解

### 2. **可测试性提升**
- 每个模块可独立单元测试
- 不再需要初始化整个 VideoOCR 类

### 3. **可扩展性提升**
- 新增 OCR 引擎：只需修改 `ocr_engine.py`
- 新增输出格式：只需修改 `result_exporter.py`
- 新增视频编码器：只需修改 `video_processor.py`

### 4. **代码复用**
- 各模块可独立导入使用
- 便于在其他项目中复用

## 迁移指南

### 从旧版本迁移

如果你有基于旧版本的代码：

1. **直接替换**：新版本 API 完全兼容，直接替换文件即可
2. **移除 pipeline_queue_size 参数**（如果使用了）：
   ```python
   # 旧代码
   ocr = VideoOCR(pipeline_queue_size=32)

   # 新代码（移除此参数）
   ocr = VideoOCR()
   ```

### 如果需要流水线功能

如果你确实需要多线程流水线处理：
1. 使用 `video_ocr_old.py` 旧版本
2. 或者基于新架构自行实现（推荐使用进程池而非线程池）

## 测试

### 语法检查
```bash
python -m py_compile video_ocr.py core/*.py
```

### 导入测试
```python
from video_ocr import VideoOCR
from core import OCREngine, VideoProcessor, FrameRenderer

print("导入成功")
```

## 未来改进方向

1. **进程池处理**：替代线程池，绕过 GIL 限制
2. **插件化架构**：支持自定义 OCR 引擎
3. **GPU 加速编解码**：使用 NVENC/NVDEC
4. **增量更新**：仅处理视频变化的部分
5. **批量处理**：同时处理多个视频文件

## 总结

这次重构在**保持完全向后兼容**的前提下：
- ✅ 删除了复杂的流水线代码
- ✅ 模块化拆分，提升可维护性
- ✅ 代码行数减少 ~35%
- ✅ 每个模块职责单一，便于测试和扩展
- ✅ 为后续功能扩展打下良好基础

---

**重构时间**: 2025-12-23
**重构方式**: 轻量级解耦（保持简洁）
**向后兼容**: 完全兼容（除 `pipeline_queue_size` 参数）
