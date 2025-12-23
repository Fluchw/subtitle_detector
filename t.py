from paddleocr import TextDetection
import numpy as np
import cv2
import paddle

# 强制使用 CPU
paddle.set_device('cpu')

model = TextDetection(model_name="PP-OCRv5_mobile_det")

# 创建测试图
imgs = []
positions = [(10, 70), (100, 70), (200, 70), (50, 50)]
for i in range(4):
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, f"TEXT{i}", positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    imgs.append(img)

results = list(model.predict(imgs, batch_size=4))

print(f"结果数量: {len(results)}")
for idx, res in enumerate(results):
    if hasattr(res, 'json') and res.json:
        res_dict = res.json.get('res', res.json)
        polys = res_dict.get('dt_polys', [])
        if len(polys) > 0:
            x = polys[0][0][0]
            print(f"输入索引 {idx} -> 检测到框，x位置约: {x:.0f}")
        else:
            print(f"输入索引 {idx} -> 未检测到")