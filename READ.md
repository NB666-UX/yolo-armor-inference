 YOLO装甲板推理与关键点显示

1. 项目完成的功能
本项目基于指定的 YOLOv26 模型，对自行拍摄的装甲板视频进行推理，并在视频中实时绘制检测框、类别、置信度，以及模型预测的 4 个关键点。

 2. 环境配置
- Python版本：3.14.7
- 主要依赖：torch, torchvision, opencv-python, ultralytics
- 运行设备：CPU

3. 模型权重准备
需将老师提供的 `best.pt` 文件放置在项目根目录下。

 4. 运行步骤
1. 安装依赖：`pip install -r requirements.txt`
2. 确保原始视频命名为 `test_video.mp4` 并放在根目录。
3. 运行命令：`python infer.py`
4. 结果保存在当前目录下的 `result_video.mp4`。

5. 视频拍摄说明
- 地点：水建学院D414实验室
- 时长：约 20 秒
- 场景：横屏拍摄，包含了发光的 B3 装甲板，光线有明暗变化。

6.运行结果展示
！[推理结果截图](./result.png)





7. 识别表现与问题处理
- 关键点排查：最初使用 `box.keypoints` 获取关键点时报错 `'Boxes' object has no attribute 'keypoints'`。经过终端打印结果排查，发现关键点数据挂在 `r.keypoints` 中。修改代码后，成功获取并绘制了 4 个真实预测的关键点。
- 光照影响：实测发现，当光柱过曝或环境较暗时，偶尔会出现漏检或关键点轻微偏移，属于真实场景下的正常波动。

