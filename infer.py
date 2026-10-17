import cv2
from ultralytics import YOLO

# ==================== 配置区 ====================
MODEL_PATH = "best.pt"               # 模型路径
INPUT_VIDEO = "test_video.mp4"       # 输入视频名
OUTPUT_VIDEO = "result_video.mp4"    # 输出视频名
# =================================================

def main():
    # 1. 加载模型
    print("正在加载模型...")
    model = YOLO(MODEL_PATH)

    # 2. 读取视频
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {INPUT_VIDEO}，请检查文件名和路径！")
        return

    # 3. 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频参数：{width}x{height}, {fps}帧/秒")

    # 4. 设置输出视频编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    print("开始逐帧推理...")

    # 5. 逐帧循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 6. 模型推理
        results = model(frame, conf=0.5, verbose=False)

        # 7. 遍历结果，画框、类别、置信度和关键点
        for r in results:
            # ----- 画关键点（直接从 r 里取，这是真正的模型预测点） -----
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()
                # kpts 的形状是 [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]]，逐点画出来
                for kp in kpts:
                    for pt in kp:
                        kp_x, kp_y = int(pt[0]), int(pt[1])
                        # 画红色实心圆点（半径5）
                        cv2.circle(frame, (kp_x, kp_y), 5, (0, 0, 255), -1)

            # ----- 画检测框、类别、置信度 -----
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                # ① 画检测框（黄色）
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # ② 画类别和置信度文字
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 8. 把画好的这一帧写入输出视频
        out.write(frame)

        # 每 30 帧打印一次进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧...")

    # 9. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"推理完成！结果视频已保存为：{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()