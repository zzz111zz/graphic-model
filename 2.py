import cv2
import subprocess
import numpy as np
import warnings
import threading
import queue
import time

warnings.filterwarnings('ignore')
import os
os.environ['ULTRALYTICS_VERBOSE'] = 'False'
from ultralytics import YOLO

# ===================== 配置 =====================
rtsp_input_url = "rtsp://127.0.0.1:25544/input"
rtsp_output_url = "rtsp://127.0.0.1:25544/output"
VIDEO_SIZE = "960x544"
FPS = 25

# 加载模型（一次加载，全局复用）
model = YOLO("yolov8n-pose.pt")

# FFmpeg 推流命令（稳定优先）
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-v', 'quiet',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', VIDEO_SIZE,
    '-r', str(FPS),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '28',
    '-threads', '2',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    rtsp_output_url
]

# 全局队列，用来给推流线程喂帧
frame_queue = queue.Queue(maxsize=10)

# 人体骨骼连线
skeleton = [
    [16,14],[14,12],[17,15],[15,13],[12,13],
    [6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11]
]

# 推流线程：只管稳定按帧率发帧
def push_stream():
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    interval = 1.0 / FPS
    last_send_time = 0
    while True:
        # 按固定帧率从队列取帧，避免忽快忽慢
        if time.time() - last_send_time >= interval:
            if not frame_queue.empty():
                img = frame_queue.get()
                try:
                    process.stdin.write(img.tobytes())
                    last_send_time = time.time()
                except:
                    break
        else:
            time.sleep(0.001)
    process.stdin.close()
    process.wait()

print("✅ 程序已启动：双线程稳定推流中...")

# 启动推流线程
threading.Thread(target=push_stream, daemon=True).start()

# 主线程：读帧 + 做YOLO识别（识别慢也不会卡推流）
cap = cv2.VideoCapture(rtsp_input_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO识别（降低尺寸+置信度，提速）
    results = model(frame, conf=0.3, verbose=False, imgsz=320)
    img = frame.copy()

    # 画关键点
    for res in results:
        keypoints = res.keypoints.data.cpu().numpy()
        for kpt in keypoints:
            for x, y, conf in kpt:
                if conf > 0.2:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            for link in skeleton:
                p1 = kpt[link[0]-1]
                p2 = kpt[link[1]-1]
                if p1[2] > 0.2 and p2[2] > 0.2:
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 1)

    # 把处理好的帧丢进队列，让推流线程稳定发送
    if not frame_queue.full():
        frame_queue.put(img)

cap.release()