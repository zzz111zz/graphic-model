# YOLOv8姿态识别RTSP视频流处理与推流项目说明
## 一、项目概述
本项目基于Python语言开发，结合OpenCV、FFmpeg与YOLOv8姿态识别模型（yolov8n-pose.pt），实现对RTSP视频流的实时人体姿态关键点检测，并将标注了骨骼关键点的视频帧通过RTSP协议稳定推流输出。项目采用双线程架构，分离视频帧处理与推流逻辑，保障推流的稳定性和帧率一致性，适用于实时人体姿态分析、行为识别等场景。

## 二、核心功能
1. **RTSP流读取**：从指定RTSP地址拉取原始视频流，支持缓冲配置优化，减少帧延迟；
2. **姿态关键点检测**：基于YOLOv8n-pose轻量化模型，实时识别视频帧中人体的骨骼关键点（如关节、骨骼连线）；
3. **关键点可视化**：在视频帧上绘制人体骨骼关键点（绿色圆点）和骨骼连线（红色线条），直观展示姿态识别结果；
4. **稳定推流输出**：通过FFmpeg将处理后的视频帧以RTSP协议推送到指定地址，采用固定帧率、双线程机制保障推流稳定性；
5. **异常处理**：具备帧读取失败、推流进程异常中断等场景的基础容错能力。

## 三、技术栈
| 技术/工具 | 版本/用途 |
|----------|-----------|
| Python   | 核心开发语言 |
| OpenCV   | 视频流读取、帧处理、图形绘制 |
| YOLOv8   | yolov8n-pose.pt模型，人体姿态关键点检测 |
| FFmpeg   | 视频帧编码与RTSP推流 |
| 多线程/队列 | 分离处理与推流线程，控制帧队列，保障帧率稳定 |
| subprocess | 调用FFmpeg进程执行推流命令 |

## 四、项目架构
### 1. 整体流程
```
RTSP输入流 → 帧读取（主线程） → YOLOv8姿态识别 → 关键点可视化 → 帧队列缓存 → 推流线程 → FFmpeg编码 → RTSP输出流
```

### 2. 线程分工
- **主线程**：负责从RTSP地址读取视频帧，调用YOLOv8模型完成姿态关键点检测，绘制骨骼关键点/连线后，将处理后的帧存入队列；
- **推流线程**：独立从队列中按固定帧率取帧，通过FFmpeg将帧编码并推送到目标RTSP地址，避免因识别耗时导致推流帧率波动。

## 五、核心配置说明
### 1. 基础参数
| 参数名 | 取值 | 说明 |
|--------|------|------|
| rtsp_input_url | rtsp://127.0.0.1:25544/input | 原始RTSP视频流输入地址 |
| rtsp_output_url | rtsp://127.0.0.1:25544/output | 处理后视频流推流地址 |
| VIDEO_SIZE | 960x544 | 视频帧分辨率（需与输入流匹配） |
| FPS | 25 | 推流输出帧率，控制推流速度 |
| 置信度阈值 | conf=0.3（模型）/0.2（关键点） | 模型识别置信度过滤，降低误检 |

### 2. YOLOv8姿态识别配置
- 模型：采用轻量化的yolov8n-pose.pt，兼顾识别速度与精度，适合实时场景；
- 识别优化：设置`imgsz=320`降低推理尺寸，提升处理速度；关闭verbose日志，减少资源占用；
- 骨骼连线：预设人体骨骼关键点连接规则（如16-14为脚踝-膝盖），仅在关键点置信度>0.2时绘制连线。

### 3. FFmpeg推流配置
```python
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # 覆盖已有文件
    '-v', 'quiet',  # 静默日志，减少输出
    '-f', 'rawvideo',  # 输入格式为原始视频帧
    '-pix_fmt', 'bgr24',  # 像素格式（匹配OpenCV输出）
    '-s', VIDEO_SIZE,  # 视频分辨率
    '-r', str(FPS),  # 输入帧率
    '-i', '-',  # 从标准输入读取帧数据
    '-c:v', 'libx264',  # H.264编码
    '-preset', 'ultrafast',  # 超快编码速度，优先实时性
    '-tune', 'zerolatency',  # 零延迟调优
    '-crf', '28',  # 视频质量（值越大质量越低，文件越小）
    '-threads', '2',  # 编码线程数
    '-f', 'rtsp',  # 输出格式为RTSP
    '-rtsp_transport', 'tcp',  # TCP传输，提升稳定性
    rtsp_output_url  # 推流目标地址
]
```

## 六、核心代码逻辑
### 1. 模型加载（全局复用）
```python
from ultralytics import YOLO
# 仅加载一次，避免重复初始化开销
model = YOLO("yolov8n-pose.pt")
```

### 2. 帧处理与关键点绘制
```python
# YOLO识别（降低尺寸+置信度，提速）
results = model(frame, conf=0.3, verbose=False, imgsz=320)
img = frame.copy()

# 人体骨骼连线规则
skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11]]

# 绘制关键点（绿色圆点）和骨骼连线（红色线条）
for res in results:
    keypoints = res.keypoints.data.cpu().numpy()
    for kpt in keypoints:
        # 绘制关键点
        for x, y, conf in kpt:
            if conf > 0.2:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
        # 绘制骨骼连线
        for link in skeleton:
            p1 = kpt[link[0]-1]
            p2 = kpt[link[1]-1]
            if p1[2] > 0.2 and p2[2] > 0.2:
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 1)
```

### 3. 双线程推流核心
```python
import queue
import threading
import time

# 全局帧队列，缓存处理后的帧（最大10帧，避免内存溢出）
frame_queue = queue.Queue(maxsize=10)

# 推流线程：按固定帧率取帧推流
def push_stream():
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    interval = 1.0 / FPS  # 帧间隔时间
    last_send_time = 0
    while True:
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

# 启动推流线程（守护线程）
threading.Thread(target=push_stream, daemon=True).start()

# 主线程：读帧+处理，将帧存入队列
cap = cv2.VideoCapture(rtsp_input_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓存，降低延迟
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 处理帧（识别+绘制）...
    if not frame_queue.full():
        frame_queue.put(img)
```

## 七、运行要求
### 1. 环境依赖
```bash
# 安装Python依赖
pip install opencv-python numpy ultralytics
# 安装FFmpeg（需加入系统环境变量）
# Windows：下载FFmpeg包并配置PATH
# Linux：apt install ffmpeg
# Mac：brew install ffmpeg
```

### 2. 硬件要求
- 处理器：双核及以上（推荐四核）；
- 内存：2GB及以上；
- 网络：确保RTSP输入流可访问，推流目标RTSP服务器（如EasyDarwin）已启动。

### 3. RTSP服务器
需部署RTSP服务器（如EasyDarwin、ZLMediaKit），用于接收推流输出，示例地址：`rtsp://127.0.0.1:25544/output`。

## 八、运行步骤
1. 配置RTSP输入/输出地址、视频分辨率、帧率等参数；
2. 确保YOLOv8n-pose.pt模型文件存在（运行时会自动下载，或手动放置到指定路径）；
3. 启动RTSP服务器，监听指定端口；
4. 运行项目代码：
   ```bash
   python 2.py
   ```
5. 通过RTSP播放器（如VLC）访问输出地址，查看标注了姿态关键点的视频流。

## 九、注意事项
1. 视频分辨率需与输入流匹配，否则会导致推流画面变形；
2. 若推流卡顿，可调整`FPS`、`imgsz`（模型推理尺寸）或`threads`（FFmpeg线程数）；
3. 网络不稳定时，建议将`rtsp_transport`设为`tcp`（默认UDP），提升传输稳定性；
4. 模型置信度阈值可根据实际场景调整，平衡识别精度与速度；
5. 运行前需确保FFmpeg已正确安装并加入系统环境变量，否则会提示“找不到ffmpeg命令”。

## 十、扩展方向
1. 增加多目标姿态跟踪，区分不同人体的骨骼关键点；
2. 加入姿态异常检测（如跌倒、举手等），触发告警；
3. 优化推流逻辑，支持断线重连；
4. 增加视频流录制功能，将处理后的视频保存为本地文件；
5. 替换为更轻量化的模型（如YOLOv8s-pose）或量化模型，进一步提升推理速度。
