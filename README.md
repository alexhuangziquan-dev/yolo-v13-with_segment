# YOLOv11-seg 行人与车辆实例分割

https://github.com/user-attachments/assets/6b9912e2-7ddf-4d50-a31a-e0480345834e



![推理效果示例](results/demo.png)

基于 YOLOv11-seg 和 COCO 2017 数据集，识别行人与 5 类车辆的完整训练/推理项目。

## 目标类别

| ID | 类别 | COCO 原始 ID | 可视化颜色 |
|----|------|-------------|-----------|
| 0  | person（行人） | 1 | 绿色 |
| 1  | bicycle（单车） | 2 | 橙色 |
| 2  | car（小汽车） | 3 | 蓝色 |
| 3  | motorcycle（摩托/电动车） | 4 | 紫色 |
| 4  | bus（巴士） | 6 | 青色 |
| 5  | truck（卡车） | 8 | 红色 |

## 项目结构

```
yolo-v11-seg/
├── configs/
│   └── train_config.yaml       # 训练超参数配置
├── data/
│   ├── coco/                   # ← 手动下载 COCO 数据集到此处
│   │   ├── annotations/
│   │   │   ├── instances_train2017.json
│   │   │   └── instances_val2017.json
│   │   └── images/
│   │       ├── train2017/
│   │       └── val2017/
│   └── pedestrian_vehicle/     # 自动生成（运行 prepare_dataset.py 后）
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── dataset.yaml
├── models/                     # 存放预训练权重（首次训练可自动下载）
├── results/                    # 推理结果输出
├── runs/                       # 训练日志（ultralytics 自动创建）
└── scripts/
    ├── prepare_dataset.py      # COCO → YOLO 格式转换
    ├── train.py                # 模型训练
    ├── evaluate.py             # 验证集评估 / 图像预测
    ├── video_test.py           # 视频推理 + 保存结果
    └── realtime.py             # 实时摄像头/流检测
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 COCO 2017 数据集

完整数据集约 **25 GB**。只想验证流程可先只下载 val2017（约 1 GB）。

```bash
cd data/coco

# 标注文件（~241 MB，必须）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# 验证集图像（~1 GB）
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/

# 训练集图像（~18 GB，训练必须）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images/

cd ../..
```

Windows（PowerShell）：
```powershell
Invoke-WebRequest http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip
```

### 3. 数据集准备

将 COCO 多边形标注转换为 YOLO 分割格式，并过滤出目标 6 类：

```bash
python scripts/prepare_dataset.py
# 可选：--coco_dir data/coco --out_dir data/pedestrian_vehicle
```

完成后在 `data/pedestrian_vehicle/` 生成图像、标签与 `dataset.yaml`。

---

## 训练

```bash
# 默认配置（configs/train_config.yaml）
python scripts/train.py

# 快速测试（nano 模型，10 轮）
python scripts/train.py --model yolo11n-seg.pt --epochs 10 --batch 8

# 更高精度
python scripts/train.py --model yolo11m-seg.pt --epochs 150 --batch 8

# 从断点恢复
python scripts/train.py --resume runs/train/ped_vehicle_seg/weights/last.pt
```

训练结果保存在 `runs/train/ped_vehicle_seg/`：

| 文件 | 说明 |
|------|------|
| `weights/best.pt` | 验证集最优模型 |
| `weights/last.pt` | 最后一轮模型 |
| `results.csv` | 训练曲线数据 |
| `confusion_matrix.png`、`PR_curve.png` | 可视化图表 |

---

## 评估

```bash
# 在 val 集评估，输出分割与检测双路 mAP
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# 对图像目录预测并保存可视化
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --predict --source path/to/images/ --save_dir results/predictions
```

---

## 视频测试

```bash
# 基本用法（结果自动保存为 results/<视频名>_pred.mp4）
python scripts/video_test.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --source  path/to/video.mp4

# 同时显示预览窗口
python scripts/video_test.py --weights best.pt --source video.mp4 --preview

# 仅检测框（跳过掩码绘制，速度更快）
python scripts/video_test.py --weights best.pt --source video.mp4 --no_mask
```

输出视频包含：半透明分割掩码（透明度 45%）、轮廓线、检测框、置信度标签，右上角实时 FPS/帧数/各类计数统计面板。

---

## 实时检测

```bash
# 默认摄像头（索引 0）
python scripts/realtime.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# 指定摄像头或 RTSP 流
python scripts/realtime.py --weights best.pt --source 1
python scripts/realtime.py --weights best.pt --source rtsp://192.168.1.100/stream

# 同时录制视频
python scripts/realtime.py --weights best.pt --save
```

**交互快捷键：**

| 键 | 功能 |
|----|------|
| Q / ESC | 退出 |
| M | 切换分割掩码显示 |
| S | 截图（保存至 results/screenshots/） |
| + / - | 提高/降低置信度阈值 |
| R | 重置统计数据 |

---

## 配置说明

编辑 `configs/train_config.yaml` 调整训练参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 预训练权重 | yolo11n-seg.pt |
| `epochs` | 训练轮数 | 100 |
| `batch` | 批大小 | 16 |
| `imgsz` | 输入图像尺寸 | 640 |
| `device` | GPU 编号或 `cpu` | 0 |
| `patience` | 早停等待轮数 | 20 |
| `optimizer` | 优化器 | AdamW |
| `lr0` | 初始学习率 | 0.01 |
| `copy_paste` | Copy-Paste 增强（分割推荐） | 0.1 |

所有 `train.py` 的 CLI 参数会覆盖 YAML 中的对应值。

---

## 常见问题

**显存不足（OOM）**：减小 `batch`（改为 8 或 4），或使用更小模型（yolo11n-seg）、更小 `imgsz`（如 416）。

**Windows 多进程报错**：在 `train_config.yaml` 中将 `workers` 改为 0 或 2。

**已有 COCO 数据不想重复下载**：`prepare_dataset.py` 用 `--coco_dir` 指向已有路径，用 `--no_copy` 跳过图像复制（仅生成标签）。
