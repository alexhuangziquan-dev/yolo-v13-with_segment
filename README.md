# YOLOv11-seg 行人与车辆实例分割

基于 YOLOv11-seg 和 COCO 2017 数据集，识别行人与 5 类车辆的完整训练/推理项目。

## 目标类别

| 目标       | 原COCO ID | 新类别ID | 类别名     |
|-----------|-----------|---------|-----------|
| 行人       | 1         | 0       | person    |
| 单车       | 2         | 1       | bicycle   |
| 小汽车     | 3         | 2       | car       |
| 摩托/电动车 | 4        | 3       | motorcycle|
| 巴士       | 6         | 4       | bus       |
| 卡车       | 8         | 5       | truck     |

## 项目结构

```
yolo-v11-seg/
├── configs/
│   └── train_config.yaml       # 训练超参数配置
├── data/
│   ├── coco/                   # ← 下载 COCO 数据集到此处
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
├── models/                     # 存放下载的预训练权重
├── results/                    # 推理结果输出
├── runs/                       # 训练日志（ultralytics 自动创建）
├── scripts/
│   ├── prepare_dataset.py      # COCO → YOLO 格式转换
│   ├── train.py                # 模型训练
│   ├── evaluate.py             # 验证集评估 / 图像预测
│   ├── video_test.py           # 视频测试 + 保存结果
│   └── realtime.py             # 实时摄像头检测
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 COCO 2017 数据集

> 完整数据集约 **25 GB**。如果只想快速验证流程，可以只下载 val2017（约 1 GB）。

```bash
# 进入 COCO 数据目录
cd data/coco

# 下载验证集图像（~1GB）
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/

# 下载训练集图像（~18GB，可选，训练必需）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images/

# 下载标注文件（~241MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ../..
```

**Windows 用户**（无 wget）：直接在浏览器下载后解压到对应目录，或使用：
```powershell
# 在 PowerShell 中
Invoke-WebRequest http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip
```

### 3. 数据集准备（COCO → YOLO 格式）

```bash
python scripts/prepare_dataset.py
```

可选参数：
```bash
python scripts/prepare_dataset.py \
    --coco_dir data/coco \
    --out_dir  data/pedestrian_vehicle
```

完成后会在 `data/pedestrian_vehicle/` 生成 YOLO 格式的图像与标签文件。

### 4. 下载预训练权重（可选，首次训练会自动下载）

```bash
# 手动下载放到 models/ 目录
# yolo11n-seg.pt（最快，适合测试）
# yolo11s-seg.pt（小型，平衡速度与精度）
# yolo11m-seg.pt（中型，推荐）
```

ultralytics 在训练时若本地没有权重会自动从官方下载。

---

## 训练

```bash
# 使用默认配置（configs/train_config.yaml）
python scripts/train.py

# 快速测试（nano 模型，只训 10 轮）
python scripts/train.py --model yolo11n-seg.pt --epochs 10 --batch 8

# 使用更大模型获得更好精度
python scripts/train.py --model yolo11m-seg.pt --epochs 150 --batch 8

# 从断点恢复
python scripts/train.py --resume runs/train/ped_vehicle_seg/weights/last.pt
```

训练结果保存在 `runs/train/ped_vehicle_seg/`：
- `weights/best.pt` — 验证集最优模型
- `weights/last.pt` — 最后一轮模型
- `results.csv` — 训练曲线数据
- `confusion_matrix.png`、`PR_curve.png` 等图表

---

## 评估

```bash
# 在 val 集上评估并输出详细指标
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# 对图像目录进行预测并保存可视化
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --predict \
    --source path/to/images/ \
    --save_dir results/predictions
```

---

## 视频测试

```bash
# 基本用法（自动保存到 results/<视频名>_pred.mp4）
python scripts/video_test.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --source  path/to/video.mp4

# 同时显示预览窗口
python scripts/video_test.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --source  path/to/video.mp4 \
    --output  results/my_output.mp4 \
    --preview

# 仅显示检测框（不绘制分割掩码，更快）
python scripts/video_test.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --source  path/to/video.mp4 \
    --no_mask
```

---

## 实时检测

```bash
# 默认摄像头（索引 0）
python scripts/realtime.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# 指定摄像头
python scripts/realtime.py --weights best.pt --source 1

# RTSP 流
python scripts/realtime.py \
    --weights best.pt \
    --source rtsp://192.168.1.100/stream

# 同时保存录像
python scripts/realtime.py --weights best.pt --save
```

**实时检测快捷键：**

| 键        | 功能              |
|----------|-----------------|
| Q / ESC  | 退出             |
| M        | 切换分割掩码显示 |
| S        | 截图（保存到 results/screenshots/） |
| + / -    | 提高/降低置信度阈值 |
| R        | 重置统计数据     |

---

## 配置说明

编辑 `configs/train_config.yaml` 调整训练参数：

| 参数          | 说明                          | 推荐值         |
|-------------|------------------------------|---------------|
| `model`     | 预训练权重                    | yolo11n/s/m-seg.pt |
| `epochs`    | 训练轮数                      | 100~200       |
| `batch`     | 批大小                        | 视显存而定     |
| `imgsz`     | 输入尺寸                      | 640           |
| `device`    | GPU 编号或 'cpu'              | 0             |
| `patience`  | 早停等待轮数                  | 20~50         |
| `copy_paste`| Copy-Paste 增强（分割推荐）   | 0.1           |

---

## 常见问题

**Q: 显存不足（OOM）**
- 减小 `batch`（改为 8 或 4）
- 使用更小的 `imgsz`（如 416）
- 选择更小的模型（yolo11n-seg）

**Q: Windows 上多进程报错**
- 在 `train_config.yaml` 中将 `workers` 改为 0 或 2

**Q: COCO 数据下载太慢**
- 使用国内镜像或先只下载 val2017 验证流程

**Q: 已有 COCO 数据集不想重复下载**
- 修改 `prepare_dataset.py` 的 `--coco_dir` 指向已有数据集路径
