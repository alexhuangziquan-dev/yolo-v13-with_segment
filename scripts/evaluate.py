"""
evaluate.py
===========
在验证集/测试集上评估 YOLOv11-seg 模型，输出详细指标。

用法：
  # 使用最优权重评估
  python scripts/evaluate.py --weights runs/train/ped_vehicle_seg/weights/best.pt

  # 自定义数据集
  python scripts/evaluate.py --weights best.pt --data data/pedestrian_vehicle/dataset.yaml

  # 对图像目录进行预测并保存结果
  python scripts/evaluate.py --weights best.pt --predict --source path/to/images/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
# 每个类别对应的颜色（BGR）
CLASS_COLORS = [
    (0, 255, 128),    # person     - 绿
    (255, 165, 0),    # bicycle    - 橙
    (0, 128, 255),    # car        - 蓝
    (255, 0, 200),    # motorcycle - 紫红
    (0, 200, 255),    # bus        - 青
    (0, 0, 255),      # truck      - 红
]


def run_validation(model, data_yaml: str, device: str, imgsz: int, conf: float, iou: float):
    """运行验证集评估并打印详细指标。"""
    print("\n" + "=" * 60)
    print("模型验证")
    print("=" * 60)

    metrics = model.val(
        data=data_yaml,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
    )

    # ── 打印分割指标 ──
    print("\n分割指标 (Segmentation):")
    print(f"  mAP@50      : {metrics.seg.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.seg.map:.4f}")
    print(f"  Precision   : {metrics.seg.mp:.4f}")
    print(f"  Recall      : {metrics.seg.mr:.4f}")

    print("\n检测指标 (Detection):")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")

    # 每类别详情
    if hasattr(metrics.seg, "ap_class_index") and metrics.seg.ap_class_index is not None:
        print("\n各类别 mAP@50-95 (分割):")
        for idx, cls_idx in enumerate(metrics.seg.ap_class_index):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            ap = metrics.seg.ap[idx] if idx < len(metrics.seg.ap) else float("nan")
            print(f"  {cls_name:12s}: {ap:.4f}")

    save_dir = Path(metrics.save_dir)
    print(f"\n结果已保存到: {save_dir}")
    return metrics


def predict_images(model, source: str, device: str, imgsz: int, conf: float, iou: float, save_dir: Path):
    """对图像目录或单张图像进行预测并保存可视化结果。"""
    print(f"\n预测目标: {source}")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=source,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=False,          # 手动绘制以完全控制样式
        stream=True,
    )

    n_saved = 0
    for r in results:
        img = r.orig_img.copy()
        img_name = Path(r.path).name

        if r.masks is not None:
            # 绘制分割掩码
            masks = r.masks.data.cpu().numpy()  # (N, H, W)
            boxes = r.boxes
            for i, mask in enumerate(masks):
                cls_id = int(boxes.cls[i])
                color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (128, 128, 128)
                # 将掩码 resize 到原图尺寸
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_bool = mask_resized > 0.5
                overlay = img.copy()
                overlay[mask_bool] = (
                    img[mask_bool] * 0.4 + np.array(color) * 0.6
                ).astype(np.uint8)
                img = overlay

                # 绘制轮廓
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, contours, -1, color, 2)

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (128, 128, 128)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = save_dir / img_name
        cv2.imwrite(str(out_path), img)
        n_saved += 1

    print(f"已保存 {n_saved} 张预测结果到: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg 模型评估")
    parser.add_argument("--weights", required=True, help="模型权重路径（best.pt）")
    parser.add_argument("--data", default="data/pedestrian_vehicle/dataset.yaml", help="数据集 YAML 路径")
    parser.add_argument("--device", default="0", help="设备：0 / cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--predict", action="store_true", help="对 --source 进行预测")
    parser.add_argument("--source", default=None, help="预测时的图像路径/目录")
    parser.add_argument("--save_dir", default="results/predictions", help="预测结果保存目录")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # 尝试在项目根目录下查找
        weights_path = ROOT / args.weights
    if not weights_path.exists():
        print(f"[错误] 权重文件不存在: {args.weights}")
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not data_yaml.exists():
        print(f"[错误] 数据集配置不存在: {data_yaml}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    print(f"加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    if args.predict:
        source = args.source
        if source is None:
            print("[错误] 预测模式需要 --source 参数")
            sys.exit(1)
        predict_images(
            model, source, args.device, args.imgsz, args.conf, args.iou,
            save_dir=ROOT / args.save_dir,
        )
    else:
        run_validation(model, str(data_yaml), args.device, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()
