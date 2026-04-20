"""
train.py
========
YOLOv11-seg 行人与车辆检测模型训练脚本。

用法：
  # 使用默认配置（configs/train_config.yaml）
  python scripts/train.py

  # 覆盖单个参数
  python scripts/train.py --model yolo11s-seg.pt --epochs 150 --batch 8

  # 从断点恢复训练
  python scripts/train.py --resume runs/train/ped_vehicle_seg/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

# 确保从项目根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            # 去掉行尾注释
            if " #" in line:
                line = line[: line.index(" #")] + "\n"
            lines.append(line)
    return yaml.safe_load("".join(lines)) or {}


def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg 训练")
    parser.add_argument("--config", default="configs/train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--model", default=None, help="模型权重，覆盖配置中的 model 参数")
    parser.add_argument("--data", default=None, help="数据集 YAML 路径，覆盖配置中的 data 参数")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="批大小")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图像尺寸")
    parser.add_argument("--device", default=None, help="设备：0 / 0,1 / cpu")
    parser.add_argument("--resume", default=None, help="从检查点恢复训练，指定 last.pt 路径")
    parser.add_argument("--name", default=None, help="实验名称")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    print(f"已加载配置: {config_path}")

    # 命令行参数覆盖配置文件
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "name": args.name,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
            print(f"  覆盖参数: {k} = {v}")

    # 恢复训练
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[错误] 检查点文件不存在: {resume_path}")
            sys.exit(1)
        cfg["model"] = str(resume_path)
        cfg["resume"] = True
        print(f"  从检查点恢复: {resume_path}")

    # 检查数据集是否存在
    data_path = Path(cfg.get("data", ""))
    if not data_path.exists():
        print(f"[错误] 数据集配置不存在: {data_path}")
        print("  请先运行: python scripts/prepare_dataset.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("训练参数汇总:")
    for k, v in cfg.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60 + "\n")

    # 导入 ultralytics 并开始训练
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    model_path = cfg.pop("model")
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print("开始训练...\n")
    results = model.train(**cfg)

    # 打印最优结果
    print("\n" + "=" * 60)
    print("训练完成！")
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    print(f"  保存目录  : {save_dir}")
    print(f"  最优权重  : {best_weights}")
    print(f"  最后权重  : {last_weights}")
    print("=" * 60)

    # 训练后自动验证
    print("\n开始最终验证...")
    metrics = model.val()
    print(f"\n验证结果:")
    print(f"  mAP50    : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 : {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
