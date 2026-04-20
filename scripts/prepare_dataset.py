"""
prepare_dataset.py
==================
将 COCO 2017 数据集转换为 YOLOv11-seg 格式，只保留目标类别。

目标类别映射：
  COCO category_id -> 新 class_id
  1  (person)      -> 0
  2  (bicycle)     -> 1
  3  (car)         -> 2
  4  (motorcycle)  -> 3
  6  (bus)         -> 4
  8  (truck)       -> 5

用法：
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --coco_dir data/coco --out_dir data/pedestrian_vehicle
"""

import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── 类别映射 ──────────────────────────────────────────────────────────────────
COCO_TO_NEW = {
    1: 0,  # person
    2: 1,  # bicycle
    3: 2,  # car
    4: 3,  # motorcycle
    6: 4,  # bus
    8: 5,  # truck
}
TARGET_COCO_IDS = set(COCO_TO_NEW.keys())
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}


def polygon_to_yolo(segmentation: list, img_w: int, img_h: int) -> list[str]:
    """将 COCO polygon 转换为归一化 YOLO 分割格式字符串列表（每个多边形一条）。"""
    results = []
    for polygon in segmentation:
        if len(polygon) < 6:  # 至少需要 3 个点
            continue
        coords = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        coords[:, 0] /= img_w
        coords[:, 1] /= img_h
        coords = np.clip(coords, 0.0, 1.0)
        flat = " ".join(f"{v:.6f}" for v in coords.flatten())
        results.append(flat)
    return results


def process_split(
    ann_file: Path,
    img_src_dir: Path,
    img_dst_dir: Path,
    lbl_dst_dir: Path,
    split_name: str,
    copy_images: bool = True,
) -> dict:
    """处理单个数据集划分（train / val）。"""
    print(f"\n{'='*60}")
    print(f"处理 {split_name} 集: {ann_file.name}")
    print(f"{'='*60}")

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建立 image_id -> image_info 映射
    id_to_img = {img["id"]: img for img in coco["images"]}

    # 过滤目标类别的标注，并按 image_id 分组
    ann_by_img: dict[int, list] = {}
    skipped_crowd = 0
    for ann in coco["annotations"]:
        if ann["category_id"] not in TARGET_COCO_IDS:
            continue
        if ann.get("iscrowd", 0):          # 跳过群体标注（无多边形）
            skipped_crowd += 1
            continue
        seg = ann.get("segmentation", [])
        if not seg or not isinstance(seg, list):
            continue
        img_id = ann["image_id"]
        ann_by_img.setdefault(img_id, []).append(ann)

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    stats = {name: 0 for name in CLASS_NAMES.values()}
    n_images = 0
    n_skipped_no_ann = 0

    for img_id, anns in tqdm(ann_by_img.items(), desc=f"  转换 {split_name}"):
        img_info = id_to_img.get(img_id)
        if img_info is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]  # e.g. "000000001234.jpg"
        img_path = img_src_dir / file_name

        label_lines = []
        for ann in anns:
            coco_cat = ann["category_id"]
            new_cls = COCO_TO_NEW[coco_cat]
            seg_strings = polygon_to_yolo(ann["segmentation"], img_w, img_h)
            for seg_str in seg_strings:
                label_lines.append(f"{new_cls} {seg_str}")
            stats[CLASS_NAMES[new_cls]] += 1

        if not label_lines:
            n_skipped_no_ann += 1
            continue

        # 保存标签
        stem = Path(file_name).stem
        lbl_file = lbl_dst_dir / f"{stem}.txt"
        lbl_file.write_text("\n".join(label_lines), encoding="utf-8")

        # 复制图像
        if copy_images:
            dst_img = img_dst_dir / file_name
            if img_path.exists():
                shutil.copy2(img_path, dst_img)
            else:
                print(f"  [警告] 图像不存在: {img_path}")

        n_images += 1

    print(f"\n  完成 {split_name} 集:")
    print(f"    有效图像: {n_images}")
    print(f"    跳过（无有效标注）: {n_skipped_no_ann}")
    print(f"    跳过群体标注: {skipped_crowd}")
    print(f"    各类别实例数:")
    for name, cnt in stats.items():
        print(f"      {name:12s}: {cnt:,}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="COCO 2017 -> YOLOv11-seg 数据集转换")
    parser.add_argument(
        "--coco_dir",
        default="data/coco",
        help="COCO 数据集根目录（包含 images/ 和 annotations/）",
    )
    parser.add_argument(
        "--out_dir",
        default="data/pedestrian_vehicle",
        help="输出目录",
    )
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="不复制图像（只生成标签文件，图像已在目标位置时使用）",
    )
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)

    # 检查输入目录
    splits = {
        "train": {
            "ann": coco_dir / "annotations" / "instances_train2017.json",
            "img_src": coco_dir / "images" / "train2017",
            "img_dst": out_dir / "images" / "train",
            "lbl_dst": out_dir / "labels" / "train",
        },
        "val": {
            "ann": coco_dir / "annotations" / "instances_val2017.json",
            "img_src": coco_dir / "images" / "val2017",
            "img_dst": out_dir / "images" / "val",
            "lbl_dst": out_dir / "labels" / "val",
        },
    }

    for split_name, cfg in splits.items():
        if not cfg["ann"].exists():
            print(f"[错误] 标注文件不存在: {cfg['ann']}")
            print(f"  请先下载 COCO 2017 数据集到 {coco_dir}/")
            print(f"  参考 README.md 中的下载说明")
            return

    all_stats: dict[str, dict] = {}
    for split_name, cfg in splits.items():
        all_stats[split_name] = process_split(
            ann_file=cfg["ann"],
            img_src_dir=cfg["img_src"],
            img_dst_dir=cfg["img_dst"],
            lbl_dst_dir=cfg["lbl_dst"],
            split_name=split_name,
            copy_images=not args.no_copy,
        )

    # 生成 dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for idx, name in CLASS_NAMES.items():
        yaml_content += f"  {idx}: {name}\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n已生成 dataset.yaml: {yaml_path}")

    print("\n" + "=" * 60)
    print("数据集准备完成！")
    print(f"输出目录: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
