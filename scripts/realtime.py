"""
realtime.py
===========
实时摄像头/视频流检测，支持多摄像头与 RTSP 流。

用法：
  # 默认摄像头（索引 0）
  python scripts/realtime.py --weights best.pt

  # 指定摄像头
  python scripts/realtime.py --weights best.pt --source 1

  # RTSP 流
  python scripts/realtime.py --weights best.pt --source rtsp://192.168.1.100/stream

  # 同时保存录像
  python scripts/realtime.py --weights best.pt --save

快捷键（运行时）：
  Q / ESC   - 退出
  M         - 切换分割掩码显示
  S         - 截图保存到 results/screenshots/
  +/-       - 调整置信度阈值
  R         - 重置统计
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CLASS_COLORS_BGR = [
    (0, 255, 128),
    (0, 165, 255),
    (255, 128, 0),
    (200, 0, 255),
    (255, 200, 0),
    (0, 0, 255),
]
ALPHA = 0.45


def draw_detections(frame: np.ndarray, result, show_mask: bool) -> np.ndarray:
    h, w = frame.shape[:2]

    if show_mask and result.masks is not None:
        for i, mask in enumerate(result.masks.data.cpu().numpy()):
            cls_id = int(result.boxes.cls[i])
            color = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
            mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            region = mask_r > 0.5
            overlay = frame.copy()
            overlay[region] = (
                frame[region] * (1 - ALPHA) + np.array(color, np.float32) * ALPHA
            ).astype(np.uint8)
            frame = overlay
            cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, color, 2)

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    conf_thresh: float,
    show_mask: bool,
    counts: dict[str, int],
    total_detections: int,
    session_time: float,
) -> np.ndarray:
    """绘制 HUD 信息面板。"""
    h, w = frame.shape[:2]

    # 顶部状态条
    cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
    mask_status = "掩码:开" if show_mask else "掩码:关"
    status = (f"FPS: {fps:5.1f}  |  置信度: {conf_thresh:.2f}  |  {mask_status}  |  "
               f"累计检测: {total_detections}  |  运行: {session_time:.0f}s")
    cv2.putText(frame, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

    # 右侧类别计数面板
    panel_w = 180
    panel_h = 20 + len(counts) * 24 + 10
    px = w - panel_w - 8
    py = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 4, py - 4), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "当前帧检测", (px, py + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (name, cnt) in enumerate(counts.items()):
        color = CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)]
        bar_len = min(cnt * 12, panel_w - 80)
        y = py + 20 + (i + 1) * 24
        cv2.rectangle(frame, (px, y - 14), (px + bar_len, y - 2), color, -1)
        cv2.putText(frame, f"{name}: {cnt}", (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # 底部快捷键提示
    hint = "Q/ESC: 退出  M: 切换掩码  S: 截图  +/-: 调整置信度"
    cv2.rectangle(frame, (0, h - 24), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, hint, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLOv11-seg 实时检测")
    parser.add_argument("--weights", required=True, help="模型权重路径")
    parser.add_argument("--source", default="0", help="摄像头索引或视频流地址（默认 0）")
    parser.add_argument("--device", default="0", help="设备：0 / cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.30, help="初始置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--save", action="store_true", help="同时保存录像到 results/")
    parser.add_argument("--window_w", type=int, default=1280, help="显示窗口宽度")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        weights_path = ROOT / args.weights
    if not weights_path.exists():
        print(f"[错误] 权重文件不存在: {args.weights}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 请安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    print(f"加载模型: {weights_path}")
    model = YOLO(str(weights_path))

    # 解析 source
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {source}")
        sys.exit(1)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"视频源: {src_w}x{src_h} @ {src_fps:.1f}fps")

    # 可选录像
    writer = None
    if args.save:
        rec_dir = ROOT / "results" / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = rec_dir / f"realtime_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rec_path), fourcc, src_fps, (src_w, src_h))
        print(f"录像保存到: {rec_path}")

    screenshot_dir = ROOT / "results" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # 状态变量
    conf_thresh = args.conf
    show_mask = True
    fps_buf = []
    total_detections = 0
    session_start = time.perf_counter()

    cv2.namedWindow("YOLOv11-seg 实时检测", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv11-seg 实时检测", args.window_w, int(args.window_w * src_h / src_w))

    print("\n实时检测已启动！按 Q 或 ESC 退出\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("视频流结束")
                break

            t0 = time.perf_counter()
            results = model.predict(
                frame,
                device=args.device,
                imgsz=args.imgsz,
                conf=conf_thresh,
                iou=args.iou,
                verbose=False,
            )
            t1 = time.perf_counter()

            fps_buf.append(1.0 / max(t1 - t0, 1e-6))
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps = sum(fps_buf) / len(fps_buf)

            result = results[0]
            n_det = len(result.boxes) if result.boxes is not None else 0
            total_detections += n_det

            frame = draw_detections(frame, result, show_mask)

            counts = {name: 0 for name in CLASS_NAMES}
            if result.boxes is not None:
                for cls_id in result.boxes.cls.tolist():
                    name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
                    counts[name] = counts.get(name, 0) + 1

            frame = draw_hud(
                frame, fps, conf_thresh, show_mask, counts,
                total_detections, time.perf_counter() - session_start,
            )

            if writer:
                writer.write(frame)

            cv2.imshow("YOLOv11-seg 实时检测", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q 或 ESC
                print("退出")
                break
            elif key == ord("m"):
                show_mask = not show_mask
                print(f"分割掩码: {'开启' if show_mask else '关闭'}")
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                ss_path = screenshot_dir / f"screenshot_{ts}.jpg"
                cv2.imwrite(str(ss_path), frame)
                print(f"截图已保存: {ss_path}")
            elif key == ord("+") or key == ord("="):
                conf_thresh = min(0.95, conf_thresh + 0.05)
                print(f"置信度阈值: {conf_thresh:.2f}")
            elif key == ord("-"):
                conf_thresh = max(0.05, conf_thresh - 0.05)
                print(f"置信度阈值: {conf_thresh:.2f}")
            elif key == ord("r"):
                total_detections = 0
                session_start = time.perf_counter()
                print("统计已重置")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"录像已保存")
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - session_start
    print(f"\n会话统计:")
    print(f"  运行时长: {elapsed:.1f}s")
    print(f"  累计检测: {total_detections} 个目标")
    print(f"  平均 FPS: {sum(fps_buf)/len(fps_buf):.1f}" if fps_buf else "")


if __name__ == "__main__":
    main()
