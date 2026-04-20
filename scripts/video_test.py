
import os
from ultralytics import YOLO
import cv2
import numpy as np  

model = YOLO("../models/20260420132042/yolo11n-seg.pt")

video_source = "../videos/test-video.mp4" 

if not os.path.exists(video_source):
    print(f"Error: Video file '{video_source}' not found.")
    exit()

cap = cv2.VideoCapture(video_source)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "../results/segmented_output.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

GREEN = (0, 255, 0)   
BLACK = (0, 0, 0)    

mask_palette = [
    (255, 255, 0),   
    (255, 0, 255),   
    (0, 165, 255),   
    (0, 0, 255),     
    (255, 0, 0)     
]
MASK_ALPHA = 0.5 

frame_count = 0  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
    result = results[0]
    
    annotated_frame = frame.copy()

    if result.masks is not None:
        raw_masks = result.masks.data.cpu().numpy() 
        
        combined_masks_overlay = np.zeros_like(frame)

        for i, mask_data in enumerate(raw_masks):
            mask_color = mask_palette[i % len(mask_palette)]

            bin_mask = (mask_data * 255).astype("uint8")
            bin_mask_resized = cv2.resize(bin_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            mask_colored = np.full_like(frame, mask_color)
            object_colored_mask = cv2.bitwise_and(mask_colored, mask_colored, mask=bin_mask_resized)
            
            combined_masks_overlay = cv2.bitwise_or(combined_masks_overlay, object_colored_mask)

        cv2.addWeighted(combined_masks_overlay, MASK_ALPHA, annotated_frame, 1.0, 0, annotated_frame)

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_text = f"{model.names[cls_id]} {conf:.2f}"
            
            thickness = 3

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN, thickness)
            
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 35), (x1 + text_width, y1), GREEN, -1) # 填充背景
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2) # 黑色字体

    if annotated_frame.shape[1] == w and annotated_frame.shape[0] == h:
        out.write(annotated_frame)
        frame_count += 1

    cv2.imshow("Video Stream Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

if frame_count > 0:
    print(f"over, {frame_count} frames, saved to {output_path}")
else:
    print("waring: no frames processed, output video not saved.")