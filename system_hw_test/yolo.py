import argparse
import json
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Load model
# model = YOLO("yolov8n.pt")
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo12n.pt")
model = YOLO("yolo11n-seg.pt")

# Common resolutions to test (width, height), ordered high to low
RESOLUTIONS = [
    (3840, 2160),  # 4K
    (2560, 1440),  # QHD
    (1920, 1080),  # Full HD
    (1280, 720),  # HD
    (1024, 576),
    (800, 600),
    (640, 480),  # VGA fallback
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cam", help="the index of the camera you want to use", type=int, default=0
)
print(parser.format_help())
args = parser.parse_args()


def mask_to_polygons(mask):
    """
    Convert a binary mask to polygons using OpenCV.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons


def save_yolo_results_to_json(results, index, output_file):
    data = []

    for result in results:
        frame_entry = {}

        # Use current time or provide your own timestamp
        frame_entry["frame"] = index
        frame_entry["timestamp"] = time.time()

        if result.masks is None or result.boxes is None:
            continue

        masks = result.masks.data.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        labels = (
            [result.names[i] for i in class_ids]
            if hasattr(result, "names")
            else [str(i) for i in class_ids]
        )
        bboxes = result.boxes.xyxy.cpu().numpy().tolist()

        frame_entry["bboxes"] = bboxes
        frame_entry["labels"] = labels
        frame_entry["polygons"] = []

        for mask in masks:
            binary_mask = (mask > 0.5).astype(np.uint8)
            polygons = mask_to_polygons(binary_mask)
            # Convert each polygon to a single-line string
            flat_polygons = ["[" + ",".join(map(str, poly)) + "]" for poly in polygons]
            frame_entry["polygons"].append(flat_polygons)

        data.append(frame_entry)

    with open(output_file, "a") as f:
        json.dump(data, f, indent=2)


def set_best_resolution(cap, resolutions):
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Give it a moment to settle
        time.sleep(0.1)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            print(f"✅ Resolution set to: {width}x{height}")
            return width, height

    print("⚠️ Could not set preferred resolution. Using default.")
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


# Open webcam
cap = cv2.VideoCapture(args.cam)

# Set the best available resolution
best_width, best_height = set_best_resolution(cap, RESOLUTIONS)

frame_index = 0
print("Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame)

    segmented_image = None

    if results:
        segmented_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    save_yolo_results_to_json(results, frame_index, "yolo_log.json")

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            detections.append(
                {
                    "class": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                }
            )

    # Print to terminal
    print(f"\nFrame {frame_index} @ {len(detections)} objects:")
    for det in detections:
        print(f"  {det['class']} ({det['confidence']:.2f}) -> {det['bbox']}")

    frame_index += 1
    if len(segmented_image) > 0:
        cv2.imshow("YOLO Object Detection", segmented_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\nDetection logging complete")
