import argparse
import datetime
import json
import time

import cv2
from ultralytics import YOLO

# Load model
# model = YOLO("yolov8n.pt")
# model = YOLO("yolo11n.pt")

model = YOLO("yolo12n.pt")

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
parser.add_argument("--cam", help="the index of the camera you want to use", type=int, default=0)
print(parser.format_help())
args = parser.parse_args()


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
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Open webcam
cap = cv2.VideoCapture(args.cam)

# Set the best available resolution
best_width, best_height = set_best_resolution(cap, RESOLUTIONS)

# Create timestamped log filename
start_time = datetime.datetime.now(datetime.UTC)
log_filename = f"detections_log_{start_time.isoformat(timespec='seconds').replace(':', '-')}Z.jsonl"
log_file = open(log_filename, "a")

frame_index = 0
print(f"Logging to: {log_filename}")
print("Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()
    datetime_str = datetime.datetime.fromtimestamp(timestamp, datetime.UTC).isoformat()

    results = model.predict(source=frame, save=False, stream=True, verbose=False)

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

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Print to terminal
    print(f"\nFrame {frame_index} @ {datetime_str} — {len(detections)} objects:")
    for det in detections:
        print(f"  {det['class']} ({det['confidence']:.2f}) -> {det['bbox']}")

    # Write to log
    json_line = json.dumps(
        {
            "frame": frame_index,
            "timestamp": timestamp,
            "datetime": datetime_str,
            "detections": detections,
        }
    )
    log_file.write(json_line + "\n")
    log_file.flush()

    frame_index += 1
    cv2.imshow("YOLOv8n Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
log_file.close()
cap.release()
cv2.destroyAllWindows()
print(f"\nDetection logging complete. Log saved to: {log_filename}")
