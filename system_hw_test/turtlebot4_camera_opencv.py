import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "../src")
from zenoh_msgs import open_zenoh_session


def listener(sample):
    bytesI = sample.payload.to_bytes()
    print(f"Received {len(sample.payload)}")
    X = np.frombuffer(bytesI, dtype=np.uint8)
    # for some reason the first 76 numbers are trash?
    # some sort of metadata header?
    Xc = X[76:187576]
    rgb = np.reshape(Xc, (250, 250, 3))
    cv2.imwrite("front_image.jpg", rgb)


if __name__ == "__main__":
    with open_zenoh_session() as session:
        camera = session.declare_subscriber("pi/oakd/rgb/preview/image_raw", listener)
        print("Zenoh is open")
        while True:
            print("Waiting for camera messages")
            time.sleep(1)
