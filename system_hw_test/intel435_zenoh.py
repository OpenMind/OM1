import sys
import time
import zenoh
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue

sys.path.insert(0, "../src")
from zenoh_idl import sensor_msgs

fx, fy = 525.0, 525.0
cx, cy = 320.0, 240.0

data_queue = queue.Queue(maxsize=1)

def image_callback(sample):
    try:
        data = sensor_msgs.Image.deserialize(sample.payload.to_bytes())

        print(f"At {data.header.stamp.sec} Received image: {data.width}x{data.height}, encoding: {data.encoding}")

        depth_array = np.frombuffer(bytes(data.data), dtype=np.uint16)
        depth_image = depth_array.reshape((data.height, data.width))

        depth_m = depth_image.astype(np.float32) / 1000.0
        close_objects = (depth_m > 0) & (depth_m <= 0.30)

        data_queue.put({
            'depth_image': depth_image,
            'close_objects': close_objects,
            'depth_m': depth_m
        })

        print(f"Found {np.sum(close_objects)} points within 30cm")

    except Exception as e:
        print(f"Callback error: {e}")

def update_plot():
    """Update plot in main thread"""
    try:
        data = data_queue.get_nowait()

        depth_image = data['depth_image']
        close_objects = data['close_objects']
        depth_m = data['depth_m']

        plt.clf()

        plt.subplot(1, 2, 1)
        display_image = depth_image.copy()
        display_image[~close_objects] = 0
        plt.imshow(display_image, cmap='viridis')
        plt.title('Depth Image (Objects < 30cm)')
        plt.colorbar(label='Depth (mm)')
        plt.axis('off')

        if np.any(close_objects):
            v, u = np.where(close_objects)
            z = depth_m[close_objects]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            ax = plt.subplot(1, 2, 2, projection='3d')
            ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
            ax.set_title(f'Point Cloud ({len(x)} points)')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Depth (m)')

            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([0, 0.3])
        else:
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, 'No objects\nwithin 30cm',
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Point Cloud')

        plt.tight_layout()
        plt.draw()

    except queue.Empty:
        pass
    except Exception as e:
        print(f"Plot update error: {e}")

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure(figsize=(12, 5))

    def zenoh_thread():
        with zenoh.open(zenoh.Config()) as session:
            session.declare_subscriber(
                "camera/camera/depth/image_rect_raw",
                image_callback,
            )

            try:
                while True:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Zenoh thread error: {e}")

    zenoh_worker = threading.Thread(target=zenoh_thread, daemon=True)
    zenoh_worker.start()

    try:
        while plt.get_fignums():
            update_plot()
            plt.pause(0.001)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        plt.close('all')
