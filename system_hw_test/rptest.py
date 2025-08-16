import argparse
import math
import sys
import threading
import time
from queue import Queue, Empty

import numpy as np
import zenoh
from matplotlib import pyplot as plot
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from rpdriver import RPDriver

sys.path.insert(0, "../src")
try:
    from zenoh_idl import sensor_msgs
except ImportError:
    print("Please run this script from inside /system_hw_test")

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--serial", type=str, help="serial port to use, when using the low level driver, e.g. /dev/cu.usbserial-0001")
parser.add_argument("--zenoh", action="store_true", help="use zenoh to connect to the robot")
parser.add_argument("--multicast", help="multicast address for zenoh", type=str, default=None)
parser.add_argument("--URID", help="your robot's URID, when using Zenoh", type=str, default="")
parser.add_argument("--type", type=str, default="go2", help="go2 or tb4")
# IMPORTANT: only support normal for reliability
parser.add_argument("--scan", choices=["normal"], default="normal",
                    help="lidar scan mode (normal only)")
parser.add_argument("--near-min", type=float, default=0.16, help="min blocking distance (m)")
parser.add_argument("--near-max", type=float, default=1.1, help="max blocking distance (m)")
args = parser.parse_args()

# ---------- Candidate paths ----------
def create_straight_line_path_from_angle(angle_degrees, length=1.0, num_points=10):
    angle_rad = math.radians(angle_degrees)
    end_x = length * math.sin(angle_rad)  # 0° forward (+y)
    end_y = length * math.cos(angle_rad)
    x_vals = np.linspace(0.0, end_x, num_points)
    y_vals = np.linspace(0.0, end_y, num_points)
    return np.array([x_vals, y_vals])

# Define 9 straight line paths separated by 15 degrees
# Center path is 0° (straight forward), then ±15°, ±30°, ±45°, ±60°
path_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]
path_length = 1.05
paths = [create_straight_line_path_from_angle(a, path_length) for a in path_angles]
print(f"Created {len(paths)} paths with angles: {path_angles}")
print(f"Each path extends {path_length}m from robot center")

# ---------- Figure ----------
fig = plot.figure()
ax1 = plot.subplot(131)
ax2 = plot.subplot(132)
ax3 = plot.subplot(133)

# Panel 1: overview
ax1.plot([0],[0],"o",color="blue")
ax1.add_patch(Circle((0,0), 0.20, color="red"))
points = ax1.plot([], [], "-", color="black")[0]
ax1.annotate("Front", xytext=(0.1, 0.3), xy=(0, 0.5))
ax1.annotate("", xytext=(0, 0), xy=(0, 1.5), arrowprops=dict(arrowstyle="->"))
ax1.set_xlim(-5,5); ax1.set_ylim(-5,5); ax1.set_aspect("equal")

# Robot config
half_width_robot = 0.20
relevant_distance_min = args.near_min
relevant_distance_max = args.near_max
sensor_mounting_angle = 180.0 if args.type != "tb4" else 270.0
angles_blanked = [] if args.type != "tb4" else [[-180.0, -160.0], [110.0, 180.0]]

# Panel 2: zoom
ax2.plot([0],[0],"o", color="blue", markersize=8, zorder=10)
ax2.add_patch(Circle((0,0), 0.20, ls="--", lw=1, ec="red", fc="none"))
if args.type == "tb4":
    ax2.add_patch(Rectangle((-0.05, -0.15), 0.20, 0.06, ls="--", fc="black"))
else:
    ax2.add_patch(Rectangle((-0.2, -0.7), 0.40, 0.70, ls="--", lw=1, ec="red", fc="none"))
pointsZoom = ax2.plot([], [], ".", color="black")[0]
m = args.near_max + 0.2
ax2.set_xlim(-m, m); ax2.set_ylim(-m, m); ax2.set_aspect("equal")

# Precreate path artists
lines = [ax2.plot([0],[0], "-", color="black")[0] for _ in paths]

# Panel 3: angle-distance strip
line = ax3.plot([0], [0], "-", color="red")[0]
ax3.set_xlim(-180, 180); ax3.set_ylim(0, 1.2); ax3.set_aspect(300)
ax3.plot([-180.0, -48.0], [1.18, 1.18], "-", color="red",   linewidth=3.0)
ax3.plot([ -42.0,  42.0], [1.18, 1.18], "-", color="black", linewidth=3.0)
ax3.plot([  48.0, 180.0], [1.18, 1.18], "-", color="green", linewidth=3.0)
ax3.annotate("Left",  xytext=(-125, 1.1), xy=(0, 0.5))
ax3.annotate("Front", xytext=( -20, 1.1), xy=(0, 0.5))
ax3.annotate("Right", xytext=(  85, 1.1), xy=(0, 0.5))

# --- Visualize blanked sectors (if any) on ax2 & ax3 ---
for b in angles_blanked:
    deg_to_rad = np.pi / 180.0
    start_angle = b[0] * deg_to_rad
    end_angle   = b[1] * deg_to_rad
    theta = np.linspace(start_angle, end_angle, 50)
    r = relevant_distance_max

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # arc on the zoom panel
    ax2.plot(x, y, "--", color="grey", linewidth=1.5)
    # two radial edges
    ax2.plot([0, x[0]],  [0, y[0]],  "--", color="grey", linewidth=1)
    ax2.plot([0, x[-1]], [0, y[-1]], "--", color="grey", linewidth=1)

    # strip on angle-distance panel
    width = abs(b[1] - b[0])
    ax3.add_patch(Rectangle((b[0], 0.2), width, 1.0, fc="grey"))

# ---------- Queue pipeline ----------
scan_queue: Queue[np.ndarray] = Queue(maxsize=1)
def enqueue_latest(data: np.ndarray):
    try:
        while True:
            scan_queue.get_nowait()
    except Empty:
        pass
    try:
        scan_queue.put_nowait(data)
    except Exception:
        pass
def ensure_streaming(lidar, wait_s=2.0):
    """Wait for data; if none, flip DTR once (handles inverted boards)."""
    ser = lidar._serial
    def have_bytes():
        try: return ser.in_waiting
        except AttributeError: return ser.inWaiting()

    # wait a moment for first packets
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if have_bytes() >= 5:
            return True
        time.sleep(0.01)

    # try flipping DTR polarity once
    try:
        current = ser.dtr
        ser.setDTR(not current)
        print(f"[serial] flipped DTR -> {ser.dtr}")
        time.sleep(1.0)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if have_bytes() >= 5:
                return True
            time.sleep(0.01)
    except Exception:
        pass
    return False


def log_lidar_banner(lidar: RPDriver):
    """Best-effort: print model/firmware/health once for diagnostics."""
    try:
        print("Info:", lidar.get_info())
    except Exception as e:
        print("Info read failed:", e)
    try:
        print("Health:", lidar.get_health())
    except Exception as e:
        print("Health read failed:", e)

# ---------- Producers ----------
def continuous_serial_robust(port: str):
    """Robust reader: always use NORMAL mode; restart on error."""
    lidar = None
    while True:
        try:
            if lidar is None:
                print("[serial] opening", port)
                lidar = RPDriver(port)
                # log_lidar_banner(lidar) ###### For DEBUG 

            # Hard reset to known-good state
            try: lidar.stop()
            except Exception: pass
            try: lidar.stop_motor()
            except Exception: pass
            try: lidar.clean_input()
            except Exception: pass

            # Spin up & start NORMAL stream
            lidar.start("normal")
            time.sleep(0.3)  # settle

            # Verify bytes are arriving; auto-flip DTR once if needed
            if not ensure_streaming(lidar, wait_s=2.0):
                raise RuntimeError("No scan bytes after start (even after DTR flip)")

            print("[serial] streaming (normal)...")
            fps_counter, t0 = 0, time.monotonic()

            # NOTE: iter_scans yields "scan" lists directly in this driver
            for scan in lidar.iter_scans(scan_type="normal", max_buf_meas=3000, min_len=5):
                arr = np.array(scan)  # (quality, angle_deg, distance_mm)
                if arr.size == 0:
                    continue
                angles_deg = arr[:, 1]
                distances_m = arr[:, 2] / 1000.0
                enqueue_latest(np.column_stack((angles_deg, distances_m)))

                fps_counter += 1
                now = time.monotonic()
                if now - t0 >= 1.0:
                    print(f"[serial] fps={fps_counter}")
                    fps_counter = 0
                    t0 = now

        except Exception as e:
            print(f"[serial] error: {e}  (restarting in 1s)")
            try:
                if lidar: lidar.stop()
            except Exception: pass
            try:
                if lidar: lidar.reset()
            except Exception: pass
            try:
                if lidar: lidar.stop_motor()
            except Exception: pass

            # hard reopen after descriptor-like errors
            if "Bad scan start response" in str(e) or "Descriptor" in str(e):
                try:
                    if lidar: lidar.disconnect()
                except Exception: pass
                try:
                    if lidar: lidar.connect()
                except Exception: pass

            time.sleep(1.0)  # keep this



def zenoh_scan(sample):
    scan = sensor_msgs.LaserScan.deserialize(sample.payload.to_bytes())
    angles = 360.0 * (np.arange(scan.angle_min, scan.angle_max, scan.angle_increment) + math.pi) / (2*math.pi)
    angles = np.flip(angles)
    enqueue_latest(np.column_stack((angles, np.array(scan.ranges))))

# ---------- Geometry ----------
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t*dx, y1 + t*dy
    return math.hypot(px - cx, py - cy)

# ---------- Processing (GUI thread) ----------
def process(data: np.ndarray):

    complexes = []
    for angle, distance in data:
        d_m = float(distance)
        # don't worry about distant objects
        if d_m > 5.0:
            continue

        # first, correctly orient the sensor zero to the robot zero
        angle = angle + sensor_mounting_angle
        if angle >= 360.0: angle -= 360.0
        elif angle < 0.0:  angle += 360.0

        # then, convert to radians
        a_rad = math.radians(angle)
        v1 = d_m * math.cos(a_rad); v2 = d_m * math.sin(a_rad)
        # convert to x and y
        # x runs backwards to forwards, y runs left to right
        x = -v2; y = -v1

        # convert the angle to -180 to + 180 range
        angle = angle - 180.0  # convert to [-180, 180]

        # this is too close, disregard
        keep = d_m >= relevant_distance_min
        for b in angles_blanked:
            if b[0] <= angle <= b[1]:
                # this is a permanent reflection based on the robot
                # disregard
                keep = False
                break
        
        # the final data ready to use for path planning
        if keep:
            complexes.append([x, y, angle, d_m])

    if not complexes:
        points.set_data([], []); pointsZoom.set_data([], []); line.set_data([], [])
        for idx, p in enumerate(paths):
            lines[idx].set_data(p[0], p[1]); lines[idx].set_color("black")
        return


    # sort data into strictly increasing angles to deal with sensor issues
    # the sensor sometimes reports part of the previous scan and part of the next scan
    # so you end up with multiple slightly different values for some angles at the
    # junction
    array = np.array(complexes)
    array = array[array[:,2].argsort()]  # sort by angle
    X, Y, A, D = array[:,0], array[:,1], array[:,2], array[:,3]

    points.set_data(X, Y)
    pointsZoom.set_data(X, Y)
    line.set_data(A, D)

    """
    Determine set of possible paths
    """
    possible_paths = np.array(range(len(paths)))
    for x, y, d in zip(X, Y, D):
        if d > relevant_distance_max or d < relevant_distance_min:
            continue
        for apath in possible_paths.copy():
            ps = paths[apath]
            if distance_point_to_line_segment(
                x, y,
                ps[0][0], ps[1][0],
                ps[0][-1], ps[1][-1]
            ) < half_width_robot:
                possible_paths = np.setdiff1d(possible_paths, np.array([apath]))
                break

    for idx, p in enumerate(paths):
        lines[idx].set_data(p[0], p[1])
        lines[idx].set_color("green" if idx in possible_paths else "red")

    ppl = possible_paths.tolist()
    left    = [p for p in ppl if path_angles[p] in (-60,-45,-30)]
    forward = [p for p in ppl if path_angles[p] in (-15,0,15)]
    right   = [p for p in ppl if path_angles[p] in (30,45,60)]
    back    = [p for p in ppl if path_angles[p] == 180]

    print(f"possible_paths: {possible_paths}")
    if ppl:
        if left:    print(f"You can turn left using paths: {left} ({[path_angles[p] for p in left]}°).")
        if forward: print(f"You can go forward using paths: {forward} ({[path_angles[p] for p in forward]}°).")
        if right:   print(f"You can turn right using paths: {right} ({[path_angles[p] for p in right]}°).")
        if back:    print(f"You can go backward using paths: {back} ({[path_angles[p] for p in back]}°).")
    else:
        print("You are surrounded by objects and cannot safely move.")


def animate(_):
    try:
        data = scan_queue.get_nowait()
    except Empty:
        return []
    process(data)
    return [points, pointsZoom, line, *lines]

# ---------- Main ----------
if __name__ == "__main__":
    if args.serial:
        print(f"Using {args.serial} as the serial port")
        t = threading.Thread(target=continuous_serial_robust, args=(args.serial,), daemon=True)
        t.start()
        ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
        plot.show()
        sys.exit(0)

    if args.zenoh:
        print("Using Zenoh to connect to robot")
        print("[INFO] Opening zenoh session...")
        conf = zenoh.Config()
        if args.multicast:
            conf.insert_json5("scouting", f'{{"multicast": {{"address": "{args.multicast}"}}}}')
        z = zenoh.open(conf)
        if args.type == "go2":
            print("[INFO] Creating Subscribers for Go2")
            z.declare_subscriber("scan", zenoh_scan)
        elif args.type == "tb4":
            print("[INFO] Creating Subscribers for TB4")
            z.declare_subscriber(f"{args.URID}/pi/scan", zenoh_scan)
        else:
            print(f"[ERROR] Unsupported robot type: {args.type}")
            sys.exit(1)
        ani = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
        plot.show()
        sys.exit(0)

    raise ValueError("You must specify either --serial or --zenoh to run this script.")
