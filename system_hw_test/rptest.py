import argparse
import math
import sys
import threading
import time

import numpy as np
import zenoh
from matplotlib import pyplot as plot
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from rpdriver import RPDriver

from .intel435.intel435_obstacle_zenoh import Intel435ObstacleDector

sys.path.insert(0, "../src")

try:
    from zenoh_msgs import open_zenoh_session, sensor_msgs
except ImportError:
    print("Please run this script from inside /system_hw_test")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--serial", help="serial port to use, when using the low level driver", type=str
)
parser.add_argument(
    "--zenoh", help="use zenoh to connect to the robot", action="store_true"
)
parser.add_argument(
    "--multicast", help="multicast address for zenoh", type=str, default=None
)
parser.add_argument(
    "--URID", help="your robot's URID, when using Zenoh", type=str, default=""
)
parser.add_argument(
    "--type", help="the type of the robot (go2 or tb4)", type=str, default="go2"
)
print(parser.format_help())

args = parser.parse_args()

# Create an instance of Intel435ObstacleDector
intel435ObstacleDector = Intel435ObstacleDector()


def create_straight_line_path_from_angle(angle_degrees, length=1.0, num_points=10):
    """Create a straight line path from origin at specified angle and length"""
    angle_rad = math.radians(angle_degrees)
    end_x = length * math.sin(angle_rad)  # sin for x because 0° is forward (positive y)
    end_y = length * math.cos(angle_rad)  # cos for y because 0° is forward (positive y)

    x_vals = np.linspace(0.0, end_x, num_points)
    y_vals = np.linspace(0.0, end_y, num_points)
    return np.array([x_vals, y_vals])


# Define 9 straight line paths separated by 15 degrees
# Center path is 0° (straight forward), then ±15°, ±30°, ±45°, ±60°
path_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]  # degrees
path_length = 1.05  # meters

paths = [
    create_straight_line_path_from_angle(angle, path_length) for angle in path_angles
]

print(f"Created {len(paths)} paths with angles: {path_angles}")
print(f"Each path extends {path_length}m from robot center")

pp = []
for path in paths:
    pairs = list(zip(path[0], path[1]))
    pp.append(pairs)

print(paths)
print(pp)

fig = plot.figure()
ax1 = plot.subplot(131)
ax2 = plot.subplot(132)
ax3 = plot.subplot(133)

center = ax1.plot([0], [0], "o", color="blue")[0]  # the robot
circle = ax1.add_patch(Circle((0, 0), 0.20, color="red"))  # the robot
points = ax1.plot([], [], "-", color="black")[0]
front = ax1.annotate("Front", xytext=(0.1, 0.3), xy=(0, 0.5))
arrow = ax1.annotate("", xytext=(0, 0), xy=(0, 1.5), arrowprops=dict(arrowstyle="->"))
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect("equal")

# Robot and sensor configuration
# UNITREE

half_width_robot = 0.20  # the width of the robot is 40 cm
relevant_distance_max = 1.1  # meters
relevant_distance_min = 0.20  # meters
sensor_mounting_angle = 172.0  # corrects for how sensor is mounted
# angles_blanked = [[-180.0, -140.0], [140.0, 180.0]]
angles_blanked = []

# Figure 2 - the zoom and the possible paths
centerZoom = ax2.plot([0], [0], "o", color="blue")[0]  # the robot

if args.type == "tb4":
    sensor_mounting_angle = 270.0
    angles_blanked = [[-180.0, -160.0], [110.0, 180.0]]
    circleZoom = ax2.add_patch(
        Circle((0, 0), 0.20, ls="--", lw=1, ec="red", fc="none")
    )  # the robot head
    outline = ax2.add_patch(
        Rectangle((-0.05, -0.15), 0.20, 0.06, ls="--", fc="black")
    )  # the robot electronics
else:
    circleZoom = ax2.add_patch(
        Circle((0, 0), 0.20, ls="--", lw=1, ec="red", fc="none")
    )  # the robot head
    outline = ax2.add_patch(
        Rectangle((-0.2, -0.7), 0.40, 0.70, ls="--", lw=1, ec="red", fc="none")
    )  # the robot body
pointsZoom = ax2.plot([], [], ".", color="black")[0]
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect("equal")

lines = []
for li in list(range(0, len(paths))):
    lines.append(ax2.plot([0], [0], "-", color="black")[0])

line = ax3.plot([0], [0], "-", color="red")[0]
ax3.set_xlim(-180, 180)
ax3.set_ylim(0, 1.2)
ax3.set_aspect(300)

ax3.plot([-180.0, -48.0], [1.18, 1.18], "-", color="red", linewidth=3.0)[0]
ax3.plot([-42.0, 42.0], [1.18, 1.18], "-", color="black", linewidth=3.0)[0]
ax3.plot([48.0, 180.0], [1.18, 1.18], "-", color="green", linewidth=3.0)[0]
ax3.annotate("Left", xytext=(-125, 1.1), xy=(0, 0.5))
ax3.annotate("Front", xytext=(-20, 1.1), xy=(0, 0.5))
ax3.annotate("Right", xytext=(85, 1.1), xy=(0, 0.5))

# display the blanked regions of the scan
for b in angles_blanked:
    deg_to_rad = np.pi / 180.0

    start_angle = b[0] * deg_to_rad
    end_angle = b[1] * deg_to_rad

    theta = np.linspace(start_angle, end_angle, 50)
    r = relevant_distance_max

    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # the arc
    ax2.plot(x, y, "--", color="grey", linewidth=1.5)

    # the straight lines
    ax2.plot([0, x[0]], [0, y[0]], "--", color="grey", linewidth=1)
    ax2.plot([0, x[-1]], [0, y[-1]], "--", color="grey", linewidth=1)

    # for panel 3 - this is in the correct units
    width = abs(b[1] - b[0])
    ax3.add_patch(Rectangle((b[0], 0.2), width, 1.0, fc="grey"))


def continuous_serial(lidar):

    for i, scan in enumerate(
        lidar.iter_scans(scan_type="express", max_buf_meas=3000, min_len=5)
    ):

        array = np.array(scan)

        # the driver sends angles in degrees between from 0 to 360
        # warning - the driver may send two or more readings per angle,
        # this can be confusing for the code
        angles = array[:, 1]

        # distances are in millimeters
        distances_mm = array[:, 2]
        distances_m = [i / 1000 for i in distances_mm]

        data = list(zip(angles, distances_m))
        array_ready = np.array(data)
        # print(f"Array {array_ready}")
        process(array_ready)


def zenoh_scan(sample):

    scan = sensor_msgs.LaserScan.deserialize(sample.payload.to_bytes())
    # print(f"Scan {scan}")

    # angle_min=-3.1241390705108643, angle_max=3.1415927410125732
    angles = list(
        map(
            lambda x: 360.0 * (x + math.pi) / (2 * math.pi),
            np.arange(scan.angle_min, scan.angle_max, scan.angle_increment),
        )
    )

    angles_final = np.flip(angles)
    # angles now run from 360.0 to 0 degrees
    data = list(zip(angles_final, scan.ranges))
    array_ready = np.array(data)
    # print(f"Array {array_ready}")
    process(array_ready)


def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    # Vector from line start to line end
    dx = x2 - x1
    dy = y2 - y1

    # If the line segment has zero length, return distance to point
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Calculate the parameter t that represents the projection of the point onto the line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # Clamp t to [0, 1] to stay within the line segment
    t = max(0, min(1, t))

    # Find the closest point on the line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Return the distance from the point to the closest point on the line segment
    return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def calculate_angle_and_distance(world_x, world_y):
    distance = math.sqrt(world_x**2 + world_y**2)

    angle_rad = math.atan2(world_y, world_x)
    angle_degrees = math.degrees(angle_rad)

    return angle_degrees, distance


def process(data):

    complexes = []

    for angle, distance in data:

        d_m = distance

        # don't worry about distant objects
        if d_m > 5.0:
            continue

        # first, correctly orient the sensor zero to the robot zero
        angle = angle + sensor_mounting_angle
        if angle >= 360.0:
            angle = angle - 360.0
        elif angle < 0.0:
            angle = 360.0 + angle

        # then, convert to radians
        a_rad = angle * math.pi / 180.0

        v1 = d_m * math.cos(a_rad)
        v2 = d_m * math.sin(a_rad)

        # convert to x and y
        # x runs backwards to forwards, y runs left to right
        x = -1 * v2
        y = -1 * v1

        # convert the angle to -180 to + 180 range
        angle = angle - 180.0

        keep = True
        for b in angles_blanked:
            if angle >= b[0] and angle <= b[1]:
                # this is a permanent reflection based on the robot
                # disregard
                keep = False
                break

        if d_m < relevant_distance_min:
            # this is too close, disregard
            keep = False

        # the final data ready to use for path planning
        if keep:
            complexes.append([x, y, angle, d_m])

    if len(intel435ObstacleDector.obstacle) > 100:
        for obstacle in intel435ObstacleDector.obstacle:
            angle, distance = calculate_angle_and_distance(obstacle["x"], obstacle["y"])
            complexes.append(
                [
                    obstacle["x"],
                    obstacle["y"],
                    angle,
                    distance,
                ]
            )

    array = np.array(complexes)

    # sort data into strictly increasing angles to deal with sensor issues
    # the sensor sometimes reports part of the previous scan and part of the next scan
    # so you end up with multiple slightly different values for some angles at the
    # junction
    sorted_indices = array[:, 2].argsort()
    array = array[sorted_indices]

    X = array[:, 0]
    Y = array[:, 1]
    A = array[:, 2]
    D = array[:, 3]

    global points
    points.set_data(X, Y)

    global pointsZoom
    pointsZoom.set_data(X, Y)

    global line
    line.set_data(A, D)

    """
    Determine set of possible paths
    """
    possible_paths = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bad_paths = []

    # all the possible conflicting points
    for x, y, d in list(zip(X, Y, D)):
        for apath in possible_paths:
            # Get the start and end points of this straight line path
            path_points = paths[apath]
            start_x, start_y = (
                path_points[0][0],
                path_points[1][0],
            )
            end_x, end_y = path_points[0][-1], path_points[1][-1]

            if apath == 9:
                # For going back, only consider obstacles that are:
                # 1. Behind the robot (negative x in robot frame)
                # 2. Within a certain distance threshold

                # Assuming robot faces positive x direction
                # Only check obstacles behind the robot
                if y >= 0:  # Skip obstacles in front of or beside the robot
                    continue

            # Calculate distance from obstacle to the line segment
            dist_to_line = distance_point_to_line_segment(
                x, y, start_x, start_y, end_x, end_y
            )

            if dist_to_line < half_width_robot:
                # too close - this path will not work
                path_to_remove = np.array([apath])
                bad_paths.append(apath)
                possible_paths = np.setdiff1d(possible_paths, path_to_remove)
                break  # no need to keep checking this path - we know this path is bad

    print(f"possible_paths: {possible_paths}")

    # convert to simple list
    ppl = possible_paths.tolist()

    left = []
    forward = []
    right = []
    backward = []

    for p in ppl:
        # Categorize paths based on angle
        angle = path_angles[p]
        if angle == -60:
            left.append(p)
        elif angle == -45:
            left.append(p)
        elif angle == -30:
            left.append(p)
        elif angle == -15:
            forward.append(p)
        elif angle == 0:
            forward.append(p)
        elif angle == 15:
            forward.append(p)
        elif angle == 30:
            right.append(p)
        elif angle == 45:
            right.append(p)
        elif angle == 60:
            right.append(p)
        elif angle == 180:
            backward.append(p)

        lines[p].set_data(paths[p][0], paths[p][1])
        lines[p].set_color("green")

    if len(ppl) > 0:
        print(f"There are {len(ppl)} possible paths.")
        if len(left) > 0:
            print(
                f"You can turn left using paths: {left} ({[path_angles[p] for p in left]}°)."
            )
        if len(forward) > 0:
            print(
                f"You can go forward using paths: {forward} ({[path_angles[p] for p in forward]}°)."
            )
        if len(right) > 0:
            print(
                f"You can turn right using paths: {right} ({[path_angles[p] for p in right]}°)."
            )
        if len(backward) > 0:
            print(
                f"You can go backward using paths: {backward} ({[path_angles[p] for p in backward]}°)."
            )
    else:
        print(
            "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
        )

    for p in bad_paths:
        # these are all the bad paths
        lines[p].set_data(paths[p][0], paths[p][1])
        lines[p].set_color("red")


if __name__ == "__main__":

    if args.serial:
        PORT_NAME = args.serial
        print(f"Using {PORT_NAME} as the serial port")
        try:
            lidar = RPDriver(PORT_NAME)
            info = lidar.get_info()
            print(f"Info: {info}")

            health = lidar.get_health()
            print(f"Health: {health}")

            # reset to clear buffers
            lidar.reset()

            subscribe_thread = threading.Thread(target=continuous_serial, args=(lidar,))
            subscribe_thread.daemon = True
            subscribe_thread.start()

            ani = FuncAnimation(fig, lambda _: None)
            plot.show()

        except KeyboardInterrupt:
            print("Program interrupted")

        finally:
            print("Exiting program")
            lidar.stop()
            lidar.disconnect()
            time.sleep(0.5)
            sys.exit(0)

        sys.exit(0)

    if args.zenoh:
        print("Using Zenoh to connect to robot")
        print("[INFO] Opening zenoh session...")
        conf = zenoh.Config()
        if args.multicast:
            conf.insert_json5(
                "scouting", f'{{"multicast": {{"address": "{args.multicast}"}}}}'
            )

        z = open_zenoh_session()

        if args.type == "go2":
            print("[INFO] Creating Subscribers for Go2")
            scans = z.declare_subscriber("scan", zenoh_scan)

        if args.type == "tb4":
            print("[INFO] Creating Subscribers for TB4")
            scans = z.declare_subscriber(f"{args.URID}/pi/scan", zenoh_scan)

        if args.type != "go2" and args.type != "tb4":
            print(f"[ERROR] Unsupported robot type: {args.type}")
            sys.exit(1)

        ani = FuncAnimation(fig, lambda _: None)
        plot.show()

        sys.exit(0)

    raise ValueError("You must specify either --serial or --zenoh to run this script.")
