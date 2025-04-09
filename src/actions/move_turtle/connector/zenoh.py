import logging
import math
import time
from dataclasses import dataclass
from queue import Queue

import zenoh

from actions.base import ActionConfig, ActionConnector
from actions.move_turtle.interface import MoveInput
from zenoh_idl import geometry_msgs, nav_msgs, sensor_msgs

rad_to_deg = 57.2958

gHazard = None
gOdom = None

def listenerHazard(data):
    global gHazard
    #logging.info(f"Hazard listener")
    gHazard = sensor_msgs.HazardDetectionVector.deserialize(data.payload.to_bytes())
    #logging.info(f"Hazard listener: {gHazard}")

def listenerOdom(data):
    global gOdom
    #logging.info(f"Odom listener")
    gOdom = nav_msgs.Odometry.deserialize(data.payload.to_bytes())
    #logging.info(f"Odom listener: {gOdom}")

def euler_from_quaternion(x, y, z, w):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

class MoveZenohConnector(ActionConnector[MoveInput]):

    def __init__(self, config: ActionConfig):
        
        super().__init__(config)

        self.angle_tolerance = 5.0
        self.distance_tolerance = 0.05  # m

        self.pending_movements = Queue()
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw_now = 0.0
        self.hazard = None

        self.session = None

        URID = getattr(self.config, "URID", None)

        if URID is None:
            logging.warning("Aborting Turtlebot4 Move system, no URID provided")
            return
        else:
            logging.info(f"Turtlebot4 Move system is using URID: {URID}")

        self.cmd_vel = f"{URID}/c3/cmd_vel"

        try:
            self.session = zenoh.open(zenoh.Config())
            logging.info(f"Zenoh move client opened {self.session}")
            logging.info(f"TurtleBot4 move listeners starting with URID: {URID}")
            self.session.declare_subscriber(f"{URID}/c3/odom", listenerOdom)
            self.session.declare_subscriber(f"{URID}/c3/hazard_detection", listenerHazard)
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

    def hazardProcessor(self):
        global gHazard  
        logging.info(f"Hazard processor: {gHazard}")
        if gHazard is not None and gHazard.detections and len(gHazard.detections) > 0:
            for haz in gHazard.detections:
                # print(f"Hazard Type:{haz.type} direction:{haz.header.frame_id}")
                if haz.type == 1:
                    
                    # clear the movement queue immediately
                    with self.pending_movements.mutex:
                        self.pending_movements.queue.clear()
                    
                    if haz.header.frame_id == "bump_front_right":
                        self.hazard = "TURN_LEFT"
                        logging.info(f"Hazard: {self.hazard}")
                        # if this hazard exists, report it immediately, and don't overwrite
                        # hazard with less important information
                        return
                    if haz.header.frame_id == "bump_front_left":
                        self.hazard = "TURN_RIGHT"
                        logging.info(f"Hazard: {self.hazard}")
                        return
                    if haz.header.frame_id == "bump_front_center":
                        self.hazard = "TURN"
                        logging.info(f"Hazard: {self.hazard}")
                        return
    
    def odomProcessor(self):
        global gOdom
        logging.info(f"Odom processor: {gOdom}")
        if gOdom is not None: 
            x = gOdom.pose.pose.orientation.x
            y = gOdom.pose.pose.orientation.y
            z = gOdom.pose.pose.orientation.z
            w = gOdom.pose.pose.orientation.w
            
            logging.info(f"TurtleBot4 Odom listener: {gOdom.pose.pose.orientation}")
            
            angles = euler_from_quaternion(x, y, z, w)

            self.yaw_now = (
                angles[2] * rad_to_deg * -1.0
            )  # we use the aviation convention of CW yaw = positive

            logging.info(f"TurtleBot4 current yaw angle {self.yaw_now}")

            # current position in world frame
            self.x = gOdom.pose.pose.position.x
            self.y = gOdom.pose.pose.position.y

    def move(self, vx, vyaw):
        """
        generate movement commands
        """
        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        logging.info("Pub twist: {} - {}".format(vx, vyaw))
        t = geometry_msgs.Twist(
            linear=geometry_msgs.Vector3(x=float(vx), y=0.0, z=0.0),
            angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(vyaw)),
        )
        self.session.put(self.cmd_vel, t.serialize())

    async def connect(self, output_interface: MoveInput) -> None:

        logging.info(f"SendThisToZenoh: {output_interface.action}")

        # if self.pending_movements.qsize() > 0:
        #     logging.info("Movement in progress: disregarding new command")
        #     return

        # if self.x == 0.0:
        #     # this value is never precisely zero EXCEPT while
        #     # booting and waiting for data to arrive
        #     logging.info("Waiting for location data")
        #     return

        # if output_interface.action == "turn left":
        #     logging.info(f"Zenoh command: {output_interface.action}")
        #     # turn 90 Deg to the left (CCW)
        #     target_yaw = self.yaw_now - 90.0
        #     if target_yaw <= -180.0: target_yaw += 360.0
        #     self.pending_movements.put([0.0, target_yaw, "turn_left"])
        # elif output_interface.action == "turn right":
        #     logging.info(f"Zenoh command: {output_interface.action}")
        #     # turn 90 Deg to the right (CW)
        #     target_yaw = self.yaw_now + 90.0
        #     if target_yaw >= 180.0: target_yaw -= 360.0
        #     self.pending_movements.put([0.0, target_yaw, "turn_right"])
        # elif output_interface.action == "move forwards":
        #     self.pending_movements.put([0.5, 0.0, "advance", self.x, self.y])
        # elif output_interface.action == "move back":
        #     self.pending_movements.put([0.5, 0.0, "back", self.x, self.y])
        # # elif output_interface.action == "avoid left obstacle":
        # #     logging.info(f"Zenoh command: {output_interface.action}")
        # #     self.pub_twist(0, 0.8)
        # # elif output_interface.action == "avoid right obstacle":
        # #     logging.info(f"Zenoh command: {output_interface.action}")
        # #     self.pub_twist(0, -0.8)
        # elif output_interface.action == "stand still":
        #     logging.info(f"Zenoh command: {output_interface.action}")
        #     # do nothing
        # else:
        #     logging.info(f"Unknown move type: {output_interface.action}")

    def tick(self) -> None:

        time.sleep(0.1)
        
        logging.info(f"Move tick")

        self.hazardProcessor()
        self.odomProcessor()

    # AVOID_LEFT_OBSTACLE = "avoid left obstacle"
    # AVOID_FRONT_OBSTACLE = "avoid front obstacle"
    # AVOID_RIGHT_OBSTACLE = "avoid right obstacle"

        if self.x == 0.0:
            # this value is never precisely zero except while
            # booting and waiting for data to arrive
            logging.info("Waiting for location data")
            return

        if self.hazard is not None:
            logging.info(f"Should be empty: {self.pending_movements}")
            # self.pending_movements.put([0.5, 0.0, "back", self.x, self.y])
            # this jams the TB4 because it overlaps with the low level recoil action
            if self.hazard == "TURN_RIGHT":
                target_yaw = self.yaw_now + 100.0
                if target_yaw >= 180.0: target_yaw -= 360.0
                self.pending_movements.put([0.0, target_yaw, "turn_right"])
            elif self.hazard == "TURN_LEFT":
                target_yaw = self.yaw_now - 100.0
                if target_yaw <= -180.0: target_yaw += 360.0
                self.pending_movements.put([0.0, target_yaw, "turn_left"])
            elif self.hazard == "TURN":
                # always turn left, after collision
                target_yaw = self.yaw_now - 100.0
                if target_yaw <= -180.0: target_yaw += 360.0
                self.pending_movements.put([0.0, target_yaw, "turn_left"])
            # clear the hazard
            self.hazard = None
            logging.info(f"Should have avoidance: {self.pending_movements}")

    #     target = list(pending_movements.queue)

    #     if len(target) > 0:
    #         logging.info(f"Zenoh Target: {target}")
    #         if target[0][2] == "turn_left":
    #             togo = abs(G.yaw_now - target[0][1])
    #             logging.info(f"tl to go: {togo}")
    #             if togo > self.angle_tolerance:
    #                 if self.yaw_now > target[0][1]:  # keep turning left
    #                     logging.info(f"keep turning left. remaining:{togo} ")
    #                     self.move(0.0, 0.3)
    #                 elif (
    #                     self.yaw_now < target[0][1]
    #                 ):  # turn to the right - you turned too far
    #                     logging.info(f"OVERSHOOT: turn right. remaining:{togo} ")
    #                     self.move(0.0, -0.1)
    #             else:
    #                 logging.info("done, pop 1 off queue")
    #                 self.pending_movements.get()
    #         elif target[0][2] == "turn_right":
    #             togo = abs(G.yaw_now - target[0][1])
    #             logging.info(f"tr to go: {togo}")
    #             if togo > self.angle_tolerance:
    #                 if self.yaw_now < target[0][1]:  # keep turning right
    #                     logging.info(f"keep turning right. remaining:{togo} ")
    #                     self.move(0.0, -0.3)
    #                 elif (
    #                     self.yaw_now > target[0][1]
    #                 ):  # turn to the left - you turned too far
    #                     logging.info(f"OVERSHOOT: turn left. remaining:{togo} ")
    #                     self.move(0.0, 0.1)
    #             else:
    #                 logging.info("done, pop 1 off queue")
    #                 self.pending_movements.get()
    #         elif target[0][2] == "advance":
    #             distance_traveled = math.sqrt(
    #                 (G.x - target[0][3]) ** 2 + (G.y - target[0][4]) ** 2
    #             )
    #             togo = abs(target[0][0] - distance_traveled)
    #             logging.info(f"distance to go: {togo}")
    #             if togo > self.distance_tolerance:
    #                 if distance_traveled < target[0][0]:  # keep advancing
    #                     logging.info(f"keep advancinG. remaining:{togo} ")
    #                     self.move(0.4, 0.0)
    #                 elif distance_traveled > target[0][0]:  # you moved too far
    #                     logging.info(f"OVERSHOOT: retreat. remaining:{togo} ")
    #                     self.move(-0.1, 0.0)
    #             else:
    #                 logging.info("done, pop 1 off queue")
    #                 self.pending_movements.get()
    #         elif target[0][2] == "back":
    #             distance_traveled = math.sqrt(
    #                 (G.x - target[0][3]) ** 2 + (G.y - target[0][4]) ** 2
    #             )
    #             togo = abs(target[0][0] - distance_traveled)
    #             logging.info(f"distance to back up: {togo}")
    #             if togo > self.distance_tolerance:
    #                 if distance_traveled < target[0][0]:  # keep backing up
    #                     logging.info(f"keep backing up. remaining:{togo} ")
    #                     self.move(-0.3, 0.0)
    #                 elif distance_traveled > target[0][0]:  # you moved too far
    #                     logging.info(f"OVERSHOOT: advance. remaining:{togo} ")
    #                     self.move(0.1, 0.0)
    #             else:
    #                 logging.info("done, pop 1")
    #                 self.pending_movements.get()
