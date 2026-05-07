from dataclasses import dataclass

topic_schemas = [
    ("odom", "nav_msgs/msg/Odometry"),
    ("cmd_vel", "geometry_msgs/msg/Twist"),
    ("cmd_vel_human", "geometry_msgs/msg/Twist"),
    ("scan", "sensor_msgs/msg/LaserScan"),
    ("joint_states", "sensor_msgs/msg/Imu"),
    ("imu/data", "sensor_msgs/msg/Imu"),
    ("tf", "tf2_msgs/msg/TFMessage"),
    ("tf_static", "tf2_msgs/msg/TFMessage"),
    ("joy", "sensor_msgs/msg/Joy"),
    # Camera streams
    ("camera/go2/image_raw", "sensor_msgs/msg/Image"),
    ("camera/realsense2_camera_node/color/image_raw", "sensor_msgs/msg/Image"),
    ("camera/realsense2_camera_node/color/image_isaac_sim_raw", "sensor_msgs/msg/Image"),
    ("camera/realsense2_camera_node/color/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("camera/realsense2_camera_node/depth/image_rect_raw", "sensor_msgs/msg/Image"),
    ("camera/realsense2_camera_node/depth/image_rect_isaac_sim_raw", "sensor_msgs/msg/Image"),
    ("camera/realsense2_camera_node/depth/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("rgb_image", "sensor_msgs/msg/Image"),
    # Unitree topcs
    ("lowstate", "unitree_go/msg/LowState"),
    ("lf/lowstate", "unitree_go/msg/LowState"),
    ("sportmodestate", "unitree_go/msg/SportModeState"),
    ("lf/sportmodestate", "unitree_go/msg/SportModeState"),
    ("utlidar/robot_pose", "geometry_msgs/msg/PoseStamped"),
    ("utlidar/cloud_deskewed", "sensor_msgs/msg/PointCloud2"),
    ("unitree_lidar/points", "sensor_msgs/msg/PointCloud2"),
    # Unitree sport API
    ("api/sport/request", "om_api/msg/OMAPIRequest"),
    ("api/sport/response", "om_api/msg/OMAPIResponse"),
    # OM system topics
    ("om/paths", "om_api/msg/Paths"),
    ("om/paths/r50", "om_api/msg/Paths"),
    ("om/paths/r100", "om_api/msg/Paths"),
    ("om/paths/r200", "om_api/msg/Paths"),
    ("om/ai/request", "om_api/msg/OMAPIRequest"),
    ("om/ai/response", "om_api/msg/OMAPIResponse"),
    ("om/asr/text", "om_api/msg/OMASRText"),
    ("om/avatar/request", "om_api/msg/OMAvatarFaceRequest"),
    ("om/avatar/response", "om_api/msg/OMAvatarFaceResponse"),
    ("image", "sensor_msgs/msg/Image"),
]


@dataclass(frozen=True)
class TopicSchemas:
    """
    Mapping from the topic to the schema name.

    Parameters
    ----------
    topic : str
        The topic on the broker to subscribe/publish to.
    schema : str
        The ROS message schema, e.g. "nav_msgs/msg/Odometry".
    """

    topic: str
    schema: str


topic_map: dict[str, TopicSchemas] = {
    topic: TopicSchemas(topic=topic, schema=schema) for topic, schema in topic_schemas
}
