import json
import logging
import os

import zenoh

logging.basicConfig(level=logging.INFO)


def create_zenoh_config(network_discovery: bool = True) -> zenoh.Config:
    """
    Create a Zenoh configuration for a client connecting to a Zenoh router.

    The connect endpoint defaults to tcp/127.0.0.1:7447 (a local router on
    the same host) but can be overridden via OM1_ZENOH_ENDPOINT — for example
    "wss/test-sim.openmind.com:8444" to reach a remote Zenoh router over a
    TLS-terminated WebSocket. For self-signed test deployments,
    OM1_ZENOH_TLS_ROOT_CA can point at the trusted CA cert file path.

    Parameters
    ----------
    network_discovery : bool, optional
        Whether to enable network discovery (default is True).

    Returns
    -------
    zenoh.Config
        The Zenoh configuration object.
    """
    config = zenoh.Config()
    if not network_discovery:
        endpoint = os.environ.get("OM1_ZENOH_ENDPOINT", "tcp/127.0.0.1:7447")
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))

        if endpoint.startswith(("wss/", "tls/", "quic/")):
            ca_path = os.environ.get("OM1_ZENOH_TLS_ROOT_CA")
            if ca_path:
                config.insert_json5(
                    "transport/link/tls/root_ca_certificate",
                    json.dumps(ca_path),
                )

    return config


def open_zenoh_session():
    """
    Open a Zenoh session.

    If OPENMIND_CLOUD_URL is set, return an OpenMindZenohSession that
    looks like a zenoh.Session to OM1 plugins but routes pub/sub through
    the OpenMind cloud broker. This is the pattern customers use in the
    cloud product.

    Otherwise, open a normal Zenoh client (local first, then network
    discovery) — same behavior as before.

    Returns
    -------
    zenoh.Session or OpenMindZenohSession
        Quack-compatible session object.
    """
    if os.environ.get("OPENMIND_CLOUD_URL"):
        return _open_cloud_session()

    local_config = create_zenoh_config(network_discovery=False)
    try:
        session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
        return session
    except Exception:
        logging.info("Falling back to network discovery...")

    config = create_zenoh_config()
    try:
        session = zenoh.open(config)
        logging.info("Zenoh client opened with network discovery")
        return session
    except Exception as e:
        logging.error(f"Error opening Zenoh client: {e}")
        raise Exception("Failed to open Zenoh session") from e


def _open_cloud_session():
    """Build an OpenMind cloud broker session.

    Required env vars:
      OPENMIND_CLOUD_URL    e.g. "wss://api.openmind.com/api/core/simulation/zenoh?api_key=..."

    Optional env vars:
      OPENMIND_CLOUD_TOKEN       legacy JWT auth (not needed when api_key is in URL)
      OPENMIND_TOPIC_MAP_JSON    JSON object of extra/override mappings:
                                 { "foo": {"topic":"foo","schema":"std_msgs/msg/String"} }

    The default topic_map covers the topics OM1 plugins subscribe/publish
    today (odom, cmd_vel, image, scan, joint_states). Customers / robots
    with different topic conventions extend via OPENMIND_TOPIC_MAP_JSON.
    """
    from cloud.zenoh_shim import OpenMindZenohSession, TopicSpec

    url = os.environ["OPENMIND_CLOUD_URL"]
    token = os.environ.get("OPENMIND_CLOUD_TOKEN") or None

    # Mirrors the broker's DEFAULT_READ_ALLOW / DEFAULT_WRITE_ALLOW and the
    # GPU zenoh-bridge allow list. Override / extend via the
    # ``OPENMIND_TOPIC_MAP_JSON`` env var.
    _CORE = [
        # (broker_topic, schema)
        ("odom", "nav_msgs/msg/Odometry"),
        ("cmd_vel", "geometry_msgs/msg/Twist"),
        ("cmd_vel_human", "geometry_msgs/msg/Twist"),
        ("scan", "sensor_msgs/msg/LaserScan"),
        # binary CDR pass-through ignores schema; placeholder here so the
        # mapping exists for plugins that subscribe in binary mode.
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
        # Unitree Go2 robot state
        ("lowstate", "unitree_go/msg/LowState"),
        ("lf/lowstate", "unitree_go/msg/LowState"),
        ("sportmodestate", "unitree_go/msg/SportModeState"),
        ("lf/sportmodestate", "unitree_go/msg/SportModeState"),
        ("utlidar/robot_pose", "geometry_msgs/msg/PoseStamped"),
        ("utlidar/cloud_deskewed", "sensor_msgs/msg/PointCloud2"),
        ("unitree_lidar/points", "sensor_msgs/msg/PointCloud2"),
        # Unitree Sport API (binary-mode only; schema unused for the put).
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
        # Historical alias used by some plugins.
        ("image", "sensor_msgs/msg/Image"),
    ]

    topic_map: dict[str, TopicSpec] = {
        broker_topic: TopicSpec(broker_topic=broker_topic, schema=schema) for broker_topic, schema in _CORE
    }

    custom = os.environ.get("OPENMIND_TOPIC_MAP_JSON")
    if custom:
        try:
            for k, v in json.loads(custom).items():
                topic_map[k] = TopicSpec(broker_topic=v["topic"], schema=v["schema"])
        except Exception as e:
            logging.warning("OPENMIND_TOPIC_MAP_JSON malformed, ignoring: %s", e)

    logging.info("OpenMind cloud session: url=%s mappings=%d", url, len(topic_map))
    return OpenMindZenohSession(url, topic_map=topic_map, token=token)


if __name__ == "__main__":
    session = open_zenoh_session()
    if session:
        logging.info("Session opened successfully")
        session.close()
    else:
        logging.error("Failed to open Zenoh session")
