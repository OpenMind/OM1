from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    world_file = LaunchConfiguration('world', default='gazebo_worlds/enhanced/office_environment.world')
    
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='gazebo_worlds/enhanced/office_environment.world'),
        
        # Launch Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', world_file],
            output='screen'
        ),
        
        # OM1 Bridge
        ExecuteProcess(
            cmd=['python3', 'scripts/om1_gazebo_bridge.py'],
            output='screen'
        ),
        
        # ROS 2 Bridge for camera
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/camera@sensor_msgs/msg/Image@gz.msgs.Image'],
            output='screen'
        ),
        
        # ROS 2 Bridge for LiDAR
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'],
            output='screen'
        ),
        
        # ROS 2 Bridge for IMU
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/imu@sensor_msgs/msg/Imu@gz.msgs.IMU'],
            output='screen'
        ),
    ])
