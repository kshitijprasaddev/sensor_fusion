from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('group8_fusion'))
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value='/data/studentProject'),
        DeclareLaunchArgument('config', default_value=str(share / 'config/fusion.yaml')),
        DeclareLaunchArgument('weights', default_value='/models/yolo11n.pt'),
        DeclareLaunchArgument('device', default_value='0'),
        DeclareLaunchArgument('rate', default_value='0.5'),
        DeclareLaunchArgument('loop', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(package='group8_fusion', executable='bag_player', output='screen', parameters=[{
            'bag': arg('bag'), 'rate': ParameterValue(arg('rate'), value_type=float),
            'loop': ParameterValue(arg('loop'), value_type=bool), 'startup_delay': 8.0}]),
        Node(package='group8_fusion', executable='fusion_node', output='screen', parameters=[{
            'bag': arg('bag'), 'config': arg('config'), 'weights': arg('weights'),
            'device': ParameterValue(arg('device'), value_type=str), 'use_sim_time': True}]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'config/fusion.rviz')],
             parameters=[{'use_sim_time': True}], condition=IfCondition(arg('rviz')), output='screen'),
    ])
