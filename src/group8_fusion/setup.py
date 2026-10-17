from glob import glob
from setuptools import setup

setup(
    name='group8_fusion', version='0.1.0',
    packages=['group8_fusion'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/group8_fusion']),
        ('share/group8_fusion', ['package.xml']),
        ('share/group8_fusion/launch', glob('launch/*.py')),
        ('share/group8_fusion/config', glob('config/*')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Team Project Group 8', maintainer_email='236782705+kshitijprasaddev@users.noreply.github.com',
    description='Camera, LiDAR and radar fusion with ROS 2 and RViz2', license='AGPL-3.0-only',
    entry_points={'console_scripts': [
        'bag_player = group8_fusion.replay:main',
        'fusion_node = group8_fusion.ros_node:main',
    ]},
)
