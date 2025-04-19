from setuptools import find_packages, setup

package_name = 'visual_odometry_pkg'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    packages=['visual_odometry_pkg'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shivu',
    maintainer_email='shi@gmail.com',
    description='Visual odometry ROS 2 node using OpenCV',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'vo_node = visual_odometry_pkg.visual_odometry_node:main',
    ],
},
)
