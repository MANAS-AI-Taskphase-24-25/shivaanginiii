import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import transformations as tf_trans
import matplotlib.pyplot as plt

class VideoOdometry(Node):
    def __init__(self):
        super().__init__('vo_node')

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/odom_path', 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.cap = cv2.VideoCapture('/home/shivu/drivingvideo.mp4')

        # features and parameters for lk optical flow
        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.prev_gray = None
        self.prev_pts = None
        self.x, self.y, self.theta = 0.0, 0.0, 0.0  # starting pose

        self.scale = 0.1  

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.05, self.process_frame)


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video.")
            self.cap.release()
            return

        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Video Feed", frame)
        cv2.waitKey(1)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)

            if next_pts is not None and status is not None:
                good_prev = self.prev_pts[status.flatten() == 1]
                good_next = next_pts[status.flatten() == 1]

                if len(good_prev) >= 6:
                    M, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)

                    if M is not None:
                        dx, dy = M[0, 2], M[1, 2]
                        dtheta = np.arctan2(M[1, 0], M[0, 0])

                        self.x += self.scale * (np.cos(self.theta) * dx - np.sin(self.theta) * dy)
                        self.y += self.scale * (np.sin(self.theta) * dx + np.cos(self.theta) * dy)
                        self.theta += dtheta

                        # convert to quaternion
                        qx, qy, qz, qw = tf_trans.quaternion_from_euler(0, 0, self.theta)

                        stamp = self.get_clock().now().to_msg()

                        odom = Odometry()
                        odom.header.stamp = stamp
                        odom.header.frame_id = "odom"
                        odom.child_frame_id = "base_link"
                        odom.pose.pose.position.x = self.x
                        odom.pose.pose.position.y = self.y
                        odom.pose.pose.position.z = 0.0
                        odom.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                        self.odom_pub.publish(odom)

                        pose = PoseStamped()
                        pose.header.stamp = stamp
                        pose.header.frame_id = "odom"
                        pose.pose = odom.pose.pose
                        self.path_msg.poses.append(pose)
                        self.path_msg.header.stamp = stamp
                        self.path_pub.publish(self.path_msg)


        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)


def main(args=None):
    rclpy.init(args=args)
    node = VideoOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
