import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import heapq
import numpy as np
import time

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('pathfinding_node')

        self.get_logger().info("Subscribing to /map...")
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)

        self.map_data = None
        self.width = 0
        self.height = 0
        self.resolution = 0.05  
        self.origin_x = 0.0
        self.origin_y = 0.0

        # stores the last path
        self.last_path = Path()

        self.get_logger().info("A* Pathfinding Node has started. Waiting for map data...")

    def map_callback(self, msg):
        """ Process the received occupancy grid map. """
        self.get_logger().info("Map received! Processing...")

        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        start = (0,0)  
        goal = (self.width - 1, self.height - 1)  

        path = self.a_star(start, goal)

        if path:
            self.publish_path(path)
        else:
            self.get_logger().warn("No valid path found!")

    def a_star(self, start, goal):
        """ A* pathfinding algorithm implementation. """
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] #denotes the possible 8 directions

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])

                if not self.is_valid(neighbor):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  

    def is_valid(self, pos):
        """ Check if a position is within bounds and not an obstacle. """
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map_data[y, x] == 0  # 0 indicates open block
        return False

    def reconstruct_path(self, came_from, current):
        """ Reconstructs path from goal to start using the came_from dictionary. """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)  
        path.reverse()
        return path

    def publish_path(self, path):
        """ Publish the computed path to the /path topic. """
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for x, y in path:       # converts the grids to coordinates
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x * self.resolution + self.origin_x
            pose.pose.position.y = y * self.resolution + self.origin_y
            path_msg.poses.append(pose) 

        self.last_path = path_msg  
        self.path_pub.publish(self.last_path)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} waypoints")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlanner()
    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            if node.last_path and len(node.last_path.poses) > 0:
                node.path_pub.publish(node.last_path)
                node.get_logger().info("Republishing last known path...")
            time.sleep(1.0)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down A* Pathfinding Node.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
