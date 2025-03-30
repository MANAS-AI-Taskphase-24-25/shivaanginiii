import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

class ChatPublisher(Node):
    def __init__(self):
        super().__init__('chat_publisher')
        self.publisher_ = self.create_publisher(String, 'chat_topic', 10)
        self.subscription = self.create_subscription(
            String,
            'chat_topic',
            self.listener_callback,
            10)
        threading.Thread(target=self.send_messages, daemon=True).start()

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')

    def send_messages(self):
        while rclpy.ok():
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                self.get_logger().info("chat ended")
                rclpy.shutdown()
                break
            
            msg = String()
            msg.data = user_input
            self.publisher_.publish(msg)
            self.get_logger().info("message sent.")

def main():
    rclpy.init()
    node = ChatPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
