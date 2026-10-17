"""Stream supplied bags with common receipt timestamps and a playback clock."""
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from .bag import Bag, restamp_sensor_cdr


def set_stamp(stamp, nanoseconds):
    stamp.sec, stamp.nanosec = divmod(int(nanoseconds), 10**9)


class Player(Node):
    def __init__(self):
        super().__init__('group8_bag_player')
        self.declare_parameter('bag', '')
        self.declare_parameter('rate', 0.5)
        self.declare_parameter('loop', True)
        self.declare_parameter('startup_delay', 4.0)
        self.bag = Bag(self.get_parameter('bag').value)
        self.rate = float(self.get_parameter('rate').value)
        if self.rate <= 0:
            raise ValueError('Playback rate must be positive')
        self.stop = threading.Event()
        self.publishers_by_topic = {}
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        for name, (_, typ, _) in self.bag.topics.items():
            if name != '/tf_static':
                self.publishers_by_topic[name] = self.create_publisher(typ, name, sensor_qos)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.static_pub = self.create_publisher(TFMessage, '/tf_static', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE))
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            if self.stop.wait(self.get_parameter('startup_delay').value):
                return
            transforms = self.bag.static_transforms()
            self.static_pub.publish(TFMessage(transforms=transforms))
            self.get_logger().warning('Replay uses BAG RECEIPT timestamps, not acquisition synchronization.')
            while rclpy.ok() and not self.stop.is_set():
                start_wall = time.monotonic()
                last_static = 0.0
                for name, timestamp, raw in self.bag.serialized_messages():
                    if name == '/tf_static':
                        continue
                    target = start_wall + (timestamp - self.bag.start) / 1e9 / self.rate
                    if self.stop.wait(max(0, target - time.monotonic())):
                        return
                    clock = Clock()
                    set_stamp(clock.clock, timestamp)
                    self.clock_pub.publish(clock)
                    # Recover late subscribers and TF buffers cleared by loop clock rewinds.
                    if time.monotonic() - last_static > 1.0:
                        self.static_pub.publish(TFMessage(transforms=transforms))
                        last_static = time.monotonic()
                    if not self.bag.topics[name][2].startswith('sensor_msgs/msg/'):
                        raise ValueError(f'Unsupported replay topic without known Header layout: {name}')
                    self.publishers_by_topic[name].publish(restamp_sensor_cdr(raw, timestamp))
                self.get_logger().info('Playback pass complete')
                if not self.get_parameter('loop').value:
                    return
                if self.stop.wait(1.0):
                    return
        except Exception as exc:
            self.get_logger().error(f'Playback failed: {exc}')
            rclpy.try_shutdown()

    def destroy_node(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.bag.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Player()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
