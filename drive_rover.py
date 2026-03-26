#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import csv
import math

def euler_from_quaternion(q):
    """Convert quaternion into euler angles (yaw)"""
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class RoverDriver(Node):
    def __init__(self):
        super().__init__('rover_driver')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        
        self.initial_x = None
        self.initial_y = None
        self.initial_yaw = None
        
        self.waypoints = []
        self.current_wp_idx = 0
        self.load_waypoints()
        
        # Run the control loop at 20Hz
        self.timer = self.create_timer(0.05, self.control_loop) 
        
    def load_waypoints(self):
        try:
            with open('path_waypoints.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.waypoints.append((float(row[0]), float(row[1])))
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints. Ready to drive!")
        except FileNotFoundError:
            self.get_logger().error("path_waypoints.csv not found! Run astar.py first.")
            
    def odom_callback(self, msg):
        # Gazebo odometry starts at 0,0. We must offset it to our actual world spawn point!
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = euler_from_quaternion(msg.pose.pose.orientation)
        
        if self.initial_x is None and len(self.waypoints) > 0:
            # Anchor the odometry to the first waypoint (our spawn location)
            self.initial_x = self.waypoints[0][0]
            self.initial_y = self.waypoints[0][1]
            
            if len(self.waypoints) > 1:
                dx = self.waypoints[1][0] - self.waypoints[0][0]
                dy = self.waypoints[1][1] - self.waypoints[0][1]
                self.initial_yaw = math.atan2(dy, dx)
            else:
                self.initial_yaw = 0.0

        if self.initial_x is not None:
            # Rotate and translate the local odometry into global map coordinates
            rot_x = odom_x * math.cos(self.initial_yaw) - odom_y * math.sin(self.initial_yaw)
            rot_y = odom_x * math.sin(self.initial_yaw) + odom_y * math.cos(self.initial_yaw)
            
            self.current_x = self.initial_x + rot_x
            self.current_y = self.initial_y + rot_y
            self.current_yaw = self.initial_yaw + odom_yaw
        
    def control_loop(self):
        # Don't start driving until we have received our first GPS coordinate
        if not self.waypoints or self.current_x is None:
            return
            
        if self.current_wp_idx >= len(self.waypoints):
            self.publisher_.publish(Twist()) # Send stop command
            self.get_logger().info("Goal reached successfully!")
            self.timer.cancel()
            return
            
        target_x, target_y = self.waypoints[self.current_wp_idx]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.hypot(dx, dy)
        
        # If we are within 0.2 meters of the waypoint, advance to the next one
        if distance < 0.2:
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                self.get_logger().info(f"Driving to Waypoint {self.current_wp_idx + 1}/{len(self.waypoints)}")
            return
            
        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - self.current_yaw
        
        # Normalize yaw error to always turn the shortest direction (-pi to pi)
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        
        cmd = Twist()
        cmd.angular.z = 1.5 * yaw_error # Proportional steering
        
        # Only drive forward if generally facing the target, otherwise pivot in place
        if abs(yaw_error) < 0.3:
            cmd.linear.x = min(0.6 * distance, 0.5) # Speed up to 0.5 m/s
        else:
            cmd.linear.x = 0.0
            
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    driver = RoverDriver()
    rclpy.spin(driver)
    driver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()