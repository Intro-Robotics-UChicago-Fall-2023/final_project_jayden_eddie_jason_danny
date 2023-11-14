#!/usr/bin/env python3

import rospy, cv_bridge
import cv2
import cv2.aruco as aruco
import numpy as np
import os
from q_learning_project.msg import QLearningReward
from q_learning_project.msg import RobotMoveObjectToTag
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import moveit_commander
import math
import csv


import threading

class FindAndPickupCan:

        def __init__(self):
                self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
                self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
                self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                        Image, self.image_callback)
                # subscribe to the lidar scan from the robot
                self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
                self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
                self.arm_len_1 = 0.2
                self.arm_len_2 = 0.274
                self.can_to_pick = 'sprite'

                self.latest_image = None
                self.new_image_flag = False
        
        def laser_callback(self, data):
            # used to check the distances in front of the robot
            
            num_readings = len(data.ranges)  

            
            front_readings_count = int((30.0 / 360.0) * num_readings)
            # get the indices for the middle of the data.ranges array
            middle_index = num_readings // 2
            start_index = middle_index - (front_readings_count // 2)
            end_index = middle_index + (front_readings_count // 2)

            # extract the ranges for the front
            front_ranges = data.ranges[start_index:end_index]
            if not front_ranges:
                print("no distance in front!")
            else:
                # calculate the minimum distance in that range, ignoring 'inf' values
                self.distance_in_front = min(r for r in front_ranges if r != float('inf'))
                print(self.distance_in_front)
                   
        
        def get_lidar_distance(self):
            return self.distance_in_front
                
        def image_callback(self, msg):
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            self.image_width = self.image.shape[1]
            self.detected_object_center = self.detect_can(self.image, self.can_to_pick)
            color_ranges = {
                'dc': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                'coke': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                'sprite': (np.array([145, 75, 75]), np.array([175, 255, 255]))
            }
            if self.color_to_pick in color_ranges:
                    lower_color, upper_color = color_ranges[self.color_to_pick]
                    mask = cv2.inRange(hsv, lower_color, upper_color)
                    M = cv2.moments(mask)
                    if M['m00'] > 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(self.image, (cx, cy), 20, (0,0,255), -1)
            self.latest_image = self.image
            self.new_image_flag = True
        
        def calculate_angles(self, x, z):
           
            r = math.sqrt(x**2 + z**2)

            
            cos_q2 = (r ** 2 - self.arm_len_1**2 - self.arm_len_2**2) / (2 * self.arm_len_1 * self.arm_len_2)
            q2 = math.acos(cos_q2)

        
            q1 = math.atan2(z, x) - math.atan2(self.arm_len_2 * math.sin(q2), self.arm_len_1 + self.arm_len_2 * math.cos(q2))

            return q1, q2

        
        def calculate_z(self, cx, cy, lidar_distance):
            image_height = self.image.shape[0]
            camera_vertical_fov = math.radians(48.8) 
            angle_from_center = ((cy - (image_height / 2)) / (image_height / 2)) * (camera_vertical_fov / 2)
            z = self.distance_in_front * math.tan(angle_from_center)

            return z

        
        def detect_can(self, image, can):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_ranges = {
                    'dc': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                    'coke': (np.array([25, 128, 50]), np.array([90, 179, 255])),
                    'sprite': (np.array([55, 40, 180]), np.array([125, 120, 230]))
                }
            target_center = None
            
            if color in color_ranges:
                    lower_color, upper_color = color_ranges[color]
                    mask = cv2.inRange(hsv, lower_color, upper_color)
                    M = cv2.moments(mask)
                    if M['m00'] > 0:
                        self.cx = int(M['m10']/M['m00'])
                        self.cy = int(M['m01']/M['m00'])
                        target_center = (cx, cy)
                        cv2.circle(self.image, (cx, cy), 20, (0,0,255), -1)
                        
            return target_center
        
        def rotate_and_find_object(self):
            rate = rospy.Rate(10)  
            while self.distance_in_front >= 0.25:
                
                if self.object_centered and self.distance_in_front <= 0.25:
                    self.vel_pub.publish(Twist())
                    rospy.loginfo("Object approached, stopping")
                    break  # Exit the loop

                # if we detect the object, center and move towards it
                elif self.detected_object_center:
                    err = self.detected_object_center[0] - self.image_width / 2
                    angular_speed = -float(err) / 1000  

                    # prop control
                    if abs(err) > self.image_width * 0.05:  
                        linear_speed = 0  
                    else:
                        
                        linear_speed = 0.1

                    
                    vel_msg = Twist()
                    vel_msg.angular.z = angular_speed
                    vel_msg.linear.x = linear_speed
                    self.vel_pub.publish(vel_msg)

                else:
                    # search for object
                    vel_msg = Twist()
                    vel_msg.angular.z = 0.1 
                    self.vel_pub.publish(vel_msg)

                rate.sleep()  

            self.vel_pub.publish(Twist())
            rospy.loginfo("Exiting rotate_and_find_object")

        def pick_up(self):
        # complete the pickup sequence
            gripper_joint_open = [0.01, 0.01]
            self.move_group_gripper.go(gripper_joint_open, wait=True)
            rospy.sleep(1)
            self.move_group_gripper.stop()
            rospy.sleep(1)
            x = 0.25
            z = self.calculate_z(self.cx, self.cy, self.distance_in_front)
            angle1, angle2 = self.calculate_angles(x, z)
            pickup_angles = [angle1, angle2, 0, 0]
            while not self.move_group_arm.go(pickup_angles, wait=True):
                rospy.logerr("Pick up motion failed at init")
                rospy.sleep(1)
                
            rospy.sleep(1)
            self.move_group_arm.stop()
              
            gripper_joint_closed = [-0.01, -0.01]  
            # Close the gripper to grasp the object
            rospy.sleep(10)
            if not self.move_group_gripper.go(gripper_joint_closed, wait=True):
                rospy.logerr("Gripper close motion failed at pickup")
                return False
            rospy.sleep(1)
            self.move_group_gripper.stop()
            rospy.sleep(1)
        
        def run(self):
            
            rate = rospy.Rate(30)  # Set an appropriate rate (e.g., 30Hz)
            
            
            while not rospy.is_shutdown():
                # Always check for new images and update the display
                if self.new_image_flag:
                    cv2.imshow("window", self.latest_image)
                    cv2.waitKey(3)
                    self.new_image_flag = False
                    
                    rate.sleep()
        
        
if __name__ == '__main__':
    rospy.init_node('pick_up_object')
    pickup_putdown = FindAndPickupCan()
    pickup_putdown.run()

            
        