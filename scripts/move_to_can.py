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
        
        def calculate_angles(self, x, y):
            q2 = math.acos((x ** 2 + y ** 2 - self.arm_len_1**2 - self.arm_len_2**2) / (2*self.arm_len_1*self.arm_len_2))
            q1 = math.atan(y/x) - math.atan((self.arm_len_2*math.sin(q2))/(self.arm_len_1 + self.arm_len_2*math.cos(q2)))
            return q1, q2

        
        def detect_can(self, image, can):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_ranges = {
                    'dc': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                    'coke': (np.array([25, 128, 50]), np.array([90, 179, 255])),
                    'sprite': (np.array([55, 40, 180]), np.array([125, 120, 230]))
                }
            target_center = None
            self.object_found = False
            if color in color_ranges:
                    lower_color, upper_color = color_ranges[color]
                    mask = cv2.inRange(hsv, lower_color, upper_color)
                    M = cv2.moments(mask)
                    if M['m00'] > 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        target_center = (cx, cy)
                        cv2.circle(self.image, (cx, cy), 20, (0,0,255), -1)
                        self.object_found = True
            return target_center
        
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

            
        