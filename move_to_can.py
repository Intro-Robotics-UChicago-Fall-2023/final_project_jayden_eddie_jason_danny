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
                self.arm_len 2 = 0.274
                
        def image_callback():
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            self.image_width = self.image.shape[1]
            color_ranges = {
                'pepsi': (np.array()),
                'coke': (np.array()),
                'mountain dew': (np.array())
            }
        
        def calculate_angles(x, y):
            q2 = math.acos((x ** 2 + y ** 2 - self.arm_len_1**2 - self.arm_len_2**2) / (2*self.arm_len_1*self.arm_len_2))
            q1 = math.atan(y/x) - math.atan((self.arm_len_2*math.sin(q2))/(self.arm_len_1 + self.arm_len_2*math.cos(q2))

        
        

            
        