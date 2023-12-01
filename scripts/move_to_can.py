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
            # robot arm inits
                self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
                self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

                # image callback
                self.image_sub = rospy.Subscriber('camera/rgb/image_raw',
                        Image, self.image_callback)
                self.bridge = cv_bridge.CvBridge()
                # for ar tags
                self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
                # subscribe to the lidar scan from the robot
                self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
                # vel subscriber
                self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
                # relevant variables, some of these will need to be reset when the robotn begins
                # a new action sequence
                self.arm_len_1 = 0.313
                self.arm_len_2 = 0.051
                self.cx = None
                self.cy = None
                self.can_to_pick = 'sprite'
                self.distance_in_front = float('inf')
                self.latest_image = None
                self.new_image_flag = False
                self.z = None
                self.detected_object_center = None
        
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
                #print(self.distance_in_front)
         
                
        def get_lidar_distance(self):
            return self.distance_in_front
                
        def image_callback(self, msg):
            self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            self.image_width = self.image.shape[1]
            self.image_height = self.image.shape[0]
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.corners, self.ids, self.rejected_points = cv2.aruco.detectMarkers(grayscale_image, self.aruco_dict)
            if self.ids is not None:  
                        cv2.aruco.drawDetectedMarkers(self.image, self.corners, self.ids)
    
            
            # color ranges for the different cans (only sprite is accurate atm, but should be able to copy from q-learning)
            color_ranges = {
                'dc': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                'coke': (np.array([145, 75, 75]), np.array([175, 255, 255])),
                'sprite': (np.array([35, 100, 100]), np.array([85, 200, 230]))
            }
                
            if self.can_to_pick in color_ranges:
                    lower_color, upper_color = color_ranges[self.can_to_pick]
                    mask = cv2.inRange(hsv, lower_color, upper_color)
                    # Find contours of the can
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # If contours are found, assume the largest one is the can
                    if contours:
                            c = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            
                            # Draw a rectangle around the detected can
                            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Calculate the Z value (vertical distance from the camera to the can)
                            KNOWN_CAN_HEIGHT = 0.12
                            CAMERA_VERTICAL_FOV = math.radians(48.8) 
                            
                            if not self.z and self.distance_in_front < 0.5:# Calculate Z using the height of the can in the image
                               
                                self.z = self.calculate_vertical_offset(y + h // 2, self.image_height, CAMERA_VERTICAL_FOV, self.distance_in_front)
                                self.z = self.z - self.z * 0.4
                                print("z = ", self.z)
                            
                            # Update the detected object center
                            self.detected_object_center = (x + w // 2, y + h // )
                
            self.latest_image = self.image
            self.new_image_flag = True
        
        def calculate_angles(self, x, z):
            """ calculates the angles q1 and q1 given x (lidar) and z (from vertical offset)"""
            print("calculating with x:", x, "z: ", z)
            r = math.sqrt(x**2 + z**2)

            cos_q2 = (r ** 2 - self.arm_len_1**2 - self.arm_len_2**2) / (2 * self.arm_len_1 * self.arm_len_2)
            print("cos_q2: ", cos_q2)
            if cos_q2 > 1:
                cos_q2 = 1
            q2 = math.acos(cos_q2)
            
        
            q1 = math.atan2(z, x) - math.atan2(self.arm_len_2 * math.sin(q2), self.arm_len_1 + self.arm_len_2 * math.cos(q2))

            return q1, q2

        
        def calculate_vertical_offset(self, can_center_y, image_height, camera_vertical_fov, distance_to_can):
            """ calculates z based on where the can is in the image + distance to the can (held constant))"""
            # Calculate the proportion of the image height from the center to the can's center
            proportion_from_center = (can_center_y - (image_height / 2)) / (image_height / 2)

            # Calculate the angle from the center to the can's center
            angle_from_center = proportion_from_center * (camera_vertical_fov / 2)

            # Calculate the vertical offset using the angle and the distance to the can
            vertical_offset = distance_to_can * math.tan(angle_from_center)
            # multiply by -1 to convert to robot coord system
            return -1 * vertical_offset

        
        def rotate_and_find_object(self):
            print("entered rotate function")
            rate = rospy.Rate(10)
            while self.distance_in_front >= 0.18:
                if self.distance_in_front <= 0.18:
                        break
                # If we detect the object, center and move towards it
                if self.detected_object_center:
                    err = self.detected_object_center[0] - self.image_width / 2
                    angular_speed = -float(err) / 1000

                    # Proportional control
                    if abs(err) > self.image_width * 0.05:
                        linear_speed = 0
                    else:
                        linear_speed = 0.1

                    vel_msg = Twist()
                    vel_msg.angular.z = angular_speed
                    vel_msg.linear.x = linear_speed
                    self.vel_pub.publish(vel_msg)
                else:
                    # Search for object
                    vel_msg = Twist()
                    vel_msg.angular.z = -0.3
                    self.vel_pub.publish(vel_msg)

                rate.sleep()

           
            self.vel_pub.publish(Twist())
            rospy.loginfo("Object approached, stopping")

            self.vel_pub.publish(Twist())
            rospy.loginfo("Exiting rotate_and_find_object")

        def pick_up(self):
        # complete the pickup sequence
            gripper_joint_open = [0.01, 0.01]
            self.move_group_gripper.go(gripper_joint_open, wait=True)
            rospy.sleep(1)
            self.move_group_gripper.stop()
            rospy.sleep(1)
            x = 0.31
            angle1, angle2 = self.calculate_angles(x, self.z)
            # convert to robot coordinate system
            angle1 = -1 * angle1
            angle2 = angle2 - (math.pi / 2)
            print("angles", angle1, angle2)
            pickup_angles = [0, angle1, angle2, 0]

            while not self.move_group_arm.go(pickup_angles, wait=True):
            
                rospy.logerr("Pick up motion failed at init")
                rospy.sleep(1)
                
            rospy.sleep(1)
            self.move_group_arm.stop()
              
            gripper_joint_closed = [-0.01, -0.01]  
            # Close the gripper to grasp the object
            rospy.sleep(8)
            if not self.move_group_gripper.go(gripper_joint_closed, wait=True):
                rospy.logerr("Gripper close motion failed at pickup")
                return False
            rospy.sleep(1)
            self.move_group_gripper.stop()
            rospy.sleep(1)
            carry_joint_values = [0, -0.819, -0.258, 0.098]

            # Move the arm to the carry position
            rospy.sleep(1)
            while not self.move_group_arm.go(carry_joint_values, wait=True):
                rospy.logerr("motion failed at pickup")
                rospy.sleep(1)
            rospy.sleep(5)
            self.move_group_arm.stop()
        

        def back_up(self):
            vel_msg = Twist()
            
            vel_msg.linear.x = -0.2
            self.vel_pub.publish(vel_msg)
            rospy.sleep(4)
            self.vel_pub.publish(Twist())
        
        def look_for_tag(self, target_id):
            rate = rospy.Rate(5)
            if self.ids is not None:
                ids_flattened = self.ids.flatten()
            while self.ids is None or target_id not in self.ids.flatten():
                
                vel_msg = Twist()
                vel_msg.angular.z = 0.2
                self.vel_pub.publish(vel_msg)
                rate.sleep()
                if self.ids is not None:
                    ids_flattened = self.ids.flatten()
                    if target_id in ids_flattened:
                        break
            self.vel_pub.publish(Twist())
            rospy.loginfo("tag found ")

            
                
            if target_id in ids_flattened:
                rospy.loginfo("AR tag found, moving to it")
                index = np.where(ids_flattened == target_id)[0][0]
                    
                    
                while self.distance_in_front > 0.3:
                     # calculate the position of the AR tag
                    target_corner = self.corners[index][0]
                    center_x = int((target_corner[0][0] + target_corner[2][0]) / 2)
                    image_center_x = self.image.shape[1] / 2
                    err_x = center_x - image_center_x
                        
                    # prop control
                    vel_msg = Twist()
                    vel_msg.angular.z = -float(err_x) / 1000 
                    vel_msg.linear.x = 0.1  

                    self.vel_pub.publish(vel_msg)
                        
                print("reached goal distance")

                    
                self.vel_pub.publish(Twist())
            
        
        def drop_can_end_sequence(self):
            gripper_joint_open = [0.01, 0.01]
            rospy.sleep(5)
            self.move_group_gripper.go(gripper_joint_open, wait=True)
            rospy.sleep(1)
            self.move_group_gripper.stop()
            rospy.sleep(1)
            home_joint_values = [0, -1, 0.325, 0.7]
            while not self.move_group_arm.go(home_joint_values, wait=True):
                rospy.logerr("Pick up motion failed at home pose")
                rospy.sleep(1)


        
        def execute_action(self):
            rospy.loginfo("looking for can")
            rospy.sleep(5)
            self.rotate_and_find_object()
            rospy.loginfo("pickup can")
            self.pick_up()
            rospy.loginfo("backing up")
            self.back_up()
            rospy.loginfo("looking for tag")
            self.look_for_tag(3)
            self.drop_can_end_sequence()
        
        def start_action_sequence(self):
            # This method will start the action sequence in a new thread
            action_thread = threading.Thread(target=self.execute_action)
            action_thread.start()
        
        def run(self):
            self.start_action_sequence()
            rate = rospy.Rate(30) 
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

            
        
