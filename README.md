# final_project_jayden_eddie_jason_danny


# Team Name
## The Happy Hydrators
# Members
## Danny Lingen, Jason Starbuck, Eddy Tay, and Jayden Francis

## Video

[![Happy Hydrators!](https://github.com/Intro-Robotics-UChicago-Fall-2023/final_project_jayden_eddie_jason_danny/blob/main/DALL%C2%B7E%202023-12-05%2023.29.06%20-%20An%20image%20of%20a%20'Beverage%20Bot%2C'%20a%20sleek%20and%20modern%20robot%2C%20delivering%20soda%20to%20a%20group%20of%20excited%20people.%20The%20robot%20has%20a%20futuristic%20design%20and%20is%20holding.jpeg)](https://youtu.be/DOVHarTF_Fo)

## Project Description: 
Our team is passionate about enhancing the convenience and efficiency of daily tasks, and we understand the widespread desire for easy access to favorite beverages without disrupting leisure time. To address this, we are excited to develop an innovative robotic solution designed to bring drinks directly to you while you relax on your couch. This solution harnesses the power of Gesture Recognition, Machine Learning (ML), and Inverse Kinematics (along wih other robotics concepts that we developed and practiced throughout the quarter). Inverse Kinematics enables the robot's arm to reach and grab beverages from any (reachable) height, ensuring adaptability to different scenarios. Gesture Recognition is incorporated to allow users to indicate their beverage choice and signal where they want it delivered (along with where the target for delivery is). Additionally, ML is employed to equip the robot with the ability to discern various hand gestures in order to allow transitions and targets between states. We aim to create a seamless, user-friendly experience that integrates advanced technology into the comfort of your home.

## System Architecture: 

The major components of our project involve gesture recognition and inverse kinematics, which were developed in tandem with common algorithms we have seen elsewhere this quarter, such as color recognition, proportional control, state management, and callback functions.

We will highlight the major components individually:

### Inverse Kinematics:
The following steps involved in developing the code to allow the robot to recognize and retrieve the can at any arbitrary (reachable) height.

1. Object recognition: We did this through color recognition by using the `image_callback()` function to draw a bounding rectangle around the object with the most contours within the specified color range. We then used this to find the object's center and position the robot appropriately in `rotate_and_find_object().`
2. Vertical and horizontal offset determination: The horizontal offset was calculated by simply using the lidar by telling the robot to stop at a fixed distance in front of the object (0.18 m). The vertical offset was a bit more complicated. We calculated this value with the `calculate_vertical_offset()` function by calculating the pixel offset between the center of the can and the center of the image by combining this with our knowledge of the horizontal distance (using LIDAR) and taking the angle from the center using the cameras vertical FOV, we were able to consistently get accurate readings for our vertical offset (z-value)
3. Calculating the appropriate angles: This was done using the `calculate_angles()` function; we directly adapted the equations we learned about in class for a 2 DOF joint system
4. Converting from the robot coordinate system to ours: Before using the angles in `pick_up(),` we convert them from the real-world coordinate system to the robot coordinate system by multiplying angle1 by -1 and subtracting 90 degrees (pi/2) from angle2.

### Computer Vision + Machine Learning
The development of this section of the code code enables the robot to recognize specific gestures, which are then linked to predefined actions (getting a specific can). The training of this model utilized a dataset available at Kaggle, supplemented with images captured by the suboptimal camera of the TurtleBot.

The MediaPipe framework was integrated to enhance accuracy. This framework detects hand landmarks and overlays these coordinates onto the existing dataset. This approach significantly improved accuracy, particularly with the lower-quality camera on the TurtleBot.

The images were resized to 150x150 pixels and fed into the neural network model. The model was trained over 100 epochs with batch sizes of 256. Following the completion of training, the model was saved for deployment on the robot.

After the model was developed, we incorperated it into our node with the `process_frames()` function which is called before our robot enters the sequence to pick up and retrieve a can. This function determines which can the robot will target after a gesture is detected for 5 seconds. It also provides a helpful output to the viewing window which allows us to see the gesture our robot is "seeing".

We also incorperate our machine learning model in the `move_to_gesture()` function, which finds the center of a hand detected via the ML model and uses proportional control to move towards the hand so it can drop it off and complete its action.


#### Model on the robot 

1. Initially, we initialize the model and load the associated class names when the system boots up.
2. Following this, we start preprocessing images to align them with the characteristics of the data used in training the model.
3. We implement structural detection in the images, convert them to black and white, and then input them into the model.
4. The model subsequently generates confidence percentages for each class. We select the class with the highest confidence percentage, convert its identifier to an integer, and return this integer corresponding to the identified class.

The machine learning model can be found in the `train.py` file in the scripts folder.


## ROS Node Diagram:
<img width="744" alt="Screenshot 2023-12-04 at 9 20 43 PM" src="https://github.com/Intro-Robotics-UChicago-Fall-2023/final_project_jayden_eddie_jason_danny/assets/102828818/c4869901-b814-46f5-9335-31e91b4459b5">


## Execution: Describe how to run your code, e.g., step-by-step instructions on what commands to run in each terminal window to execute your project code.
In order to run our project can can clone this git repository and run the following commands, each in a different terminal window.

Window 1: enter catkin_ws and run roscore
Window 2: ssh into a turtlebot, connect it to your IP address and run bringup
Window 3: ssh into the same turtlebot, connect it to your IP address, and run bringup_cam
Window 4: to correctly compress images run: rosrun image_transport republish compressed in:=raspicam_node/image raw out:=camera/rgb/image_raw
Window 5: run roslaunch turtlebot3_manipulation_bringup turtlebot3_manipulation_bringup.launch
Window 6: run roslaunch turtlebot3_manipulation_moveit_config move_group.launch

Finally, you are ready to run our program, our main node is located in the move_to_can.py file, however, this node requires access to the mp_hand_gesture directory which contains our machine learning model.

In order for our script to run correctly, you MUST execute the command rosrun final_project move_to_can.py in the final_project_jayden_eddie_jason_danny directory in order to ensure that all the dependencies are taken care of.

## Challenges, Future Work, and Takeaways: These should take a similar form and structure to how you approached these in the previous projects (1 paragraph each for the challenges and future work and a few bullet points for takeaways)

## Execution: Describe how to run your code, e.g., step-by-step instructions on what commands to run in each terminal window to execute your project code.
In order to run our project can can clone this git repository and run the following commands, each in a different terminal window.

Window 1: enter catkin_ws and run roscore
Window 2: ssh into a turtlebot, connect it to your IP address and run bringup
Window 3: ssh into the same turtlebot, connect it to your IP address, and run bringup_cam
Window 4: to correctly compress images run: rosrun image_transport republish compressed in:=raspicam_node/image raw out:=camera/rgb/image_raw
Window 5: run roslaunch turtlebot3_manipulation_bringup turtlebot3_manipulation_bringup.launch
Window 6: run roslaunch turtlebot3_manipulation_moveit_config move_group.launch

Finally, you are ready to run our program, our main node is located in the move_to_can.py file, however, this node requires access to the mp_hand_gesture directory which contains our machine learning model.

In order for our script to run correctly, you MUST execute the command rosrun final_project move_to_can.py in the final_project_jayden_eddie_jason_danny directory in order to ensure that all the dependencies are taken care of.

## Challenges, Future Work, and Takeaways: These should take a similar form and structure to how you approached these in the previous projects (1 paragraph each for the challenges and future work and a few bullet points for takeaways)

### Challenges

This project was a testimate to our ability to adapt, develop, and incorperate all of the lessons and knowledge we attained over the course of this fantastic introduction to robotics. Our first challenge came with the development of our inverse kinematics algorithm. While adapting the equations we learned about in class to code was a simple task, we were far from a solution after our first iteration. We first had to accurately measure the vertical offset of the can (which we did using the size of the can in pixels, and a little more trig), and then we had to convert from the real world coordinate system to the coordinate system of the robot, which was something we only realized after hours of trying and failing to obtain a correct result.

The machine learning component also had significant challenges associated with it. Despite achieving high accuracy in controlled settings, the model struggled to perform effectively in real-world environments. This discrepancy might be attributed to factors like inadequate camera resolution, but it required us to retrain the model with some of our own pictures from the robot camera to achieve a more accurate model.

Finally, connecting all of the pieces was a challenge in itself, ensuring that everything ran smoothly, and the edge cases with color recognition and arm positions, and state management was all handled appropriately so we could be confident in our demo took many additional hours of debugging and testing our framework.

### Future Work

Future work for the Happy Hydrators includes several avenues that we could explore to enhance the system's performance and utility. The first is to improve the sensor integration. Upgrading the camera or integrating additional sensors could enhance the system's ability to perceive and interact with its environment accurately. This may include using higher-resolution cameras, depth sensors, or LIDAR to improve object recognition and spatial awareness.

We also would have liked to use a machine learning algorithm for the cans, in addition to the gestures. We did attempt to do this with a pretrained model that we found on roboflow, but we could not find anything that had a high enough accuracy for our purposes. Additionally, we tried training our own model, but because of the crushed nature of our cans, we were not able to find enough traning images to get an accurate model for our specific use cans with the 3 cans we had access too. We do believe this is something we could have done if we found clean training data (or had smaller cans, for instance).


### Takeaways

This was a fantastic project to wrap up what has been an extremely enjoyable and informative quarter for all of us. We were able to successfully incorperate new robotics algorithms (ML/gesture recognition, inverse kinematics), and build upon our existing knowledge about robotics that we have developed throughout the quarter (color recognition, proportional control, state management, etc). 

In the end, we are all extremely happy not only with what we created, but also with the process that allowed us to develop our final project this quarter. Learning about important robotics concepts and algorithms and then having the freedom to explore, to fail, and to suffer through a freeform robotics project has made us all better programmers, thinkers, and communicators. We beleive this format helped to develop our own understanding a ton and (speaking personally) my excitement about robotics has only risen despite all the hours of debugging and testing this quarter required.

This was also a great project to really allow us to discover how to effectively work as a team, and divvying up work appropriately is a takeaway I can see being invaluable in professional as well as academic settings.

As a final thought, we would like to sincerely thank the teaching staff of intro to robotics, we all had an amazing time in this class and learned more than we would have thought possible in a 10 week quarter, we are very greatful for your help and guidance.











[below is our project overview from the Project Proposal, which we kept for posterity]













## Motivation (2-3 sentences): Why is your team interested in pursuing this project?
Our team is driven by the desire to enhance convenience and efficiency in everyday tasks. We recognize the universal appeal of effortlessly accessing your favorite beverages without interrupting leisure time, so we're excited to develop a robotic solution that delivers this comfort right to your couch.

## Main Components (1 paragraph): What are the main topics covered in your project? What robotics algorithm(s) will you be exploring in your project? Are there any other main components to your project?
Our project encompasses several critical topics within robotics and artificial intelligence. We will delve into reinforcement learning, specifically employing Q-learning for decision-making processes, such as identifying the correct beverage to retrieve. Additionally, we will implement Markov Decision Processes (MDPs) to devise optimal pathfinding strategies that navigate around obstacles. A notable feature of our project is the integration of voice recognition to facilitate user-robot interaction. 
We plan to go beyond the classroom concepts by developing a gesture recognition model, enabling non-verbal communication with the robot for drink selection. A unique challenge we aim to tackle is programming the robot to place the drink to target accurately: the user's hand. 

## Final Product (1 bullet point per deliverable level, 2-3 sentences per bullet point): For each of the following, what would success look like. 
### Minimum Viable Product (MVP): 
Success for our MVP would be characterized by the robot's ability to accurately recognize a specific hand gesture, allowing it to initiate a straightforward mission to locate and retrieve a single beverage can. The robot will then deliver the can by placing it at a predetermined location, such as directly in front of the user, without the need to differentiate between multiple options or adapt to a moving target.

### Solid End Result: 
For a successful result, the robot would be capable of identifying the user's hand, maneuvering through a complex environment to a designated cooler, and selecting the correct beverage can based on learned preferences. On its return journey, the robot would respond to vocal prompts to hasten its pace and adeptly place the can into the user's potentially moving hand.

### Stretch Goal: 
The stretch goal would see the robot's functionality significantly enhanced, with speech recognition replacing hand signals for beverage selection. In addition, the robot would dynamically alter its pathfinding in real-time, executing new navigational maneuvers as instructed verbally by the user, showcasing a remarkable level of adaptive, interactive robotics.

## Timeline: Provide a rough draft timeline for the major milestones of your project.
## Our rough draft timeline for our project's major milestones:
### By November 15 (Wednesday): 
Complete the development and testing of the hand gesture recognition algorithm, ensuring it can reliably interpret user gestures for initiating the beverage retrieval process.
### By November 17 (Friday): 
Initiate the development of the pathfinding algorithm to navigate around obstacles efficiently. Finalize the robot's pick-up and hand-placement movements, enabling it to handle the beverage can with precision.
### By November 28 (Tuesday): 
Aim to have the pathfinding algorithm fully operational. Begin integrating voice level detection to adjust the robot's speed in response to the user's vocal commands. Also, start refining the moving hand placement algorithm to accurately place the beverage in the user's hand.
### By December 1 (Friday): 
Focus on debugging and fine-tuning all systems to ensure seamless operation. Complete documentation, including comprehensive comments within the code for clarity and a detailed README file to guide users and developers in understanding and possibly contributing to the project.

## Resources: Describe what materials you plan to use (e.g., additional sensors, objects for the robot to pick up, a maze environment) as well as details about the turtlebot(s) that you plan to use (how many turtlebots, whether or not you want the OpenManipulator arm(s) attached). 
### We plan to utilize the following resources:
### Microphone Sensor: 
The primary sensor required will be a microphone to facilitate noise recognition. We anticipate using the built-in microphone of a laptop to capture voice commands and detect changes in voice levels that will influence the robot's speed.
### Beverages and Maze Environment: 
We will provide mock beverage cans that the robot will be trained to identify and retrieve. These will be placed within a maze-like environment, which will serve as the testbed for the robot's navigation and pathfinding capabilities.
### Turtlebot with Manipulator Arm: 
A single Turtlebot equipped with an OpenManipulator arm will be used. The arm is essential for the physical tasks of picking up and placing the beverages.

### [Optional] Budget: If you would like additional materials for your project, please develop a budget with estimated costs and links for the components you are requesting.
Not requesting a budget. 

## Risks (2-3 sentences): What do you see as the largest risks to your success in this project?
The most significant risks that could impact our project's success include technical issues with the Turtlebot's camera system, such as malfunctions, excessive lag, or insufficient resolution, which could hinder effective hand gesture recognition. Additionally, potential connectivity issues or inadequate sound differentiation with the laptop microphones may impair our robot's responsiveness to voice commands. Moreover, the availability of Turtlebots and the time-intensive nature of physical testing pose considerable challenges, as hardware malfunctions or limited access could drastically reduce our testing opportunities, potentially preventing us from achieving our desired outcome.

