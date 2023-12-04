# final_project_jayden_eddie_jason_danny


# Team Name
## The Happy Hydrators
# Members
## Danny Lingen, Jason Starbuck, Eddy Tay, and Jayden Francis

## Project Description: 
Our team is passionate about enhancing the convenience and efficiency of daily tasks, and we understand the widespread desire for easy access to favorite beverages without disrupting leisure time. To address this, we are excited to develop an innovative robotic solution designed to bring drinks directly to you while you relax on your couch. This solution harnesses the power of Gesture Recognition, Machine Learning (ML)/Training Data Sets, and Inverse Kinematics. Inverse Kinematics enables the robot's arm to reach and grab beverages from any height, ensuring adaptability to different scenarios. Gesture Recognition is incorporated to allow users to indicate their beverage choice and signal where they want it delivered. Additionally, ML and Training Sets are employed to equip the robot with the ability to discern various hand gestures and recognize different beverage types. We aim to create a seamless, user-friendly experience that integrates advanced technology into the comfort of your home.

## System Architecture: 

The major components of our project involve gesture recognition and inverse kinematics, which were developed in tandem with common algorithms we have seen elsewhere this quarter like color recognition, proportional control, state management, and callback functions.

We will highlight the major components individually:

### Inverse Kinematics:
developing the code to allow the robot to recognize and retrieve the can at any arbitrary (reachable) height involved the following steps.

1. Object recognition: We did this through color recognition by using the `image_callback()` function to draw a bounding rectangle around the object with the most contours within the specified color range, we then use this to find the center of the object and position the robot appropriately in `rotate_and_find_object()`.
2. Vertical and horizontal offset determination: The horizontal offset was calculated by simply using the lidar, by telling the robot to stop at a fixed distance in front of the object (0.18 m). The vertical offset was a bit more complicated. We calculated this value with the `calculate_vertical_offset()` function by calculating the pixel offset between the center of the can and the center of the image, by combining this with our knowledge of the horizontal distance (using LIDAR) and taking the angle from the center using the cameras vertical FOV, we were able to consistantly get accurate readings for our vertical offset (z-value)
3. Calculating the appropriate angles: This was done using the `calculate_angles()` function, we directly adapted the equations we learned about in class for a 2 DOF joint system
4. Converting from the robots coordinate system to ours: Before using the angles in `pick_up()`, we convert them from the real world coordinate system to the robots coordinate system by multiplying angle1 by -1 and subtracting 90 degrees (pi/2) from angle2. 

## ROS Node Diagram:

## Execution: Describe how to run your code, e.g., step-by-step instructions on what commands to run in each terminal window to execute your project code.

## Challenges, Future Work, and Takeaways: These should take a similar form and structure to how you approached these in the previous projects (1 paragraph each for the challenges and future work and a few bullet points for takeaways)

### Challenges

Developing a robotic system for beverage delivery, leveraging machine learning, gesture recognition, and inverse kinematics, encounters several notable challenges. A primary difficulty lies in harmonizing the software components—comprising machine learning models, gesture recognition algorithms, and inverse kinematics—with physical elements like robotic arms, sensors, and cameras. Ensuring that the physical hardware effectively executes sophisticated software algorithms is essential.

In particular, gesture and recognition posed significant obstacles. Despite achieving high accuracy in controlled settings, these models struggled to perform effectively in real-world environments. This discrepancy might be attributed to factors like inadequate camera resolution. Moreover, converting theoretical arm angles into practical, real-world movements added layers of complexity, presenting further challenges in applying inverse kinematics to the robot's arm. These issues underscore the intricate task of translating advanced computational models into reliable, physical robotic actions.

### Future Work

Future work for the Happy Hydrators includes several avenues that we can explore to enhance the system's performance and utility. The first is to improve the sensor integration. Upgrading the camera or integrating additional sensors could enhance the system's ability to perceive and interact with its environment accurately. This may include using higher-resolution cameras, depth sensors, or LIDAR to improve object recognition and spatial awareness.
The second is a robust ML Algorithm. Refining the ML algorithms to handle real-world variability better and training the system with a more diverse data set can increase its accuracy and reliability in different lighting conditions and environments. The current design struggles with different lighting conditions. We suspect this may be due to camera quality. The third is to Enhance the Mechanical Design of the robotic arm. Improving the robotic arm's design for more precise and stable movements can help in accurately picking and placing objects, especially in varied and dynamic home environments. The current design may overshoot or undershoot the beverage can.

### Takeaways
In the takeaway from this project, we discovered a significant gap between our theoretical model and practical implementation, particularly concerning the performance of the machine learning (ML) and training models. While our ML models showed impressive accuracy in controlled environments—achieving 98% accuracy in recognizing hand gestures and 91% in identifying cans—these results did not translate seamlessly into real-world applications with the TurtleBot's camera system. The robot became finicky and often failed to function as expected in practical scenarios. This discrepancy highlighted a crucial disconnect between the code we developed and its integration with the physical machine. It was a valuable lesson in the complexities of translating theoretical accuracy into real-world reliability, emphasizing the need for more robust testing and calibration in diverse, real-life conditions.


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
Not requesting a budget 

## Risks (2-3 sentences): What do you see as the largest risks to your success in this project?
The most significant risks that could impact our project's success include technical issues with the Turtlebot's camera system, such as malfunctions, excessive lag, or insufficient resolution, which could hinder effective hand gesture recognition. Additionally, potential connectivity issues or inadequate sound differentiation with the laptop microphones may impair our robot's responsiveness to voice commands. Moreover, the availability of Turtlebots and the time-intensive nature of physical testing pose considerable challenges, as hardware malfunctions or limited access could drastically reduce our testing opportunities, potentially preventing us from achieving our desired outcome.

