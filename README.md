# final_project_jayden_eddie_jason_danny


# Team Name
## The Happy Hydrators
# Members
## Danny Lingen, Jason Starbuck, Eddy Tay, and Jayden Francis
# Git Repo:
## https://github.com/Intro-Robotics-UChicago-Fall-2023/drink_fetcher

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

