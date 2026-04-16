import gym 
import numpy as np
import pybullet as p
import pybullet_data
import time

class NavigationEnv(gym.Env): #creating a custom RL environment gym.Env is base for all RL environments 
    def __init__(self, render=False): #render = false means that no graphics
        super(NavigationEnv, self).__init__() #initializes the Gym framwork internally
        #connect to pybullet 
        self.render = render 
        if self.render: 
            self.client = p.connect(p.GUI) #show simualtion window (debugging)

        else:
            self.client = p.connect(p.DIRECT) #hidden fast simulation (trainig)


        #setting simulation settings 
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #loads built in assetts( planes and robots)
        p.setGravity(0, 0, -9.8)    #sets gravity

        #setting action
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), #-1 full revrerse/left turn 
            high=np.array([1.0,1.0]), #1 full forward/right turn 
            dtype=np.float32
        )

        #observation space
        self.observation_space = gym.spaces.Box(
            low = -10, 
            high = 10, 
            shape= (4,), 
            dtype = np.float32
        )

        #[robot_x, robot_y, goal_x, goal_y]


        #internal variables
        self.robot = None
        self.goal = None
        self.steps = 0

    def reset(self):
        #reset world + robot
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        #return initial observation
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0,0,0.1])

        #set goal 
        self.goal = np.random.uniform(-2,2, size=2)

        #reset step counter
        self.steps = 0

        return self._get_obs()


    def step(self, action):
        self.steps += 1
        forward, turn = action #returns forward speed and turn speed
        #apply action 
        p.resetBaseVelocity(
            self.robot, 
            linearVelocity = [forward, 0,0], 
            angularVelocity = [0, 0, turn]
        )
        #simulate physic 
        p.stepSimulation()
        if self.render: 
            time.sleep(1/240)

        obs = self._get_obs()
        #compute reward
        reward = self._compute_reward()
        #check done condition 
        done = self._check_done()
        #return everything RL needs 

        return obs, reward, done, {}
    

    def _get_obs(self): 

        #get robot position
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)

        #extract x and y 
        robot_xy = np.array(robot_pos[:2])

        return np.concatenate([robot_xy, self.goal])
    

    def _compute_reward(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(robot_pos[:2])  

        distance = np.linalg.norm(robot_xy - self.goal)


        reward = -distance
        
        if distance < 0.2: 
            reward += 10

        return reward


    def _check_done(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(robot_pos[:2])

        distance = np.linalg.norm(robot_xy - self.goal)

        if distance < 0.2: 
            return True

        if self.steps > 1000: #exceed time limit
            return True
        
        return False