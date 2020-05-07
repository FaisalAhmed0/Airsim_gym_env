import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
import numpy as np
import airsim
import math
''' 
        Observation: 
        Type: Box(12)
        Num Observation                Min            Max
        0   Quad postion x            -Inf            Inf
        1   Quad postion y            -Inf            Inf
        2   Quad postion z            -Inf            Inf
        3   Quad Velocity x           -Inf            Inf
        4   Quad Velocity y           -Inf            Inf
        5   Quad Velocity z           -Inf            Inf
        6   Quad Orientation roll     -pi/2           pi/2
        7   Quad Orientation pitch    -pi/2           pi/2
        8   Quad Orientation yaw      -pi/2           pi/2
        9  Quad Angular_velocity x    -Inf            Inf
        10  Quad Angular_velocity y   -Inf            Inf
        11  Quad Angular_velocity z   -Inf            Inf
        
    Actions:
        Type: Box(4)                   
        Num Action                     Min            Max   
        0   roll rate                  -1             +1
        1   pitch rate                 -1             +1
        2   yaw rate                   -1             +1
        0   throttle                    0             +1

    Goal positon: (8,5,16)
    This goal in Unity Frame of reference   

    Maximum time steps = 300    
'''
class AirsimEnv(gym.Env):  
    def __init__(self):

        observation_low = -1 * np.array([np.inf,np.inf,np.inf, # positon lower limits
                                        np.inf,np.inf,np.inf,  # linear velocity lower limits
                                        np.pi/2,np.pi/2,np.inf, # orientation lower limits
                                        np.inf,np.inf,np.inf]) # angular velocity lower limits
        observation_high = np.array([np.inf,np.inf,np.inf, # positon upper limits
                                    np.inf,np.inf,np.inf,  # linear velocity upper limits
                                    np.pi/2,np.pi/2,np.inf, # orientation upper limits
                                    np.inf,np.inf,np.inf]) # angular velocity upper limits

        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
        

        actions_low = np.array([-1, -1, -1, 0])
        actions_high = np.array([1, 1, 1, 1])
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)
        
        
        # The goal is specified in unity global coordinate system 
        # then transformed to the quad coordinate system to compute the reward.
        self.goal = np.array([8,5,16])

        self.time_step = 0 
        self.done = False
        self.max_eps_steps = 300

        self.seed()
        
        self.airsimClient = airsim.MultirotorClient()
        self.airsimClient.confirmConnection()
        self.airsimClient.enableApiControl(True)
        self.airsimClient.armDisarm(True)
        self.isReseted  = False
        time.sleep(0.1)

    def step(self, action):
        # Check if the user has reseted the environment and the action is whtih in the bound 
        assert self.isReseted , "Environments should be reseted before taking an action" 
        assert self.action_space.contains(action), "Action is out of bound"
        self.time_step += 1
        # comit the action, 
        self.airsimClient.moveByAngleRatesThrottleAsync(float(action[0]), # roll rate
                                                        float(action[1]), # pitch rate
                                                        float(action[2]), # yaw rate
                                                        float(action[3]), # throttle
                                                        4)   # duration
        time.sleep(0.1)
        quad_state = self.airsimClient.getMultirotorState().kinematics_estimated
        state,state_tup = self._extract_state(quad_state)
        quad_postion = state_tup[0]

        goal_quad_frame = self._r_u_to_q(self.goal)
        # a z position high than 0.7 means the quad is under the Environment plane/terrain
        if quad_postion[2] > 0.7:
            print("Collided")
            self.done = True
            reward = -1000

        if not self.done:
            reward = self._compute_DTG_reward(quad_postion, self.goal_quad_frame)
            l2_dis = self._compute_l2_distance(quad_postion, self.goal_quad_frame)
            # Solved !
            if l2_dis <= 0.5:
                self.done = True
            elif l2_dis >300:
                self.done = True  

        state_dict = self._state_to_dict(state)
        actions_dict = self._action_to_dict(action)
        info = {'time_step':self.time_step, 'state': state_dict,'action' :actions_dict,'reward': reward }

        if self.time_step >= self.max_eps_steps:
            self.done = True
        done = self.done
        return state, reward, done, info

    def reset(self):
        self.airsimClient.reset()
        self.airsimClient.confirmConnection()
        self.airsimClient.enableApiControl(True)
        self.airsimClient.armDisarm(True)
        quad_state = self.airsimClient.getMultirotorState().kinematics_estimated
        state,state_tup = self._extract_state(quad_state)
        time.sleep(0.1)
        self.isReseted = True

        return state

    
    # compute the distance to goal (DTG) reward
    def _compute_DTG_reward(self,quad_postion, goal):
        weights = [1,1,0.5]
        l2_dis = - np.sqrt(( weights[0] * (quad_postion[0] - goal[0])**2) 
                        + (  weights[1] * (quad_postion[1] - goal[1])**2)
                        + (  weights[2] * (quad_postion[2] - goal[2])**2) )
        return l2_dis

    def _extract_state(self, quad_state):
        position_x = quad_state.position.x_val
        position_y = quad_state.position.y_val
        position_z = quad_state.position.z_val
        # print(position_x, position_y, position_z)
        position = np.array([position_x, position_y, position_z ])
        # position = quad_state.position.to_numpy_array()
        linear_velocity = quad_state.linear_velocity.to_numpy_array()
        orientation_quaternions = quad_state.orientation.to_numpy_array()
        orientation_angles,_ = self._conver_quaternoins_to_euler_angles(orientation_quaternions)
        angular_velocity = quad_state.angular_velocity.to_numpy_array()

        state = np.concatenate((position ,linear_velocity, orientation_angles, angular_velocity))

        return state, (position, linear_velocity, orientation_angles, angular_velocity)

    # Convet a state vector to a dictionary fot logging purposes
    def _state_to_dict(self, state):
        state_dict = {'x position': state[0],
                      'y position': state[1],
                      'z position': state[2],
                      'x linear velocity': state[3],
                      'y linear velocity': state[4],
                      'z linear velocity': state[5],
                      'roll orientation' : state[6],
                      'pitch orientation': state[7],
                      'yaw orientation'  : state[8],
                      'x angular velocity': state[9],
                      'y angular velocity': state[10],
                      'z angular velocity': state[11]}
        return state_dict

    # Convet an action vector to a dictionary fot logging purposes
    def _action_to_dict(self, action):
        action_dict = {'roll rate' : action[0],
                       'pitch rate': action[1],
                       'yaw rate'  : action[2],
                       'throttle'  : action[3] }
        return action_dict

    # compute the l2/euclidian distance between two points
    def _compute_l2_distance(self, pos1, pos2):
        l2_dis = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2 )
        return l2_dis

    # Convert a quaternioin values to euler angles to check for 
    def _conver_quaternoins_to_euler_angles(self, q):
        # roll pitch yaw
        euler_angles_rad = np.array([
                        math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)), # roll
                        math.asin(2*(q[0]*q[2] - q[3]*q[1])),                             # pitch
                        math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))  # yae
                        ])
        euler_angles_degrees = euler_angles_rad * (180/np.pi)

        return euler_angles_rad, euler_angles_degrees

    # transform the location from unity frame of reference to the quad frame of reference
    def _r_u_to_q(self, point):
        R = np.array([[0,0,1],[1,0,0],[0,-1,0]])
        p_quad_frame = R.dot(point)
        return p_quad_frame
