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
                                        np.inf,np.inf,np.inf, # angular velocity lower limits
                                        np.inf, np.inf]) # goal x,y,z position in local frame  
        observation_high = np.array([np.inf,np.inf,np.inf, # positon upper limits
                                    np.inf,np.inf,np.inf,  # linear velocity upper limits
                                    np.pi/2,np.pi/2,np.inf, # orientation upper limits
                                    np.inf,np.inf,np.inf, # angular velocity upper limits
                                    np.inf, np.inf]) # goal x,y,z position in local frame

        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
        

        actions_low = np.array([-np.inf, -np.inf, -np.inf])
        actions_high = np.array([np.inf, np.inf, np.inf])
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)
        
        
        # The goal is specified in unity global coordinate system 
        # then transformed to the quad coordinate system to compute the reward.
        # self.goal = np.array([goal[0],goal[1],goal[2]])

        self.time_step = 0 
        self.done = False
        self.max_eps_steps = 1500
        self.max_episode_steps = 1500

        self.seed()
        
        self.airsimClient = airsim.MultirotorClient()
        self.airsimClient.confirmConnection()
        self.airsimClient.enableApiControl(True)
        self.airsimClient.armDisarm(True)
        self.isReseted  = False

        self.state_mean = np.array([0, 0, 1.412217970937490531e-05, 0, 0, 1.647552423179149753e-02, 0, 0, 3.141592653589793116e+00, 0, 0 ,0, 6.5, 3.141])
        self.state_s_k = np.zeros(14)

        self.reward_mean = 0
        self.reward_s_k = 0


        self.k = 0
        # self.goal = [10 * np.random.randn()
        #             , 10 * np.random.randn()
        #             , np.random.uniform(-5,-10)]
        # time.sleep(0.1)

    # def setGoal(self, goal):
    #     # goal = self._r_u_to_q(np.array(goal[0],goal[1],goal[2]))
    #     # goal = [2.5, 4.9, -5]
    #     self.goal = [goal[0], goal[1], goal[2]]

    def step(self, action):
        self.k += 1
        # Check if the user has reseted the environment and the action is whtih in the bound 
        # print(action)
        assert self.isReseted , "Environments should be reseted before taking an action" 
        assert self.action_space.contains(action), "Action is out of bound"
        self.time_step += 1
        # comit the action, 
        # print(f'action0 {float(action[0])}')
        # print(f'action1 {float(action[1])}')
        # print(f'action2 {float(action[2])}')
        # print(f'action3 {float(action[3])}')
        # self.airsimClient.moveByAngleRatesThrottleAsync(float(action[0]), # roll rate
        #                                                 float(action[1]), # pitch rate
        #                                                 float(action[2]), # yaw rate
        #                                                 0.6, # throttle
        #                                                 0.1).join()   # duration
        if not self.done:
            self.airsimClient.moveByAngleRatesZAsync(float(action[0]),
                                                 float(action[1]),
                                                 float(action[2]),
                                                 self.goal[2],
                                                 0.01).join()
        # time.sleep(0.15)
        quad_state = self.airsimClient.getMultirotorState().kinematics_estimated
        state,state_tup = self._extract_state(quad_state)
        quad_postion = state_tup[0]

        goal_quad_frame = self.goal
        reward = self._compute_DTG_reward(state_tup, goal_quad_frame, action)
        # a z position high than 0.7 means the quad is under the Environment plane/terrain
        # TODO: add collision detiction

        # if self.time_step > 5:
        #     collision_info = self.airsimClient.simGetCollisionInfo()
        #     is_collided = collision_info.has_collided
        #     if is_collided:
        #         print("collision detcted")
        #         self.done = True
        #         reward = -1000

        # if (np.abs(state[7]) > 0.6 or np.abs(state[6]) > 0.6 ):
        #     print('extreme angle orientation')
        #     self.done = True
            # reward = -5000

        if quad_postion[2] > 0.7:
            print("Collided")
            self.done = True
            reward = -10000

        if not self.done:
            reward = self._compute_DTG_reward(state_tup, goal_quad_frame, action)
            l2_dis = self._compute_l2_distance(quad_postion, goal_quad_frame)
            # print(f'l2 dis {l2_dis}')
            # Solved !
            if l2_dis <= 2:
                # self.done = True    
                print("Solved!")
                if self.num_reached_goals == 0:
                    reward = 5000
                    print(f"solved at {quad_postion} fot goal number {self.num_reached_goals+1}")
                elif self.num_reached_goals == 1:
                    reward = 10000
                    print(f"solved at {quad_postion} fot goal number {self.num_reached_goals+1}")
                else:
                    reward = 15000
                    print(f"solved at {quad_postion} fot goal number {self.num_reached_goals+1}")
                    self.done = True
                if not self.done:
                    # self.goal_ind += 1
                    self.goal += self.GenerateGoal(range_min=4, range_max=6, init_point=state_tup[0])
                    self.goal = self.GoalsToQuadFrame(self.goal)
                print(f"new goal is {self.goal}")
                self.num_reached_goals += 1
                
            # elif l2_dis <= 0.3:
            #     print('0.3 region')
            #     reward = 1000
            # elif l2_dis <=0.5:
            #     print('print 0.5 region')
            #     reward = 500
            elif l2_dis >100:
                print("too far")
                self.done = True
                reward = -10000
        state_dict = self._state_to_dict(state)
        actions_dict = self._action_to_dict(action)
        info = {'time_step':self.time_step, 'state': state_dict,'action' :actions_dict}

        if self.time_step >= self.max_eps_steps:
            print("over time")
            self.done = True
            # reward = -5000
        done = self.done


        # print(f'state {state}')
        # print(f'reward {reward}')

        # Normalize the state and reward
        self.state_mean_prev = self.state_mean
        self.state_mean = self.state_mean_prev + ((1/self.k)*(state - self.state_mean_prev))
        self.state_s_k += (state - self.state_mean_prev) * (state - self.state_mean)
        if self.k == 2:
            state_std = np.zeros(14)
            state_std += 2e-6
        else:
            state_std = np.sqrt(self.state_s_k/(max(self.k-1,2)))
        state = (state - self.state_mean) / state_std

        self.reward_mean_prev = self.reward_mean
        self.reward_mean = self.reward_mean_prev + ((1/self.k)*(reward - self.reward_mean_prev))
        self.reward_s_k += (reward - self.reward_mean_prev) * (reward - self.reward_mean)
        reward_std = np.sqrt(self.reward_s_k/(self.k-1))

        self.reward_bef += reward
        print(f"reward before norm is {reward} sum {self.reward_bef}")
        reward = (reward ) / reward_std
        self.reward_af += reward
        print(f"reward after norm is {reward} sum {self.reward_af}")



        # print(f"state {state} shape {state.shape}")
        # print(f"mean {self.state_mean} shape {self.state_mean.shape}")
        # print(f"state_std {state_std} shape {state_std.shape}")

        # print(f"norm state {state} shape {state.shape}")
        return state, reward, done, info

    def reset(self):
        self.k += 1
        self.reward_bef = 0
        self.reward_af = 0
        # set the goals
        # goals = [self.GenerateGoal(range_min=5, range_max=8), self.GenerateGoal(range_min=9, range_max=12), self.GenerateGoal(range_min=13, range_max=15)]
        # self.goals = self.GoalsToQuadFrame(goal)
        self.goal = self.GoalsToQuadFrame(self.GenerateGoal(range_min=5, range_max=8))
        self.num_reached_goals = 0
        self.goal_ind = 0
        print(f"goal is {self.goal}")

        self.airsimClient.confirmConnection()
        self.airsimClient.reset()
        self.airsimClient.enableApiControl(True)
        self.airsimClient.armDisarm(True)

        quad_state = self.airsimClient.getMultirotorState().kinematics_estimated
        # self.randomizeGoal()
        state,state_tup = self._extract_state(quad_state)
        self.isReseted = True
        self.done = False
        self.time_step = 0 
        time.sleep(0.1)

        # State and reward Normalization
        self.state_mean_prev = self.state_mean.copy()
        self.state_mean = self.state_mean_prev + ((1/self.k)*(state - self.state_mean_prev))
        self.state_s_k += (state - self.state_mean_prev) * (state - self.state_mean)
        if self.k == 1:
            state_std = np.zeros(14)
            state_std += 1e-6
        else:
            state_std = np.sqrt(self.state_s_k/(max(self.k-1,2)))



        # print(f"state {state} shape {state.shape}")
        # print(f"mean {self.state_mean} shape {self.state_mean.shape}")
        # print(f"state_std {state_std} shape {state_std.shape}")
        # print(f"K {self.k}")


        state = (state - self.state_mean) / state_std
        # print(f"norm state {state} shape {state.shape}")
        # print(f"Goal position {self.goal}")
        return state

    
    # compute the distance to goal (DTG) reward
    def _compute_DTG_reward(self, state_tup, goal, actions):
        quad_postion = state_tup[0]
        quad_vel = state_tup[1]
        angles = state_tup[2]

        weights = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
        l2_dis = (( weights[0] * (quad_postion[0] - goal[0])**2) 
                        + ( weights[1] * (quad_postion[1] - goal[1])**2) +  weights[2] * (quad_postion[2] - goal[2])**2) 
        # vel_pen = weights[3] * quad_vel[0]**2 + weights[4] * quad_vel[1]**2 + weights[5] * quad_vel[2]**2
        # actions_pen = (weights[6] * (actions[0]**2)) + (weights[7] * (actions[1]**2)) +(weights[8] * (actions[2]**2))
        # angles_pen = np.sum(0.1*angles**2)
        # print(f'l2_dis {l2_dis}')
        # print(f'vel_pen {vel_pen}')
        # print(f'actions_pen {actions_pen}')
        return - (l2_dis )#+  vel_pen + actions_pen + angles_pen)

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
        d = self._compute_l2_distance(position, self.goal)
        angle = np.arctan2((self.goal[1] - position_y), (self.goal[0] - position_x))
        # angle *= (180 / np.pi)


        state = np.concatenate((position ,linear_velocity, orientation_angles, angular_velocity))
        state = np.append(state, d)
        state = np.append(state, angle)
        # print(f"state {state}")
        # print(f"angle in degrees is {angle * 180 / np.pi}")
        # import pdb; pdb.set_trace()
        return state, (position, linear_velocity, orientation_angles, angular_velocity, d, angle)

    def randomizeGoal(self):
        self.goal = [np.random.uniform(-30, 30)
                            , np.random.uniform(-30, 30)
                            , np.random.uniform(-5,-15)]

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
                       'throttle'  : 0.6 }
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
        if p_quad_frame[2] > 0:
            p_quad_frame[2] *= -1
        return p_quad_frame

    def GoalsToQuadFrame(self, goals):
        R = np.array([[0,0,1],[1,0,0],[0,-1,0]])

        goals = np.array(goals)
        goals_quad_frame = goals.dot(R)

        goals_quad_frame[2] = np.clip(goals_quad_frame[2], -10, -4)
        return goals_quad_frame

    # def GenerateGoal(self, range_min = 8, range_max = 10):
    #     count = 1
    #     d = np.random.uniform(low=-1,high=1,size=(3,)) * count
    #     dis = np.sqrt(d[0]**2 + d[1]**2+ d[2]**2)
    #     while not (dis >= range_min and dis <= range_max):
    #         d = np.random.uniform(low=-1,high=1,size=(3,)) * count
    #         dis = np.sqrt(d[0]**2 + d[1]**2+ d[2]**2)
    #         if dis < range_min:
    #             count += 1
    #         else:
    #             count -= 1
    #     return d
    def GenerateGoal(self, range_min =5 ,range_max = 10, init_point=[0,0,0]):
        count = 1.
        num_iter = 0
        d = np.random.uniform(low=-1,high=1,size=(3,)) * count
        dis = np.sqrt((d[0]-init_point[0])**2 + (d[1]-init_point[1])**2+ (d[2]-init_point[2])**2)

        while not (dis < range_max and dis > range_min):
            num_iter += 300
            d = np.random.uniform(low=-1,high=1,size=(3,)) * count
            dis = np.sqrt((d[0]-init_point[0])**2 + (d[1]-init_point[1])**2+ (d[2]-init_point[2])**2)

            if dis <= range_min:
                count += 0.1
            elif dis > range_max:
                count -= 0.1
            elif num_iter >= 300:
                break

        # print(f"dis {dis}")
        return d + init_point



