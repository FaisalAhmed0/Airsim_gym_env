import gym
import gym_airsim
import torch
import numpy as np
import argparse
import ast

parser = argparse.ArgumentParser()

# parser.add_argument("--goal", type=str, required=True)
# args = parser.parse_args()

# goal = ast.literal_eval(args.goal)
# print(f"Goal is {goal} type {type(goal)}")


path_to_model = '/home/isra/Documents/airsim_exp_results_random_seed_0/pyt_save/model.pt'
actor_critic = torch.load(path_to_model)
print(f"Actor critic object {actor_critic}")

# transform the location from unity frame of reference to the quad frame of reference
# def _r_u_to_q(point):
#     R = np.array([[0,0,1],[1,0,0],[0,-1,0]])
#     p_quad_frame = R.dot(point)
#     return p_quad_frame

# set up the envionment
def Env(goal):
	env = gym.make('airsim_gym-v0')
	# env.setGoal(goal)
	return env

if __name__ == '__main__':
	policy = actor_critic.act
	print(f"policy object {policy}")
	num_trails = 100
	# goal = _r_u_to_q(goal)
	env = gym.make('airsim_gym-v0')
	success_times = 0
	for i in range(num_trails):
		state = env.reset()
		done = False
		total_reward = 0
		while not done:
			action = policy(torch.as_tensor(state, dtype=torch.float32))
			state, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				break
		if total_reward > 3000:
			success_times +=1
		print(f"Performance {total_reward}")
	print(f"Success rate {success_times/num_trails}")
