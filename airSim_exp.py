from spinup import ppo_pytorch as ppo
import torch 
import gym
import gym_airsim


def envFunc():
	env = gym.make('airsim_gym-v0')
	return env

# Setup the environment function and hyperparameters
env_fn = envFunc
ac_kwargs = dict(hidden_sizes=[64, 64])
logger_kwargs = dict(output_dir='/home/faisal/Documents/airsim_exp_results_seed_0',exp_name='airsim_fixed_goal')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 1234 ,max_ep_len=400, steps_per_epoch=4000, epochs=375, logger_kwargs=logger_kwargs)
logger_kwargs = dict(output_dir='/home/isra/Documents/airsim_exp_results_seed_5',exp_name='airsim_fixed_goal')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 12345 ,max_ep_len=400, steps_per_epoch=4000, epochs=375, logger_kwargs=logger_kwargs)
logger_kwargs = dict(output_dir='/home/isra/Documents/airsim_exp_results_seed_10',exp_name='airsim_fixed_goal')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 123456 ,max_ep_len=400, steps_per_epoch=4000, epochs=375, logger_kwargs=logger_kwargs)


