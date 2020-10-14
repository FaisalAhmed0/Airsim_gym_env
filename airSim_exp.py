from spinup import ppo_pytorch as ppo
import torch 
import gym
import gym_airsim


# Check this, it may not work 
def envFunc():
	env = gym.make('airsim_gym-v0')
	return env

# Setup the environment function and hyperparameters
env_fn = envFunc
ac_kwargs = dict(hidden_sizes=[64, 64])
logger_kwargs = dict(output_dir='/home/isra/Documents/airsim_exp_results_random_seed_0',exp_name='random_goals')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 0 ,max_ep_len=500, steps_per_epoch=4000, epochs=250, logger_kwargs=logger_kwargs)
logger_kwargs = dict(output_dir='/home/isra/Documents/airsim_exp_results_random_seed_5',exp_name='random_goals')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 5 ,max_ep_len=500, steps_per_epoch=4000, epochs=250, logger_kwargs=logger_kwargs)
logger_kwargs = dict(output_dir='/home/isra/Documents/airsim_exp_results_random_seed_10',exp_name='random_goals')
ppo(env_fn=envFunc, ac_kwargs=ac_kwargs, seed= 10 ,max_ep_len=500, steps_per_epoch=4000, epochs=250, logger_kwargs=logger_kwargs)


