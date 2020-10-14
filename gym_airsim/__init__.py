from gym.envs.registration import register

register(
	id = 'airsim_gym-v0',
	entry_point= 'gym_airsim.envs:AirsimEnv'
	)