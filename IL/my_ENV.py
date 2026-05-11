import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, Env_id, Env_matrix):
        super(EnvWrapper, self).__init__(env)
        original_observation_space = env.observation_space
        extra_dims = Env_matrix.shape[1]
        low = np.concatenate([original_observation_space.low, np.full(extra_dims, -np.inf)])
        high = np.concatenate([original_observation_space.high, np.full(extra_dims, np.inf)])
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        self.Env_id = Env_id
        np.random.seed(Env_id)
        self.multiplicative_matrix = Env_matrix
        self.states = None
        self.spur_corr_init = None

    def reset(self, **kwargs):

        states, _ = self.env.reset(**kwargs)
        self.states = states
        dim_c, dim_mu = self.multiplicative_matrix.shape
        self.spur_corr_init = np.matmul(np.random.uniform(-1, 1, size=dim_c), self.multiplicative_matrix)
        observation = np.concatenate([self.states, self.spur_corr_init])
        return observation

    def step(self, action):
        dim_c, dim_mu = self.multiplicative_matrix.shape
        spur_corr = np.matmul(self.states[:dim_c], self.multiplicative_matrix)
        states, reward, done, info, _ = self.env.step(action)
        self.states = states
        observation = np.concatenate([self.states, spur_corr])
        return observation, reward, done, info

