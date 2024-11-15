import random

import matplotlib.pyplot as plt

from my_envs.env_functions import *


class FindTarget(MetaEnv):
    def __init__(self, to_render: bool = False):
        super().__init__()
        self.name: str = 'FindTarget'
        self.to_render: bool = to_render
        self.fig, self.ax, self.plot_rate = None, None, None
        if self.to_render:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
            self.plot_rate = 0.001

        self.field: np.ndarray = np.zeros((10, 10))
        self.target = None
        self.main_agent = None

    def name(self):
        return self.name

    def reset(self, seed: int = 123) -> Tuple[Any, Dict]:
        # return: obs, info
        # random.seed(seed)
        # np.random.seed(seed)
        self.target = (random.randint(0, 10), random.randint(0, 10))
        self.main_agent = (random.randint(0, 10), random.randint(0, 10))
        info = {'field': self.field, 'target': self.target, 'main_agent': self.main_agent}
        obs = []
        return obs, info

    def sample_action(self) -> Any:
        return random.choice([0, 1, 2, 3, 4])

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        # return: obs, reward, done, truncated, info
        pass

    def close(self) -> None:
        plt.close()



def main():
    env = FindTarget(to_render=True)
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # action = policy(observation)  # User-defined policy function
        action = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()



if __name__ == '__main__':
    main()


