import matplotlib.pyplot as plt

from my_envs.env_functions import *
from my_envs.env_plot_functions import *


class FindTarget(MetaMultiAgentEnv):
    def __init__(self, to_render: bool = False):
        super().__init__()
        self.name: str = 'FindTarget'

        self.height: int = 10
        self.weight: int = 10
        self.field: np.ndarray = np.zeros((self.height, self.weight))
        self.target = None
        self.agents_loc = {}

        # for rendering
        self.to_render: bool = to_render
        self.fig, self.ax, self.plot_rate = None, None, None
        if self.to_render:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
            self.plot_rate = 0.001

    def reset(self, seed: int = 123) -> Tuple[Any, Dict]:
        # return: obs, info
        # random.seed(seed)
        # np.random.seed(seed)
        self.target = (random.randint(0, 10), random.randint(0, 10))
        self.agents_loc = {'agent_0': [random.randint(0, 10), random.randint(0, 10)]}
        info = {'field': self.field, 'target': self.target, 'main_agent': self.agents_loc}
        obs = []
        return obs, info

    def sample_action(self, agent_name) -> Any:
        pass

    def sample_actions(self) -> Any:
        assert len(self.agents_loc) != 0, 'A reset is needed'
        actions = {a_name: random.choice([0, 1, 2, 3, 4]) for a_name, _ in self.agents_loc.items()}
        return actions

    def step(self, actions: dict) -> Tuple[dict, dict, dict, dict, dict]:
        # return: obs, rewards, dones, truncated, info
        for agent_name, action in actions.items():
            prev_loc = self.agents_loc[agent_name]
            next_loc = self.agents_loc[agent_name]
            action = actions[agent_name]
            # 1 - up, 2 - right, 3 - down, 4 - left, 0 - wait
            if action == 1:
                next_loc[0] += 1
            if action == 2:
                next_loc[1] += 1
            if action == 3:
                next_loc[0] -= 1
            if action == 4:
                next_loc[1] -= 1
            if next_loc[0] < 0 or next_loc[0] >= self.height:
                next_loc[0] = prev_loc[0]
            if next_loc[1] < 0 or next_loc[1] >= self.weight:
                next_loc[1] = prev_loc[1]

            self.agents_loc[agent_name] = next_loc

        if self.to_render:
            render_field(self.ax[0], info={
                'env_name': self.name,
                'target': self.target,
                'agents_loc': self.agents_loc,
                'field': self.field,
            })
            plt.pause(self.plot_rate)
        # return: obs, rewards, dones, truncated, info
        dones = {a_name: False for a_name, _ in self.agents_loc.items()}
        truncated = {a_name: False for a_name, _ in self.agents_loc.items()}
        return {}, {}, {}, {}, {}

    def close(self) -> None:
        plt.close()



def main():
    env = FindTarget(to_render=True)
    observations, info = env.reset(seed=42)
    for _ in range(1000):
        # action = policy(observations)  # User-defined policy function
        actions = env.sample_actions()
        observations, rewards, terminated, truncated, info = env.step(actions)

        if any(terminated.values()) or any(truncated.values()):
            observations, info = env.reset()

    env.close()


if __name__ == '__main__':
    main()


