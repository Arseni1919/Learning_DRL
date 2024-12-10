import matplotlib.pyplot as plt

from my_envs.env_functions import *
from my_envs.env_plot_functions import *


class FindTarget(MetaMultiAgentEnv):
    def __init__(self, to_render: bool = False):
        super().__init__()
        self.name: str = 'FindTarget'

        self.height: int = 20
        self.width: int = 20
        self.field: np.ndarray = create_rand_field(self.width, self.height, 0.05)
        self.target = None
        self.agents_loc: dict = {}
        self.iteration: int = 0
        self.max_iter: int = 500
        self.num_agents = 1
        # self.num_agents = 10

        # rewards
        self.r_step = -0.01
        self.r_collision = -0.1
        self.r_dying = -1
        self.r_target = 5

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
        self.iteration = 0
        self.target = choose_unoccupied_loc(self.field)
        self.agents_loc = create_agents_loc(self.field, self.num_agents)
        info = {'field': self.field, 'target': self.target, 'main_agent': self.agents_loc}
        obs = []
        return obs, info

    def sample_action(self, agent_name) -> Any:
        pass

    def sample_actions(self) -> Any:
        assert len(self.agents_loc) != 0, 'A reset is needed'
        actions = {a_name: random.choice([0, 1, 2, 3, 4]) for a_name, _ in self.agents_loc.items()}
        # actions = {a_name: 1 for a_name, _ in self.agents_loc.items()}
        return actions

    def step(self, actions: dict) -> Tuple[dict, dict, dict, dict, dict]:
        # return: obs, rewards, dones, truncated, info
        self.iteration += 1
        out_obs = {}
        for agent_name, action in actions.items():
            prev_loc = self.agents_loc[agent_name][:]
            next_loc = self.agents_loc[agent_name]
            action = actions[agent_name]
            # 1 - up, 2 - right, 3 - down, 4 - left, 0 - wait
            if action == 1:
                next_loc[1] += 1
            if action == 2:
                next_loc[0] += 1
            if action == 3:
                next_loc[1] -= 1
            if action == 4:
                next_loc[0] -= 1

            if next_loc[0] < 0 or next_loc[0] >= self.width:
                next_loc[0] = prev_loc[0]
            if next_loc[1] < 0 or next_loc[1] >= self.height:
                next_loc[1] = prev_loc[1]

            if self.field[next_loc[0], next_loc[1]] == 1:
                next_loc = prev_loc

            self.agents_loc[agent_name] = next_loc

        out_obs = {}
        for agent_name, action in actions.items():
            out_obs[agent_name] = build_agent_obs(
                2, agent_name, self.agents_loc[agent_name],
                self.field, self.width, self.height, self.agents_loc, self.target
            )
        out_rewards = {}
        for agent_name, action in actions.items():
            out_rewards[agent_name] = self.r_step
            next_loc = self.agents_loc[agent_name]
            if next_loc == self.target:
                out_rewards[agent_name] = self.r_target
                continue
            if self.iteration >= self.max_iter:
                out_rewards[agent_name] = self.r_dying


        if self.to_render:
            render_field(self.ax[0], info={
                'env_name': self.name,
                'target': self.target,
                'agents_loc': self.agents_loc,
                'field': self.field,
                'iteration': self.iteration
            })
            render_agent_view(self.ax[1], info={'agent_obs': out_obs['agent_0']})
            plt.pause(self.plot_rate)

        done = self.iteration >= self.max_iter
        out_dones = {a_name: done for a_name, _ in self.agents_loc.items()}
        out_truncated = {a_name: False for a_name, _ in self.agents_loc.items()}
        out_info = {'iteration': self.iteration, 'agents_loc': self.agents_loc}
        # return: obs, rewards, dones, truncated, info
        return out_obs, out_rewards, out_dones, out_truncated, out_info

    def close(self) -> None:
        plt.close()



def main():
    env = FindTarget(to_render=True)
    # env = FindTarget(to_render=False)
    observations, info = env.reset(seed=42)
    for i in range(1_000_000):
        # action = policy(observations)  # User-defined policy function
        actions = env.sample_actions()
        observations, rewards, terminated, truncated, info = env.step(actions)

        if any(terminated.values()) or any(truncated.values()):
            observations, info = env.reset()
        print(f'\r{i=}', end='')

    env.close()


if __name__ == '__main__':
    main()


