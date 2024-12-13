import matplotlib.pyplot as plt

from my_envs.sup_env_globals import *
from my_envs.sup_env_plot_functions import *


class EnvTigerDeer(MetaMultiAgentEnv):
    def __init__(self, to_render: bool = False):
        super().__init__()
        self.name: str = 'EnvTigerDeer'

        self.height: int = 45
        self.width: int = 45
        self.field: np.ndarray = create_rand_field(self.width, self.height, 0.01)
        self.agents_dict = OrderedDict()
        self.tigers_dict = OrderedDict()
        self.deer_dict = OrderedDict()
        self.iteration: int = 0
        self.n_episode: int = 0
        self.max_iter: int = 500

        # self.num_tigers, self.num_deer = 202, 40
        self.num_tigers, self.num_deer = 101, 20
        # self.num_tigers, self.num_deer = 10, 2

        # obs
        self.obs_radius_tiger = 4
        self.obs_radius_deer = 1

        # HP globals
        self.hp_tiger_start = 10
        self.hp_tiger_step = -0.1
        self.hp_tiger_eat = 8

        self.hp_deer_start = 5
        self.hp_deer_step = 0.1
        self.hp_deer_attacked = -1

        # rewards
        self.r_dying = -1
        self.r_attack = 1
        self.r_being_attacked = -0.1

        # for rendering
        self.to_render: bool = to_render
        self.fig, self.ax, self.plot_rate = None, None, None
        if self.to_render:
            self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
            self.plot_rate = 0.001

    def reset(self, seed: int = 123, **kwargs) -> Tuple[Any, Dict]:
        # return: obs, info
        # random.seed(seed)
        # np.random.seed(seed)
        self.iteration = 0
        self.n_episode = kwargs['n_episode'] if 'n_episode' in kwargs else 0
        self._create_agents()
        info = {'field': self.field}
        obs = self._build_obs()
        return obs, info

    def sample_action(self, agent_name) -> Any:
        if agent_name in self.tigers_dict:
            # 0 - do nothing, 1-4 - move, 5-8 - attack
            return random.randint(0, 9)
        if agent_name in self.deer_dict:
            # 0 - do nothing, 1-4 - move
            return random.randint(0, 5)
        raise RuntimeError('nope')

    def sample_actions(self) -> Any:
        assert len(self.tigers_dict) != 0, 'A reset is needed'
        assert len(self.deer_dict) != 0, 'A reset is needed'
        actions = {}
        for t_name, _ in self.tigers_dict.items():
            # 0 - do nothing, 1-4 - move, 5-8 - attack
            action = random.randint(0, 9)
            actions[t_name] = action
        for d_name, _ in self.deer_dict.items():
            # 0 - do nothing, 1-4 - move
            action = random.randint(0, 5)
            actions[d_name] = action
        return actions

    def step(self, actions: dict) -> Tuple[dict, dict, dict, dict]:
        # return: obs, rewards, dones, info
        self.iteration += 1

        # update moves and attack positions
        self._execute_actions(actions)
        self._update_hp_reward_alive_params()

        out_obs = self._build_obs()

        out_rewards = {}
        for d_name, d_params in self.deer_dict.items():
            out_rewards[d_name] = d_params['reward']
        for t_name, t_params in self.tigers_dict.items():
            out_rewards[t_name] = t_params['reward']

        if self.to_render:
            info = {
                'env_name': self.name,
                'tigers_dict': self.tigers_dict,
                'deer_dict': self.deer_dict,
                'field': self.field,
                'iteration': self.iteration,
                'n_episode': self.n_episode,
                'out_obs': out_obs,
            }
            render_tiger_deer_field(self.ax[0], info)
            # render_td_agent_view(self.ax[1], info)
            plt.pause(self.plot_rate)

        out_dones = {}
        t_counter = 0
        d_counter = 0
        for d_name, d_params in self.deer_dict.items():
            out_dones[d_name] = d_params['alive']
            d_counter += d_params['alive']
        for t_name, t_params in self.tigers_dict.items():
            out_dones[t_name] = t_params['alive']
            t_counter += t_params['alive']
        finished = t_counter < 2 or d_counter == 0

        out_info = {
            'iteration': self.iteration, 'tigers_dict': self.tigers_dict, 'deer_dict': self.deer_dict,
            'finished': finished,
        }
        # return: obs, rewards, dones, truncated, info
        return out_obs, out_rewards, out_dones, out_info

    def close(self) -> None:
        plt.close()

    def _create_agents(self):
        field = self.field.copy()
        for i in range(self.num_tigers):
            tiger_name = f'tiger_{i}'
            [target_x, target_y] = choose_unoccupied_loc(field)
            self.tigers_dict[tiger_name] = {
                'num': i,
                'type': 'tiger',
                'alive': True,
                'loc': [target_x, target_y],
                'attack': [target_x, target_y],
                'hp': self.hp_tiger_start,
                'reward': 0,
                'obs_radius': self.obs_radius_tiger,
            }
            field[target_x, target_y] = 1
            self.agents_dict[tiger_name] = self.tigers_dict[tiger_name]
        for i in range(self.num_deer):
            deer_name = f'deer_{i}'
            [target_x, target_y] = choose_unoccupied_loc(field)
            self.deer_dict[deer_name] = {
                'num': i,
                'type': 'deer',
                'alive': True,
                'loc': [target_x, target_y],
                'hp': self.hp_deer_start,
                'reward': 0,
                'obs_radius': self.obs_radius_deer,
            }
            field[target_x, target_y] = 1
            self.agents_dict[deer_name] = self.deer_dict[deer_name]
        self.agents_dict = {**self.tigers_dict, **self.deer_dict}

    def _execute_actions(self, actions: dict) -> None:
        # execute actions
        forbidden_locs = [params['loc'][:] for params in self.tigers_dict.values() if params['alive']]
        forbidden_locs.extend([params['loc'][:] for params in self.deer_dict.values() if params['alive']])

        for t_name, t_params in self.tigers_dict.items():
            if not t_params['alive']:
                continue
            t_loc = t_params['loc']
            # 0 - do nothing, 1-4 - move, 5-8 - attack
            prev_loc = t_loc[:]
            action = actions[t_name]
            if action == 0:
                t_params['attack'] = t_loc[:]
                continue
            if action == 1:
                t_loc[1] += 1
            if action == 2:
                t_loc[0] += 1
            if action == 3:
                t_loc[1] -= 1
            if action == 4:
                t_loc[0] -= 1
            if t_loc[0] < 0 or t_loc[0] >= self.width:
                t_loc[0] = prev_loc[0]
            if t_loc[1] < 0 or t_loc[1] >= self.height:
                t_loc[1] = prev_loc[1]
            if self.field[t_loc[0], t_loc[1]] == 1:
                t_params['loc'] = prev_loc
            if t_loc in forbidden_locs:
                t_params['loc'] = prev_loc
            forbidden_locs.append(t_params['loc'])
            t_loc = t_params['loc']
            t_params['attack'] = t_loc
            if action == 5:
                t_params['attack'] = [t_loc[0], t_loc[1] + 1]
            if action == 6:
                t_params['attack'] = [t_loc[0] + 1, t_loc[1]]
            if action == 7:
                t_params['attack'] = [t_loc[0], t_loc[1] - 1]
            if action == 8:
                t_params['attack'] = [t_loc[0] - 1, t_loc[1]]

        for d_name, d_params in self.deer_dict.items():
            if not d_params['alive']:
                continue
            # 0 - do nothing, 1-4 - move
            d_loc = d_params['loc']
            prev_loc = d_loc[:]
            action = actions[d_name]
            if action == 0:
                continue
            if action == 1:
                d_loc[1] += 1
            if action == 2:
                d_loc[0] += 1
            if action == 3:
                d_loc[1] -= 1
            if action == 4:
                d_loc[0] -= 1
            if d_loc[0] < 0 or d_loc[0] >= self.width:
                d_loc[0] = prev_loc[0]
            if d_loc[1] < 0 or d_loc[1] >= self.height:
                d_loc[1] = prev_loc[1]
            if self.field[d_loc[0], d_loc[1]] == 1:
                d_params['loc'] = prev_loc
            if d_loc in forbidden_locs:
                d_params['loc'] = prev_loc
            forbidden_locs.append(d_params['loc'])

    def _update_hp_reward_alive_params(self):
        processed_names = []
        for d_name, d_params in self.deer_dict.items():
            d_params['reward'] = 0
        for t_name, t_params in self.tigers_dict.items():
            t_params['reward'] = 0
        deer_field = np.ones(self.field.shape) * -1
        for d_name, d_params in self.deer_dict.items():
            if not d_params['alive']:
                continue
            deer_field[d_params['loc'][0], d_params['loc'][1]] = d_params['num']
        for t_name_1, t_name_2 in combinations(self.tigers_dict.keys(), 2):
            if not self.tigers_dict[t_name_1]['alive'] or not self.tigers_dict[t_name_1]['alive']:
                continue
            attack_1 = self.tigers_dict[t_name_1]['attack']
            attack_2 = self.tigers_dict[t_name_2]['attack']
            if attack_1 != attack_2:
                continue
            if deer_field[attack_1[0], attack_1[1]] == -1:
                continue
            deer_name = f'deer_{int(deer_field[attack_1[0], attack_1[1]])}'
            self.tigers_dict[t_name_1]['hp'] += self.hp_tiger_eat
            self.tigers_dict[t_name_2]['hp'] += self.hp_tiger_eat
            self.deer_dict[deer_name]['hp'] += self.hp_deer_attacked
            self.tigers_dict[t_name_1]['reward'] = self.r_attack
            self.tigers_dict[t_name_2]['reward'] = self.r_attack
            self.deer_dict[deer_name]['reward'] = self.r_being_attacked
            processed_names.append(t_name_1)
            processed_names.append(t_name_2)
            processed_names.append(deer_name)
        for d_name, d_params in self.deer_dict.items():
            if not d_params['alive']:
                continue
            if d_name in processed_names:
                continue
            d_params['hp'] += self.hp_deer_step
            if d_params['hp'] <= 0:
                d_params['alive'] = False
                d_params['reward'] = self.r_dying
        for t_name, t_params in self.tigers_dict.items():
            if not t_params['alive']:
                continue
            if t_name in processed_names:
                continue
            t_params['hp'] += self.hp_tiger_step
            t_params['alive'] = t_params['hp'] > 0

    def _build_obs(self) -> Dict[str, np.ndarray]:
        """
        The observation space is:
        - 3x3 map with 5 channels for deer
        - 9x9 map with 5 channels for tigers

        The channels:
        - obstacle/off the map
        - my_team_presence
        - my_team_hp
        - other_team_presence
        - other_team_hp
        """
        t_team_presence, d_team_presence = np.zeros(self.field.shape), np.zeros(self.field.shape)
        t_team_hp, d_team_hp = np.zeros(self.field.shape), np.zeros(self.field.shape)
        for i_name, i_params in self.agents_dict.items():
            if i_params['type'] == 'tiger':
                t_team_presence[i_params['loc'][0], i_params['loc'][1]] = 1
                t_team_hp[i_params['loc'][0], i_params['loc'][1]] = i_params['hp']
            elif i_params['type'] == 'deer':
                d_team_presence[i_params['loc'][0], i_params['loc'][1]] = 1
                d_team_hp[i_params['loc'][0], i_params['loc'][1]] = i_params['hp']
            else:
                raise RuntimeError('nope')
        obs = {}
        for i_name, i_params in self.agents_dict.items():
            radius = self.obs_radius_tiger if i_params['type'] == 'tiger' else self.obs_radius_deer
            curr_loc = i_params['loc']
            i_obs: np.ndarray = np.zeros((2*radius + 1, 2*radius + 1, 5))
            for x_offset in range(-radius, radius+1):
                for y_offset in range(-radius, radius+1):
                    i_obs_x = x_offset + radius
                    i_obs_y = y_offset + radius
                    i_x = curr_loc[0] + x_offset
                    i_y = curr_loc[1] + y_offset
                    if i_x < 0 or i_x >= self.width or i_y < 0 or i_y >= self.height:
                        continue
                    #  - obstacle/off the map
                    i_obs[i_obs_x, i_obs_y, 0] = self.field[i_x, i_y]
                    if i_params['type'] == 'tiger':
                        #  - my_team_presence
                        i_obs[i_obs_x, i_obs_y, 1] = t_team_presence[i_x, i_y]
                        #  - my_team_hp
                        i_obs[i_obs_x, i_obs_y, 2] = t_team_hp[i_x, i_y]
                        #  - other_team_presence
                        i_obs[i_obs_x, i_obs_y, 3] = d_team_presence[i_x, i_y]
                        #  - other_team_hp
                        i_obs[i_obs_x, i_obs_y, 4] = d_team_hp[i_x, i_y]
                    else:
                        #  - my_team_presence
                        i_obs[i_obs_x, i_obs_y, 1] = d_team_presence[i_x, i_y]
                        #  - my_team_hp
                        i_obs[i_obs_x, i_obs_y, 2] = d_team_hp[i_x, i_y]
                        #  - other_team_presence
                        i_obs[i_obs_x, i_obs_y, 3] = t_team_presence[i_x, i_y]
                        #  - other_team_hp
                        i_obs[i_obs_x, i_obs_y, 4] = t_team_hp[i_x, i_y]

            obs[i_name] = i_obs
        return obs




def main():
    eps_counter = 0
    env = EnvTigerDeer(to_render=True)
    # env = EnvTigerDeer(to_render=False)
    observations, info = env.reset(seed=42)
    for i in range(1_000_000):
        # actions = policy(observations)  # User-defined policy function
        actions = env.sample_actions()
        observations, rewards, terminated, info = env.step(actions)

        if info['finished']:
            eps_counter += 1
            observations, info = env.reset(n_episode=eps_counter)
        print(f'\r{eps_counter=}, {i=}', end='')

    env.close()


if __name__ == '__main__':
    main()


