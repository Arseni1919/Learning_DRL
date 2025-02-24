from abc import ABC

import torch

from alg_VDN_sup import *
from plot_functions import *
from global_functions import *
from my_envs.env_tiger_deer import EnvTigerDeer
from algs_MARL.alg_metaclass import AlgMetaClass

Transitions = namedtuple('Transitions', ('obs', 'actions', 'next_obs', 'rewards', 'terminated'))

class ReplayMemory:
    def __init__(self, capacity: int = 100000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transitions(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AlgVDN(AlgMetaClass):

    def __init__(
            self, 
            n_agents: int, 
            agents_names: List[str], 
            n_actions: int, 
            obs_shape: tuple, 
            params: dict
    ):
        self.n_agents = n_agents
        self.agents_names = agents_names
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.params = params
        self.eps_threshold = 1.0
        self.device = params['device']

        # init NNs + memory
        self.input_shape = math.prod(obs_shape)
        self.policy_nn_dict = {}
        self.target_nn_dict = {}
        nn_params_list = []
        for agent_name in agents_names:
            self.policy_nn_dict[agent_name] = DQN(self.input_shape, n_actions).to(self.device).to(dtype=torch.float32)
            self.target_nn_dict[agent_name] = DQN(self.input_shape, n_actions).to(self.device).to(dtype=torch.float32)
            self.target_nn_dict[agent_name].load_state_dict( self.policy_nn_dict[agent_name].state_dict())
            nn_params_list.append(list(self.policy_nn_dict[agent_name].parameters()))
        united_nn_params_list = list(itertools.chain.from_iterable(nn_params_list))
        self.optimizer: optim.AdamW = optim.AdamW(united_nn_params_list, lr=self.params['LR'], amsgrad=True)
        self.memory = ReplayMemory(capacity=self.params['capacity'])

    def select_actions(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        env = kwargs['env']
        n_step = kwargs['n_step']
        EPS_START = self.params['EPS_START']
        EPS_DECAY = self.params['EPS_DECAY']
        EPS_END = self.params['EPS_END']
        self.eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * n_step / EPS_DECAY)
        alive_agents = [a_name for a_name in self.agents_names if a_name in obs_dict]
        actions = {}
        if random.random() < self.eps_threshold:
            for agent_name in alive_agents:
                action = env.sample_action(agent_name)
                actions[agent_name] = torch.tensor(action, device=self.device, dtype=torch.int64)
            return actions
        with torch.no_grad():
            for agent_name in alive_agents:
                agent_policy = self.policy_nn_dict[agent_name]
                agent_obs = obs_dict[agent_name]
                policy_output = agent_policy(agent_obs)
                action = torch.argmax(policy_output, dim=0).to(dtype=torch.int64)
                actions[agent_name] = action
        return actions

    def add_to_memory(self, obs, actions, new_obs, rewards, terminated) -> None:
        new_obs = {k: torch.tensor(v, device=self.device, dtype=torch.float32).flatten() for k, v in new_obs.items()}
        rewards = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in rewards.items()}
        terminated = {k: torch.tensor(v, device=self.device, dtype=torch.bool) for k, v in terminated.items()}
        self.memory.push(obs, actions, new_obs, rewards, terminated)

    def target_soft_copy(self) -> None:
        TAU = self.params['TAU']
        for agent_name in self.agents_names:
            a_policy_state_dict = self.policy_nn_dict[agent_name].state_dict()
            a_target_state_dict = self.target_nn_dict[agent_name].state_dict()
            for key in a_policy_state_dict:
                a_target_state_dict[key] = TAU * a_policy_state_dict[key] + (1 - TAU) * a_target_state_dict[key]
            self.target_nn_dict[agent_name].load_state_dict(a_target_state_dict)

    def learn(self) -> None:
        BATCH_SIZE = self.params['BATCH_SIZE']
        GAMMA = self.params['GAMMA']
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transitions(*zip(*transitions))
        obs, actions, next_obs, rewards, terminated = batch
        self.optimizer.zero_grad()
        tot_q_values = torch.tensor([])
        tot_q_values_expected = torch.tensor([])
        for agent_name in self.agents_names:
            batch_obs, action_batch, batch_next_obs, reward_batch, batch_terminated = [], [], [], [], []
            for i in range(BATCH_SIZE):
                if agent_name in obs[i]:
                    batch_obs.append(obs[i][agent_name])
                    action_batch.append(actions[i][agent_name])
                    agent_next_obs = next_obs[i][agent_name] if agent_name in next_obs[i] else torch.zeros(self.input_shape, device=self.device, dtype=torch.float32)
                    batch_next_obs.append(agent_next_obs)
                    reward_batch.append(rewards[i][agent_name])
                    batch_terminated.append(terminated[i][agent_name])
            batch_obs = torch.stack(batch_obs)
            action_batch = torch.stack(action_batch).unsqueeze(1)
            batch_next_obs = torch.stack(batch_next_obs)
            reward_batch = torch.stack(reward_batch)
            batch_terminated = torch.stack(batch_terminated)
            q_values = self.policy_nn_dict[agent_name](batch_obs).gather(1, action_batch)  # Get Q-values for all actions
            # tot_q_values += q_values
            tot_q_values = torch.cat((tot_q_values, q_values))
            with torch.no_grad():
                next_q_values = self.target_nn_dict[agent_name](batch_next_obs).max(1).values * (1 - batch_terminated.float())
                q_value_expected = reward_batch[0] + GAMMA * next_q_values

                tot_q_values_expected = torch.cat((tot_q_values_expected, q_value_expected))
            criterion = nn.MSELoss()
            loss: nn.Module = criterion(q_values, q_value_expected)
            loss.backward()


        # criterion = nn.SmoothL1Loss()
        # tot_q_values_expected = tot_q_values_expected.unsqueeze(1)
        # criterion = nn.MSELoss()
        # loss: nn.Module = criterion(tot_q_values, tot_q_values_expected)
        # self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()
        self.target_soft_copy()
        # nn.utils.clip_grad_value_(policy_net.parameters(), 100)


@use_profiler(save_dir='../stats/alg_vdn.pstat')
def main():
    # parameters
    # n_episodes = 1_000_000
    n_episodes = 10000
    # n_episodes = 300
    # n_episodes = 100
    # n_episodes = 3
    params = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.99,
        'EPS_START': 0.9,
        'EPS_END': 0.05,
        'EPS_DECAY': 10000,
        'TAU': 0.005,  # update rate of the target network
        'LR': 1e-5,  # learning rate of the AdamW optimizer
        'capacity': 10000,
        # device = torch.device(
        #     "cuda" if torch.cuda.is_available() else
        #     "mps" if torch.backends.mps.is_available() else
        #     "cpu"
        # )
        'device': 'cpu',
    }

    # SEED
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    # for plots
    to_plot = True
    # to_plot = False
    running_reward = 0
    running_rewards: List[int] = []
    ep_rewards: List[int] = []
    total_i_counter = 0
    if to_plot:
        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_rate = 0.001

    # create env + parameters after env init
    # map_name = "3m"
    # env = StarCraft2Env(map_name=map_name)
    # env = EnvTigerDeer(to_render=True)
    # env_info = env.get_env_info()
    # n_actions = env_info["n_actions"]
    # n_agents = env_info["n_agents"]
    # obs_shape = env_info['obs_shape']

    # env = EnvTigerDeer(to_render=True)

    # num_tigers, num_deer = 202, 40
    # num_tigers, num_deer = 101, 20
    # num_tigers, num_deer = 51, 10
    # num_tigers, num_deer = 20, 10
    # num_tigers, num_deer = 10, 4
    num_tigers, num_deer = 2, 20

    # env = EnvTigerDeer(num_tigers=num_tigers, num_deer=num_deer, to_render=True)
    env: EnvTigerDeer = EnvTigerDeer(num_tigers=num_tigers, num_deer=num_deer, to_render=False)
    n_agents = env.num_tigers
    agents_names = env.tigers_names
    n_actions = env.n_actions_tiger
    obs_shape = env.obs_shape_tiger

    # create alg
    alg: AlgVDN = AlgVDN(n_agents, agents_names, n_actions, obs_shape, params)  # alg

    # main loop
    # for i_episode in count():
    for i_episode in range(n_episodes):
        observations, info = env.reset()
        episode_reward = 0

        for i_step in count():
            observations = {k: torch.tensor(v, device=alg.device, dtype=torch.float32).flatten() for k, v in observations.items()}
            actions = alg.select_actions(observations, n_step=total_i_counter, env=env)  # alg
            actions_for_env = {agent_name: int(action.item()) for agent_name, action in actions.items()}
            new_observations, rewards, terminated, info = env.step(actions_for_env)
            finished = info['finished']

            alg.add_to_memory(observations, actions, new_observations, rewards, terminated)  # alg
            alg.learn()  # alg

            observations = new_observations

            # stats update and finish check
            total_i_counter += 1
            episode_reward += sum([rewards[a_name] for a_name in agents_names])
            print(f'\r{alg.eps_threshold=: .2f}, {i_episode=}, {i_step=}, {total_i_counter=}', end='')
            if finished:
                break

        # print + plot
        running_reward = 0.01 * episode_reward + (1 - 0.01) * running_reward
        running_rewards.append(running_reward)
        ep_rewards.append(episode_reward)
        print(f'\n{episode_reward=} | {running_reward=:.2f}', end='\n')
        if to_plot:
            plot_running_rewards(ax, info={
                'env_name': env.name, 'running_rewards': running_rewards, 'ep_rewards': ep_rewards
            })
            plt.pause(plot_rate)

    print('\n--- Completed ---')
    env.close()
    plt.show()
    plt.show()
    plt.show()


if __name__ == '__main__':
    main()