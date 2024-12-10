import torch

from globals import *


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class AgentIQL:
    def __init__(
            self,
            num: int,
            n_actions: int,
            obs_shape: int,
            LR: float,
            EPS_START: float, EPS_DECAY: float, EPS_END: float,
            TAU: float,
            device: str,
    ):
        self.num: int = num
        self.name = f'agent_{num}'
        self.n_actions: int = n_actions
        self.obs_shape: int = obs_shape
        self.LR: float = LR
        self.EPS_START: float = EPS_START
        self.EPS_DECAY: float = EPS_DECAY
        self.EPS_END: float = EPS_END
        self.TAU: float = TAU
        self.device: str = device

        self.policy_net: nn.Module = DQN(obs_shape, n_actions).to(device)
        self.target_net: nn.Module = DQN(obs_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer: optim.AdamW = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory: ReplayMemory = ReplayMemory(10000)

    def select_action(self, state: torch.Tensor, n_step: int, env):
        avail_actions = env.get_avail_agent_actions(self.num)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        eps_threshold: float = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * n_step / self.EPS_DECAY)
        if random.random() > eps_threshold:
            with torch.no_grad():
                policy_out = self.policy_net(state)
                max_value = policy_out[:, avail_actions_ind].max(1).values
                action = torch.where(policy_out == max_value)[1].unsqueeze(0)
                return action
        else:
            action = torch.tensor([[np.random.choice(avail_actions_ind)]], device=self.device, dtype=torch.long)
            return action

    def add_to_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def target_soft_copy(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.TAU * policy_net_state_dict[key] + (1 - self.TAU) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)


class AlgIQL:
    def __init__(
            self,
            n_agents: int,
            n_actions: int,
            obs_shape: int,
            LR: float,
            EPS_START: float, EPS_DECAY: float, EPS_END: float,
            TAU: float,
            BATCH_SIZE: int,
            GAMMA: float,
            device: str,
    ):
        self.n_agents: int = n_agents
        self.n_actions: int = n_actions
        self.obs_shape: int = obs_shape
        self.LR: float = LR
        self.EPS_START: float = EPS_START
        self.EPS_DECAY: float = EPS_DECAY
        self.EPS_END: float = EPS_END
        self.TAU: float = TAU
        self.BATCH_SIZE: int = BATCH_SIZE
        self.GAMMA: float = GAMMA
        self.device: str = device
        self.agents, self.agents_dict = [], {}
        for i in range(n_agents):
            new_agent = AgentIQL(i, n_actions, obs_shape, LR, EPS_START, EPS_DECAY, EPS_END, TAU, device)
            self.agents.append(new_agent)
            self.agents_dict[f'agent_{i}'] = new_agent

    def select_actions(self, obs: list, n_step: int, env) -> List[int]:
        actions = []
        for agent_id in range(self.n_agents):
            state = torch.tensor(obs[agent_id], device=self.device, dtype=torch.float32).unsqueeze(0)
            action = self.agents[agent_id].select_action(state, n_step, env)
            actions.append(action)
        return actions

    def add_to_memories(self, obs, actions, new_obs, reward, terminated):
        for agent_id in range(self.n_agents):
            state = torch.tensor(obs[agent_id], device=self.device, dtype=torch.float32).unsqueeze(0)
            action = actions[agent_id]
            next_state = None if terminated else torch.tensor(new_obs[agent_id], device=self.device, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], device=self.device)
            self.agents[agent_id].add_to_memory(state, action, next_state, reward)

    def learn(self):
        for agent_id in range(self.n_agents):
            memory = self.agents[agent_id].memory
            policy_net = self.agents[agent_id].policy_net
            target_net = self.agents[agent_id].target_net
            optimizer = self.agents[agent_id].optimizer
            optimize_model(memory, self.device, policy_net, target_net, optimizer, self.BATCH_SIZE, self.GAMMA)
            self.agents[agent_id].target_soft_copy()


def optimize_model(
        memory: ReplayMemory,
        device: str,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        BATCH_SIZE: int,
        GAMMA: float

) -> None:
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)
    criterion = nn.SmoothL1Loss()
    loss: nn.Module = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
