import math
from random import sample

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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def select_action(
        state, n_step: int, policy_net: nn.Module, env, device,
        EPS_START: float, EPS_DECAY: float, EPS_END: float
) -> torch.Tensor:
    eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * n_step / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_sr(ax, info):
    ax.cla()
    episode_durations = info['episode_durations']
    ax.plot(episode_durations)

    # ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    # ax.set_xlim([min(n_agents_list), max(n_agents_list)])
    # ax.set_ylim([0, 1 + 0.1])
    # ax.set_xticks(n_agents_list)
    # ax.set_xlabel('N agents', fontsize=27)
    # ax.set_ylabel('Success Rate', fontsize=27)
    # # ax.set_title(f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.')
    # # set_plot_title(ax, f'{img_dir[:-4]} Map | time limit: {time_to_think_limit} sec.', size=11)
    # set_plot_title(ax, f'{img_dir[:-4]}', size=30)
    # # set_legend(ax, size=27)
    # labelsize = 20
    # ax.xaxis.set_tick_params(labelsize=labelsize)
    # ax.yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()
