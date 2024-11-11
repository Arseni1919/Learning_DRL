import torch

from globals import *

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCriticNet, self).__init__()
        self.saved_actions: List[SavedAction] = []
        self.rewards: List[int | float] = []
        self.backbone = Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)


    def forward(self, x: Tensor):
        features = self.backbone(x)
        actor_out = F.softmax(self.actor_head(features), dim=-1)
        critic_out = self.critic_head(features)
        return actor_out, critic_out


def select_action(state: np.ndarray, ac_net: ActorCriticNet, device):
    state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(device=device)
    probs, state_value = ac_net(state)
    m: torch.distributions.Distribution = Categorical(probs)
    action: Tensor = m.sample()
    # ac_net.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item(), m.log_prob(action), state_value