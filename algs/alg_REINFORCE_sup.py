from globals import *

class Policy(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Policy, self).__init__()
        self.saved_log_probs: List[Tensor] = []
        self.rewards: List[int | float] = []
        self.nn = Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: Tensor):
        action_scores = self.nn(x)
        return F.softmax(action_scores, dim=1)


def select_action(state: np.ndarray, policy: nn.Module, device):
    state: Tensor = torch.from_numpy(state).float().unsqueeze(0).to(device=device)
    probs = policy(state)
    m: torch.distributions.Distribution = Categorical(probs)
    action: Tensor = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode(policy: Policy, optimizer: optim.Optimizer, device, GAMMA: float, EPS: float):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.appendleft(R)
    returns = torch.tensor(returns, device=device, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        # Note that we use a negative because optimizers use gradient descent,
        # whilst the REINFORCE rule assumes gradient ascent.
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    policy.rewards = []
    policy.saved_log_probs = []