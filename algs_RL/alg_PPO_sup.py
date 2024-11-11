import torch

from globals import *
from plot_functions import *
from global_functions import *


class ReplayBuffer:

    def __init__(self, update_time_interval: int):
        self.update_time_interval: int = update_time_interval
        self.memory = deque(maxlen=self.update_time_interval)

    def add(self, state, action, logprob, reward, state_value, done):
        self.memory.append((state, action, logprob, reward, state_value, done))

    def get_all_memory(self):
        states, actions, logprobs, rewards, state_values, dones = zip(*self.memory)
        return states, actions, logprobs, rewards, state_values, dones

    def clear(self):
        self.memory = deque(maxlen=self.update_time_interval)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_std_init: float, lr: float = 1e-4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.learning_rate: float = lr
        self.nn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.action_var: torch.Tensor = torch.full((action_dim,), self.action_std_init**2)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std**2)

    def forward(self, x):
        out = self.nn(x)
        return out


class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, lr: float = 1e-4):
        super(Critic, self).__init__()
        self.learning_rate: float = lr

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            update_time_interval: int,
            action_std_init: float,
            actor_lr: float,
            critic_lr: float,
            k_epochs: int,
            gamma: float,
            eps_clip: float,
            device: str
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.update_time_interval = update_time_interval
        self.action_std: float = action_std_init
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.k_epochs: int = k_epochs
        self.gamma: float = gamma
        self.eps_clip: float = eps_clip
        self.device = device


        self.actor: Actor = Actor(state_dim, hidden_dim, action_dim, action_std_init, actor_lr).to(device)
        self.actor_old: Actor = Actor(state_dim, hidden_dim, action_dim, action_std_init, actor_lr).to(device)
        self.critic: Critic = Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)
        self.critic_old: Critic= Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)

        self.memory = ReplayBuffer(update_time_interval)

        self._update_target_networks()  # initialize target networks

        self.MSELoss = nn.MSELoss()

    def _update_target_networks(self):

        for old_param, param in zip(self.actor_old.parameters(), self.actor.parameters()):
            # old_param.data.copy_(tau * param.data + (1 - tau) * old_param.data)
            old_param.data.copy_(param.data)

        for old_param, param in zip(self.critic_old.parameters(), self.critic.parameters()):
            old_param.data.copy_(param.data)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.actor.set_action_std(self.action_std)
        self.actor_old.set_action_std(self.action_std)

    def act(self, state):
        with torch.no_grad():
            state: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_mean = self.actor_old(state).detach()
            cov_mat = torch.diag(self.actor_old.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample().detach()
            logprob = dist.log_prob(action).detach()
            state_val = self.critic_old(state).detach()

        action = action.detach().cpu().numpy().flatten()
        return action, logprob, state_val

    def get_pure_action(self, state, *args, **kwargs):
        with torch.no_grad():
            state: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_mean = self.actor_old(state).detach()
            cov_mat = torch.diag(self.actor_old.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample().detach()
        action = action.detach().cpu().numpy().flatten()
        return action


    def store_transition(self, state, action, logprob, reward, state_value, done):
        self.memory.add(state, action, logprob, reward, state_value, done)

    def learn(self):
        if len(self.memory) < self.update_time_interval:
            return

        raw_states, raw_actions, raw_logprobs, raw_rewards, raw_state_values, raw_dones  = self.memory.get_all_memory()

        # build Monte-Carlo estimates of returns
        mc_rewards = []
        discounted_reward = 0
        for i_reward, i_done in zip(reversed(raw_rewards), reversed(raw_dones)):
            if i_done:
                discounted_reward = 0
            discounted_reward = i_reward + (self.gamma * discounted_reward)
            mc_rewards.insert(0, discounted_reward)

        old_states = torch.tensor(raw_states, dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(raw_actions, dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(raw_logprobs, dtype=torch.float32).to(self.device)
        # normalizing the rewards
        mc_rewards = torch.tensor(mc_rewards, dtype=torch.float32).to(self.device)
        mc_rewards = (mc_rewards - mc_rewards.mean()) / (mc_rewards.std() + 1e-7)
        old_state_values = torch.tensor(raw_state_values, dtype=torch.float32).to(self.device)

        advantages = mc_rewards.detach() - old_state_values.detach()

        # optimize policy for k epochs
        for i_epoch in range(self.k_epochs):
            # input: old_states, old_actions
            actions_mean = self.actor(old_states)
            cov_mat = torch.diag(self.actor.action_var)
            cov_mat = cov_mat.expand(actions_mean.shape[0], cov_mat.shape[0], cov_mat.shape[1])
            # actions_var = self.actor.action_var.expand_as(actions_mean).to(self.device)
            dist = MultivariateNormal(actions_mean, cov_mat)
            if self.actor.action_dim == 1:
                old_actions = old_actions.reshape(-1, self.action_dim)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = self.critic(old_states).squeeze()

            # finding the ratio: (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = - torch.min(surr1, surr2) + 0.5 * self.MSELoss(state_values, mc_rewards) - 0.01 * dist_entropy

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.mean().backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        # UPDATE TARGET NETWORKS
        self._update_target_networks()

        self.memory.clear()
