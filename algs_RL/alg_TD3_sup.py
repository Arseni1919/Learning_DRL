import torch

from globals import *
from plot_functions import *
from global_functions import *


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.memory = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, lr: float = 1e-4):
        super(Actor, self).__init__()
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

    def forward(self, x):
        out = self.nn(x)
        return out


class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, lr: float = 1e-4):
        super(Critic, self).__init__()
        self.learning_rate: float = lr

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x


class TD3:

    def __init__(
            self,
            state_dim, action_dim, hidden_dim,
            buffer_size: int, batch_size: int,
            actor_lr: float, critic_lr: float, tau: float, gamma: float, device: str
    ):
        self.actor: Actor = Actor(state_dim, hidden_dim, action_dim, actor_lr).to(device)
        self.actor_target: Actor = Actor(state_dim, hidden_dim, action_dim, actor_lr).to(device)
        self.critic1: Critic = Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)
        self.critic1_target: Critic= Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)
        self.critic2: Critic = Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)
        self.critic2_target: Critic = Critic(state_dim, action_dim, hidden_dim, critic_lr).to(device)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size: int = batch_size
        self.tau: float = tau
        self.gamma: float = gamma
        self.device = device

        self._update_target_networks(tau=1)  # initialize target networks

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, state, noise=0.0):
        state: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().to('cpu').numpy()[0]
        action = np.clip(action + noise, -1, 1)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # UPDATE CRITIC
        self.critic1.optimizer.zero_grad()

        with torch.no_grad():
            next_actions: torch.Tensor = self.actor_target(next_states)
            noise = torch.normal(0, 0.2, size=next_actions.shape)
            noise = torch.clip(noise, -0.5, 0.5)
            next_actions = torch.clip(next_actions + noise, -1, 1)
            target1_q_values = self.critic1_target(next_states, next_actions)
            target2_q_values = self.critic2_target(next_states, next_actions)
            min_target = torch.min(target1_q_values, target2_q_values)
            target_q_values = rewards + (1 - dones) * self.gamma * min_target

        current1_q_values = self.critic1(states, actions)
        critic1_loss = nn.MSELoss()(current1_q_values, target_q_values)
        critic1_loss.backward()
        self.critic1.optimizer.step()

        current2_q_values = self.critic2(states, actions)
        critic2_loss = nn.MSELoss()(current2_q_values, target_q_values)
        critic2_loss.backward()
        self.critic2.optimizer.step()

        # UPDATE ACTOR
        self.actor.optimizer.zero_grad()

        actor_loss = - self.critic1(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # UPDATE TARGET NETWORKS
        self._update_target_networks()
