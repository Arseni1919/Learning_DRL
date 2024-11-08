import numpy as np
import torch
from torch.distributions import Normal

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


class ValueNetwork(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, lr: float = 1e-4):
        super(ValueNetwork, self).__init__()
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


class SoftQNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, lr: float = 1e-4):
        super(SoftQNetwork, self).__init__()
        self.learning_rate: float = lr

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, log_std_min: float = -20, log_std_max: float = 2, lr: float = 1e-4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.learning_rate: float = lr
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        action = action.cpu()
        return action[0]


class SAC:

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            actor_lr: float,
            value_lr: float,
            soft_q_lr: float,
            gamma: float,
            soft_tau: float,
            buffer_size: int,
            batch_size: int,
            device: str
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.value_lr = value_lr
        self.soft_q_lr = soft_q_lr
        self.gamma: float = gamma
        self.soft_tau: float = soft_tau
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.device = device
        self.log_std_min: int = -20
        self.log_std_max: int = 2

        self.actor: Actor = Actor(state_dim, action_dim, hidden_dim, self.log_std_min, self.log_std_max, self.actor_lr).to(device)
        self.value_nn: ValueNetwork = ValueNetwork(state_dim, hidden_dim, self.value_lr).to(device)
        self.value_nn_target: ValueNetwork = ValueNetwork(state_dim, hidden_dim, self.value_lr).to(device)
        self.soft_q_nn_1: SoftQNetwork = SoftQNetwork(state_dim, action_dim, hidden_dim, self.soft_q_lr).to(device)
        self.soft_q_nn_2: SoftQNetwork = SoftQNetwork(state_dim, action_dim, hidden_dim, self.soft_q_lr).to(device)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self._update_target_networks()  # initialize target networks

        self.MSELoss = nn.MSELoss()

    def _update_target_networks(self, tau: float = 1.0):

        for target_param, param in zip(self.value_nn_target.parameters(), self.value_nn.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
            # target_param.data.copy_(param.data)

    def act(self, state):
        with torch.no_grad():
            state: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state).detach()
            action = action.detach().cpu().numpy().flatten()
        return action

    def get_pure_action(self, state, *args, **kwargs):
        with torch.no_grad():
            state: torch.Tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state).detach()
            action = action.detach().cpu().numpy().flatten()
        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

        predicted_q_values_1 = self.soft_q_nn_1(states, actions)
        predicted_q_values_2 = self.soft_q_nn_2(states, actions)
        predicted_value = self.value_nn(states)
        new_actions, log_probs, z_values, means, log_stds = self.actor.evaluate(states)

        # Training Q function
        target_values = self.value_nn_target(next_states).squeeze().unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * target_values

        q_value_loss_1 = self.MSELoss(predicted_q_values_1, target_q_values.detach())
        self.soft_q_nn_1.optimizer.zero_grad()
        q_value_loss_1.backward()
        self.soft_q_nn_1.optimizer.step()

        q_value_loss_2 = self.MSELoss(predicted_q_values_2, target_q_values.detach())
        self.soft_q_nn_2.optimizer.zero_grad()
        q_value_loss_2.backward()
        self.soft_q_nn_2.optimizer.step()

        # Training Value function
        predicted_new_q_values = torch.min(self.soft_q_nn_1(states, new_actions), self.soft_q_nn_2(states, new_actions))
        target_values_func = predicted_new_q_values - log_probs
        value_loss = self.MSELoss(predicted_value, target_values_func.detach())

        self.value_nn.optimizer.zero_grad()
        value_loss.backward()
        self.value_nn.optimizer.step()

        # Training Actor function
        actor_loss = (log_probs - predicted_new_q_values).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self._update_target_networks(tau=self.soft_tau)




