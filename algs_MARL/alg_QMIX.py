import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from smac.env import StarCraft2Env


# Define the Agent's Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Mixer(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(Mixer, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Hypernetworks for weights and biases
        self.hyper_w1 = nn.Linear(state_dim, self.n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, states):
        """
        agent_qs: Tensor of shape [batch_size, n_agents]
        states: Tensor of shape [batch_size, state_dim]
        """
        # bs = agent_qs.size(0)  # batch size
        bs = 1

        # Ensure agent_qs is reshaped to [batch_size, n_agents, 1]
        # agent_qs = agent_qs.view(bs, self.n_agents, 1)
        agent_qs.unsqueeze(0)

        # Compute w1, b1, w2, b2
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.hidden_dim)

        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)

        # Forward pass through the Mixer
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)  # Shape: [batch_size, 1, hidden_dim]
        q_total = torch.bmm(hidden, w2) + b2  # Shape: [batch_size, 1, 1]

        return q_total.squeeze(-1)  # Shape: [batch_size]


# QMIX Algorithm
class QMIX:
    def __init__(self, env, n_agents, state_dim, obs_dim, n_actions, lr=0.001, gamma=0.99):
        self.env = env
        self.n_agents = n_agents
        self.gamma = gamma
        self.lr = lr

        # Agent and Mixer Networks
        self.q_networks = [QNetwork(obs_dim, n_actions) for _ in range(n_agents)]
        self.mixer = Mixer(n_agents, state_dim)
        self.target_mixer = Mixer(n_agents, state_dim)

        # Optimizers
        self.agent_optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.q_networks]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr)

    def get_joint_q_values(self, observations, state):
        agent_qs = torch.stack([net(obs) for net, obs in zip(self.q_networks, observations)], dim=1)
        q_total = self.mixer(agent_qs, state)
        return q_total

    def update(self, batch):
        observations, actions, rewards, next_observations, state = batch
        # Compute current Q-values
        current_q_total = self.get_joint_q_values(observations, state)

        # Compute target Q-values
        target_q_total = rewards + self.gamma * self.get_joint_q_values(next_observations, state)

        # Loss and optimization
        loss = torch.nn.functional.mse_loss(current_q_total, target_q_total)
        self.mixer_optimizer.zero_grad()
        _ = [opt.zero_grad() for opt in self.agent_optimizers]
        loss.backward()
        self.mixer_optimizer.step()
        _ = [opt.step() for opt in self.agent_optimizers]


# Main Training Loop
def train_qmix():
    env = StarCraft2Env(map_name="2m_vs_1z")
    env.reset()

    n_agents = env.n_agents
    state_dim = env.get_state().shape[0]
    obs_dim = env.get_obs_agent(0).shape[0]
    n_actions = env.n_actions

    qmix = QMIX(env, n_agents, state_dim, obs_dim, n_actions)

    for episode in range(1000):
        done = False
        env.reset()
        episode_reward = 0

        while not done:
            observations = [torch.tensor(env.get_obs_agent(i), dtype=torch.float32) for i in range(n_agents)]
            state = torch.tensor(env.get_state(), dtype=torch.float32)

            # Select valid actions
            actions = []
            for i in range(n_agents):
                q_values = qmix.q_networks[i](observations[i])
                avail_actions = env.get_avail_agent_actions(i)
                avail_actions = torch.tensor(avail_actions, dtype=torch.bool)

                # Mask invalid actions
                masked_q_values = q_values.clone()
                masked_q_values[~avail_actions] = -float('inf')

                action = torch.argmax(masked_q_values).item()
                actions.append(action)

            # Step in the environment
            reward, done, _ = env.step(actions)
            episode_reward += reward

            # Dummy update with one-step batch
            new_observations = [torch.tensor(env.get_obs_agent(i), dtype=torch.float32) for i in range(n_agents)]
            qmix.update((observations, actions, reward, new_observations, state))

        print(f"Episode {episode} | Total Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    train_qmix()




######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from smac.env import StarCraft2Env
from typing import List, Tuple


class QMixNetwork(nn.Module):
    def __init__(self, state_dim, n_agents, mixing_embed_dim=32):
        super(QMixNetwork, self).__init__()

        # Hypernetwork for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim * n_agents)
        )

        # Hypernetwork for bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

        # Final layer
        self.V = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

        self.n_agents = n_agents

    def forward(self, indiv_qs, states):
        batch_size = indiv_qs.size(0)

        # Generate weights and bias
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.n_agents, 1)
        b1 = self.hyper_b1(states).view(batch_size, 1)

        # Compute weighted sum of individual Q-values
        hidden = F.elu(torch.bmm(indiv_qs.unsqueeze(-2), w1).squeeze(-2) + b1)

        # Value function
        v = self.V(states)

        # Final Q-value
        q_tot = hidden + v

        return q_tot


class QMixAgent:
    def __init__(self,
                 state_dim,
                 action_dims,
                 n_agents,
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_decay=0.99,
                 epsilon_min=0.05,
                 buffer_size=10000,
                 batch_size=32):

        self.state_dim = state_dim
        self.action_dims = action_dims
        self.n_agents = n_agents

        # Q-Networks (local and target)
        self.q_networks = self._build_q_networks()
        self.target_q_networks = [net.clone() for net in self.q_networks]

        # Mixing Network
        self.mixer = QMixNetwork(state_dim, n_agents)
        self.target_mixer = QMixNetwork(state_dim, n_agents)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.q_networks[0].parameters()},
            {'params': self.mixer.parameters()}
        ], lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Replay Buffer
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def _build_q_networks(self):
        # Create separate Q-networks for each agent with their specific action space
        return [
            nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            ) for action_dim in self.action_dims
        ]

    def select_actions(self, state, available_actions):
        # Epsilon-greedy action selection respecting available actions
        actions = []

        for agent_id, (q_network, max_action) in enumerate(zip(self.q_networks, self.action_dims)):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state)

            if random.random() < self.epsilon:
                # Random action from available actions for this agent
                agent_available_actions = [a for a in range(max_action) if a in available_actions[agent_id]]
                action = random.choice(agent_available_actions)
            else:
                # Compute Q-values
                q_values = q_network(state_tensor)

                # Filter Q-values by available actions
                available_q_values = q_values[[a for a in range(max_action) if a in available_actions[agent_id]]]

                # Select max action from available actions
                action = available_actions[agent_id][available_q_values.argmax().item()]

            actions.append(action)

        return actions

    def store_transition(self, state, actions, rewards, next_state, done):
        # Store transition in replay buffer
        self.memory.append((state, actions, rewards, next_state, done))

        # Limit buffer size
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values for each agent
        current_q_values = torch.stack([
            q_network(states).gather(1, actions[:, agent_id].unsqueeze(1)).squeeze(1)
            for agent_id, q_network in enumerate(self.q_networks)
        ], dim=1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = torch.stack([
                q_network(next_states).max(dim=1)[0]
                for q_network in self.target_q_networks
            ], dim=1)

            # Compute target Q-tot
            target_q_tot = self.target_mixer(next_q_values.unsqueeze(-1), next_states)

            # Compute current Q-tot
            current_q_tot = self.mixer(current_q_values.unsqueeze(-1), states)

            # Target computation
            target = rewards.unsqueeze(-1) + self.gamma * target_q_tot * (1 - dones.unsqueeze(-1))

        # Compute loss
        loss = F.mse_loss(current_q_tot, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.q_networks[0].parameters()) + list(self.mixer.parameters()),
            max_norm=10
        )

        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_networks(self):
        # Soft update of target networks
        tau = 0.005
        for target_net, net in zip(self.target_q_networks, self.q_networks):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def train_qmix_on_smac(map_name='3m', episodes=5000, max_steps=2000):
    # Initialize SMAC environment
    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()

    # Get environment dimensions
    state_dim = env_info["state_shape"]
    n_agents = env_info["n_agents"]

    # Get action dimensions for each agent
    action_dims = [env.get_total_actions()]
    for agent_id in range(n_agents):
        env.reset()
        env.get_avail_agent_actions(agent_id)

    # Initialize QMix Agent
    qmix_agent = QMixAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        n_agents=n_agents
    )

    # Training loop
    for episode in range(episodes):
        # Reset environment
        env.reset()
        terminated = False
        episode_reward = 0

        for step in range(max_steps):
            # Get current state
            state = env.get_state()

            # Get available actions for each agent
            available_actions = [env.get_avail_agent_actions(agent_id) for agent_id in range(n_agents)]

            # Select actions
            actions = qmix_agent.select_actions(state, available_actions)

            # Execute actions
            reward, terminated, _ = env.step(actions)

            # Get next state
            next_state = env.get_state()

            # Store transition
            qmix_agent.store_transition(state, actions, reward, next_state, terminated)

            # Train agent
            loss = qmix_agent.train()

            # Update target networks periodically
            if step % 100 == 0:
                qmix_agent.update_target_networks()

            episode_reward += reward

            if terminated:
                break

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {episode_reward}")

    env.close()


# Run training
if __name__ == "__main__":
    train_qmix_on_smac()