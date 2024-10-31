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


class DDPG:

    def __init__(
            self,
            state_dim, action_dim, hidden_dim,
            buffer_size, batch_size,
            actor_lr, critic_lr, tau, gamma
    ):
        # self.actor: nn.Module = Actor()
        pass