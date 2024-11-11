from algs_RL.alg_Double_DQN_sup import *
from algs_RL.alg_DQN_sup import *


import matplotlib.pyplot as plt
import torch
from torch.onnx.symbolic_opset11 import unsqueeze

from alg_DQN_sup import *


def optimize_model(
        memory: ReplayMemory,
        device: torch.device,
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
        policy_actions = policy_net(non_final_next_states).argmax(1).unsqueeze(1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, policy_actions).squeeze()
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)
    criterion = nn.SmoothL1Loss()
    loss: nn.Module = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():

    # PARAMETERS
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        n_episodes = 600
    else:
        n_episodes = 50
    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: float = 1000
    TAU: float = 0.005  # update rate of the target network
    LR: float = 1e-4  # learning rate of the AdamW optimizer
    # render_mode="rgb_array"
    render_mode = "human"
    # render_mode = None
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())
    # for plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_rate = 0.001
    episode_durations: List[int] = []

    # INIT ENV
    # env = gym.make("LunarLander-v3", render_mode="human")
    # env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
    # env = gym.make('CliffWalking-v0', render_mode="human")
    # env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # PARAMETERS AFTER ENV INIT
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    policy_net: nn.Module = DQN(n_observations, n_actions).to(device)
    target_net: nn.Module = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer: optim.AdamW = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory: ReplayMemory = ReplayMemory(10000)

    steps_done: int = 0

    for i_episode in range(n_episodes):
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        for t in count():
            # main interation with an env
            action: torch.Tensor = select_action(state, steps_done, policy_net, env, device, EPS_START, EPS_DECAY, EPS_END)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done: bool = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
            steps_done += 1
            # aftermath
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(memory, device, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
            if done:
                episode_durations.append(t + 1)
                plot_sr(ax[0], info={'episode_durations': episode_durations})
                plt.pause(plot_rate)
                break
            

    print('Complete')
    env.close()
    plt.show()


if __name__ == '__main__':
    main()