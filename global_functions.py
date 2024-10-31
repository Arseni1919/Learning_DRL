from globals import *


def run_CartPole(select_action, ac_net, device, render_mode: str = 'human', n_episodes: int = 1):
    env = gym.make("CartPole-v1", render_mode=render_mode)

    for i_episode in range(n_episodes):
        state, _ = env.reset()

        for t in count():
            action = select_action(state, ac_net, device)
            state, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            if done:
                break
